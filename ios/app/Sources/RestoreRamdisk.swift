import Foundation

/// The restore ramdisk, with our own two programs inside it.
///
/// After a restore writes the system, ChefKiss's guide has a person mount the
/// result on a Mac and run the filesystem patcher over the shared cache. That
/// step is the last one that needs a computer, and it does not have to: the
/// same patcher can run in the guest, in the same ramdisk boot, the moment the
/// restore is done writing. What it needs is to be in the ramdisk.
///
/// The ramdisk is an `IM4P` wrapping a raw HFSX volume, and adding a file to
/// HFS+ means inserting into its catalog B-tree -- splitting nodes, allocating
/// blocks, rewriting headers. Replacing the contents of a file that is already
/// there needs none of that: the catalog record stays where it is, the blocks
/// stay allocated to it, and all that changes are the bytes in those blocks and
/// two fields in the record. So this replaces three files the restore has no
/// use for:
///
/// - two NFC firmware blobs, megabytes each, for a secure element this machine
///   does not have;
/// - `com.apple.syslogd.plist`, a launchd job the ramdisk ships **disabled**,
///   which becomes the job that starts our daemon.
///
/// The plist needs one thing more. Nearly every file in the ramdisk is HFS+
/// compressed -- the contents live in an extended attribute and the data fork
/// is empty -- so there is nothing to overwrite. Clearing the compressed flag
/// and giving the record a block off the volume's own free list fixes that, and
/// is still only a matter of writing fields that are already there. The two
/// firmware blobs are among the handful of files stored plainly, and their
/// blocks are kept as they are.
///
/// Nothing is added, nothing is deleted, and the image keeps its exact size --
/// so the result goes back into the IM4P it came from without re-wrapping it.
enum RestoreRamdisk {
    /// Where our programs go, and what they replace.
    ///
    /// The paths matter to the daemon too: it execs the patcher by path, and
    /// the job plist names the daemon by path.
    static let daemonPath =
        "/usr/standalone/firmware/nfrestore/firmware/jcop-prod/JCOP-01.30-006-P.bin"
    static let patcherPath =
        "/usr/standalone/firmware/nfrestore/firmware/jcop-prod/JCOP-11.04-012.2-P.bin"
    static let jobPath = "/System/Library/LaunchDaemons/com.apple.syslogd.plist"

    /// The job that starts the daemon. `RunAtLoad`, because nothing else in the
    /// ramdisk would ever ask for it.
    static var job: Data {
        Data("""
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
        "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
        \t<key>Label</key>
        \t<string>com.orchard.patcher</string>
        \t<key>ProgramArguments</key>
        \t<array>
        \t\t<string>\(daemonPath)</string>
        \t</array>
        \t<key>RunAtLoad</key>
        \t<true/>
        \t<key>StandardOutPath</key>
        \t<string>/dev/console</string>
        \t<key>StandardErrorPath</key>
        \t<string>/dev/console</string>
        </dict>
        </plist>

        """.utf8)
    }

    enum Failure: LocalizedError {
        case notIM4P
        case notHFS
        case missing(String)
        case tooBig(String, Int, Int)
        case noPrograms
        case noRoom

        var errorDescription: String? {
            switch self {
            case .notIM4P: return L("RAM-диск не IM4P — не тот файл?")
            case .notHFS:  return L("В RAM-диске не найден том HFS+")
            case .missing(let path): return L("В RAM-диске нет файла %@", path)
            case .tooBig(let path, let size, let room):
                return L("Не влезает в %@: %d байт при месте на %d", path, size, room)
            case .noPrograms: return L("В приложении нет программ для гостя")
            case .noRoom: return L("В RAM-диске не осталось свободных блоков")
            }
        }
    }

    /// The two programs, as the build puts them into the app.
    static var programs: (daemon: URL, patcher: URL)? {
        let bundle = Bundle.main.bundleURL.appendingPathComponent("guest")
        let daemon = bundle.appendingPathComponent("orchard_patcher")
        let patcher = bundle.appendingPathComponent("orchard_fs_patcher")
        guard FileManager.default.fileExists(atPath: daemon.path),
              FileManager.default.fileExists(atPath: patcher.path)
        else { return nil }
        return (daemon, patcher)
    }

    /// Makes the patched ramdisk out of the stock one. The result is a copy;
    /// the original from the IPSW is left alone.
    static func build(stock: URL, into destination: URL,
                      note: (String) -> Void = { _ in }) throws {
        guard let programs = programs else { throw Failure.noPrograms }

        try? FileManager.default.removeItem(at: destination)
        try copy(stock, to: destination)

        let handle = try FileHandle(forUpdating: destination)
        defer { try? handle.close() }

        let volume = try Volume(handle: handle, base: try payloadOffset(of: handle))
        try volume.replace(jobPath, with: job, mode: 0o644)
        try volume.replace(daemonPath, with: try Data(contentsOf: programs.daemon), mode: 0o755)
        try volume.replace(patcherPath, with: try Data(contentsOf: programs.patcher), mode: 0o755)
        try handle.synchronize()
        note(L("Подготовка: RAM-диск с патчером собран"))
    }

    /// A copy that does not hold the whole hundred megabytes in memory.
    private static func copy(_ source: URL, to destination: URL) throws {
        guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
            throw Failure.missing(destination.lastPathComponent)
        }
        let reader = try FileHandle(forReadingFrom: source)
        defer { try? reader.close() }
        let writer = try FileHandle(forWritingTo: destination)
        defer { try? writer.close() }
        var done = false
        while !done {
            try autoreleasepool {
                guard let chunk = try reader.read(upToCount: 8 << 20), !chunk.isEmpty else {
                    done = true
                    return
                }
                try writer.write(contentsOf: chunk)
            }
        }
    }

    /// Where the HFSX volume starts inside the `IM4P`.
    ///
    /// The container is a SEQUENCE of "IM4P", a four-character type, a version
    /// string and the payload as an OCTET STRING; the volume is the payload's
    /// contents, uncompressed for a ramdisk. Walking the header by hand keeps a
    /// hundred megabytes off the heap -- the parser in `IMG4` would have to
    /// hold the payload to hand it back.
    private static func payloadOffset(of handle: FileHandle) throws -> UInt64 {
        try handle.seek(toOffset: 0)
        guard let header = try handle.read(upToCount: 256) else { throw Failure.notIM4P }
        var at = 0
        func element() throws -> (tag: UInt8, start: Int, length: Int) {
            guard at + 2 <= header.count else { throw Failure.notIM4P }
            let tag = header[header.startIndex + at]
            var length = Int(header[header.startIndex + at + 1])
            at += 2
            if length & 0x80 != 0 {
                let count = length & 0x7F
                guard count > 0, count <= 4, at + count <= header.count else { throw Failure.notIM4P }
                length = 0
                for i in 0..<count { length = (length << 8) | Int(header[header.startIndex + at + i]) }
                at += count
            }
            return (tag, at, length)
        }

        let outer = try element()                       // SEQUENCE
        guard outer.tag == 0x30 else { throw Failure.notIM4P }
        let magic = try element()                       // IA5String "IM4P"
        guard magic.tag == 0x16, magic.length == 4,
              String(decoding: header[(header.startIndex + magic.start)..<(header.startIndex + magic.start + 4)],
                     as: UTF8.self) == "IM4P"
        else { throw Failure.notIM4P }
        at = magic.start + magic.length
        let type = try element()                        // IA5String, the four-character type
        at = type.start + type.length
        let version = try element()                     // IA5String
        at = version.start + version.length
        let payload = try element()                     // OCTET STRING
        guard payload.tag == 0x04 else { throw Failure.notIM4P }
        return UInt64(payload.start)
    }

    // MARK: - HFS+

    /// Just enough HFS+ to find a file and overwrite it where it lies.
    ///
    /// Everything here is big-endian and fixed-width, which is what makes the
    /// whole thing short: the volume header says where the catalog is, the
    /// catalog's leaves are a linked list, and a file record carries its own
    /// extents. Nothing is allocated and no key is ever compared -- the leaves
    /// are read in order and the names matched as they come.
    private final class Volume {
        private let handle: FileHandle
        private let base: UInt64
        private let blockSize: UInt32
        private let totalBlocks: UInt32
        private let allocation: Fork
        private let catalog: Fork
        private let nodeSize: UInt32
        private let firstLeaf: UInt32

        struct Fork {
            var logicalSize: UInt64
            var totalBlocks: UInt32
            var extents: [(start: UInt32, count: UInt32)]
        }

        /// A file the catalog holds: where its record is in the image, and
        /// where its contents are.
        struct Entry {
            var isFolder: Bool
            var id: UInt32
            var recordOffset: UInt64     // of the record's data, in the image
            var fork: Fork
        }

        init(handle: FileHandle, base: UInt64) throws {
            self.handle = handle
            self.base = base
            let header = try Volume.read(handle, at: base + 1024, count: 512)
            let signature = String(decoding: header[0..<2], as: UTF8.self)
            guard signature == "H+" || signature == "HX" else { throw Failure.notHFS }
            self.blockSize = Volume.be32(header, 40)
            self.totalBlocks = Volume.be32(header, 44)
            self.allocation = Volume.fork(header, 112)
            self.catalog = Volume.fork(header, 272)

            // Node zero carries the tree's header record, and every offset in
            // the tree is counted in the node size it states -- so node zero
            // itself has to be found without one: it starts where the fork
            // starts.
            let start = base + UInt64(catalog.extents.first?.start ?? 0) * UInt64(blockSize)
            let node = try Volume.read(handle, at: start, count: 512)
            self.nodeSize = UInt32(Volume.be16(node, 14 + 18))
            self.firstLeaf = Volume.be32(node, 14 + 10)
        }

        /// Replaces a file's contents. A file with blocks enough for them keeps
        /// the blocks it has; one without -- a compressed file keeps nothing in
        /// its data fork -- is given blocks off the volume's free list.
        func replace(_ path: String, with data: Data, mode: UInt16) throws {
            guard let entry = try find(path), !entry.isFolder else { throw Failure.missing(path) }
            var extents = entry.fork.extents
            let room = extents.reduce(0) { $0 + Int($1.count) * Int(blockSize) }
            if room < data.count {
                let needed = UInt32((data.count + Int(blockSize) - 1) / Int(blockSize))
                extents = [(try allocate(needed), needed)]
                // The fork's own fields: how many blocks it holds, and where
                // they are. The seven extents after the first are cleared, so
                // nothing of what the record used to say is left behind.
                try Volume.write(handle, at: entry.recordOffset + 100, Data(Volume.bytes32(needed)))
                var descriptors = Volume.bytes32(extents[0].start) + Volume.bytes32(needed)
                descriptors += [UInt8](repeating: 0, count: 7 * 8)
                try Volume.write(handle, at: entry.recordOffset + 104, Data(descriptors))
                // Whatever the old blocks were, they are not this file's any
                // more. They stay marked as taken: this image is made for one
                // restore and thrown away with it, and a leak inside it costs
                // nothing next to the code that would tidy it up.
            }

            var written = 0
            for extent in extents where written < data.count {
                let chunk = min(Int(extent.count) * Int(blockSize), data.count - written)
                let start = data.index(data.startIndex, offsetBy: written)
                try Volume.write(handle, at: base + UInt64(extent.start) * UInt64(blockSize),
                                 data[start..<data.index(start, offsetBy: chunk)])
                written += chunk
            }

            // What is left of the record: the length of what is in the blocks,
            // the mode -- a launchd job has to be readable and a program has to
            // be executable, and a firmware blob is neither -- root as the
            // owner, and the compressed flag off, because the contents are no
            // longer in an extended attribute and the kernel must not look for
            // them there.
            try Volume.write(handle, at: entry.recordOffset + 88, Data(Volume.bytes64(UInt64(data.count))))
            try Volume.write(handle, at: entry.recordOffset + 42, Data(Volume.bytes16(mode | 0o100000)))
            try Volume.write(handle, at: entry.recordOffset + 32, Data(Volume.bytes64(0)))   // owner, group
            try Volume.write(handle, at: entry.recordOffset + 41, Data([0]))                 // ownerFlags
        }

        /// Takes a run of blocks the volume is not using, and marks it taken.
        ///
        /// The allocation file is a bit per block, most significant bit first;
        /// the volume header's count of free blocks has to follow.
        private func allocate(_ blocks: UInt32) throws -> UInt32 {
            var bitmap = try read(allocation, count: Int((totalBlocks + 7) / 8))
            var run = 0
            var start = 0
            for block in 0..<Int(totalBlocks) {
                let mask = UInt8(0x80) >> UInt8(block & 7)
                if bitmap[block >> 3] & mask != 0 {
                    run = 0
                    continue
                }
                if run == 0 { start = block }
                run += 1
                guard run == Int(blocks) else { continue }
                for taken in start..<(start + run) {
                    bitmap[taken >> 3] |= UInt8(0x80) >> UInt8(taken & 7)
                }
                try write(allocation, Data(bitmap))
                let free = Volume.be32(try Volume.read(handle, at: base + 1024 + 48, count: 4), 0)
                try Volume.write(handle, at: base + 1024 + 48, Data(Volume.bytes32(free - blocks)))
                return UInt32(start)
            }
            throw Failure.noRoom
        }

        /// Walks a path from the root, one component at a time.
        func find(_ path: String) throws -> Entry? {
            var parent: UInt32 = 2                       // the root folder's own id
            var entry: Entry?
            for name in path.split(separator: "/") {
                guard let found = try lookup(String(name), in: parent) else { return nil }
                parent = found.id
                entry = found
            }
            return entry
        }

        /// Reads the leaves in order looking for one name under one parent.
        private func lookup(_ name: String, in parent: UInt32) throws -> Entry? {
            let wanted = Array(name.utf16)
            var index = firstLeaf
            while index != 0 {
                let node = try Volume.read(handle, at: imageOffset(ofNode: index), count: Int(nodeSize))
                let next = Volume.be32(node, 0)
                let records = Int(Volume.be16(node, 10))
                for record in 0..<records {
                    let offset = Int(Volume.be16(node, Int(nodeSize) - (record + 1) * 2))
                    guard offset + 8 < node.count else { continue }
                    let keyLength = Int(Volume.be16(node, offset))
                    guard Volume.be32(node, offset + 2) == parent else { continue }
                    let nameLength = Int(Volume.be16(node, offset + 6))
                    guard nameLength == wanted.count else { continue }
                    var chars = [UInt16]()
                    chars.reserveCapacity(nameLength)
                    for i in 0..<nameLength { chars.append(Volume.be16(node, offset + 8 + i * 2)) }
                    guard chars == wanted else { continue }

                    // The record's data follows the key, aligned to two bytes.
                    var at = offset + 2 + keyLength
                    if at % 2 != 0 { at += 1 }
                    let type = Volume.be16(node, at)
                    let recordOffset = imageOffset(ofNode: index) + UInt64(at)
                    if type == 1 {                       // a folder
                        return Entry(isFolder: true, id: Volume.be32(node, at + 8),
                                     recordOffset: recordOffset,
                                     fork: Fork(logicalSize: 0, totalBlocks: 0, extents: []))
                    }
                    if type == 2 {                       // a file
                        return Entry(isFolder: false, id: Volume.be32(node, at + 8),
                                     recordOffset: recordOffset,
                                     fork: Volume.fork(node, at + 88))
                    }
                }
                index = next
            }
            return nil
        }

        // MARK: Reading the layout

        /// Reads a fork whole. Only the small ones are read this way -- the
        /// allocation bitmap of a hundred-megabyte volume is a few kilobytes.
        private func read(_ fork: Fork, count: Int) throws -> [UInt8] {
            var bytes = [UInt8]()
            var read = 0
            for extent in fork.extents where read < count {
                let chunk = min(Int(extent.count) * Int(blockSize), count - read)
                bytes += try Volume.read(handle, at: base + UInt64(extent.start) * UInt64(blockSize),
                                         count: chunk)
                read += chunk
            }
            return bytes
        }

        private func write(_ fork: Fork, _ data: Data) throws {
            var written = 0
            for extent in fork.extents where written < data.count {
                let chunk = min(Int(extent.count) * Int(blockSize), data.count - written)
                let start = data.index(data.startIndex, offsetBy: written)
                try Volume.write(handle, at: base + UInt64(extent.start) * UInt64(blockSize),
                                 data[start..<data.index(start, offsetBy: chunk)])
                written += chunk
            }
        }

        /// Where a byte offset inside the catalog fork sits in the image. The
        /// fork is a list of extents of allocation blocks, so an offset past
        /// the first one carries over into the next.
        private func imageOffset(inCatalog offset: UInt64) -> UInt64 {
            var seen: UInt64 = 0
            for extent in catalog.extents {
                let length = UInt64(extent.count) * UInt64(blockSize)
                if offset < seen + length {
                    return base + UInt64(extent.start) * UInt64(blockSize) + (offset - seen)
                }
                seen += length
            }
            return base + UInt64(catalog.extents.first?.start ?? 0) * UInt64(blockSize)
        }

        /// Where a node of the catalog starts in the image.
        private func imageOffset(ofNode node: UInt32) -> UInt64 {
            imageOffset(inCatalog: UInt64(node) * UInt64(nodeSize))
        }

        private static func fork(_ bytes: [UInt8], _ at: Int) -> Fork {
            var extents = [(start: UInt32, count: UInt32)]()
            for i in 0..<8 {
                let start = be32(bytes, at + 16 + i * 8)
                let count = be32(bytes, at + 20 + i * 8)
                if count != 0 { extents.append((start, count)) }
            }
            return Fork(logicalSize: be64(bytes, at), totalBlocks: be32(bytes, at + 12), extents: extents)
        }

        private static func be16(_ bytes: [UInt8], _ at: Int) -> UInt16 {
            guard at + 2 <= bytes.count else { return 0 }
            return UInt16(bytes[at]) << 8 | UInt16(bytes[at + 1])
        }

        private static func be32(_ bytes: [UInt8], _ at: Int) -> UInt32 {
            guard at + 4 <= bytes.count else { return 0 }
            return (0..<4).reduce(UInt32(0)) { $0 << 8 | UInt32(bytes[at + $1]) }
        }

        private static func be64(_ bytes: [UInt8], _ at: Int) -> UInt64 {
            guard at + 8 <= bytes.count else { return 0 }
            return (0..<8).reduce(UInt64(0)) { $0 << 8 | UInt64(bytes[at + $1]) }
        }

        private static func bytes16(_ value: UInt16) -> [UInt8] { [UInt8(value >> 8), UInt8(value & 0xFF)] }

        private static func bytes32(_ value: UInt32) -> [UInt8] {
            (0..<4).reversed().map { UInt8((value >> (8 * UInt32($0))) & 0xFF) }
        }

        private static func bytes64(_ value: UInt64) -> [UInt8] {
            (0..<8).reversed().map { UInt8((value >> (8 * UInt64($0))) & 0xFF) }
        }

        private static func read(_ handle: FileHandle, at offset: UInt64, count: Int) throws -> [UInt8] {
            try handle.seek(toOffset: offset)
            return Array(try handle.read(upToCount: count) ?? Data())
        }

        private static func write(_ handle: FileHandle, at offset: UInt64, _ data: Data) throws {
            try handle.seek(toOffset: offset)
            try handle.write(contentsOf: data)
        }
    }
}
