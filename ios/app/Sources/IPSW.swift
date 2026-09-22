import Foundation
import Compression

/// The IPSW, read where it lies.
///
/// A restore firmware is six gigabytes and the phone has better uses for them:
/// unpacking it whole would need the space twice over. So nothing here unpacks.
/// The central directory is read, the few small components a restore personalises
/// are pulled out one at a time, and the filesystem image — which Apple stores
/// uncompressed — is handed to the restore as a range of the original file, read
/// as it goes over the wire.
///
/// Zip64 is not optional here: the image alone is over four gigabytes, so the
/// sizes and offsets live in the extra field rather than in the header.
struct IPSW {
    struct Member {
        var path: String
        /// 0 stored, 8 deflate.
        var method: Int
        var compressedSize: Int
        var plainSize: Int
        /// Where the compressed bytes start, past the local header.
        var dataOffset: Int

        var isStored: Bool { method == 0 }
    }

    enum Failure: LocalizedError {
        case notZip
        case corrupt(String)
        case missing(String)
        case unsupported(String)

        var errorDescription: String? {
            switch self {
            case .notZip:              return L("Это не архив прошивки")
            case .corrupt(let what):   return L("Архив повреждён: %@", what)
            case .missing(let what):   return L("В прошивке нет файла %@", what)
            case .unsupported(let w):  return L("В архиве есть то, что я не умею разбирать: %@", w)
            }
        }
    }

    let url: URL
    private(set) var members: [String: Member] = [:]

    init(url: URL) throws {
        self.url = url
        try readDirectory()
    }

    // MARK: - The central directory

    private static func u16(_ d: [UInt8], _ at: Int) -> Int { Int(d[at]) | Int(d[at + 1]) << 8 }
    private static func u32(_ d: [UInt8], _ at: Int) -> Int {
        (0..<4).reduce(0) { $0 | Int(d[at + $1]) << (8 * $1) }
    }
    private static func u64(_ d: [UInt8], _ at: Int) -> Int {
        (0..<8).reduce(0) { $0 | Int(d[at + $1]) << (8 * $1) }
    }

    private mutating func readDirectory() throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let size = Int(try handle.seekToEnd())
        guard size > 22 else { throw Failure.notZip }

        // The end record is last, behind a comment nobody sets on an IPSW.
        let tailLength = min(size, 66 * 1024)
        try handle.seek(toOffset: UInt64(size - tailLength))
        let tail = [UInt8](handle.readData(ofLength: tailLength))
        guard let eocd = IPSW.find(0x0605_4B50, in: tail) else { throw Failure.notZip }

        var count = IPSW.u16(tail, eocd + 10)
        var directoryOffset = IPSW.u32(tail, eocd + 16)
        var directorySize = IPSW.u32(tail, eocd + 12)

        // Zip64: the real numbers are in a record of its own, found through the
        // locator that sits just before the end record.
        if directoryOffset == 0xFFFF_FFFF || count == 0xFFFF || directorySize == 0xFFFF_FFFF {
            guard let locator = IPSW.find(0x0706_4B50, in: tail) else {
                throw Failure.unsupported("zip64")
            }
            let recordAt = IPSW.u64(tail, locator + 8)
            try handle.seek(toOffset: UInt64(recordAt))
            let record = [UInt8](handle.readData(ofLength: 56))
            guard IPSW.u32(record, 0) == 0x0606_4B50 else { throw Failure.corrupt("zip64 record") }
            count = IPSW.u64(record, 32)
            directorySize = IPSW.u64(record, 40)
            directoryOffset = IPSW.u64(record, 48)
        }

        try handle.seek(toOffset: UInt64(directoryOffset))
        let directory = [UInt8](handle.readData(ofLength: directorySize))

        var at = 0
        for _ in 0..<count {
            guard at + 46 <= directory.count, IPSW.u32(directory, at) == 0x0201_4B50 else {
                throw Failure.corrupt("central directory")
            }
            let method = IPSW.u16(directory, at + 10)
            var compressed = IPSW.u32(directory, at + 20)
            var plain = IPSW.u32(directory, at + 24)
            let nameLength = IPSW.u16(directory, at + 28)
            let extraLength = IPSW.u16(directory, at + 30)
            let commentLength = IPSW.u16(directory, at + 32)
            var localOffset = IPSW.u32(directory, at + 42)

            let nameAt = at + 46
            let name = String(decoding: directory[nameAt..<nameAt + nameLength], as: UTF8.self)

            // The zip64 extra field fills in whichever of the three overflowed,
            // in this order and only those.
            var extraAt = nameAt + nameLength
            let extraEnd = extraAt + extraLength
            while extraAt + 4 <= extraEnd {
                let tag = IPSW.u16(directory, extraAt)
                let size = IPSW.u16(directory, extraAt + 2)
                if tag == 0x0001 {
                    var field = extraAt + 4
                    if plain == 0xFFFF_FFFF, field + 8 <= extraEnd { plain = IPSW.u64(directory, field); field += 8 }
                    if compressed == 0xFFFF_FFFF, field + 8 <= extraEnd { compressed = IPSW.u64(directory, field); field += 8 }
                    if localOffset == 0xFFFF_FFFF, field + 8 <= extraEnd { localOffset = IPSW.u64(directory, field) }
                }
                extraAt += 4 + size
            }
            at = extraEnd + commentLength

            guard !name.hasSuffix("/") else { continue }

            // Only the local header says how long the name and extra field are
            // *there*; the central copies may differ.
            try handle.seek(toOffset: UInt64(localOffset))
            let local = [UInt8](handle.readData(ofLength: 30))
            guard local.count == 30, IPSW.u32(local, 0) == 0x0403_4B50 else {
                throw Failure.corrupt("local header for \(name)")
            }
            let dataOffset = localOffset + 30 + IPSW.u16(local, 26) + IPSW.u16(local, 28)
            members[name] = Member(path: name, method: method, compressedSize: compressed,
                                   plainSize: plain, dataOffset: dataOffset)
        }
    }

    private static func find(_ signature: Int, in bytes: [UInt8]) -> Int? {
        guard bytes.count >= 4 else { return nil }
        var probe = bytes.count - 4
        while probe >= 0 {
            if u32(bytes, probe) == signature { return probe }
            probe -= 1
        }
        return nil
    }

    // MARK: - Getting things out

    func member(_ path: String) throws -> Member {
        guard let member = members[path] else { throw Failure.missing(path) }
        return member
    }

    /// One component, in memory. For the small ones — a kernel, a device tree,
    /// a ticket-sized blob — not for the filesystem image.
    func read(_ path: String) throws -> [UInt8] {
        let member = try self.member(path)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: UInt64(member.dataOffset))
        let raw = handle.readData(ofLength: member.compressedSize)
        guard raw.count == member.compressedSize else { throw Failure.corrupt(path) }
        switch member.method {
        case 0:
            return [UInt8](raw)
        case 8:
            var out = Data(count: member.plainSize)
            let written: Int = out.withUnsafeMutableBytes { destination in
                raw.withUnsafeBytes { source in
                    compression_decode_buffer(
                        destination.bindMemory(to: UInt8.self).baseAddress!, member.plainSize,
                        source.bindMemory(to: UInt8.self).baseAddress!, raw.count,
                        nil, COMPRESSION_ZLIB)
                }
            }
            guard written == member.plainSize else { throw Failure.corrupt(path) }
            return [UInt8](out)
        default:
            throw Failure.unsupported(L("способ сжатия %d", member.method))
        }
    }

    /// Writes a component to disk without holding it in memory: the ramdisk is
    /// a hundred megabytes and lands in the app's own folder.
    func extract(_ path: String, to destination: URL,
                 progress: ((Double) -> Void)? = nil) throws {
        let member = try self.member(path)
        let reader = try FileHandle(forReadingFrom: url)
        defer { try? reader.close() }
        try reader.seek(toOffset: UInt64(member.dataOffset))

        FileManager.default.createFile(atPath: destination.path, contents: nil)
        guard let writer = try? FileHandle(forWritingTo: destination) else {
            throw Failure.corrupt(destination.lastPathComponent)
        }
        defer { try? writer.close() }

        let chunk = 4 * 1024 * 1024
        var left = member.compressedSize
        if member.isStored {
            // See `RestorePrep.localCopy`'s own pool: the same undrained-
            // autorelease growth applies to any tight read/write loop dispatched
            // as one long GCD block, and a member can be hundreds of megabytes.
            while left > 0 {
                try autoreleasepool {
                    let piece = reader.readData(ofLength: min(chunk, left))
                    guard !piece.isEmpty else { throw Failure.corrupt(path) }
                    writer.write(piece)
                    left -= piece.count
                    progress?(Double(member.compressedSize - left) / Double(member.compressedSize))
                }
            }
            return
        }

        // Raw DEFLATE, decoded a window at a time.
        //
        // The input buffer is held here rather than inside a `withUnsafeBytes`:
        // the decoder keeps a pointer into it between calls, and it has to stay
        // where it was. The loop runs until the decoder says END — draining it
        // matters, since the last call with input left can still owe output.
        var stream = compression_stream(dst_ptr: UnsafeMutablePointer<UInt8>(bitPattern: 1)!, dst_size: 0,
                                        src_ptr: UnsafePointer<UInt8>(bitPattern: 1)!, src_size: 0,
                                        state: nil)
        guard compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB)
                == COMPRESSION_STATUS_OK
        else { throw Failure.corrupt(path) }
        defer { compression_stream_destroy(&stream) }

        let input = UnsafeMutablePointer<UInt8>.allocate(capacity: chunk)
        let output = UnsafeMutablePointer<UInt8>.allocate(capacity: chunk)
        defer { input.deallocate(); output.deallocate() }
        stream.src_size = 0

        // Same undrained-autorelease risk as the stored branch above, for the
        // same reason — see the note there.
        while true {
            let (status, produced): (compression_status, Int) = try autoreleasepool {
                if stream.src_size == 0, left > 0 {
                    let piece = reader.readData(ofLength: min(chunk, left))
                    guard !piece.isEmpty else { throw Failure.corrupt(path) }
                    piece.copyBytes(to: input, count: piece.count)
                    stream.src_ptr = UnsafePointer(input)
                    stream.src_size = piece.count
                    left -= piece.count
                    progress?(Double(member.compressedSize - left) / Double(member.compressedSize))
                }

                stream.dst_ptr = output
                stream.dst_size = chunk
                // Only the last of the input may be finalised; doing it earlier
                // ends the stream on whatever has been read so far.
                let flags = left == 0 ? Int32(COMPRESSION_STREAM_FINALIZE.rawValue) : 0
                let status = compression_stream_process(&stream, flags)
                let produced = chunk - stream.dst_size
                if produced > 0 { writer.write(Data(bytes: output, count: produced)) }
                return (status, produced)
            }

            switch status {
            case COMPRESSION_STATUS_END: return
            case COMPRESSION_STATUS_OK:
                // No input left, nothing produced and not finished: the archive
                // ended in the middle of a stream.
                if left == 0, stream.src_size == 0, produced == 0 { throw Failure.corrupt(path) }
            default: throw Failure.corrupt(path)
            }
        }
    }

    /// Where a stored member's bytes are in the IPSW itself.
    ///
    /// The filesystem image is stored, not deflated, which is what makes an
    /// offline restore on a phone possible at all: it can be read straight out
    /// of the firmware as ASR asks for it, with nothing unpacked first.
    func rangeOfStored(_ path: String) throws -> (offset: Int, length: Int) {
        let member = try self.member(path)
        guard member.isStored else { throw Failure.unsupported(L("%@ сжат, а нужен как есть", path)) }
        return (member.dataOffset, member.plainSize)
    }

    // MARK: - The manifest

    /// The build identity a fresh install uses, for this device.
    func buildIdentity(deviceClass: String = "n104ap", erase: Bool = true) throws -> [String: Any] {
        let raw = try read("BuildManifest.plist")
        guard let plist = try PropertyListSerialization.propertyList(from: Data(raw), options: [], format: nil) as? [String: Any],
              let identities = plist["BuildIdentities"] as? [[String: Any]]
        else { throw Failure.corrupt("BuildManifest.plist") }
        for identity in identities {
            guard let info = identity["Info"] as? [String: Any],
                  (info["DeviceClass"] as? String)?.lowercased() == deviceClass,
                  (info["RestoreBehavior"] as? String) == (erase ? "Erase" : "Update")
            else { continue }
            return identity
        }
        throw Failure.missing(deviceClass)
    }

    /// The path a component of the build identity lives at inside the IPSW.
    static func path(of component: String, in identity: [String: Any]) -> String? {
        guard let manifest = identity["Manifest"] as? [String: Any],
              let entry = manifest[component] as? [String: Any],
              let info = entry["Info"] as? [String: Any]
        else { return nil }
        return info["Path"] as? String
    }
}
