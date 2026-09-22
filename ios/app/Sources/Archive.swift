import Compression
import Foundation

/// Reading `.ipa` and writing `.tar`, on the phone.
///
/// The guest cannot be relied on to unpack a zip: `unzip` is missing from a bare
/// bootstrap, while `tar` is there on every image. So the phone opens the `.ipa`
/// itself and hands the guest a tar, which is a format simple enough to write in
/// a page of code.
///
/// Nothing here needs a third-party library: a zip entry holds raw DEFLATE, and
/// that is exactly what Apple's Compression framework calls `COMPRESSION_ZLIB`.
enum Archive {
    enum Failure: LocalizedError {
        case notZip
        case unsupported(String)
        case corrupt(String)

        var errorDescription: String? {
            switch self {
            case .notZip:
                return L("Это не .ipa: внутри нет оглавления zip.")
            case .unsupported(let what):
                return L("В архиве есть то, что я не умею разбирать: %@", what)
            case .corrupt(let what):
                return L("Архив повреждён: %@", what)
            }
        }
    }

    /// One file out of the archive, with the permissions it was stored with.
    struct Entry {
        var path: String
        var data: Data
        /// The unix mode zip keeps in the high half of its external attributes.
        /// It carries the executable bit, and without that bit the app that
        /// lands in the guest cannot be launched at all.
        var mode: UInt16
        var isDirectory: Bool
        var isSymlink: Bool
    }

    // MARK: - Reading a zip

    private static func u16(_ d: Data, _ at: Int) -> Int { Int(d[at]) | Int(d[at + 1]) << 8 }
    private static func u32(_ d: Data, _ at: Int) -> Int {
        Int(d[at]) | Int(d[at + 1]) << 8 | Int(d[at + 2]) << 16 | Int(d[at + 3]) << 24
    }

    /// Everything in the archive, in the order the central directory lists it.
    ///
    /// The central directory is read rather than the stream of local headers:
    /// only the directory is guaranteed to carry the external attributes, and
    /// those are where the executable bit lives.
    static func entries(ofZip url: URL) throws -> [Entry] {
        let zip = try Data(contentsOf: url, options: .mappedIfSafe)
        guard zip.count > 22 else { throw Failure.notZip }

        // End of central directory, found by walking back over the comment.
        var eocd = -1
        let lowest = max(0, zip.count - 22 - 0xFFFF)
        var probe = zip.count - 22
        while probe >= lowest {
            if u32(zip, probe) == 0x0605_4B50 { eocd = probe; break }
            probe -= 1
        }
        guard eocd >= 0 else { throw Failure.notZip }

        let count = u16(zip, eocd + 10)
        var offset = u32(zip, eocd + 16)
        if offset == 0xFFFF_FFFF || count == 0xFFFF {
            throw Failure.unsupported("zip64")
        }

        var result: [Entry] = []
        result.reserveCapacity(count)

        for _ in 0..<count {
            guard offset + 46 <= zip.count, u32(zip, offset) == 0x0201_4B50 else {
                throw Failure.corrupt("central directory")
            }
            let method = u16(zip, offset + 10)
            let compressed = u32(zip, offset + 20)
            let plain = u32(zip, offset + 24)
            let nameLength = u16(zip, offset + 28)
            let extraLength = u16(zip, offset + 30)
            let commentLength = u16(zip, offset + 32)
            let external = u32(zip, offset + 38)
            let localOffset = u32(zip, offset + 42)

            let nameStart = offset + 46
            let name = String(decoding: zip[nameStart..<nameStart + nameLength], as: UTF8.self)
            offset = nameStart + nameLength + extraLength + commentLength

            // The high half is the unix mode, when the archive came from a unix
            // packer. Anything else gets sensible defaults instead.
            var mode = UInt16((external >> 16) & 0xFFFF)
            let directory = name.hasSuffix("/")
            let symlink = (mode & 0xF000) == 0xA000
            if mode & 0o777 == 0 { mode = directory ? 0o755 : 0o644 }

            if directory {
                result.append(Entry(path: String(name.dropLast()), data: Data(),
                                    mode: mode, isDirectory: true, isSymlink: false))
                continue
            }

            // The local header repeats the name and extra field, and only it
            // says how long they are *here* — the central copy can differ.
            guard localOffset + 30 <= zip.count, u32(zip, localOffset) == 0x0403_4B50 else {
                throw Failure.corrupt("local header for \(name)")
            }
            let dataStart = localOffset + 30 + u16(zip, localOffset + 26) + u16(zip, localOffset + 28)
            guard dataStart + compressed <= zip.count else { throw Failure.corrupt(name) }
            let raw = zip[dataStart..<dataStart + compressed]

            let content: Data
            switch method {
            case 0:  content = Data(raw)
            case 8:  content = try inflate(Data(raw), expecting: plain, named: name)
            default: throw Failure.unsupported(L("способ сжатия %d", method))
            }
            guard content.count == plain else { throw Failure.corrupt(name) }

            result.append(Entry(path: name, data: content, mode: mode,
                                isDirectory: false, isSymlink: symlink))
        }
        return result
    }

    /// Raw DEFLATE, which is what zip stores and what COMPRESSION_ZLIB decodes.
    private static func inflate(_ data: Data, expecting size: Int, named: String) throws -> Data {
        if size == 0 { return Data() }
        var out = Data(count: size)
        let written: Int = out.withUnsafeMutableBytes { destination in
            data.withUnsafeBytes { source in
                compression_decode_buffer(
                    destination.bindMemory(to: UInt8.self).baseAddress!, size,
                    source.bindMemory(to: UInt8.self).baseAddress!, data.count,
                    nil, COMPRESSION_ZLIB)
            }
        }
        guard written == size else { throw Failure.corrupt(named) }
        return out
    }

    // MARK: - Writing a tar

    private static let block = 512

    /// Writes the entries as a ustar archive, straight to disk.
    ///
    /// Streamed rather than assembled in memory: an unpacked app is tens of
    /// megabytes, and the phone has better uses for them.
    static func writeTar(_ entries: [Entry], to url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let out = try FileHandle(forWritingTo: url)
        defer { try? out.close() }

        for entry in entries {
            let body = entry.isSymlink ? Data() : entry.data
            let link = entry.isSymlink ? String(decoding: entry.data, as: UTF8.self) : ""
            let type: UInt8 = entry.isDirectory ? 0x35 : (entry.isSymlink ? 0x32 : 0x30)

            // ustar splits a long name between two fields, and cannot hold one
            // longer than both together. GNU's long-name record has no such
            // limit, and the guest's tar reads it.
            if splitName(entry.path) == nil {
                var header = Data(count: block)
                let name = Data(entry.path.utf8)
                fill(&header, "././@LongLink", 0, 100)
                fill(&header, octal(0, width: 7), 100, 8)
                fill(&header, octal(0, width: 7), 108, 8)
                fill(&header, octal(0, width: 7), 116, 8)
                fill(&header, octal(name.count + 1, width: 11), 124, 12)
                fill(&header, octal(0, width: 11), 136, 12)
                header[156] = 0x4C            // 'L'
                fill(&header, "ustar  ", 257, 8)
                seal(&header)
                out.write(header)
                out.write(pad(name + Data([0])))
            }

            out.write(try makeHeader(for: entry, type: type, link: link, size: body.count))
            if !body.isEmpty { out.write(pad(body)) }
        }
        // Two empty blocks end the archive.
        out.write(Data(count: block * 2))
    }

    private static func makeHeader(for entry: Entry, type: UInt8, link: String, size: Int) throws -> Data {
        var header = Data(count: block)
        let (prefix, name) = splitName(entry.path) ?? ("", String(entry.path.suffix(100)))
        fill(&header, name, 0, 100)
        fill(&header, octal(Int(entry.mode & 0o7777), width: 7), 100, 8)
        fill(&header, octal(0, width: 7), 108, 8)          // uid: root
        fill(&header, octal(0, width: 7), 116, 8)          // gid: wheel
        fill(&header, octal(size, width: 11), 124, 12)
        fill(&header, octal(Int(Date().timeIntervalSince1970), width: 11), 136, 12)
        header[156] = type
        fill(&header, link, 157, 100)
        fill(&header, "ustar", 257, 6)
        fill(&header, "00", 263, 2)
        fill(&header, "root", 265, 32)
        fill(&header, "wheel", 297, 32)
        fill(&header, prefix, 345, 155)
        seal(&header)
        return header
    }

    /// ustar keeps the last component in `name` and the rest in `prefix`.
    /// Returns nil when the path fits in neither, which is the long-name case.
    private static func splitName(_ path: String) -> (prefix: String, name: String)? {
        if path.utf8.count <= 100 { return ("", path) }
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        for index in 1..<parts.count {
            let head = parts[0..<index].joined(separator: "/")
            let tail = parts[index...].joined(separator: "/")
            if head.utf8.count <= 155, tail.utf8.count <= 100 { return (head, tail) }
        }
        return nil
    }

    /// The checksum is computed with its own field read as spaces.
    private static func seal(_ header: inout Data) {
        for index in 148..<156 { header[index] = 0x20 }
        let sum = header.reduce(0) { $0 + Int($1) }
        fill(&header, octal(sum, width: 6), 148, 8)
        header[154] = 0
        header[155] = 0x20
    }

    private static func octal(_ value: Int, width: Int) -> String {
        String(format: "%0\(width)o", value)
    }

    private static func fill(_ header: inout Data, _ text: String, _ at: Int, _ size: Int) {
        let bytes = Array(text.utf8.prefix(size))
        for (index, byte) in bytes.enumerated() { header[at + index] = byte }
    }

    private static func pad(_ data: Data) -> Data {
        let remainder = data.count % block
        return remainder == 0 ? data : data + Data(count: block - remainder)
    }
}
