import Foundation
import Darwin

/// A folder shared with the guest: `Shared` in the app's Documents, which the
/// Files app shows, served over WebDAV on the phone's loopback.
///
/// The guest reaches it at `http://10.0.2.2:8080`: under the emulator's NAT,
/// 10.0.2.2 is the host, and a connection to it lands on the phone's own
/// 127.0.0.1. macOS mounts WebDAV itself (Finder → Go → Connect to Server),
/// so nothing has to be installed in the guest, and the folder works both
/// ways — files put in it on the phone appear in the guest and the other way
/// round.
///
/// Why WebDAV and not a shared-folder device: macOS guests share folders over
/// virtio-fs, which QEMU serves only through an external `virtiofsd` process,
/// and an iOS app cannot start one. WebDAV needs nothing but a socket.
///
/// The server is the small subset Finder uses: OPTIONS, PROPFIND, GET/HEAD
/// with a single byte range, PUT (streamed to disk, chunked or not),
/// DELETE, MKCOL, MOVE, COPY, PROPPATCH (acknowledged), and LOCK/UNLOCK —
/// accepted without real locking, because Finder mounts a server that cannot
/// lock read-only. One thread per connection; blocking sockets.
final class SharedFolder {
    static let shared = SharedFolder()
    static let port: UInt16 = 8080
    static let guestURL = "http://10.0.2.2:\(port)"

    static var directory: URL { VMConfig.documents.appendingPathComponent("Shared", isDirectory: true) }

    private var listener: Int32 = -1
    private let lock = NSLock()

    var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return listener >= 0
    }

    func start() {
        lock.lock(); defer { lock.unlock() }
        guard listener < 0 else { return }
        try? FileManager.default.createDirectory(at: Self.directory, withIntermediateDirectories: true)

        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return note(L("Общая папка: не удалось открыть сокет")) }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = Self.port.bigEndian
        // The loopback only: the emulator's NAT is the one client it is for.
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, listen(fd, 16) == 0 else {
            close(fd)
            return note(L("Общая папка: порт %d занят", Int(Self.port)))
        }
        listener = fd
        let thread = Thread { [weak self] in self?.acceptLoop(fd) }
        thread.name = "orchard.shared-folder"
        thread.start()
        note(L("Общая папка: %@ (папка Shared в «Файлах»)", Self.guestURL))
    }

    func stop() {
        lock.lock(); defer { lock.unlock() }
        guard listener >= 0 else { return }
        close(listener)
        listener = -1
    }

    private func note(_ text: String) { LogCapture.shared.note(text) }

    private func acceptLoop(_ fd: Int32) {
        while true {
            let client = accept(fd, nil, nil)
            if client < 0 {
                if errno == EINTR { continue }
                return      // closed by stop()
            }
            var one: Int32 = 1
            setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
            let thread = Thread { DAVConnection(fd: client, root: Self.directory).run() }
            thread.name = "orchard.shared-folder.conn"
            thread.start()
        }
    }
}

// MARK: - One client connection

private final class DAVConnection {
    private let fd: Int32
    private let root: URL
    private var buffer = [UInt8]()
    private var start = 0          // read position in `buffer`
    private var current = ""       // "METHOD /path depth", for the log

    /// Every request and its status go to the emulator log, up to a limit:
    /// when the guest's WebDAV client refuses to mount, the only account of
    /// why is what it asked and what it was told.
    private static var logged = 0
    private static let logLock = NSLock()
    private static func log(_ line: String) {
        logLock.lock()
        defer { logLock.unlock() }
        guard logged < 400 else { return }
        logged += 1
        LogCapture.shared.note("DAV " + line)
    }

    init(fd: Int32, root: URL) {
        self.fd = fd
        self.root = root.standardizedFileURL
    }

    func run() {
        defer { close(fd) }
        while let request = readRequest() {
            let keepAlive = handle(request)
            if !keepAlive { return }
        }
    }

    // MARK: Reading

    private struct Request {
        let method: String
        let path: String            // decoded, always starting with "/"
        let headers: [String: String]   // lower-cased names
        let version: String
        func header(_ name: String) -> String? { headers[name] }
    }

    /// More bytes from the socket into the buffer; false at EOF or error.
    private func fill() -> Bool {
        if start > 0 && start == buffer.count {
            buffer.removeAll(keepingCapacity: true)
            start = 0
        } else if start >= 65536 {
            buffer.removeFirst(start)
            start = 0
        }
        var chunk = [UInt8](repeating: 0, count: 65536)
        let n = chunk.withUnsafeMutableBytes { recv(fd, $0.baseAddress, 65536, 0) }
        if n <= 0 { return false }
        buffer.append(contentsOf: chunk[0..<n])
        return true
    }

    private func readLine() -> String? {
        while true {
            if let i = buffer[start...].firstIndex(of: 0x0A) {
                var end = i
                if end > start && buffer[end - 1] == 0x0D { end -= 1 }
                let line = String(decoding: buffer[start..<end], as: UTF8.self)
                start = i + 1
                return line
            }
            if buffer.count - start > 65536 { return nil }   // no header is that long
            if !fill() { return nil }
        }
    }

    private func readRequest() -> Request? {
        guard var first = readLine() else { return nil }
        while first.isEmpty { guard let next = readLine() else { return nil }; first = next }
        let parts = first.split(separator: " ", omittingEmptySubsequences: true)
        guard parts.count >= 3 else { return nil }
        var headers: [String: String] = [:]
        while let line = readLine(), !line.isEmpty {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }
        return Request(method: String(parts[0]).uppercased(),
                       path: Self.decodePath(String(parts[1])),
                       headers: headers,
                       version: String(parts[2]))
    }

    /// The request body, handed over in pieces; returns false if the client went away.
    private func readBody(_ r: Request, into sink: (ArraySlice<UInt8>) -> Void) -> Bool {
        if r.header("expect")?.lowercased() == "100-continue" {
            send("HTTP/1.1 100 Continue\r\n\r\n")
        }
        if r.header("transfer-encoding")?.lowercased().contains("chunked") == true {
            while true {
                guard let sizeLine = readLine() else { return false }
                let hex = sizeLine.split(separator: ";").first.map(String.init) ?? ""
                guard let size = Int(hex.trimmingCharacters(in: .whitespaces), radix: 16) else { return false }
                if size == 0 {
                    while let trailer = readLine(), !trailer.isEmpty {}
                    return true
                }
                guard readExactly(size, into: sink) else { return false }
                _ = readLine()      // the CRLF after each chunk
            }
        }
        let length = Int(r.header("content-length") ?? "0") ?? 0
        return readExactly(length, into: sink)
    }

    private func readExactly(_ count: Int, into sink: (ArraySlice<UInt8>) -> Void) -> Bool {
        var left = count
        while left > 0 {
            if start == buffer.count, !fill() { return false }
            let take = min(left, buffer.count - start)
            sink(buffer[start..<(start + take)])
            start += take
            left -= take
        }
        return true
    }

    private func discardBody(_ r: Request) -> Bool { readBody(r) { _ in } }

    // MARK: Writing

    private func send(_ text: String) { send(Array(text.utf8)[...]) }

    private func send(_ bytes: ArraySlice<UInt8>) {
        bytes.withUnsafeBytes { raw in
            var offset = 0
            while offset < raw.count {
                let n = Darwin.send(fd, raw.baseAddress! + offset, raw.count - offset, 0)
                if n <= 0 { return }
                offset += n
            }
        }
    }

    private func respond(_ status: Int, _ reason: String, headers: [String: String] = [:],
                         body: String = "", keepAlive: Bool) {
        let bytes = Array(body.utf8)
        Self.log("\(current) → \(status)")
        var head = "HTTP/1.1 \(status) \(reason)\r\n"
        head += "Content-Length: \(bytes.count)\r\n"
        head += "Date: \(Self.httpDate(Date()))\r\n"
        head += "Server: Orchard\r\n"
        head += "Connection: \(keepAlive ? "keep-alive" : "close")\r\n"
        for (k, v) in headers { head += "\(k): \(v)\r\n" }
        head += "\r\n"
        send(head)
        if !bytes.isEmpty { send(bytes[...]) }
    }

    // MARK: Dispatch

    /// Serves one request; returns whether the connection stays open.
    private func handle(_ r: Request) -> Bool {
        current = "\(r.method) \(r.path)" + (r.header("depth").map { " depth=\($0)" } ?? "")
        let keepAlive = r.header("connection")?.lowercased() != "close" && r.version != "HTTP/1.0"
        guard let url = resolve(r.path) else {
            _ = discardBody(r)
            respond(403, "Forbidden", keepAlive: keepAlive)
            return keepAlive
        }
        switch r.method {
        case "OPTIONS":
            _ = discardBody(r)
            respond(200, "OK", headers: [
                "DAV": "1, 2",
                "MS-Author-Via": "DAV",
                "Allow": "OPTIONS, GET, HEAD, PUT, DELETE, MKCOL, MOVE, COPY, PROPFIND, PROPPATCH, LOCK, UNLOCK",
            ], keepAlive: keepAlive)
        case "PROPFIND":
            guard discardBody(r) else { return false }
            propfind(r, url, keepAlive: keepAlive)
        case "PROPPATCH":
            guard discardBody(r) else { return false }
            let body = Self.xmlHeader + "<D:multistatus xmlns:D=\"DAV:\"><D:response><D:href>\(Self.xml(Self.encodePath(r.path)))</D:href>"
                + "<D:propstat><D:prop/><D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response></D:multistatus>"
            respond(207, "Multi-Status", headers: ["Content-Type": "application/xml; charset=utf-8"],
                    body: body, keepAlive: keepAlive)
        case "GET", "HEAD":
            guard discardBody(r) else { return false }
            return get(r, url, headOnly: r.method == "HEAD", keepAlive: keepAlive)
        case "PUT":
            return put(r, url, keepAlive: keepAlive)
        case "DELETE":
            guard discardBody(r) else { return false }
            if (try? FileManager.default.removeItem(at: url)) != nil {
                respond(204, "No Content", keepAlive: keepAlive)
            } else {
                respond(404, "Not Found", keepAlive: keepAlive)
            }
        case "MKCOL":
            guard discardBody(r) else { return false }
            if FileManager.default.fileExists(atPath: url.path) {
                respond(405, "Method Not Allowed", keepAlive: keepAlive)
            } else if (try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)) != nil {
                respond(201, "Created", keepAlive: keepAlive)
            } else {
                respond(409, "Conflict", keepAlive: keepAlive)
            }
        case "MOVE", "COPY":
            guard discardBody(r) else { return false }
            moveOrCopy(r, url, copy: r.method == "COPY", keepAlive: keepAlive)
        case "LOCK":
            guard discardBody(r) else { return false }
            lock(r, url, keepAlive: keepAlive)
        case "UNLOCK":
            guard discardBody(r) else { return false }
            respond(204, "No Content", keepAlive: keepAlive)
        default:
            _ = discardBody(r)
            respond(405, "Method Not Allowed", keepAlive: keepAlive)
        }
        return keepAlive
    }

    // MARK: Methods

    private func propfind(_ r: Request, _ url: URL, keepAlive: Bool) {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return respond(404, "Not Found", keepAlive: keepAlive)
        }
        var body = Self.xmlHeader + "<D:multistatus xmlns:D=\"DAV:\">"
        body += entry(href: r.path, url: url, isDir: isDir.boolValue)
        if isDir.boolValue, r.header("depth") != "0" {
            let children = (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []
            let base = r.path.hasSuffix("/") ? r.path : r.path + "/"
            for name in children.sorted() {
                let child = url.appendingPathComponent(name)
                var childIsDir: ObjCBool = false
                guard FileManager.default.fileExists(atPath: child.path, isDirectory: &childIsDir) else { continue }
                body += entry(href: base + name, url: child, isDir: childIsDir.boolValue)
            }
        }
        body += "</D:multistatus>"
        respond(207, "Multi-Status", headers: ["Content-Type": "application/xml; charset=utf-8"],
                body: body, keepAlive: keepAlive)
    }

    private func entry(href: String, url: URL, isDir: Bool) -> String {
        let attrs = (try? FileManager.default.attributesOfItem(atPath: url.path)) ?? [:]
        let modified = attrs[.modificationDate] as? Date ?? Date()
        let created = attrs[.creationDate] as? Date ?? modified
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        var path = href
        if isDir && !path.hasSuffix("/") { path += "/" }
        var props = "<D:displayname>\(Self.xml(url.lastPathComponent))</D:displayname>"
        props += "<D:getlastmodified>\(Self.httpDate(modified))</D:getlastmodified>"
        props += "<D:creationdate>\(Self.isoDate(created))</D:creationdate>"
        props += "<D:supportedlock><D:lockentry><D:lockscope><D:exclusive/></D:lockscope>"
            + "<D:locktype><D:write/></D:locktype></D:lockentry></D:supportedlock>"
        if isDir {
            props += "<D:resourcetype><D:collection/></D:resourcetype>"
            // Finder checks the free space before it copies anything in.
            if let values = try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
               let free = values.volumeAvailableCapacityForImportantUsage {
                props += "<D:quota-available-bytes>\(free)</D:quota-available-bytes>"
                props += "<D:quota-used-bytes>0</D:quota-used-bytes>"
            }
        } else {
            props += "<D:resourcetype/>"
            props += "<D:getcontentlength>\(size)</D:getcontentlength>"
            props += "<D:getcontenttype>application/octet-stream</D:getcontenttype>"
        }
        return "<D:response><D:href>\(Self.xml(Self.encodePath(path)))</D:href><D:propstat><D:prop>\(props)</D:prop>"
            + "<D:status>HTTP/1.1 200 OK</D:status></D:propstat></D:response>"
    }

    private func get(_ r: Request, _ url: URL, headOnly: Bool, keepAlive: Bool) -> Bool {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            respond(404, "Not Found", keepAlive: keepAlive)
            return keepAlive
        }
        if isDir.boolValue {
            respond(200, "OK", headers: ["Content-Type": "text/plain; charset=utf-8"],
                    body: "Orchard shared folder\n", keepAlive: keepAlive)
            return keepAlive
        }
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            respond(403, "Forbidden", keepAlive: keepAlive)
            return keepAlive
        }
        defer { try? handle.close() }
        let attrs = (try? FileManager.default.attributesOfItem(atPath: url.path)) ?? [:]
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0
        let modified = attrs[.modificationDate] as? Date ?? Date()
        var first: Int64 = 0
        var last: Int64 = size - 1
        var status = (200, "OK")
        if let range = r.header("range"), range.hasPrefix("bytes="), size > 0 {
            let spec = range.dropFirst(6).split(separator: ",").first.map(String.init) ?? ""
            let ends = spec.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
            if ends.count == 2 {
                if ends[0].isEmpty, let suffix = Int64(ends[1]) {
                    first = max(0, size - suffix)
                } else {
                    first = Int64(ends[0]) ?? 0
                    if let e = Int64(ends[1]) { last = min(e, size - 1) }
                }
                if first > last || first >= size {
                    respond(416, "Range Not Satisfiable", headers: ["Content-Range": "bytes */\(size)"],
                            keepAlive: keepAlive)
                    return keepAlive
                }
                status = (206, "Partial Content")
            }
        }
        let length = size == 0 ? 0 : last - first + 1
        Self.log("\(current) → \(status.0) \(length) bytes")
        var head = "HTTP/1.1 \(status.0) \(status.1)\r\n"
        head += "Content-Length: \(length)\r\n"
        head += "Content-Type: application/octet-stream\r\n"
        head += "Accept-Ranges: bytes\r\n"
        head += "Last-Modified: \(Self.httpDate(modified))\r\n"
        if status.0 == 206 { head += "Content-Range: bytes \(first)-\(last)/\(size)\r\n" }
        head += "Connection: \(keepAlive ? "keep-alive" : "close")\r\n\r\n"
        send(head)
        if headOnly || length == 0 { return keepAlive }
        do {
            try handle.seek(toOffset: UInt64(first))
            var left = length
            while left > 0 {
                let want = Int(min(left, 1 << 20))
                guard let data = try handle.read(upToCount: want), !data.isEmpty else { return false }
                send(Array(data)[...])
                left -= Int64(data.count)
            }
        } catch {
            return false
        }
        return keepAlive
    }

    private func put(_ r: Request, _ url: URL, keepAlive: Bool) -> Bool {
        let existed = FileManager.default.fileExists(atPath: url.path)
        let temp = url.deletingLastPathComponent()
            .appendingPathComponent(".orchard-upload-\(UUID().uuidString)")
        guard FileManager.default.createFile(atPath: temp.path, contents: nil),
              let handle = try? FileHandle(forWritingTo: temp) else {
            _ = discardBody(r)
            respond(409, "Conflict", keepAlive: keepAlive)
            return keepAlive
        }
        var failed = false
        let complete = readBody(r) { piece in
            guard !failed else { return }
            do { try piece.withUnsafeBytes { try handle.write(contentsOf: Data($0)) } } catch { failed = true }
        }
        try? handle.close()
        guard complete, !failed else {
            try? FileManager.default.removeItem(at: temp)
            if complete { respond(507, "Insufficient Storage", keepAlive: keepAlive) }
            return complete && keepAlive
        }
        _ = try? FileManager.default.removeItem(at: url)
        do {
            try FileManager.default.moveItem(at: temp, to: url)
            respond(existed ? 204 : 201, existed ? "No Content" : "Created", keepAlive: keepAlive)
        } catch {
            try? FileManager.default.removeItem(at: temp)
            respond(409, "Conflict", keepAlive: keepAlive)
        }
        return keepAlive
    }

    private func moveOrCopy(_ r: Request, _ url: URL, copy: Bool, keepAlive: Bool) {
        guard let destination = r.header("destination"),
              let destPath = Self.pathOfDestination(destination),
              let target = resolve(destPath) else {
            return respond(400, "Bad Request", keepAlive: keepAlive)
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            return respond(404, "Not Found", keepAlive: keepAlive)
        }
        let existed = FileManager.default.fileExists(atPath: target.path)
        if existed {
            if r.header("overwrite")?.uppercased() == "F" {
                return respond(412, "Precondition Failed", keepAlive: keepAlive)
            }
            try? FileManager.default.removeItem(at: target)
        }
        do {
            if copy {
                try FileManager.default.copyItem(at: url, to: target)
            } else {
                try FileManager.default.moveItem(at: url, to: target)
            }
            respond(existed ? 204 : 201, existed ? "No Content" : "Created", keepAlive: keepAlive)
        } catch {
            respond(409, "Conflict", keepAlive: keepAlive)
        }
    }

    private func lock(_ r: Request, _ url: URL, keepAlive: Bool) {
        // A lock-null resource: Finder locks a name before it writes to it.
        var created = false
        if !FileManager.default.fileExists(atPath: url.path) {
            created = FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        let token = "opaquelocktoken:\(UUID().uuidString.lowercased())"
        let body = Self.xmlHeader + "<D:prop xmlns:D=\"DAV:\"><D:lockdiscovery><D:activelock>"
            + "<D:locktype><D:write/></D:locktype><D:lockscope><D:exclusive/></D:lockscope>"
            + "<D:depth>0</D:depth><D:timeout>Second-3600</D:timeout>"
            + "<D:locktoken><D:href>\(token)</D:href></D:locktoken>"
            + "</D:activelock></D:lockdiscovery></D:prop>"
        respond(created ? 201 : 200, created ? "Created" : "OK",
                headers: ["Content-Type": "application/xml; charset=utf-8", "Lock-Token": "<\(token)>"],
                body: body, keepAlive: keepAlive)
    }

    // MARK: Paths

    /// The file a request path names, or nil if it would leave the shared folder.
    private func resolve(_ path: String) -> URL? {
        var url = root
        for part in path.split(separator: "/") where !part.isEmpty && part != "." {
            if part == ".." { return nil }
            url.appendPathComponent(String(part))
        }
        let resolved = url.standardizedFileURL
        guard resolved.path == root.path || resolved.path.hasPrefix(root.path + "/") else { return nil }
        return resolved
    }

    private static func decodePath(_ target: String) -> String {
        var path = target
        if let schemeEnd = path.range(of: "://") {
            // An absolute URI: keep only its path.
            let rest = path[schemeEnd.upperBound...]
            path = rest.firstIndex(of: "/").map { String(rest[$0...]) } ?? "/"
        }
        if let q = path.firstIndex(of: "?") { path = String(path[..<q]) }
        let decoded = path.removingPercentEncoding ?? path
        return decoded.hasPrefix("/") ? decoded : "/" + decoded
    }

    private static func pathOfDestination(_ destination: String) -> String? {
        let path = decodePath(destination)
        return path.isEmpty ? nil : path
    }

    private static func encodePath(_ path: String) -> String {
        var allowed = CharacterSet.urlPathAllowed
        allowed.remove(charactersIn: "?#[]@!$&'()*+,;=")
        return path.addingPercentEncoding(withAllowedCharacters: allowed) ?? path
    }

    // MARK: Formatting

    private static let xmlHeader = "<?xml version=\"1.0\" encoding=\"utf-8\"?>"

    private static func xml(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }

    private static let httpFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "GMT")
        f.dateFormat = "EEE, dd MMM yyyy HH:mm:ss 'GMT'"
        return f
    }()

    private static func httpDate(_ date: Date) -> String {
        objc_sync_enter(httpFormatter); defer { objc_sync_exit(httpFormatter) }
        return httpFormatter.string(from: date)
    }

    private static func isoDate(_ date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }
}
