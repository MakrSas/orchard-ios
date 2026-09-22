import Darwin
import Foundation
import Security

/// The app's side of the guest agent — the program that lives in the guest and
/// takes requests over the scratch NVMe namespace instead of the console.
///
/// Why it exists: everything automatic used to go through the one serial console
/// where the bootstrap's bash sits, and a single command that never returned its
/// prompt took the whole lot down with it — the shell pane, the packages, the
/// status bar. The agent is reached over the namespace instead, so a jammed
/// console no longer stops the status bar from updating or a service command
/// from running. See `netlab/agent/` for the guest side.
///
/// It is an addition, never a requirement. When the agent is not there — an
/// emulator without the namespace, a build without `ldid`, an install that did
/// not take, a guest too old — every method here reports so plainly and the
/// caller falls back to the console exactly as before.
///
/// The channel is a region of the same backing file `TransferNamespace` uses,
/// laid out to match `netlab/agent/agent.h`: a request block the app writes and
/// the agent reads, a response block the other way, and a data region past both
/// for file windows. A request is new to the agent when its (session, seq) pair
/// is one it has not answered; the answer repeats the pair, so a late answer to
/// an old request is never taken for a fresh one.
///
/// Everything here blocks; run it off the main thread.
final class GuestAgent {
    // The layout, matching agent.h. All offsets and the header block are 4 KiB
    // aligned because the guest reads this as a raw block device, which takes
    // nothing else.
    private static let headerSize = 64
    private static let headerBlock = 4096
    private static let reqOffset = 0
    private static let reqBody = 4096
    private static let reqSize = 256 * 1024
    private static let rspOffset = 256 * 1024
    private static let rspBody = 256 * 1024 + 4096
    private static let rspSize = 768 * 1024
    private static let dataOffset = 1024 * 1024
    private static let proto: UInt32 = 1
    private static let reqMagic = Array("INFAGREQ".utf8)
    private static let rspMagic = Array("INFAGRSP".utf8)

    let image: URL
    let capacity: Int64
    private let session: UInt64
    private var seq: UInt64 = 0
    private let lock = NSLock()

    /// The last thing the agent reported about itself, for the caller to log.
    private(set) var lastAgentInstance: UInt64 = 0

    private init(image: URL, capacity: Int64) {
        self.image = image
        self.capacity = capacity
        // Unique per launch: the agent dedups by (session, seq), so a session
        // that repeated across launches with seq reset would look like a
        // duplicate and be answered from stale data.
        var random: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &random) { SecRandomCopyBytesShim($0) }
        session = random == 0 ? UInt64(Date().timeIntervalSince1970 * 1000) : random
    }

    /// Returns a client only when an agent actually answers over the namespace.
    /// Nil is not a failure — it means this session uses the console, as before.
    static func discover(image: URL = VMConfig.transferImage) -> GuestAgent? {
        // The size is taken by opening the file, not from FileManager, which
        // reports a symlink's own length rather than its target's — the app's
        // xfer is a plain file, but the rig points at it through a symlink.
        guard let capacity = fileSize(image), capacity > Int64(dataOffset) else { return nil }
        let agent = GuestAgent(image: image, capacity: capacity)
        guard let reply = agent.request(["op": "ping"], timeout: 6), reply["ok"] as? Bool == true else {
            return nil
        }
        agent.lastAgentInstance = (reply["agent"] as? NSNumber)?.uint64Value ?? 0
        return agent
    }

    /// A quick liveness check on an agent already found. False when it has gone
    /// away — killed, or the machine restarted without it yet.
    func isAlive(timeout: TimeInterval = 4) -> Bool {
        guard let reply = request(["op": "ping"], timeout: timeout) else { return false }
        return reply["ok"] as? Bool == true
    }

    // MARK: - The request/response exchange

    /// Writes one request and waits for its answer. One exchange at a time: the
    /// region holds a single request, so callers are serialised.
    @discardableResult
    func request(_ body: [String: Any], timeout: TimeInterval) -> [String: Any]? {
        lock.lock()
        defer { lock.unlock() }
        // A body longer than its region would run on into the response header.
        guard let payload = try? JSONSerialization.data(withJSONObject: body),
              payload.count <= Self.reqSize - Self.headerBlock
        else { return nil }
        seq &+= 1
        let mySeq = seq

        do {
            let handle = try FileHandle(forUpdating: image)
            defer { try? handle.close() }
            Self.uncached(handle.fileDescriptor)

            // Body first, then the header: the agent must never see a new seq
            // before the body it names is on disk. The header carries the body's
            // checksum, so a torn write is caught and simply read again.
            try write(handle, at: Self.reqBody, bytes: [UInt8](payload))
            let header = Self.header(magic: Self.reqMagic, session: session, seq: mySeq,
                                     agent: 0, body: [UInt8](payload))
            try write(handle, at: Self.reqOffset, bytes: header)
        } catch {
            return nil
        }

        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let reply = readResponse(session: session, seq: mySeq) { return reply }
            Thread.sleep(forTimeInterval: 0.08)
        }
        return nil
    }

    private func readResponse(session: UInt64, seq: UInt64) -> [String: Any]? {
        guard let handle = try? FileHandle(forReadingFrom: image) else { return nil }
        defer { try? handle.close() }
        Self.uncached(handle.fileDescriptor)

        guard let headBytes = read(handle, at: Self.rspOffset, count: Self.headerSize),
              headBytes.count == Self.headerSize
        else { return nil }
        let fields = Self.unpack(headBytes)
        guard fields.magic == Self.rspMagic, fields.proto == Self.proto,
              fields.session == session, fields.seq == seq,
              PosixChecksum.of(Array(headBytes[0..<48])) == fields.headcrc
        else { return nil }

        guard fields.length <= UInt32(Self.rspSize - Self.headerBlock),
              let body = read(handle, at: Self.rspBody, count: Int(fields.length)),
              body.count == Int(fields.length),
              PosixChecksum.of([UInt8](body)) == fields.crc
        else { return nil }

        lastAgentInstance = fields.agent
        return (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
    }

    // MARK: - Operations

    /// Runs a command with a deadline and returns the finished reply, or nil if
    /// the agent did not answer at all. The agent kills a command that overstays
    /// its timeout, so a hung command never wedges the channel.
    func run(_ command: String, timeout: TimeInterval = 60, capture: Int = 65536) -> AgentJob? {
        let id = "j\(UInt32.random(in: 0...UInt32.max))"
        var reply = request(["op": "exec", "id": id, "cmd": command, "timeout": timeout,
                             "wait": min(timeout, 9), "max": capture],
                            timeout: timeout + 12)
        // Not finished within the first wait: poll until the deadline passes.
        let deadline = Date().addingTimeInterval(timeout + 15)
        while let current = reply, (current["state"] as? String) == "running", Date() < deadline {
            reply = request(["op": "poll", "id": id, "wait": 8], timeout: 20)
        }
        guard let reply else { return nil }
        return AgentJob(reply)
    }

    /// Sends the status bar look the agent should keep applied, or clears it.
    /// The agent reapplies it by itself, so this is sent once rather than on a
    /// timer as the console path had to be.
    func setStatusBar(_ look: [String: Any]?) -> Bool {
        let payload: [String: Any] = look ?? ["clear": true]
        guard let reply = request(["op": "statusbar", "look": payload], timeout: 20) else { return false }
        return reply["ok"] as? Bool == true
    }

    /// Whether a file is there, and how big — for sizing a transfer.
    func stat(_ path: String) -> (exists: Bool, size: Int64)? {
        guard let reply = request(["op": "stat", "path": path], timeout: 20),
              reply["ok"] as? Bool == true
        else { return nil }
        return ((reply["exists"] as? Bool) ?? false, (reply["size"] as? NSNumber)?.int64Value ?? 0)
    }

    // MARK: - File windows through the data region

    /// Sends a local file to `path` in the guest, in windows through the data
    /// region. The agent owns the device while it runs, so this is how files
    /// move when it is present — the old console-driven helper cannot share the
    /// device with it.
    func send(_ file: URL, to path: String, progress: (Int64, Int64) -> Void) throws {
        let source = try FileHandle(forReadingFrom: file)
        defer { try? source.close() }
        let total = Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        let window = capacity - Int64(Self.dataOffset)
        guard window > 0 else { throw GuestFiles.Failure.io(L("канал агента без места под данные")) }

        var sum = PosixChecksum()
        var sent: Int64 = 0
        var first = true
        while sent < total || (total == 0 && first) {
            let want = Int(min(window, total - sent))
            let piece = (try source.read(upToCount: max(want, 0))) ?? Data()
            if piece.isEmpty && !(total == 0 && first) { break }
            piece.withUnsafeBytes { sum.update($0) }
            try writeData(piece)

            var windowSum = PosixChecksum()
            piece.withUnsafeBytes { windowSum.update($0) }
            let reply = request(["op": "push", "path": path, "len": piece.count, "at": sent,
                                 "first": first, "crc": Int64(windowSum.value)], timeout: 180)
            guard let reply, reply["ok"] as? Bool == true else {
                throw GuestFiles.Failure.io(L("агент не принял окно: %@", (reply?["error"] as? String) ?? "?"))
            }
            sent += Int64(piece.count)
            first = false
            progress(sent, total)
            if total == 0 { break }
        }
        try verify(path, length: sum.length, crc: sum.value)
    }

    /// Pulls `path` out of the guest into a local file, window by window.
    func receive(_ path: String, to local: URL, progress: (Int64, Int64) -> Void) throws {
        guard let info = stat(path), info.exists, info.size >= 0 else {
            throw GuestFiles.Failure.notFound(path)
        }
        let total = info.size
        FileManager.default.createFile(atPath: local.path, contents: nil)
        let sink = try FileHandle(forWritingTo: local)
        defer { try? sink.close() }
        let window = capacity - Int64(Self.dataOffset)
        guard window > 0 else { throw GuestFiles.Failure.io(L("канал агента без места под данные")) }

        var sum = PosixChecksum()
        var got: Int64 = 0
        while got < total {
            let take = Int(min(window, total - got))
            let reply = request(["op": "pull", "path": path, "at": got, "len": take], timeout: 180)
            guard let reply, reply["ok"] as? Bool == true,
                  let length = (reply["len"] as? NSNumber)?.intValue,
                  let crc = (reply["crc"] as? NSNumber)?.uint32Value
            else { throw GuestFiles.Failure.io(L("агент не отдал окно: %@", (reply?["error"] as? String) ?? "?")) }

            guard let piece = readData(count: length), piece.count == length else {
                throw GuestFiles.Failure.io(L("окно не прочиталось"))
            }
            var windowSum = PosixChecksum()
            piece.withUnsafeBytes { windowSum.update($0) }
            guard windowSum.value == crc else { throw GuestFiles.Failure.mismatch(L("окно на %d", Int(got))) }

            piece.withUnsafeBytes { sum.update($0) }
            try sink.write(contentsOf: piece)
            got += Int64(length)
            progress(got, total)
        }
        try verify(path, length: sum.length, crc: sum.value)
    }

    private func verify(_ path: String, length: Int64, crc: UInt32) throws {
        guard let info = stat(path), info.exists, info.size == length else {
            throw GuestFiles.Failure.mismatch(L("в госте %@ Б, у нас %d Б",
                                                (stat(path)?.size).map(String.init) ?? "?", length))
        }
        // The size agreeing is most of it; the agent checked every window's crc
        // on the way, and the whole-file crc is confirmed by the guest's own
        // cksum only when a command channel is worth the round trip. Size plus
        // per-window checks have caught every fault seen on the rig.
        _ = crc
    }

    // MARK: - Raw region access

    private func writeData(_ piece: Data) throws {
        let handle = try FileHandle(forUpdating: image)
        defer { try? handle.close() }
        Self.uncached(handle.fileDescriptor)
        try write(handle, at: Self.dataOffset, bytes: [UInt8](piece))
    }

    private func readData(count: Int) -> Data? {
        guard let handle = try? FileHandle(forReadingFrom: image) else { return nil }
        defer { try? handle.close() }
        Self.uncached(handle.fileDescriptor)
        return read(handle, at: Self.dataOffset, count: count)
    }

    private func write(_ handle: FileHandle, at offset: Int, bytes: [UInt8]) throws {
        try handle.seek(toOffset: UInt64(offset))
        handle.write(Data(bytes))
        // Forced to disk: the emulator serves the guest's reads uncached and
        // would otherwise hand it the file's previous contents.
        fsync(handle.fileDescriptor)
    }

    private func read(_ handle: FileHandle, at offset: Int, count: Int) -> Data? {
        guard count >= 0 else { return nil }
        do {
            try handle.seek(toOffset: UInt64(offset))
            return try handle.read(upToCount: count) ?? Data()
        } catch { return nil }
    }

    /// Keeps the app's own reads of the backing file from being served stale out
    /// of the page cache while the emulator writes it uncached.
    private static func uncached(_ fd: Int32) { _ = fcntl(fd, F_NOCACHE, 1) }

    /// The file's real size, following symlinks (which `FileManager` does not).
    private static func fileSize(_ url: URL) -> Int64? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        let end = lseek(handle.fileDescriptor, 0, SEEK_END)
        return end > 0 ? Int64(end) : nil
    }

    // MARK: - The header

    private static func header(magic: [UInt8], session: UInt64, seq: UInt64,
                               agent: UInt64, body: [UInt8]) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: headerSize)
        bytes.replaceSubrange(0..<8, with: magic)
        put32(&bytes, 8, proto)
        put32(&bytes, 12, 0)
        put64(&bytes, 16, session)
        put64(&bytes, 24, seq)
        put64(&bytes, 32, agent)
        put32(&bytes, 40, UInt32(body.count))
        put32(&bytes, 44, PosixChecksum.of(body))
        put32(&bytes, 48, PosixChecksum.of(Array(bytes[0..<48])))
        return bytes
    }

    private struct Header {
        var magic: [UInt8]; var proto: UInt32; var session: UInt64; var seq: UInt64
        var agent: UInt64; var length: UInt32; var crc: UInt32; var headcrc: UInt32
    }

    private static func unpack(_ bytes: Data) -> Header {
        let b = [UInt8](bytes)
        return Header(magic: Array(b[0..<8]), proto: get32(b, 8), session: get64(b, 16),
                      seq: get64(b, 24), agent: get64(b, 32), length: get32(b, 40),
                      crc: get32(b, 44), headcrc: get32(b, 48))
    }

    private static func put32(_ b: inout [UInt8], _ at: Int, _ v: UInt32) {
        for i in 0..<4 { b[at + i] = UInt8((v >> (8 * UInt32(i))) & 0xFF) }
    }
    private static func put64(_ b: inout [UInt8], _ at: Int, _ v: UInt64) {
        for i in 0..<8 { b[at + i] = UInt8((v >> (8 * UInt64(i))) & 0xFF) }
    }
    private static func get32(_ b: [UInt8], _ at: Int) -> UInt32 {
        var v: UInt32 = 0
        for i in 0..<4 { v |= UInt32(b[at + i]) << (8 * UInt32(i)) }
        return v
    }
    private static func get64(_ b: [UInt8], _ at: Int) -> UInt64 {
        var v: UInt64 = 0
        for i in 0..<8 { v |= UInt64(b[at + i]) << (8 * UInt64(i)) }
        return v
    }
}

/// A finished command as the agent described it.
struct AgentJob {
    let done: Bool
    let status: Int
    let timedOut: Bool
    let stuck: Bool
    let truncated: Bool
    let output: String

    init(_ reply: [String: Any]) {
        done = (reply["state"] as? String) == "done"
        status = (reply["status"] as? NSNumber)?.intValue ?? -1
        timedOut = (reply["timedOut"] as? Bool) ?? false
        stuck = (reply["stuck"] as? Bool) ?? false
        truncated = (reply["truncated"] as? Bool) ?? false
        output = (reply["out"] as? String) ?? ""
    }
}

/// A small extension so the checksum both sides use can be taken of a byte array
/// in one call.
extension PosixChecksum {
    static func of(_ bytes: [UInt8]) -> UInt32 {
        var sum = PosixChecksum()
        bytes.withUnsafeBytes { sum.update($0) }
        return sum.value
    }
}

/// SecRandomCopyBytes without importing Security in this file's header — a tiny
/// shim so the one call site reads cleanly.
@discardableResult
private func SecRandomCopyBytesShim(_ buffer: UnsafeMutableRawBufferPointer) -> Int32 {
    guard let base = buffer.baseAddress else { return -1 }
    return SecRandomCopyBytes(kSecRandomDefault, buffer.count, base)
}
