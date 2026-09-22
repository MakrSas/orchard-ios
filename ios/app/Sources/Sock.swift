import Foundation
import Darwin

/// A blocking loopback socket.
///
/// The servers we talk to (VNC, QMP, serial) live inside this very process, so
/// there is no network to speak of — no reachability, no interface changes, no
/// retries worth abstracting. Plain BSD sockets on a dedicated thread give a
/// straight answer, `errno` included, where a higher-level stack can sit in an
/// indeterminate state and report nothing at all.
final class Sock {
    private(set) var fd: Int32 = -1

    init() {}

    /// Takes over a descriptor somebody else opened — an accepted connection,
    /// for one.
    init(adopting descriptor: Int32) { fd = descriptor }

    /// Connects to 127.0.0.1:port. Returns nil on success, or a description of
    /// what went wrong.
    func connect(port: UInt16, timeout: TimeInterval = 20) -> String? {
        close()

        let s = socket(AF_INET, SOCK_STREAM, 0)
        guard s >= 0 else { return "socket(): \(String(cString: strerror(errno)))" }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = INADDR_LOOPBACK.bigEndian

        var tv = timeval(tv_sec: Int(timeout), tv_usec: 0)
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        var one: Int32 = 1
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, socklen_t(MemoryLayout<Int32>.size))
        // Do not let a closed peer raise SIGPIPE and take the app down.
        setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))

        let rc = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.connect(s, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        if rc != 0 {
            let reason = String(cString: strerror(errno))
            Darwin.close(s)
            return "connect(127.0.0.1:\(port)): \(reason)"
        }

        fd = s
        return nil
    }

    /// Reads exactly `count` bytes, or fails.
    func readExactly(_ count: Int) -> Data? {
        guard count > 0 else { return Data() }
        var buffer = [UInt8](repeating: 0, count: count)
        var filled = 0
        while filled < count {
            let n = buffer[filled...].withUnsafeMutableBufferPointer { p in
                Darwin.recv(fd, p.baseAddress, count - filled, 0)
            }
            if n <= 0 { return nil }
            filled += n
        }
        return Data(buffer)
    }

    /// Reads whatever is available, up to `max` bytes.
    func readSome(max: Int = 16 * 1024) -> Data? {
        var buffer = [UInt8](repeating: 0, count: max)
        let n = buffer.withUnsafeMutableBufferPointer { p in
            Darwin.recv(fd, p.baseAddress, max, 0)
        }
        if n <= 0 { return nil }
        return Data(buffer[0..<n])
    }

    @discardableResult
    func write(_ bytes: [UInt8]) -> Bool {
        var sent = 0
        while sent < bytes.count {
            let n = bytes[sent...].withUnsafeBufferPointer { p in
                Darwin.send(fd, p.baseAddress, bytes.count - sent, 0)
            }
            if n <= 0 { return false }
            sent += n
        }
        return true
    }

    /// Lets `readSome` wait as long as it takes. For a reader whose silence is
    /// normal — a console with nothing to say — a receive timeout is not an
    /// error but a trap: the first quiet stretch ends the loop for good.
    func waitIndefinitely() {
        guard fd >= 0 else { return }
        var tv = timeval(tv_sec: 0, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    }

    func close() {
        if fd >= 0 {
            Darwin.close(fd)
            fd = -1
        }
    }

    deinit { close() }
}
