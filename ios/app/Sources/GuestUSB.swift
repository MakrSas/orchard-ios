import Foundation
import Darwin

/// The USB host for the guest's own port, and usbmux on top of it.
///
/// The emulated device exports its USB port through a socket: it dials out and
/// expects whoever listens to act as the host. The stock setup answers that with
/// a Linux VM running usbmuxd, which is the one thing that VM is for. This is
/// that host, in the app: it enumerates the device, finds the usbmux interface,
/// runs the mux protocol and the cut-down TCP inside it, and hands out channels
/// to the ports the guest's services listen on — `restored` during a restore,
/// lockdownd on an installed system.
///
/// It is the same role `netlab/muxd.py` plays on the rig, so the two agree about
/// the wire formats: the emulator.s USB socket protocol header for the socket, and usbmuxd's own
/// `device.c` for the mux header (sixteen bytes from version two on) and the TCP
/// header whose window is shifted down by eight bits.
///
/// Everything here blocks; the link owns a thread of its own.
final class GuestUSB {
    /// What the guest's services listen on.
    enum Port: UInt16 {
        /// `restored`, which drives a restore and answers a plain QueryType.
        case restored = 0xF27E
        /// Where the restore's filesystem image goes, once restored asks for it.
        case asr = 12345
    }

    enum Failure: LocalizedError {
        case usb(String)
        case refused(UInt16)
        case gone
        /// The device answered NAK for the whole of a transfer's patience.
        ///
        /// Fatal where something was expected — a descriptor, a packet the
        /// device asked for — and not an error at all on a read that was merely
        /// looking: there, NAK means "nothing yet".
        case notReady

        var errorDescription: String? {
            switch self {
            case .usb(let what):  return L("USB гостя: %@", what)
            case .refused(let p): return L("Гость отказал в порту %d", Int(p))
            case .gone:           return L("Гость отключился")
            case .notReady:       return L("устройство не отвечает (NAK)")
            }
        }
    }

    // MARK: - The socket the emulator dials

    private let socketName: String
    private var listener: Int32 = -1
    private var link: Int32 = -1

    /// Set once the device has answered and the mux link is up.
    private(set) var deviceReady = false
    /// The device's serial, as it describes itself. Only printable characters:
    /// the guest pads the string with blanks and control bytes.
    private(set) var serial = ""
    private(set) var productID: UInt16 = 0

    /// Called on the link's thread as the device appears and goes away.
    var onReady: ((Bool) -> Void)?

    init(socketName: String = VMConfig.usbSocketName) {
        self.socketName = socketName
    }

    /// Starts listening. The emulator dials this socket once, as it starts up,
    /// so this has to be in place before the machine does — and it stays put for
    /// the whole run, because there is no second attempt.
    func listen() throws {
        let directory = VMConfig.socketDirectory
        let path = directory + "/" + socketName
        unlink(path)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw Failure.usb("socket(): \(errnoText())") }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        addr.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        // AF_UNIX keeps the path itself, and 104 bytes cannot hold an app
        // container path — hence the bare name, resolved against the directory
        // the emulator chdir's into.
        let name = Array(socketName.utf8)
        guard name.count < MemoryLayout.size(ofValue: addr.sun_path) else {
            Darwin.close(fd)
            throw Failure.usb(L("имя сокета длиннее, чем позволяет AF_UNIX"))
        }
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            raw.copyBytes(from: name)
        }

        let saved = FileManager.default.currentDirectoryPath
        FileManager.default.changeCurrentDirectoryPath(directory)
        defer { FileManager.default.changeCurrentDirectoryPath(saved) }

        let bound = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                bind(fd, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bound == 0 else {
            let reason = errnoText()
            Darwin.close(fd)
            throw Failure.usb("bind(\(socketName)): \(reason)")
        }
        guard Darwin.listen(fd, 1) == 0 else {
            let reason = errnoText()
            Darwin.close(fd)
            throw Failure.usb("listen(): \(reason)")
        }
        listener = fd

        let thread = Thread { [weak self] in self?.serve() }
        thread.name = "orchard.usb-host"
        thread.stackSize = 512 * 1024
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func stop() {
        deviceReady = false
        if link >= 0 { Darwin.close(link); link = -1 }
        if listener >= 0 { Darwin.close(listener); listener = -1 }
    }

    private func errnoText() -> String { String(cString: strerror(errno)) }

    /// Waits for the emulator, then brings the device up and pumps the link.
    private func serve() {
        let accepted = accept(listener, nil, nil)
        guard accepted >= 0 else { return }
        var one: Int32 = 1
        setsockopt(accepted, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        link = accepted
        LogCapture.shared.note(L("USB гостя: эмулятор подключился"))

        // The guest's USB stack answers long after the emulator dials, and the
        // emulator does not dial twice, so bring-up is retried on this link.
        while link >= 0 && !deviceReady {
            Thread.sleep(forTimeInterval: 3)
            do {
                try bringUp()
                deviceReady = true
                LogCapture.shared.note(L("USB гостя: устройство %04x, серийный %@",
                                         Int(productID), serial))
                onReady?(true)
            } catch Failure.gone {
                break
            } catch {
                continue
            }
        }

        pump()
        deviceReady = false
        onReady?(false)
    }

    // MARK: - Enumeration

    private var endpointIn: UInt8 = 0
    private var endpointOut: UInt8 = 0
    private var maxPacket = 512

    private func bringUp() throws {
        try reset()
        let device = try controlIn(request: 6, value: 0x0100, index: 0, length: 18)
        productID = UInt16(device[10]) | (UInt16(device[11]) << 8)
        serial = try string(at: device[16])
        let configurations = Int(device[17])

        // Newest configuration first, as usbmuxd does: on a device offering
        // several, the mux interface is in the last one.
        for index in stride(from: configurations - 1, through: 0, by: -1) {
            let head = try controlIn(request: 6, value: UInt16(0x0200 | index), index: 0, length: 9)
            let total = Int(head[2]) | (Int(head[3]) << 8)
            let raw = try controlIn(request: 6, value: UInt16(0x0200 | index), index: 0, length: UInt16(total))
            guard let found = GuestUSB.findMux(raw) else { continue }
            endpointIn = found.endpointIn
            endpointOut = found.endpointOut
            maxPacket = found.maxPacket
            try controlOut(request: 9, value: UInt16(raw[5]), index: 0)
            try handshake()
            return
        }
        throw Failure.usb(L("у гостя нет интерфейса usbmux"))
    }

    /// The usbmux interface, by class 255 / subclass 254 / protocol 2, and its
    /// two bulk endpoints.
    private static func findMux(_ raw: [UInt8]) -> (endpointIn: UInt8, endpointOut: UInt8, maxPacket: Int)? {
        var at = 0
        var inMux = false
        var epIn: UInt8 = 0, epOut: UInt8 = 0, packet = 0
        while at + 2 <= raw.count {
            let length = Int(raw[at]), kind = raw[at + 1]
            if length == 0 { break }
            if kind == 4, at + 8 < raw.count {
                inMux = raw[at + 5] == 255 && raw[at + 6] == 254 && raw[at + 7] == 2
            } else if kind == 5, inMux, at + 6 < raw.count, raw[at + 3] & 0x03 == 2 {
                let address = raw[at + 2]
                packet = Int(raw[at + 4]) | (Int(raw[at + 5]) << 8)
                if address & 0x80 != 0 { epIn = address & 0x0F } else { epOut = address & 0x0F }
            }
            at += length
        }
        guard epIn != 0, epOut != 0 else { return nil }
        return (epIn, epOut, packet == 0 ? 512 : packet)
    }

    // MARK: - The socket's own framing

    private enum Packet: UInt8 { case request = 1, response = 2, reset = 3, cancel = 4 }
    private enum Token: UInt8 { case setup = 0x2D, tokenIn = 0x69, tokenOut = 0xE1 }

    private var nextTransfer: UInt64 = 1
    private let linkLock = NSLock()

    private func send(_ bytes: [UInt8]) throws {
        var sent = 0
        while sent < bytes.count {
            let n = bytes[sent...].withUnsafeBufferPointer { p in
                Darwin.send(link, p.baseAddress, bytes.count - sent, 0)
            }
            guard n > 0 else { throw Failure.gone }
            sent += n
        }
    }

    private func receive(_ count: Int) throws -> [UInt8] {
        var out = [UInt8](repeating: 0, count: count)
        var filled = 0
        while filled < count {
            let n = out[filled...].withUnsafeMutableBufferPointer { p in
                Darwin.recv(link, p.baseAddress, count - filled, 0)
            }
            guard n > 0 else { throw Failure.gone }
            filled += n
        }
        return out
    }

    private func reset() throws {
        linkLock.lock(); defer { linkLock.unlock() }
        try send([Packet.reset.rawValue])
        Thread.sleep(forTimeInterval: 0.3)
    }

    /// One USB packet, retried while the device answers NAK — which is not an
    /// error but "not ready", exactly as a real host controller sees it.
    @discardableResult
    private func transfer(_ token: Token, endpoint: UInt8, data: [UInt8] = [],
                          length: Int? = nil, retries: Int = 200,
                          delay: TimeInterval = 0.01) throws -> [UInt8] {
        linkLock.lock(); defer { linkLock.unlock() }
        let want = length ?? data.count
        for _ in 0..<max(1, retries) {
            let id = nextTransfer
            nextTransfer += 1

            var request = [UInt8]()
            request.append(Packet.request.rawValue)
            request.append(0)                               // addr
            request.append(contentsOf: le32(UInt32(bitPattern: Int32(token.rawValue))))
            request.append(endpoint)
            request.append(contentsOf: le64(id))
            request.append(contentsOf: le32(0))             // stream
            request.append(0)                               // short_not_ok
            request.append(0)                               // int_req
            request.append(contentsOf: le16(UInt16(want)))
            if token != .tokenIn { request.append(contentsOf: data) }
            try send(request)

            while true {
                let kind = try receive(1)[0]
                guard kind == Packet.response.rawValue else { throw Failure.usb("packet type \(kind)") }
                let header = try receive(20)
                let replyID = u64(header, 6)
                let status = Int32(bitPattern: u32(header, 14))
                let length = Int(u16(header, 18))
                let pid = Int32(bitPattern: u32(header, 1))
                var body = [UInt8]()
                if length > 0, status != -6, pid == Int32(Token.tokenIn.rawValue) {
                    body = try receive(length)
                }
                // An ASYNC reply is a promise; the real one follows with the
                // same id.
                if replyID != id || status == -6 { continue }
                if status == -2 { break }                    // NAK: ask again
                guard status == 0 else { throw Failure.usb("status \(status)") }
                return body
            }
            Thread.sleep(forTimeInterval: delay)
        }
        throw Failure.notReady
    }

    private func control(_ type: UInt8, request: UInt8, value: UInt16, index: UInt16,
                         length: UInt16 = 0, data: [UInt8] = []) throws -> [UInt8] {
        var setup = [type, request]
        setup += le16(value) + le16(index) + le16(length)
        try transfer(.setup, endpoint: 0, data: setup)

        let incoming = type & 0x80 != 0
        var out = [UInt8]()
        if length > 0 {
            if incoming {
                out = try transfer(.tokenIn, endpoint: 0, length: Int(length))
            } else {
                try transfer(.tokenOut, endpoint: 0, data: Array(data.prefix(Int(length))))
            }
        }
        // The status stage runs the other way and carries nothing.
        try transfer(incoming ? .tokenOut : .tokenIn, endpoint: 0, length: 0)
        return out
    }

    private func controlIn(request: UInt8, value: UInt16, index: UInt16, length: UInt16) throws -> [UInt8] {
        try control(0x80, request: request, value: value, index: index, length: length)
    }

    private func controlOut(request: UInt8, value: UInt16, index: UInt16) throws {
        _ = try control(0x00, request: request, value: value, index: index)
    }

    private func string(at index: UInt8) throws -> String {
        guard index != 0 else { return "" }
        let raw = try controlIn(request: 6, value: UInt16(0x0300) | UInt16(index), index: 0x0409, length: 255)
        // A string descriptor is UTF-16, little-endian, behind a length byte.
        let end = min(Int(raw[0]), raw.count)
        guard end > 2 else { return "" }
        var units = [UInt16]()
        var at = 2
        while at + 1 < end {
            units.append(UInt16(raw[at]) | (UInt16(raw[at + 1]) << 8))
            at += 2
        }
        let text = String(decoding: units, as: UTF16.self)
        return text.filter { !$0.unicodeScalars.contains { s in s.value < 0x20 } }
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - The mux protocol

    /// The largest packet the device will take in one go.
    ///
    /// `muxd.py` sends up to 48 KB and gets away with it on small traffic, but
    /// under a restore the guest stops taking them: its mux driver posts 16 KB
    /// receive buffers, which is what `usbmuxd` calls the MRU. Anything larger
    /// and the device simply never accepts the packet.
    private static let usbMTU = 16384

    private static let muxMagic: UInt32 = 0xFEED_FACE
    private enum Proto: UInt32 { case version = 0, control = 1, setup = 2, tcp = 6 }

    private var muxVersion: UInt32 = 0
    private var txSeq: UInt16 = 0
    private var rxSeq: UInt16 = 0xFFFF
    private var inbound = [UInt8]()

    private func handshake() throws {
        try sendMux(.version, header: be32(2) + be32(0) + be32(0))
        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline {
            guard let (proto, body) = try readMux() else {
                Thread.sleep(forTimeInterval: 0.01)
                continue
            }
            if proto == Proto.version.rawValue, body.count >= 12 {
                muxVersion = min(u32be(body, 0), 2)
                if muxVersion >= 2 { try sendMux(.setup, payload: [0x07]) }
                return
            }
        }
        throw Failure.usb(L("mux не ответил на VERSION"))
    }

    /// Held across a whole outgoing packet.
    ///
    /// The mux numbers every packet it sends, and the device drops anything out
    /// of order. Two threads write here — the one driving a transfer and the
    /// one answering the device — so the number and the bytes it belongs to
    /// have to leave together. Recursive because `sendTcp` takes it and then
    /// calls in here. `muxd.py` gets this for free: one thread owns the link.
    private let sendLock = NSRecursiveLock()

    private func sendMux(_ proto: Proto, header: [UInt8] = [], payload: [UInt8] = []) throws {
        sendLock.lock(); defer { sendLock.unlock() }
        let size = (muxVersion < 2 ? 8 : 16) + header.count + payload.count
        var head = be32(proto.rawValue) + be32(UInt32(size))
        if muxVersion >= 2 {
            if proto == .setup { txSeq = 0; rxSeq = 0xFFFF }
            head += be32(GuestUSB.muxMagic) + be16(txSeq) + be16(rxSeq)
            txSeq &+= 1
        }
        let packet = head + header + payload
        try transfer(.tokenOut, endpoint: endpointOut, data: packet, retries: 2000, delay: 0.001)
        // A transfer that ends on a packet boundary needs a zero-length packet,
        // or the device waits for the rest of it.
        if packet.count % maxPacket == 0 {
            try transfer(.tokenOut, endpoint: endpointOut, data: [], retries: 2000, delay: 0.001)
        }
    }

    /// Held across a whole incoming packet, for the same reason as `sendLock`.
    ///
    /// Two threads read: the one pumping the link and whichever one is waiting
    /// on a channel. The bytes arrive in pieces and are reassembled in
    /// `inbound`, so without this they interleave and the buffer stops making
    /// sense — the device's acknowledgements are then lost, its window never
    /// reopens, and a long transfer stops dead with both threads still busy.
    private let recvLock = NSRecursiveLock()

    /// Reads whatever the device has, and returns one whole mux packet if the
    /// bytes so far make up one.
    private func readMux() throws -> (UInt32, [UInt8])? {
        recvLock.lock(); defer { recvLock.unlock() }
        if let packet = takeMux() { return packet }
        // A NAK here is the device saying "nothing for you", which is the usual
        // answer on an idle link — not a failure. Treating it as one was enough
        // to stop the mux from ever coming up.
        let chunk: [UInt8]
        do {
            chunk = try transfer(.tokenIn, endpoint: endpointIn, length: GuestUSB.usbMTU,
                                 retries: 1, delay: 0)
        } catch Failure.notReady {
            return nil
        }
        guard !chunk.isEmpty else { return nil }
        inbound += chunk
        return takeMux()
    }

    private func takeMux() -> (UInt32, [UInt8])? {
        guard inbound.count >= 8 else { return nil }
        let proto = u32be(inbound, 0)
        let total = Int(u32be(inbound, 4))
        guard total >= 8, inbound.count >= total else { return nil }
        let raw = Array(inbound[0..<total])
        inbound.removeFirst(total)
        var headerSize = 8
        if muxVersion >= 2, raw.count >= 16 {
            // Echoing the device's own number back is what keeps a session alive.
            rxSeq = u16be(raw, 12)
            headerSize = 16
        }
        return (proto, Array(raw[headerSize...]))
    }

    // MARK: - TCP inside the mux

    /// One connection to a port on the guest.
    final class Channel {
        fileprivate let host: GuestUSB
        fileprivate let localPort: UInt16
        let remotePort: UInt16
        fileprivate var txSeq: UInt32 = 0
        fileprivate var txAck: UInt32 = 0
        /// How much of ours the device has acknowledged, and how much more it
        /// is willing to hold. Without both, a long transfer overruns it.
        fileprivate var txAcked: UInt32 = 0
        fileprivate var peerWindow = 0
        fileprivate var inbox = [UInt8]()
        fileprivate var closed = false
        /// Sent but not yet acknowledged, starting at `txAcked` -- kept around
        /// so a stall can resend it instead of just waiting forever. See `write`.
        fileprivate var unacked = [UInt8]()
        /// The last time `txAcked` actually moved. A window stuck at zero room
        /// is normal for a slow guest; no acknowledgement at all for seconds
        /// on end, with bytes still outstanding, is a lost packet.
        fileprivate var lastAckProgress = Date()

        fileprivate init(host: GuestUSB, localPort: UInt16, remotePort: UInt16) {
            self.host = host
            self.localPort = localPort
            self.remotePort = remotePort
        }

        func write(_ bytes: [UInt8]) throws { try host.write(self, bytes) }
        func read(_ count: Int, timeout: TimeInterval = 20) throws -> [UInt8] {
            try host.read(self, count: count, timeout: timeout)
        }
        /// Whatever has arrived, as soon as anything has. ASR's plists come with
        /// no length in front of them, so its reader cannot ask for a count.
        func readAvailable(timeout: TimeInterval = 20) throws -> [UInt8] {
            try host.readAvailable(self, timeout: timeout)
        }
        func close() { host.close(self) }
    }

    private var channels: [UInt16: Channel] = [:]
    private var nextLocalPort: UInt16 = 0xF001
    private let muxLock = NSLock()

    /// Opens a connection to a port on the guest. Blocks until the guest
    /// answers, which during a restore can be a while.
    func connect(to port: Port, timeout: TimeInterval = 30) throws -> Channel {
        try connect(to: port.rawValue, timeout: timeout)
    }

    func connect(to port: UInt16, timeout: TimeInterval = 30) throws -> Channel {
        muxLock.lock()
        let channel = Channel(host: self, localPort: nextLocalPort, remotePort: port)
        nextLocalPort = nextLocalPort == 0xFFFE ? 0xF001 : nextLocalPort + 1
        channels[channel.localPort] = channel
        muxLock.unlock()

        try sendTcp(channel, flags: 0x02)                    // SYN
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            try serviceMux(timeout: 0.2)
            muxLock.lock()
            let established = channel.txAck != 0
            let dead = channel.closed
            muxLock.unlock()
            // Dropped from the table on the way out: a refused port is knocked
            // on again, and thirty dead channels would otherwise pile up.
            if dead { close(channel); throw Failure.refused(port) }
            if established { return channel }
        }
        close(channel)
        throw Failure.refused(port)
    }

    private func sendTcp(_ channel: Channel, flags: UInt8, payload: [UInt8] = []) throws {
        try sendTcpAt(channel, seq: channel.txSeq, flags: flags, payload: payload)
        channel.txSeq &+= UInt32(payload.count) + (flags & 0x02 != 0 ? 1 : 0)
    }

    /// Same wire shape as `sendTcp`, at a sequence number given explicitly
    /// instead of the channel's current one -- for `retransmit`, which has to
    /// send bytes under the sequence number they went out under the first
    /// time, not a new one.
    private func sendTcpAt(_ channel: Channel, seq: UInt32, flags: UInt8, payload: [UInt8] = []) throws {
        sendLock.lock(); defer { sendLock.unlock() }
        var header = be16(channel.localPort) + be16(channel.remotePort)
        header += be32(seq) + be32(channel.txAck)
        header += [5 << 4, flags]
        header += be16(UInt16((256 * 1024) >> 8)) + be16(0) + be16(0)
        try sendMux(.tcp, header: header, payload: payload)
    }

    /// How long without so much as a partial acknowledgement, while bytes are
    /// still outstanding, before assuming one of them was actually lost
    /// rather than the guest just being slow to get to them.
    private static let retransmitAfter: TimeInterval = 2

    /// Resends everything sent but not yet acknowledged, at the sequence
    /// numbers it originally went out under. A lost packet and a lost
    /// acknowledgement look identical from here, and either way the guest's
    /// own mux treats a repeat of bytes it already has as an ordinary
    /// duplicate and just re-acknowledges it -- the same thing a real TCP
    /// stack does on a retransmit.
    private func retransmit(_ channel: Channel) throws {
        muxLock.lock()
        let seq = channel.txAcked
        let payload = channel.unacked
        channel.lastAckProgress = Date()    // don't fire again next loop tick
        muxLock.unlock()
        guard !payload.isEmpty else { return }

        let mtu = GuestUSB.usbMTU - 16 - 20
        var at = 0
        while at < payload.count {
            let end = min(at + mtu, payload.count)
            try sendTcpAt(channel, seq: seq &+ UInt32(at), flags: 0x10, payload: Array(payload[at..<end]))
            at = end
        }
    }

    /// Reads from the device and files away what comes, returning as soon as
    /// one packet has been handled.
    ///
    /// Returning early is the whole point. A caller waiting for room in the
    /// device's window needs to look again the moment an acknowledgement lands,
    /// and sitting out the rest of the timeout instead cost the restore more
    /// than everything else put together: the window is 128 KB, and waiting
    /// 200 ms to notice it had reopened held the transfer to half a megabyte a
    /// second.
    private func serviceMux(timeout: TimeInterval) throws {
        let deadline = Date().addingTimeInterval(timeout)
        repeat {
            // An empty read is the normal answer on a quiet link, and asking
            // again at once turns this into a spin that holds the link away
            // from whoever is actually transferring. A millisecond of patience
            // is what `muxd.py` waits, and it costs nothing here.
            guard let (proto, body) = try readMux() else {
                Thread.sleep(forTimeInterval: 0.001)
                continue
            }
            guard proto == Proto.tcp.rawValue, body.count >= 20 else { continue }
            let source = u16be(body, 0), destination = u16be(body, 2)
            let sequence = u32be(body, 4), acknowledged = u32be(body, 8)
            let flags = body[13]
            let payload = Array(body[20...])
            _ = source
            muxLock.lock()
            let channel = channels[destination]
            muxLock.unlock()
            guard let channel else { continue }

            if flags & 0x04 != 0 {                            // RST
                muxLock.lock(); channel.closed = true; channels[destination] = nil; muxLock.unlock()
                continue
            }
            muxLock.lock()
            channel.peerWindow = Int(u16be(body, 14)) << 8
            if flags & 0x10 != 0, acknowledged != channel.txAcked {
                let newlyAcked = Int(acknowledged &- channel.txAcked)
                if newlyAcked > 0, newlyAcked <= channel.unacked.count {
                    channel.unacked.removeFirst(newlyAcked)
                }
                channel.txAcked = acknowledged
                channel.lastAckProgress = Date()
            }
            muxLock.unlock()

            if flags & 0x02 != 0, flags & 0x10 != 0 {         // SYN|ACK
                muxLock.lock(); channel.txAck = sequence &+ 1; muxLock.unlock()
                try sendTcp(channel, flags: 0x10)
                continue
            }
            if !payload.isEmpty {
                muxLock.lock()
                channel.txAck = sequence &+ UInt32(payload.count)
                channel.inbox += payload
                muxLock.unlock()
                try sendTcp(channel, flags: 0x10)
            }
            if flags & 0x01 != 0 {                            // FIN
                muxLock.lock(); channel.closed = true; muxLock.unlock()
            }
            return
        } while Date() < deadline
    }

    fileprivate func write(_ channel: Channel, _ bytes: [UInt8]) throws {
        // The device's mux refuses PSH outright, so data rides on a plain ACK,
        // and it is sent only as far as the device says it can hold. Past
        // that this waits for the window to reopen -- and if nothing gets
        // acknowledged for a couple of seconds while bytes are still
        // outstanding, `retransmit` resends them (see there for why that's
        // safe). A single dropped USB packet used to stall a multi-gigabyte
        // restore forever with no way back; a real TCP stack would have
        // recovered from it on its own.
        var at = 0
        let mtu = GuestUSB.usbMTU - 16 - 20
        while at < bytes.count {
            muxLock.lock()
            let inFlight = Int(channel.txSeq &- channel.txAcked)
            let room = channel.peerWindow - inFlight
            let dead = channel.closed
            let stalled = inFlight > 0 && Date().timeIntervalSince(channel.lastAckProgress) > GuestUSB.retransmitAfter
            muxLock.unlock()
            if dead { throw Failure.gone }
            if stalled { try retransmit(channel) }
            guard room > 0 else {
                try serviceMux(timeout: 0.005)
                if Date() > writeDeadline(channel) { throw Failure.usb(L("гость не разбирает присланное")) }
                continue
            }
            let end = min(at + min(mtu, room), bytes.count)
            let chunk = Array(bytes[at..<end])
            try sendTcp(channel, flags: 0x10, payload: chunk)
            muxLock.lock(); channel.unacked += chunk; muxLock.unlock()
            at = end
            waited[channel.localPort] = nil
        }
        waited[channel.localPort] = nil
    }

    /// When to give up waiting for the device to make room. Reset by every byte
    /// that does go out, so a slow guest is fine and a stuck one is not.
    private var waited: [UInt16: Date] = [:]

    private func writeDeadline(_ channel: Channel) -> Date {
        if let since = waited[channel.localPort] { return since }
        let deadline = Date().addingTimeInterval(120)
        waited[channel.localPort] = deadline
        return deadline
    }

    fileprivate func read(_ channel: Channel, count: Int, timeout: TimeInterval) throws -> [UInt8] {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            muxLock.lock()
            let have = channel.inbox.count
            let dead = channel.closed
            muxLock.unlock()
            if have >= count {
                muxLock.lock()
                let out = Array(channel.inbox[0..<count])
                channel.inbox.removeFirst(count)
                muxLock.unlock()
                return out
            }
            if dead && have == 0 { throw Failure.gone }
            try serviceMux(timeout: 0.2)
        }
        throw Failure.usb(L("гость не ответил вовремя"))
    }

    fileprivate func readAvailable(_ channel: Channel, timeout: TimeInterval) throws -> [UInt8] {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            muxLock.lock()
            let out = channel.inbox
            channel.inbox.removeAll()
            let dead = channel.closed
            muxLock.unlock()
            if !out.isEmpty { return out }
            if dead { throw Failure.gone }
            try serviceMux(timeout: 0.2)
        }
        throw Failure.usb(L("гость не ответил вовремя"))
    }

    fileprivate func close(_ channel: Channel) {
        muxLock.lock()
        channels[channel.localPort] = nil
        let alreadyClosed = channel.closed
        channel.closed = true
        muxLock.unlock()
        if !alreadyClosed { try? sendTcp(channel, flags: 0x04) }
    }

    /// Keeps the link serviced while nobody is reading a channel, so the device
    /// sees its acknowledgements and the session stays alive.
    private func pump() {
        while link >= 0 {
            do { try serviceMux(timeout: 0.2) }
            catch { break }
        }
    }

    // MARK: - Little helpers

    private func le16(_ v: UInt16) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8)] }
    private func le32(_ v: UInt32) -> [UInt8] { (0..<4).map { UInt8((v >> (8 * $0)) & 0xFF) } }
    private func le64(_ v: UInt64) -> [UInt8] { (0..<8).map { UInt8((v >> (8 * UInt64($0))) & 0xFF) } }
    private func be16(_ v: UInt16) -> [UInt8] { [UInt8(v >> 8), UInt8(v & 0xFF)] }
    private func be32(_ v: UInt32) -> [UInt8] { (0..<4).reversed().map { UInt8((v >> (8 * $0)) & 0xFF) } }
    private func u16(_ b: [UInt8], _ at: Int) -> UInt16 { UInt16(b[at]) | (UInt16(b[at + 1]) << 8) }
    private func u32(_ b: [UInt8], _ at: Int) -> UInt32 {
        (0..<4).reduce(UInt32(0)) { $0 | (UInt32(b[at + $1]) << (8 * UInt32($1))) }
    }
    private func u64(_ b: [UInt8], _ at: Int) -> UInt64 {
        (0..<8).reduce(UInt64(0)) { $0 | (UInt64(b[at + $1]) << (8 * UInt64($1))) }
    }
    private func u16be(_ b: [UInt8], _ at: Int) -> UInt16 { (UInt16(b[at]) << 8) | UInt16(b[at + 1]) }
    private func u32be(_ b: [UInt8], _ at: Int) -> UInt32 {
        (0..<4).reduce(UInt32(0)) { ($0 << 8) | UInt32(b[at + $1]) }
    }
}
