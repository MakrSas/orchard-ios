import Foundation
import CoreGraphics

/// Minimal RFB 3.8 client, enough to drive the emulator's own VNC server.
///
/// Only the Raw encoding is requested. That is wasteful on a network, but the
/// server is inside this very process, so the pixels never leave the device and
/// the decoder stays simple enough to trust.
///
/// The whole session runs on one thread with blocking reads: the protocol is a
/// strict request/response sequence, and written that way it reads top to bottom.
final class VNCClient: GuestDisplay {
    private struct Size: Equatable {
        var width: Int
        var height: Int
    }

    private(set) var status: GuestDisplayStatus = .disconnected
    var onStatus: ((GuestDisplayStatus) -> Void)?
    var onFrame: ((CGImage) -> Void)?

    private let sock = Sock()
    private let port: UInt16
    private var thread: Thread?
    private var running = false

    private var size = Size(width: 0, height: 0)
    private var pixels: [UInt32] = []

    init(port: UInt16 = 5900) {
        self.port = port
    }

    // MARK: - Session

    func connect() {
        guard thread == nil else { return }
        report(.connecting)
        running = true

        let thread = Thread { [weak self] in self?.run() }
        thread.name = "inferno.vnc"
        thread.stackSize = 512 * 1024
        self.thread = thread
        thread.start()
    }

    func disconnect() {
        running = false
        sock.close()
        thread = nil
        report(.disconnected)
    }

    private func report(_ new: GuestDisplayStatus) {
        switch new {
        case .connected(let w, let h): LogCapture.shared.note(L("VNC: подключено, экран %d×%d", w, h))
        case .failed(let why):         LogCapture.shared.note("VNC: \(why)")
        default:                       break
        }
        DispatchQueue.main.async {
            self.status = new
            self.onStatus?(new)
        }
    }

    /// The server only starts listening once the emulator is up, so keep trying.
    private func run() {
        var attempt = 0
        while running {
            attempt += 1
            if let problem = sock.connect(port: port) {
                if attempt == 1 || attempt % 10 == 0 {
                    LogCapture.shared.note(L("VNC: %@ — попытка %d", problem, attempt))
                }
                Thread.sleep(forTimeInterval: 1)
                continue
            }
            LogCapture.shared.note(L("VNC: сокет открыт, рукопожатие"))
            if session() { return }        // finished cleanly
            sock.close()
            Thread.sleep(forTimeInterval: 1)
        }
    }

    /// Returns true when the caller should stop retrying.
    private func session() -> Bool {
        guard let _ = sock.readExactly(12) else {
            report(.failed(L("сервер не прислал версию протокола"))); return false
        }
        sock.write(Array("RFB 003.008\n".utf8))

        guard let count = sock.readExactly(1), count[0] > 0,
              let types = sock.readExactly(Int(count[0]))
        else {
            report(.failed(L("сервер отказал в соединении"))); return false
        }
        guard types.contains(1) else {
            report(.failed(L("сервер требует пароль VNC"))); return true
        }
        sock.write([1])                                  // security type: None

        guard let result = sock.readExactly(4),
              result.be32(at: 0) == 0
        else {
            report(.failed(L("сервер отклонил рукопожатие"))); return false
        }

        sock.write([1])                                  // shared session
        guard let header = sock.readExactly(24) else {
            report(.failed(L("нет параметров экрана"))); return false
        }
        let w = Int(header.be16(at: 0))
        let h = Int(header.be16(at: 2))
        let nameLen = Int(header.be32(at: 20))
        if nameLen > 0, sock.readExactly(nameLen) == nil {
            report(.failed(L("оборвано имя сервера"))); return false
        }

        size = Size(width: w, height: h)
        pixels = [UInt32](repeating: 0xFF00_0000, count: max(w * h, 1))
        setPixelFormat()
        setEncodings()
        report(.connected(width: w, height: h))
        requestUpdate(incremental: false)

        while running {
            guard let type = sock.readExactly(1) else {
                report(.failed(L("соединение закрыто"))); return false
            }
            switch type[0] {
            case 0:
                if !readFramebufferUpdate() { return false }
            case 1:                                       // SetColourMapEntries
                guard let h = sock.readExactly(5),
                      sock.readExactly(Int(h.be16(at: 3)) * 6) != nil else { return false }
            case 2:
                break                                     // Bell
            case 3:                                       // ServerCutText
                guard let h = sock.readExactly(7),
                      sock.readExactly(Int(h.be32(at: 3))) != nil else { return false }
            default:
                report(.failed(L("непонятное сообщение сервера"))); return false
            }
        }
        return true
    }

    /// 32bpp true colour laid out so the buffer can be handed to CoreGraphics
    /// unchanged (little-endian 0xAARRGGBB).
    private func setPixelFormat() {
        sock.write([0, 0, 0, 0,
                    32, 24, 0, 1,
                    0, 255, 0, 255, 0, 255,
                    16, 8, 0,
                    0, 0, 0])
    }

    private func setEncodings() {
        sock.write([2, 0, 0, 1, 0, 0, 0, 0])              // Raw
    }

    private func requestUpdate(incremental: Bool) {
        let w = UInt16(size.width), h = UInt16(size.height)
        sock.write([3, incremental ? 1 : 0,
                    0, 0, 0, 0,
                    UInt8(w >> 8), UInt8(w & 0xFF),
                    UInt8(h >> 8), UInt8(h & 0xFF)])
    }

    private func readFramebufferUpdate() -> Bool {
        guard let header = sock.readExactly(3) else { return false }
        var remaining = Int(header.be16(at: 1))

        while remaining > 0 {
            guard let rect = sock.readExactly(12) else { return false }
            let x = Int(rect.be16(at: 0))
            let y = Int(rect.be16(at: 2))
            let w = Int(rect.be16(at: 4))
            let h = Int(rect.be16(at: 6))
            let encoding = Int32(bitPattern: rect.be32(at: 8))

            guard encoding == 0 else {
                report(.failed(L("неподдерживаемая кодировка %d", Int(encoding)))); return false
            }
            if w > 0, h > 0 {
                guard let body = sock.readExactly(w * h * 4) else { return false }
                blit(body, x: x, y: y, w: w, h: h)
            }
            remaining -= 1
        }

        publish()
        requestUpdate(incremental: true)
        return true
    }

    private func blit(_ body: Data, x: Int, y: Int, w: Int, h: Int) {
        guard size.width > 0 else { return }
        body.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: UInt32.self)
            for row in 0..<h {
                let dstStart = (y + row) * size.width + x
                guard dstStart >= 0, dstStart + w <= pixels.count else { continue }
                for col in 0..<w {
                    // Force the alpha byte: the server leaves it as padding.
                    pixels[dstStart + col] = src[row * w + col] | 0xFF00_0000
                }
            }
        }
    }

    private func publish() {
        guard size.width > 0, size.height > 0 else { return }
        let w = size.width, h = size.height
        let data = pixels.withUnsafeBufferPointer { Data(buffer: $0) }
        guard let provider = CGDataProvider(data: data as CFData) else { return }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let image = CGImage(
            width: w, height: h,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: info,
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent)
        else { return }
        DispatchQueue.main.async { self.onFrame?(image) }
    }

    // MARK: - Input

    /// Absolute pointer position in framebuffer coordinates: a tap lands exactly
    /// where the finger is, which is what the emulated multitouch panel expects.
    func send(touch: CGPoint, pressed: Bool) {
        guard case .connected = status else { return }
        let cx = UInt16(max(0, min(size.width - 1, Int(touch.x))))
        let cy = UInt16(max(0, min(size.height - 1, Int(touch.y))))
        sock.write([5, pressed ? 1 : 0,
                    UInt8(cx >> 8), UInt8(cx & 0xFF),
                    UInt8(cy >> 8), UInt8(cy & 0xFF)])
    }

    func send(functionKey: UInt32, pressed: Bool) {
        guard functionKey >= 1, functionKey <= 12 else { return }
        sendKey(0xFFBE + functionKey - 1, pressed: pressed)   // XK_F1 and up
    }

    private func sendKey(_ keysym: UInt32, pressed: Bool) {
        guard case .connected = status else { return }
        sock.write([4, pressed ? 1 : 0, 0, 0,
                    UInt8((keysym >> 24) & 0xFF), UInt8((keysym >> 16) & 0xFF),
                    UInt8((keysym >> 8) & 0xFF), UInt8(keysym & 0xFF)])
    }
}

extension Data {
    func be16(at offset: Int) -> UInt16 {
        let i = index(startIndex, offsetBy: offset)
        return UInt16(self[i]) << 8 | UInt16(self[index(after: i)])
    }

    func be32(at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        for k in 0..<4 {
            value = value << 8 | UInt32(self[index(startIndex, offsetBy: offset + k)])
        }
        return value
    }
}
