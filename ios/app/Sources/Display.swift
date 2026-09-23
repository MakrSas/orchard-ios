import CoreGraphics
import Foundation

/// Where the guest's picture comes from and where its touches go.
///
/// Two implementations: the emulator's VNC server over the loopback, and the
/// emulator's framebuffer read straight out of this process. They are
/// interchangeable so the choice can stay a switch in the settings rather than
/// a rebuild.
protocol GuestDisplay: AnyObject {
    var onFrame: ((CGImage) -> Void)? { get set }
    var onStatus: ((GuestDisplayStatus) -> Void)? { get set }
    var status: GuestDisplayStatus { get }

    func connect()
    func disconnect()

    /// Absolute position in framebuffer pixels: a tap lands where the finger
    /// is, which is what the emulated multitouch panel expects.
    func send(touch: CGPoint, pressed: Bool)
    /// The machine wires the device buttons to F1..F10.
    func send(functionKey: UInt32, pressed: Bool)
}

enum GuestDisplayStatus: Equatable {
    case disconnected
    case connecting
    case connected(width: Int, height: Int)
    case failed(String)

    var size: (width: Int, height: Int)? {
        if case .connected(let w, let h) = self { return (w, h) }
        return nil
    }
}

/// The guest's framebuffer, read where it already is.
///
/// The emulator is a library inside this process, so its display surface is
/// simply memory this app can look at. The library keeps track of which rows
/// the guest redrew and hands over only those; nothing is compared, encoded or
/// sent anywhere. What is left is the copy itself, and the frame that copy is
/// published into.
///
/// Frames rotate through three buffers. An image handed to SwiftUI must not be
/// written to while it is still on screen, and rotating gives every published
/// frame two more frames of life — long enough for the view to have moved on.
/// The alternative, copying the whole screen for each frame, is what the VNC
/// path already does and what this is here to avoid.
final class EmbeddedDisplay: GuestDisplay {
    private typealias ReadFn = @convention(c) (UnsafeMutableRawPointer?, Int, UnsafeMutablePointer<UInt32>?) -> Int32
    private typealias TouchFn = @convention(c) (Int32, Int32, Bool) -> Void
    private typealias KeyFn = @convention(c) (UInt32, Bool) -> Void
    /// Fills two counters: frames the machine showed, and refreshes the main
    /// loop got round to. Missing from older builds of the library.
    private typealias StatsFn = @convention(c) (UnsafeMutablePointer<UInt64>?) -> Void

    /// Matches OrchardFrameResult in ui/orchard-embed.h.
    private enum Result: Int32 {
        case none = 0
        case ok = 1
        case resize = 2
    }

    /// The info block is eight uint32 fields; read by index so no struct
    /// layout has to be assumed across the language boundary.
    private enum Field {
        static let width = 0, height = 1, stride = 2
        static let x = 3, y = 4, w = 5, h = 6
        static let count = 8
    }

    private static let bufferCount = 3

    private(set) var status: GuestDisplayStatus = .disconnected
    var onFrame: ((CGImage) -> Void)?
    var onStatus: ((GuestDisplayStatus) -> Void)?

    private let read: ReadFn
    private let touchFn: TouchFn
    private let keyFn: KeyFn
    private let statsFn: StatsFn?

    private var thread: Thread?
    private var running = false

    private var buffers: [UnsafeMutableRawPointer] = []
    private var bufferBytes = 0
    private var current = 0
    private var size = (width: 0, height: 0)

    /// Available only once the emulator library is loaded and its machine is up.
    init?(bridge: QemuBridge = .shared) {
        guard let read = bridge.symbol("orchard_display_read"),
              let touch = bridge.symbol("orchard_input_touch"),
              let key = bridge.symbol("orchard_input_function_key")
        else { return nil }
        self.read = unsafeBitCast(read, to: ReadFn.self)
        self.touchFn = unsafeBitCast(touch, to: TouchFn.self)
        self.keyFn = unsafeBitCast(key, to: KeyFn.self)
        self.statsFn = bridge.symbol("orchard_display_stats").map { unsafeBitCast($0, to: StatsFn.self) }
    }

    deinit {
        buffers.forEach { $0.deallocate() }
    }

    // MARK: - Session

    func connect() {
        guard thread == nil else { return }
        report(.connecting)
        running = true

        let thread = Thread { [weak self] in self?.pump() }
        thread.name = "orchard.display"
        thread.stackSize = 256 * 1024
        thread.qualityOfService = .userInteractive
        self.thread = thread
        thread.start()
    }

    func disconnect() {
        running = false
        thread = nil
        report(.disconnected)
    }

    private func report(_ new: GuestDisplayStatus) {
        if case .connected(let w, let h) = new {
            LogCapture.shared.note(L("Экран: встроенный вывод, %d×%d", w, h))
        }
        DispatchQueue.main.async {
            self.status = new
            self.onStatus?(new)
        }
    }

    /// Reads at the rate the emulator redraws. A pass with nothing to show
    /// costs one lock and a comparison, so polling is cheaper than arranging to
    /// be woken.
    private func pump() {
        let info = UnsafeMutablePointer<UInt32>.allocate(capacity: Field.count)
        info.initialize(repeating: 0, count: Field.count)
        defer { info.deallocate() }

        var tally = Tally()

        while running {
            let beforeRead = DispatchTime.now().uptimeNanoseconds
            let outcome = Result(rawValue: read(buffers.isEmpty ? nil : buffers[current],
                                                bufferBytes, info)) ?? .none
            let afterRead = DispatchTime.now().uptimeNanoseconds
            tally.readNanos += afterRead - beforeRead

            switch outcome {
            case .resize:
                resize(width: Int(info[Field.width]), height: Int(info[Field.height]))
            case .ok:
                mirror(rect: (x: Int(info[Field.x]), y: Int(info[Field.y]),
                              w: Int(info[Field.w]), h: Int(info[Field.h])),
                       stride: Int(info[Field.stride]))
                publish()
                current = (current + 1) % EmbeddedDisplay.bufferCount
                tally.delivered += 1
            case .none:
                tally.idle += 1
            }
            tally.handNanos += DispatchTime.now().uptimeNanoseconds - afterRead
            closeSecond(&tally)
            report(&tally)
            Thread.sleep(forTimeInterval: 1.0 / 60)
        }
    }

    /// What the last quarter of a minute of the loop looked like.
    private struct Tally {
        var delivered = 0
        var idle = 0
        var readNanos: UInt64 = 0
        var handNanos: UInt64 = 0
        var since = DispatchTime.now().uptimeNanoseconds
        /// Frames the machine showed, one entry for each second closed so far.
        var presents: [Int] = []
        var refreshes: UInt64 = 0
        /// When the second still open began.
        var secondSince = DispatchTime.now().uptimeNanoseconds
    }

    /// Asks the machine, as each second closes, how many frames it showed in it.
    ///
    /// An average over a quarter of a minute cannot tell a swipe from a still
    /// screen: two seconds at thirty frames among thirteen at none read as four.
    /// Counted a second at a time, the swipe reads as thirty. The library's
    /// counters restart on every read, so this is the one place that reads them.
    private func closeSecond(_ tally: inout Tally) {
        let now = DispatchTime.now().uptimeNanoseconds
        guard now - tally.secondSince >= 1_000_000_000 else { return }
        var counters: [UInt64] = [0, 0]
        if let statsFn { counters.withUnsafeMutableBufferPointer { statsFn($0.baseAddress) } }
        tally.presents.append(Int(counters[0]))
        tally.refreshes += counters[1]
        tally.secondSince = now
    }

    /// Says once a quarter of a minute where the frames went.
    ///
    /// The frame counter under the screen only ever knew about the frames that
    /// arrived; it could not tell a guest drawing ten times a second from a
    /// guest drawing forty and losing thirty on the way. These numbers can: what
    /// the machine showed, what reached the screen, and how often QEMU's main
    /// loop — the thing that competes with the vCPUs for the big lock — got
    /// round to asking for a redraw at all. The seconds in which the guest drew
    /// anything are summed up apart, since those are the ones someone watched.
    private func report(_ tally: inout Tally) {
        let now = DispatchTime.now().uptimeNanoseconds
        let elapsed = Double(now - tally.since) / 1_000_000_000
        // Once every fifteen seconds, averaged over the window. A line a second
        // pushed everything else in the log off the screen, and the numbers it
        // carried were the same fifteen times over.
        guard elapsed >= 15 else { return }
        defer {
            // The open second carries over: its frames are still in the
            // machine's counters, and a clock started afresh would fold them
            // into the next second.
            let open = tally.secondSince
            tally = Tally()
            tally.secondSince = open
        }
        // Always, for a macOS guest still being brought up: whether frames
        // reach the app at all, and whether what arrives is anything but black.
        // A blank screen has three different causes — the machine never
        // presents, it presents but nothing is read, or it presents black —
        // and only these three numbers tell them apart.
        if VMConfig.macGuest {
            LogCapture.shared.note(L("Экран: машина показала %d кадров, до приложения дошло %d, последний кадр — %@",
                                     tally.presents.reduce(0, +), tally.delivered, frameLooksBlank()
                                        ? L("чёрный") : L("есть изображение")))
        }

        guard Settings.shared.showFPS, statsFn != nil else { return }

        // A window in which the machine showed nothing and nothing reached the
        // screen says only that the guest was still, so it is left out.
        let shown = tally.presents.reduce(0, +)
        guard shown > 0 || tally.delivered > 0 else { return }
        let drawing = tally.presents.filter { $0 > 0 }.sorted()
        let milliseconds = { (nanos: UInt64) in Double(nanos) / 1_000_000 / elapsed }

        LogCapture.shared.note(L(
            "Кадры: гость показал %.0f/с (в секунды, когда рисовал: медиана %d, лучшая %d, от 30 и выше — %d из %d); дошло %.0f/с, вхолостую %.0f/с; главный цикл %.0f/с; чтение %.0f мс/с, выдача %.0f мс/с",
            Double(shown) / elapsed,
            drawing.isEmpty ? 0 : drawing[drawing.count / 2], drawing.last ?? 0,
            drawing.filter { $0 >= 30 }.count, drawing.count,
            Double(tally.delivered) / elapsed, Double(tally.idle) / elapsed,
            Double(tally.refreshes) / elapsed, milliseconds(tally.readNanos), milliseconds(tally.handNanos)))
    }

    /// Whether the newest frame is black everywhere, sampled on a grid rather
    /// than read in full: a boot logo or a line of text crosses it.
    private func frameLooksBlank() -> Bool {
        guard !buffers.isEmpty, bufferBytes > 0 else { return true }
        let newest = buffers[(current + EmbeddedDisplay.bufferCount - 1) % EmbeddedDisplay.bufferCount]
        let pixels = newest.assumingMemoryBound(to: UInt32.self)
        let count = bufferBytes / 4
        let step = max(1, count / 4096)
        for i in Swift.stride(from: 0, to: count, by: step) where pixels[i] & 0x00FF_FFFF != 0 {
            return false
        }
        return true
    }

    private func resize(width: Int, height: Int) {
        guard width > 0, height > 0 else { return }
        buffers.forEach { $0.deallocate() }
        bufferBytes = width * height * 4
        buffers = (0..<EmbeddedDisplay.bufferCount).map { _ in
            let buffer = UnsafeMutableRawPointer.allocate(byteCount: bufferBytes, alignment: 16)
            // Black rather than whatever was in the page: the first frame after
            // a resize may not cover the screen.
            memset(buffer, 0, bufferBytes)
            return buffer
        }
        current = 0
        size = (width, height)
        report(.connected(width: width, height: height))
    }

    /// Copies what was just read into the other buffers, so every one of them
    /// stays a complete frame while only the changed rows are ever touched.
    private func mirror(rect: (x: Int, y: Int, w: Int, h: Int), stride: Int) {
        guard rect.w > 0, rect.h > 0, buffers.count > 1 else { return }
        let source = buffers[current]
        let span = rect.w * 4
        for (index, destination) in buffers.enumerated() where index != current {
            for row in rect.y..<(rect.y + rect.h) {
                let offset = row * stride + rect.x * 4
                memcpy(destination.advanced(by: offset), source.advanced(by: offset), span)
            }
        }
    }

    private func publish() {
        guard size.width > 0, bufferBytes > 0, current < buffers.count else { return }
        let buffer = buffers[current]
        // No copy: the bytes stay ours, and rotation keeps them still for long
        // enough that the image on screen never sees them change.
        guard let data = CFDataCreateWithBytesNoCopy(nil, buffer.assumingMemoryBound(to: UInt8.self),
                                                     bufferBytes, kCFAllocatorNull),
              let provider = CGDataProvider(data: data)
        else { return }

        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
                                | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let image = CGImage(
            width: size.width, height: size.height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: size.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: info,
            provider: provider,
            decode: nil, shouldInterpolate: false,
            intent: .defaultIntent)
        else { return }

        DispatchQueue.main.async { self.onFrame?(image) }
    }

    // MARK: - Input

    func send(touch: CGPoint, pressed: Bool) {
        guard size.width > 0 else { return }
        touchFn(Int32(touch.x), Int32(touch.y), pressed)
    }

    func send(functionKey: UInt32, pressed: Bool) {
        keyFn(functionKey, pressed)
    }
}
