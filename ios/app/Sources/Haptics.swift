import CoreHaptics
import Foundation

/// The guest's vibration, felt on the phone.
///
/// The guest drives its taptic engine the way it plays sound: its audio server
/// renders a waveform for the actuator, and the kernel sends it down the i2s
/// port the AOP drives for it. The emulator library reads that port and
/// condenses it into frames of ten milliseconds — how hard the actuator was
/// driven and at what frequency — which `inferno_haptics_read` hands over.
/// Core Haptics does not take a waveform, but it takes exactly that: while the
/// guest keeps driving, one continuous event plays, and every frame moves its
/// intensity and sharpness.
///
/// Where there is nothing to play it on — a Mac, an iPad without a taptic
/// engine — Core Haptics says so and none of this starts. Nor does anything
/// arrive from a guest without sound: the actuator belongs to the audio
/// hardware, which the machine describes only when guest audio is on.
final class HostHaptics {
    static let shared = HostHaptics()

    /// Whether this device has an engine to play the guest's vibration on.
    static let isSupported = CHHapticEngine.capabilitiesForHardware().supportsHaptics

    private typealias ReadFn = @convention(c) (UnsafeMutableRawPointer?, Int, UInt32) -> Int

    private let lock = NSLock()
    /// Bumped by every start and stop, so that a reader left over from an
    /// earlier start — it may be waiting in the library for another half
    /// second — lets go instead of sharing the frames with its successor.
    private var generation = 0
    private var following = false

    /// Starts following the guest's actuator. Main thread. Does nothing if
    /// already following, if the machine is not running, if the switch is off,
    /// or where there is no engine to play on.
    func start(bridge: QemuBridge = .shared) {
        guard !following, Settings.shared.guestHaptics, bridge.state == .running else { return }
        // A Mac or an iPad has no taptic engine to play on, which is nothing
        // worth a line in the log at every start.
        guard HostHaptics.isSupported else { return }
        guard let symbol = bridge.symbol("inferno_haptics_read") else {
            LogCapture.shared.note(L("Вибрация: в этой сборке библиотеки её нет."))
            return
        }
        let read = unsafeBitCast(symbol, to: ReadFn.self)
        following = true
        let mine = lock.withLock { () -> Int in
            generation += 1
            return generation
        }

        let thread = Thread { [weak self] in self?.pump(read, generation: mine) }
        thread.name = "inferno.haptics"
        thread.stackSize = 256 * 1024
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    /// Main thread.
    func stop() {
        guard following else { return }
        following = false
        lock.withLock { generation += 1 }
    }

    private func isCurrent(_ mine: Int) -> Bool {
        lock.withLock { generation == mine }
    }

    private func pump(_ read: ReadFn, generation mine: Int) {
        let capacity = 64
        // Two floats a frame, as InfernoHapticFrame in ui/inferno-embed.h lays them out.
        let frames = UnsafeMutablePointer<Float>.allocate(capacity: capacity * 2)
        defer { frames.deallocate() }
        let player = HapticPlayer()
        var lastFrame = DispatchTime.now().uptimeNanoseconds

        while isCurrent(mine) {
            // Short waits while a player is kept, so that it is silenced and
            // stopped on time; long ones otherwise, which cost nothing.
            let count = read(frames, capacity, player.isPlaying ? 20 : 500)
            let now = DispatchTime.now().uptimeNanoseconds
            if count > 0 { lastFrame = now }
            for index in 0..<count {
                player.play(level: frames[index * 2], hertz: frames[index * 2 + 1], now: now)
            }
            // A stream stopped in the middle of a vibration sends no closing
            // frame; only the silence says so.
            if count == 0, now - lastFrame > 60_000_000 { player.still(now: now) }
            player.tick(now: now)
        }
        player.stop()
    }

    /// The guest's own Core Haptics drives its actuator at 80 Hz for a
    /// sharpness of 0, 135 Hz for 0.5 and 230 Hz for 1 — read off the
    /// actuator's port on the rig. That is 80 · 2.875^sharpness, and read
    /// backwards it gives the sharpness a frequency was rendered for.
    fileprivate static func sharpness(hertz: Float) -> Float {
        let value = log(hertz / 80) / log(Float(230) / 80)
        return min(max(value, 0), 1)
    }

    /// The library reads the intensity the guest's own engine asked for out
    /// of the drive signal, so it goes on as it is: the guest turns intensity
    /// into amplitude the same way Core Haptics here does, 6 dB for every
    /// halving.
    fileprivate static func intensity(level: Float) -> Float {
        min(max(level, 0), 1)
    }
}

/// Core Haptics, driven one frame at a time.
///
/// Used from the reader's thread only. The engine's handlers arrive from
/// elsewhere and only leave a flag, which the next vibration picks up.
private final class HapticPlayer {
    /// The shortest vibration worth playing. A tap in the guest can be a single
    /// frame, and ten milliseconds of vibration is not felt at all.
    private static let shortestPulse: UInt64 = 20_000_000
    /// How long a silent player is kept before it is stopped. The guest's
    /// stream stutters whenever its cores fall behind, and a gap of a few
    /// frames is that rather than the end of the vibration: a kept player picks
    /// up again at once instead of being started anew.
    private static let hold: UInt64 = 250_000_000

    private var engine: CHHapticEngine?
    private var player: CHHapticAdvancedPatternPlayer?
    private var sharpness: Float = 0.5
    /// When the current stretch of driving began, and when the player is to
    /// fall silent — never sooner than `shortestPulse` after it began.
    private var roseAt: UInt64 = 0
    private var silentAt: UInt64?
    private var silenced = false
    private var reportedFirst = false
    private var lastError = ""

    private let flags = NSLock()
    private var engineLost = true

    var isPlaying: Bool { player != nil }

    func play(level: Float, hertz: Float, now: UInt64) {
        guard level > 0 else { return still(now: now) }
        if hertz > 0 { sharpness = HostHaptics.sharpness(hertz: hertz) }
        // Players do not survive their engine stopping or resetting.
        if player != nil, flags.withLock({ engineLost }) { player = nil }

        let fresh = player == nil || silenced
        if player == nil { begin(level: level, hertz: hertz) }
        guard player != nil else { return }
        if fresh { roseAt = now }
        silentAt = nil
        silenced = false
        send(intensity: HostHaptics.intensity(level: level))
    }

    /// The guest stopped driving.
    func still(now: UInt64) {
        guard player != nil, silentAt == nil else { return }
        silentAt = max(now, roseAt + HapticPlayer.shortestPulse)
    }

    /// Silences a player whose pulse is over, and stops one that has been
    /// silent for long enough.
    func tick(now: UInt64) {
        guard player != nil, let silentAt, now >= silentAt else { return }
        if !silenced {
            send(intensity: 0)
            silenced = true
        }
        if now >= silentAt + HapticPlayer.hold { stop() }
    }

    func stop() {
        guard let player else { return }
        self.player = nil
        silentAt = nil
        silenced = false
        try? player.stop(atTime: CHHapticTimeImmediate)
    }

    private func send(intensity: Float) {
        guard let player else { return }
        do {
            try player.sendParameters([
                CHHapticDynamicParameter(parameterID: .hapticIntensityControl, value: intensity, relativeTime: 0),
                // Added to the event's own sharpness, which is 0.5.
                CHHapticDynamicParameter(parameterID: .hapticSharpnessControl,
                                         value: sharpness - 0.5, relativeTime: 0),
            ], atTime: CHHapticTimeImmediate)
        } catch {
            self.player = nil
            note(error)
        }
    }

    private func begin(level: Float, hertz: Float) {
        do {
            let engine = try runningEngine()
            let event = CHHapticEvent(eventType: .hapticContinuous, parameters: [
                CHHapticEventParameter(parameterID: .hapticIntensity, value: 1),
                CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.5),
            ], relativeTime: 0, duration: 30)
            let player = try engine.makeAdvancedPlayer(with: CHHapticPattern(events: [event], parameters: []))
            // A vibration longer than the event is still one vibration.
            player.loopEnabled = true
            try player.start(atTime: CHHapticTimeImmediate)
            self.player = player
            if !reportedFirst {
                reportedFirst = true
                LogCapture.shared.note(L("Вибрация: гость включил актуатор — уровень %.2f, %.0f Гц",
                                         Double(level), Double(hertz)))
            }
        } catch {
            note(error)
        }
    }

    private func runningEngine() throws -> CHHapticEngine {
        let lost = flags.withLock { () -> Bool in
            defer { engineLost = false }
            return engineLost
        }
        if let engine, !lost { return engine }

        let engine = try self.engine ?? makeEngine()
        self.engine = engine
        try engine.start()
        return engine
    }

    private func makeEngine() throws -> CHHapticEngine {
        let engine = try CHHapticEngine()
        engine.playsHapticsOnly = true
        // Started again for the next vibration, so an idle guest costs the
        // phone nothing.
        engine.isAutoShutdownEnabled = true
        engine.stoppedHandler = { [weak self] _ in self?.markLost() }
        engine.resetHandler = { [weak self] in self?.markLost() }
        return engine
    }

    private func markLost() {
        flags.withLock { engineLost = true }
    }

    /// Once per kind of failure: in the background every frame fails the same way.
    private func note(_ error: Error) {
        let text = error.localizedDescription
        guard text != lastError else { return }
        lastError = text
        LogCapture.shared.note(L("Вибрация: Core Haptics не запустился — %@", text))
    }
}
