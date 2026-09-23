import SwiftUI
import os
import CoreGraphics
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#else
import AppKit
#endif

#if os(iOS)
@main
struct OrchardApp: App {
    init() {
        Bootstrap.prepareDocuments()
        // Start capturing before anything can fail, so the reason is on screen.
        LogCapture.shared.start()
        LogCapture.shared.note(L("Сборка приложения: %@", BuildInfo.stamp))
        LogCapture.shared.noteDevice()
        // Must happen before the emulator asks for its translation buffer.
        JIT.prepare()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .preferredColorScheme(.dark)
        }
    }
}

/// The real insets of the device, taken from the window.
///
/// A `GeometryProxy` inside a view that ignores the safe area reports nothing
/// useful about it — the area has already been given up by the time the reader
/// measures. The window still knows, and it keeps reporting the island's share
/// of the top even with the status bar hidden, which is the number that matters
/// for keeping the guest's picture out from under it.
enum DeviceInsets {
    static var current: ScreenInsets {
        let insets = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap(\.windows)
            .first { $0.isKeyWindow }?
            .safeAreaInsets ?? .zero
        return ScreenInsets(top: insets.top, bottom: insets.bottom, left: insets.left, right: insets.right)
    }
}
#else
/// A Mac window has no island and no home indicator to keep clear of.
enum DeviceInsets {
    static var current: ScreenInsets { ScreenInsets() }
}
#endif

// MARK: - Model

/// The guest's picture, kept apart from everything else.
///
/// It arrives as fast as the guest draws, and anything watching the object that
/// holds it is rebuilt just as often. That was the control menu: opened, it was
/// thrown back to its first line dozens of times a second and could not be
/// scrolled at all. Only the screen needs the frames, so only the screen watches
/// them.
final class GuestFrame: ObservableObject {
    @Published var image: CGImage?
    /// How many of those arrived in the last second.
    @Published private(set) var fps: Double = 0

    private var counted = 0
    private var since = Date()

    func deliver(_ frame: CGImage) {
        image = frame
        counted += 1
        // Once a second, not once a frame: the number is for looking at, and
        // republishing it thirty times a second would redraw the label as often
        // as the picture.
        let elapsed = Date().timeIntervalSince(since)
        if elapsed >= 1 {
            fps = Double(counted) / elapsed
            counted = 0
            since = Date()
        }
    }
}

final class VMModel: ObservableObject {
    let picture = GuestFrame()
    #if canImport(UIKit)
    /// The pointer when the screen works as a trackpad; see Trackpad.swift.
    let trackpad = Trackpad()
    #endif
    @Published var displayStatus: GuestDisplayStatus = .disconnected
    @Published var qemuState: QemuBridge.State = .idle
    /// Whether the guest ever got itself an address over the USB link.
    @Published var networkUp = false
    /// What the file transfer is doing, for the banner at the bottom.
    @Published var transfer: TransferState?
    @Published var missing: [String] = VMConfig.missingFiles()
    @Published var jit: JIT.Availability = JIT.status
    /// QEMU is not re-entrant, and it lives inside this process. Once a machine
    /// has been started, a second one in the same process would take the app
    /// down with it, so starting again means relaunching.
    @Published var hasRun = false

    let serial = SerialConsole()
    /// The shell on a socket of its own, away from the kernel's chatter. It
    /// needs the same working link the file transfer does, and asks for it the
    /// same way.
    private(set) lazy var shell = ShellChannel(
        serial: serial,
        linkUp: { [weak self] in self?.linkIsUp ?? false })

    /// The agent inside the guest, once it is found or installed — the app's way
    /// past the single console. Nil while it is being brought up, and for good
    /// on a build or an emulator that cannot have one, in which case everything
    /// falls back to the console as before. Read and written on the main thread.
    private(set) var guestAgent: GuestAgent?
    private var agentBringUpStarted = false
    private lazy var qmp = QMPClient(port: config.qmpPort)
    /// What the next start uses instead of the settings. A restore boots the
    /// same machine with a ramdisk and without the network device, and that is
    /// a property of the run rather than something to save.
    var configOverride: VMConfig?

    var config: VMConfig { configOverride ?? Settings.shared.config }

    /// Chosen when the machine starts and kept for its lifetime: the built-in
    /// path needs the emulator library to be loaded before it can exist at all.
    private var display: GuestDisplay?

    /// Returns nil when the built-in path was asked for and the library does
    /// not have it. Falling back to VNC would be worse than saying so: the
    /// machine was started without a VNC server, so the client would sit there
    /// retrying a port nobody is listening on.
    private func makeDisplay() -> GuestDisplay? {
        var chosen: GuestDisplay
        if config.builtInDisplay {
            guard let embedded = EmbeddedDisplay() else { return nil }
            chosen = embedded
        }
        else {
            chosen = VNCClient(port: config.vncPort)
        }
        chosen.onFrame = { [weak self] image in self?.picture.deliver(image) }
        chosen.onStatus = { [weak self] status in self?.displayStatus = status }
        return chosen
    }

    var framebufferSize: (width: Int, height: Int)? { displayStatus.size }

    var isRunning: Bool { qemuState == .running }

    func refreshFiles() {
        missing = VMConfig.missingFiles()
        needsRestore = missing.isEmpty && !VMConfig.systemInstalled
    }

    /// Everything is in place, but the disk has no system on it yet.
    @Published var needsRestore = VMConfig.missingFiles().isEmpty && !VMConfig.systemInstalled

    /// StikDebug attaches after launch, so the answer changes over time.
    func refreshJIT() {
        jit = JIT.prepare()
        if !jit.isAvailable {
            // Record which allocation strategies the kernel does allow, so the
            // reason is in the log instead of a guess.
            _ = JIT.diagnose()
        }
    }

    func start() {
        refreshFiles()
        refreshJIT()
        // Said out loud, and into the log. The button is disabled in this case,
        // so from outside it is "I press Start and nothing happens" — and the
        // log people send with that report has nothing in it at all.
        guard missing.isEmpty else {
            LogCapture.shared.note(L("Запуск отменён: не хватает файлов — %@", missing.joined(separator: ", ")))
            return
        }
        // A blank disk boots into nothing and looks like a hang. A restore is
        // started the other way round — it brings its own ramdisk — so only a
        // plain start is refused here.
        guard config.restoreRamdiskPath != nil || VMConfig.systemInstalled else {
            LogCapture.shared.note(L("Запуск отменён: на диске ещё нет системы — сначала «Восстановление»"))
            return
        }
        // Which accelerator, said before anything else: under HVF there is no
        // translator, and a report that does not say which one ran is half a
        // report.
        let virtualized = config.virtualization
        if HVF.isInBuild {
            LogCapture.shared.note(HVF.description)
            LogCapture.shared.note(virtualized ? L("Ускорение: HVF") : L("Ускорение: TCG"))
        }
        // Starting without executable memory does not fail — it wedges the
        // vCPU on the first generated instruction, which is far harder to read
        // than a refusal. HVF generates no code, so it needs none.
        guard virtualized || jit.isAvailable else {
            LogCapture.shared.note(L("Запуск отменён: JIT недоступен."))
            return
        }

        // The built-in display registers itself with the machine at the one
        // moment that is safe: after qemu_init, before the main loop.
        if config.builtInDisplay, !config.headless {
            QemuBridge.shared.afterInit = {
                guard let attach = QemuBridge.shared.symbol("orchard_display_attach") else { return }
                unsafeBitCast(attach, to: (@convention(c) () -> Void).self)()
            }
        }

        QemuBridge.shared.onStateChange = { [weak self] state in
            guard let self else { return }
            self.qemuState = state
            if state != .running { HostHaptics.shared.stop() }
            #if os(macOS)
            VMModel.holdAwake(state == .running)
            #endif
            if state == .running {
                // Give qemu_init time to open its sockets.
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    if !self.config.headless {
                        if let display = self.makeDisplay() {
                            self.display = display
                            display.connect()
                        }
                        else {
                            let why = L("в этой сборке библиотеки нет встроенного вывода")
                            LogCapture.shared.note(L("Экран: %@. Переключитесь на VNC в параметрах.", why))
                            self.displayStatus = .failed(why)
                        }
                    }
                    self.serial.follow()
                    self.serial.attachInput(port: self.config.serialPort)
                    // The shell is opened without waiting for anyone to look at
                    // its pane. It waits for the bootstrap's bash by itself, and
                    // having it from the start is what keeps the rest working:
                    // the console channel needs no link, while the guest puts
                    // its own end of the USB network down once it has booted,
                    // and a shell asked for later would be waiting on that.
                    //
                    // Not for macOS: its serial console carries the kernel's
                    // log and no shell, so there is no bash to wait for.
                    if !VMConfig.macGuest { self.shell.connect() }
                }
                // Everything in this block drives an iPhone guest: its battery,
                // taptic engine, status bar, the agent the app installs into it
                // over the console, its packages, and the USB network link it
                // brings up. A macOS guest has none of them, and several would
                // type commands into its console to find out.
                if !VMConfig.macGuest {
                    // The library is loaded by now, so the battery can be handed
                    // over before the guest's driver first asks for it.
                    HostBattery.shared.start()
                    // Nothing arrives until the guest vibrates, and nothing at all
                    // from a guest started without sound.
                    HostHaptics.shared.start()
                    // The status bar: once SpringBoard is up, and again whenever the
                    // phone's own connection changes while the guest follows it.
                    PhoneNetwork.shared.onChange = { [weak self] in
                        guard Settings.shared.statusBarMode == GuestStatusBar.Mode.phone.rawValue else { return }
                        self?.paintStatusBar()
                    }
                    PhoneNetwork.shared.start()
                    // Bring up the agent in the background: install it if this is a
                    // fresh guest, find it if it is already there. It takes over the
                    // status bar and the service commands from the console. Started
                    // after a delay so a fresh boot has reached a shell first.
                    DispatchQueue.main.asyncAfter(deadline: .now() + 20) { self.bringUpAgent() }
                    self.paintStatusBarWhenReady(delay: 30)
                    if self.config.network { self.watchNetwork() }
                    if Settings.shared.autoRepairPackages { self.preparePackages() }
                }
                self.serial.onGuestDeath = { [weak self] in self?.guestDied() }
                if VMConfig.macGuest { self.watchBoot() }
                // Report what the machine is doing once it has had time to boot.
                DispatchQueue.main.asyncAfter(deadline: .now() + 15) {
                    self.inspectMachine()
                }
                // And where the busy thread actually is — the decisive datum.
                DispatchQueue.main.asyncAfter(deadline: .now() + 25) {
                    Sampler.report { LogCapture.shared.note($0) }
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 60) {
                    Sampler.report { LogCapture.shared.note($0) }
                }
                // And once the guest is at its login window or desktop, where
                // the time that matters for frames goes.
                DispatchQueue.main.asyncAfter(deadline: .now() + 150) {
                    Sampler.report { LogCapture.shared.note($0) }
                }
            }
        }
        hasRun = true
        // The emulator appends to its console log rather than starting it
        // afresh (see VMConfig), so the run before is cleared away here.
        _ = truncate(VMConfig.guestConsoleLog.path, 0)
        // reims-vgpu appends to its logs for the life of the file, not of the
        // run, so a phone that had started the machine sixteen times handed
        // over a 13 MB failure log of which this run was 0.7 MB. The run before
        // is kept beside it as .prev, as emulator.log keeps its own.
        let fm = FileManager.default
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        for name in ["reims-vgpu-fail", "reims-vgpu-draw"] {
            let log = tmp.appendingPathComponent(name + ".log")
            let prev = tmp.appendingPathComponent(name + ".prev.log")
            guard fm.fileExists(atPath: log.path) else { continue }
            try? fm.removeItem(at: prev)
            try? fm.moveItem(at: log, to: prev)
            let published = VMConfig.documents.appendingPathComponent(name + ".prev.log")
            try? fm.removeItem(at: published)
            try? fm.copyItem(at: prev, to: published)
        }
        // The machine's own switches win over the app's: they are what makes a
        // stock macOS kernel boot at all.
        QemuBridge.shared.environment = Settings.shared.emulatorEnvironment
            .merging(VMConfig.guestEnvironment) { _, machine in machine }
        QemuBridge.shared.start(arguments: config.arguments())
    }

    /// Stops the machine the only safe way there is.
    ///
    /// Swiping the app away kills the process mid-write; QMP `quit` lets QEMU
    /// unwind its main loop and flush the disks first.
    func shutdown() {
        guard isRunning else { return }
        LogCapture.shared.note(L("Выключение: отправляю QMP quit…"))
        VMModel.publishDisplayLogs()
        qmp.quit { report in LogCapture.shared.note(report) }
        display?.disconnect()
        shell.disconnect()
        serial.stop()
    }

    // MARK: Network

    /// Asks the emulator whether the guest ever configured its end of the link.
    var linkIsUp: Bool {
        guard let fn = QemuBridge.shared.symbol("orchard_net_link_up") else { return false }
        return unsafeBitCast(fn, to: (@convention(c) () -> Bool).self)()
    }

    /// Tells the guest to configure the USB interface itself.
    ///
    /// The emulator can replug the device, reset the bus and re-enumerate it,
    /// and on an image that has been used for a while none of that helps: iOS
    /// reports the link connected, starts a DHCP request, then announces
    /// `NETWORK_CONNECTION 0` and disables the controller. What does help is
    /// saying so from inside, which is what the stock guide has always told
    /// people to do by hand.
    func fixNetwork() {
        // Through the agent when there is one: it reaches the guest off the
        // console, so this works even while the console is busy — which is when
        // the network most often needs a nudge.
        if let agent = guestAgent {
            DispatchQueue.global(qos: .utility).async {
                let job = agent.run("/usr/sbin/ipconfig set en0 DHCP", timeout: 90)
                DispatchQueue.main.async {
                    if job != nil {
                        LogCapture.shared.note(L("Сеть (агент): попросил гостя поднять en0."))
                    } else if !agent.isAlive() {
                        self.guestAgent = nil
                        LogCapture.shared.note(L("Агент пропал — сеть подниму через консоль."))
                        self.fixNetwork()
                    }
                }
            }
            return
        }
        // Never in the middle of somebody else's conversation with the console:
        // a command landing between two lines of a file transfer breaks it, and
        // the poke can always wait for the next round.
        let sent = serial.ifFree {
            LogCapture.shared.note(L("Сеть: прошу гостя поднять en0…"))
            serial.send("/usr/sbin/ipconfig set en0 DHCP\n")
        }
        if !sent { LogCapture.shared.note(L("Сеть: консоль занята, попрошу позже.")) }
    }

    /// Downloads a package and installs it, showing both in the same banner as
    /// every other transfer — so the manager can be closed the moment it starts
    /// and the guest's screen watched instead.
    func installFromRepo(name: String, id: String, version: String, url: URL, size: Int64) {
        guard transfer?.isRunning != true else { return }
        if let why = transferBlocker() { transfer = .failed(why); return }
        transfer = .running(title: L("↓ %@", name), done: 0, total: size)
        LogCapture.shared.note(L("Пакеты: качаю %@", name))

        Task {
            do {
                var request = URLRequest(url: url)
                request.setValue("Telesphoreo APT-HTTP/1.0.592", forHTTPHeaderField: "User-Agent")
                request.setValue("iPhone12,1", forHTTPHeaderField: "X-Machine")
                request.setValue("14.0", forHTTPHeaderField: "X-Firmware")
                request.timeoutInterval = 60

                let (stream, response) = try await URLSession.shared.bytes(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                    throw GuestFiles.Failure.io(L("репозиторий ответил %d",
                                                  (response as? HTTPURLResponse)?.statusCode ?? 0))
                }

                let total = max(response.expectedContentLength, size)
                var body = Data()
                body.reserveCapacity(Int(max(total, 0)))
                var shown = Date.distantPast
                for try await byte in stream {
                    body.append(byte)
                    if Date().timeIntervalSince(shown) > 0.1 {
                        shown = Date()
                        self.transfer = .running(title: L("↓ %@", name), done: Int64(body.count), total: total)
                    }
                }

                let file = FileManager.default.temporaryDirectory
                    .appendingPathComponent("\(id)_\(version).deb")
                try? FileManager.default.removeItem(at: file)
                try body.write(to: file)

                self.transfer = nil
                self.installDEB(file)
            }
            catch {
                self.transfer = .failed(error.localizedDescription)
                LogCapture.shared.note(L("Пакеты: %@ — %@", name, error.localizedDescription))
            }
        }
    }

    /// Takes an app from the catalogue: downloaded here, where the network is
    /// real, then handed to the guest by the same installer the `.ipa` button
    /// uses — so it gets the same channel, the same checking and the same
    /// banner.
    func installCatalogApp(name: String, url: URL, size: Int64) {
        guard transfer?.isRunning != true else { return }
        if let why = transferBlocker() { transfer = .failed(why); return }
        transfer = .running(title: L("↓ %@", name), done: 0, total: size)
        LogCapture.shared.note(L("Каталог: качаю %@", name))

        Task {
            do {
                var request = URLRequest(url: url)
                request.timeoutInterval = 120
                let (stream, response) = try await URLSession.shared.bytes(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else {
                    throw GuestFiles.Failure.io(L("источник ответил %d",
                                                  (response as? HTTPURLResponse)?.statusCode ?? 0))
                }

                let total = max(response.expectedContentLength, size)
                var body = Data()
                body.reserveCapacity(Int(max(total, 0)))
                var shown = Date.distantPast
                for try await byte in stream {
                    body.append(byte)
                    if Date().timeIntervalSince(shown) > 0.1 {
                        shown = Date()
                        self.transfer = .running(title: L("↓ %@", name), done: Int64(body.count), total: total)
                    }
                }

                let file = FileManager.default.temporaryDirectory
                    .appendingPathComponent(GuestFiles.safeName(name) + ".ipa")
                try? FileManager.default.removeItem(at: file)
                try body.write(to: file)

                self.transfer = nil
                self.installIPA(file)
            }
            catch {
                self.transfer = .failed(error.localizedDescription)
                LogCapture.shared.note(L("Каталог: %@ — %@", name, error.localizedDescription))
            }
        }
    }

    /// What the guest has installed, for the package manager to mark.
    func installedPackages() async -> [String: String] {
        let serial = self.serial
        let files = self.files
        return await withCheckedContinuation { done in
            DispatchQueue.global(qos: .userInitiated).async {
                done.resume(returning: (try? GuestPackages.installed(serial: serial, files: files)) ?? [:])
            }
        }
    }

    /// Removes a package the guest has.
    func removePackage(_ package: String) {
        guard isRunning, transfer?.isRunning != true else { return }
        transfer = .running(title: L("Удаляю %@…", package), done: 0, total: 0)
        let serial = self.serial
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let said = try GuestPackages.remove(package, serial: serial)
                DispatchQueue.main.async {
                    self.transfer = .finished(L("Пакет удалён.") + (said.isEmpty ? "" : "\n" + said))
                    LogCapture.shared.note(L("Пакеты: %@ удалён.", package))
                }
            } catch {
                DispatchQueue.main.async {
                    self.transfer = .failed(error.localizedDescription)
                    LogCapture.shared.note(L("Пакеты: %@ — %@", package, error.localizedDescription))
                }
            }
        }
    }

    /// Restarts the guest's SpringBoard, which is how a freshly installed tweak
    /// gets loaded.
    func respring() {
        guard isRunning else { return }
        let serial = self.serial
        LogCapture.shared.note(L("Перезапускаю SpringBoard…"))
        DispatchQueue.global(qos: .userInitiated).async {
            do { try GuestPackages.respring(serial: serial) }
            catch { LogCapture.shared.note(L("SpringBoard: %@", error.localizedDescription)) }
        }
        // A new SpringBoard knows nothing of the status bar override.
        paintedStatusBar = nil
        paintStatusBarWhenReady(delay: 20)
    }

    // MARK: The agent

    /// How many times a transient agent bring-up has been retried, so a guest
    /// that is simply slow to reach a shell is waited out, but a permanent
    /// refusal is not hammered.
    private var agentAttempts = 0

    /// Finds or installs the guest agent, off the main thread, and once it is up
    /// hands it the status bar so the console is out of that loop. Safe to call
    /// again — it does nothing while a bring-up is in flight or already done.
    func bringUpAgent(force: Bool = false) {
        guard isRunning else { return }
        if force { agentBringUpStarted = false; guestAgent = nil; agentAttempts = 0 }
        guard !agentBringUpStarted, guestAgent == nil else { return }
        agentBringUpStarted = true
        DispatchQueue.global(qos: .utility).async { [weak self] in
            guard let self else { return }
            let outcome = GuestAgentSetup.bringUp(serial: self.serial) { line in
                LogCapture.shared.note(L("Агент: %@", line))
            }
            DispatchQueue.main.async {
                switch outcome {
                case .installed(let agent):
                    self.guestAgent = agent
                    self.agentAttempts = 0
                    LogCapture.shared.note(L("Агент готов: команды и строка состояния идут мимо консоли."))
                    // Hand it the status bar straight away, and let it keep it.
                    self.paintStatusBar(force: true)
                    self.syncTimeZone()
                case .unavailable(let why, let retry):
                    // Not an error: the console path stays. A guest that has not
                    // reached a shell yet is worth trying again; a build or an
                    // emulator that cannot have an agent is not.
                    self.agentBringUpStarted = false
                    if retry, self.agentAttempts < 8, self.isRunning {
                        self.agentAttempts += 1
                        DispatchQueue.main.asyncAfter(deadline: .now() + 30) { self.bringUpAgent() }
                    } else {
                        LogCapture.shared.note(L("Агент не поднят (%@) — работаю через консоль.", why))
                    }
                }
            }
        }
    }

    // MARK: The guest's status bar

    /// The helper arguments last drawn, so that nothing is redrawn for nothing.
    private var paintedStatusBar: String?
    private var paintQueued = false

    /// Draws what the status bar settings ask for.
    ///
    /// Coalesced for a second: a text field or a stepper changes many times in
    /// a row, and every change would otherwise be a console conversation.
    func paintStatusBar(force: Bool = false) {
        guard isRunning else { return }
        if force { paintedStatusBar = nil }
        guard !paintQueued else { return }
        paintQueued = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
            guard let self else { return }
            self.paintQueued = false
            let look = GuestStatusBar.requested()
            let arguments = look?.arguments ?? "-z"
            // Nothing drawn and nothing wanted: the helper need not even be
            // carried in.
            guard arguments != self.paintedStatusBar, look != nil || self.paintedStatusBar != nil else { return }

            // The agent's way: hand it the look and it keeps it applied by
            // itself, through resprings and a busy console alike. This is sent
            // once, not on a timer.
            if let agent = self.guestAgent {
                DispatchQueue.global(qos: .utility).async {
                    let ok = agent.setStatusBar(look?.json)
                    DispatchQueue.main.async {
                        if ok {
                            self.paintedStatusBar = look == nil ? nil : arguments
                            LogCapture.shared.note(L("Строка состояния гостя (агент): %@", arguments))
                        } else if agent.isAlive() {
                            LogCapture.shared.note(L("Строка состояния гостя: агент не принял."))
                        } else {
                            // The agent went away; drop back to the console.
                            self.guestAgent = nil
                            LogCapture.shared.note(L("Агент пропал — возвращаюсь на консоль для строки состояния."))
                            self.paintStatusBar(force: true)
                        }
                    }
                }
                return
            }

            // Said out loud: a change that silently did nothing is what made
            // this look broken on the phone.
            guard self.serial.interactive else {
                LogCapture.shared.note(L("Строка состояния гостя: консоль гостя не готова, попробую позже."))
                return
            }
            let serial = self.serial
            DispatchQueue.global(qos: .utility).async {
                let outcome = Result { try serial.exclusive { try GuestStatusBar.apply(look, shell: GuestShell(serial: serial)) } }
                DispatchQueue.main.async {
                    switch outcome {
                    case .success:
                        self.paintedStatusBar = look == nil ? nil : arguments
                        LogCapture.shared.note(L("Строка состояния гостя: %@", arguments))
                    case .failure(let error):
                        LogCapture.shared.note(L("Строка состояния гостя: не вышло — %@", error.localizedDescription))
                    }
                }
            }
        }
    }

    /// Which round of repainting is current; an older one stops when it sees
    /// a newer one has started.
    private var statusBarRounds = 0

    /// Keeps the status bar drawn while the guest comes up and afterwards.
    ///
    /// SpringBoard starts its status bar server some way into the boot, and an
    /// override sent before that is lost. Asking the guest whether SpringBoard
    /// is there was tried, and cost the console: `ps -A` hung on a busy guest
    /// and took bash down with it, and the shell pane, the packages and this
    /// all typed into a console nobody read after that. The override is safe
    /// to send twice, so it is simply sent again — every half minute while the
    /// guest boots, then every minute, which also brings it back soon after
    /// SpringBoard restarts on its own. Each time only if the console is free:
    /// nothing waits for this.
    private func paintStatusBarWhenReady(delay: TimeInterval) {
        statusBarRounds += 1
        let round = statusBarRounds
        let pauses: [TimeInterval] = [delay, 30, 30, 30, 30]

        func tick(_ step: Int) {
            guard isRunning, round == statusBarRounds else { return }
            repaintStatusBarQuietly()
            // The time zone rides the same rounds. Once the guest has the
            // phone's zone this does nothing, and a zone the phone changed to
            // reaches the guest within a minute.
            syncTimeZone()
            let next = step + 1 < pauses.count ? pauses[step + 1] : 60
            DispatchQueue.main.asyncAfter(deadline: .now() + next) { tick(step + 1) }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { tick(0) }
    }

    /// One repaint that gives way to anybody else on the console and says
    /// nothing unless it fails.
    private func repaintStatusBarQuietly() {
        // With the agent up there is nothing to do here: it reapplies the
        // override on its own, including after a respring, without the console.
        if guestAgent != nil { return }
        guard serial.interactive, let look = GuestStatusBar.requested() else { return }
        let serial = self.serial
        DispatchQueue.global(qos: .utility).async {
            var outcome: Result<Void, Error>?
            serial.ifFree {
                outcome = Result { try GuestStatusBar.apply(look, shell: GuestShell(serial: serial)) }
            }
            guard let outcome else { return }
            DispatchQueue.main.async {
                switch outcome {
                case .success:
                    if self.paintedStatusBar != look.arguments {
                        LogCapture.shared.note(L("Строка состояния гостя: %@", look.arguments))
                    }
                    self.paintedStatusBar = look.arguments
                case .failure(let error):
                    LogCapture.shared.note(L("Строка состояния гостя: не вышло — %@", error.localizedDescription))
                }
            }
        }
    }

    // MARK: The guest's time zone

    /// The zone the guest was last given, so the same one is not sent again.
    private var guestTimeZone: String?
    private var timeZoneInFlight = false
    /// A failure already written to the log, so a retry every minute does not
    /// write it again.
    private var timeZoneComplaint: String?

    /// Gives the guest the phone's time zone, when the settings ask for it.
    /// See `GuestTimeZone` for how. Through the agent when there is one,
    /// otherwise over the console only when nobody else is using it; quiet
    /// unless the zone changes or the guest says something unexpected.
    func syncTimeZone(force: Bool = false) {
        guard isRunning, Settings.shared.guestTimeZone, let zone = GuestTimeZone.phone else { return }
        if force { guestTimeZone = nil }
        guard zone != guestTimeZone, !timeZoneInFlight else { return }
        let agent = guestAgent
        guard agent != nil || serial.interactive else { return }
        timeZoneInFlight = true
        let command = GuestTimeZone.command(for: zone)
        let serial = self.serial
        DispatchQueue.global(qos: .utility).async {
            var answer: String?
            if let agent {
                answer = agent.run(command, timeout: 30)?.output
            } else {
                _ = serial.ifFree { answer = GuestShell(serial: serial).text(command, timeout: 30) }
            }
            DispatchQueue.main.async {
                self.timeZoneInFlight = false
                // No answer: the console was busy or the guest slow. The next
                // round asks again.
                guard let answer else { return }
                switch GuestTimeZone.outcome(of: answer, zone: zone) {
                case .set:
                    LogCapture.shared.note(L("Часовой пояс гостя: %@", zone))
                    self.guestTimeZone = zone
                case .missing:
                    // Asking again would only get the same answer.
                    LogCapture.shared.note(L("Часовой пояс гостя: в образе нет пояса %@.", zone))
                    self.guestTimeZone = zone
                case .unexpected:
                    let said = String(answer.prefix(120))
                    guard said != self.timeZoneComplaint else { return }
                    self.timeZoneComplaint = said
                    LogCapture.shared.note(L("Часовой пояс гостя: не вышло — %@", said))
                }
            }
        }
    }

    /// Installs a `.deb` without Cydia: the guest is too slow for Cydia to
    /// survive its own packager, and this path has nothing watching a clock.
    func installDEB(_ url: URL) {
        guard transfer?.isRunning != true else { return }
        if let why = transferBlocker() { transfer = .failed(why); return }
        let name = url.lastPathComponent
        transfer = .running(title: L("→ пакет %@", name), done: 0, total: 0)
        LogCapture.shared.note(L("Пакеты: ставлю %@", name))
        let serial = self.serial
        let files = self.files
        DispatchQueue.global(qos: .userInitiated).async {
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            var phase = L("→ пакет %@", name)
            do {
                var last = Date.distantPast
                let said = try GuestPackages.installDeb(url, serial: serial, files: files, progress: { done, total in
                    guard Date().timeIntervalSince(last) > 0.1 || done == total else { return }
                    last = Date()
                    DispatchQueue.main.async {
                        self.transfer = .running(title: phase, done: done, total: total)
                    }
                }, note: { line in
                    phase = line
                    DispatchQueue.main.async { self.transfer = .running(title: line, done: 0, total: 0) }
                })
                DispatchQueue.main.async {
                    self.transfer = .finished(L("Пакет установлен.") + (said.isEmpty ? "" : "\n" + said))
                    LogCapture.shared.note(L("Пакеты: %@ установлен.", name))
                }
            } catch {
                DispatchQueue.main.async {
                    self.transfer = .failed(error.localizedDescription)
                    LogCapture.shared.note(L("Пакеты: %@ — %@", name, error.localizedDescription))
                }
            }
        }
    }

    /// Re-does the part of the package repair that a guest reboot undoes.
    ///
    /// The remount and the root helper do not survive a restart, and without
    /// them Cydia fails with `cydo returned an error code (2)` — an error that
    /// says nothing about why. Only the quick half runs here; the slow half is
    /// needed once per image and stays on the button.
    private func preparePackages(delay: TimeInterval = 45) {
        let serial = self.serial
        var attempts = 0

        func attempt() {
            guard isRunning else { return }
            guard serial.interactive else {
                attempts += 1
                guard attempts < 150 else { return }
                DispatchQueue.main.asyncAfter(deadline: .now() + 4) { attempt() }
                return
            }
            DispatchQueue.global(qos: .utility).async {
                do {
                    let complaints = try GuestPackages.prepare(serial: serial)
                    let tail = complaints.isEmpty ? "" : " " + complaints.joined(separator: " ")
                    LogCapture.shared.note(L("Пакеты: гость подготовлен.") + tail)
                }
                catch {
                    LogCapture.shared.note(L("Пакеты: подготовить не вышло — %@", error.localizedDescription))
                }
            }
        }

        // Not ten seconds in: the guest is still booting then, and it puts the
        // root back read-only on its way up — the remount answered 0 and meant
        // nothing.
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { attempt() }
    }

    /// The guest panicked and the watchdog is restarting it.
    ///
    /// Everything the package manager needs — the writable root, the folders,
    /// the root helper — is undone by that, and the guest gives no other sign:
    /// the app goes on running, the console goes on printing, and Cydia is
    /// quietly broken again. So the same preparation that runs at startup runs
    /// again, once the machine has had time to come back up.
    private func guestDied() {
        LogCapture.shared.note(L("Гость упал: %@", serial.guestDeathReason))
        // The status bar override lived in the SpringBoard that just went away.
        paintedStatusBar = nil
        // The agent instance died with the guest. It starts itself again from
        // the launchd cache on the way back up, so the client is dropped and
        // found afresh once the machine has had time to boot; the console keeps
        // the status bar meanwhile.
        guestAgent = nil
        agentBringUpStarted = false
        DispatchQueue.main.asyncAfter(deadline: .now() + 120) { [weak self] in self?.bringUpAgent() }
        paintStatusBarWhenReady(delay: 150)
        guard Settings.shared.autoRepairPackages, isRunning else { return }
        LogCapture.shared.note(L("Гость упал в панику — поднимаю менеджер пакетов заново, когда вернётся."))
        preparePackages(delay: 150)
    }

    /// Puts the guest's package manager back together — the fix for Cydia's
    /// `cydo returned an error code (2)`.
    ///
    /// Offered as a button rather than done at every boot because it writes to
    /// the guest's own system volume; the part that does not survive a reboot
    /// is the remount, so pressing it again after one is normal.
    func repairPackages() {
        guard isRunning, transfer?.isRunning != true else { return }
        transfer = .running(title: L("Чиню менеджер пакетов…"), done: 0, total: 0)
        LogCapture.shared.note(L("Пакеты: чиню dpkg в госте…"))
        let serial = self.serial
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let complaints = try GuestPackages.repair(serial: serial, note: { line in
                    DispatchQueue.main.async {
                        self.transfer = .running(title: line, done: 0, total: 0)
                    }
                })
                let tail = complaints.isEmpty ? "" : "\n" + complaints.joined(separator: "\n")
                DispatchQueue.main.async {
                    self.transfer = .finished(L("Менеджер пакетов починен. Попробуйте Cydia снова.") + tail)
                    LogCapture.shared.note(L("Пакеты: готово.") + tail)
                }
            } catch {
                DispatchQueue.main.async {
                    self.transfer = .failed(error.localizedDescription)
                    LogCapture.shared.note(L("Пакеты: не вышло — %@", error.localizedDescription))
                }
            }
        }
    }

    /// Watches the link and, if it never comes up, uses the guest's own shell.
    ///
    /// Spread out on purpose: on a phone the guest can be four minutes from
    /// power-on to a shell, and a command sent before that shell exists lands
    /// nowhere. Five tries covers the slow case; if the link is still down
    /// after that, it is not going to come up by itself.
    private func watchNetwork() {
        let tries = 5
        var attempts = 0

        func check() {
            guard isRunning else { return }
            if linkIsUp {
                if !networkUp {
                    networkUp = true
                    LogCapture.shared.note(L("Сеть: гость получил адрес."))
                }
                // Keep watching. iOS takes an address, uses it, and then
                // announces the link down and stops receiving — and a watcher
                // that stopped at the good news never saw that happen.
                DispatchQueue.main.asyncAfter(deadline: .now() + 20, execute: check)
                return
            }
            if networkUp {
                networkUp = false
                attempts = 0
                LogCapture.shared.note(L("Сеть: гость погасил связь."))
            }
            if Settings.shared.netAutoFix, attempts < tries {
                attempts += 1
                LogCapture.shared.note(L("Сеть: адреса всё ещё нет, попытка %d из %d.", attempts, tries))
                fixNetwork()
                DispatchQueue.main.asyncAfter(deadline: .now() + 45, execute: check)
                return
            }
            // Keep looking, quietly: the guest may still sort itself out.
            DispatchQueue.main.asyncAfter(deadline: .now() + 60, execute: check)
        }

        // Long enough for an unhurried boot to have got there by itself.
        DispatchQueue.main.asyncAfter(deadline: .now() + 90, execute: check)
    }

    // MARK: Files

    private lazy var files = GuestFiles(
        serial: serial,
        linkUp: { [weak self] in self?.linkIsUp ?? false },
        bringNetworkUp: { [weak self] in DispatchQueue.main.async { self?.fixNetwork() } },
        agent: { [weak self] in self?.guestAgent })

    /// Checks what can be checked on the spot, so a transfer that cannot work
    /// says why at once instead of after a minute of waiting.
    private func transferBlocker() -> String? {
        if !config.network { return GuestFiles.Failure.networkOff.localizedDescription }
        if !serial.interactive { return GuestFiles.Failure.noShell.localizedDescription }
        return nil
    }

    func sendToGuest(_ url: URL) {
        guard transfer?.isRunning != true else { return }
        if let why = transferBlocker() { transfer = .failed(why); return }
        let name = url.lastPathComponent
        transfer = .running(title: "→ \(name)", done: 0, total: 0)
        LogCapture.shared.note(L("Файлы: отправляю %@ в гостя", name))
        let files = self.files
        DispatchQueue.global(qos: .userInitiated).async {
            // Files picked from the Files app are lent, not given.
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let started = Date()
            do {
                var last = Date.distantPast
                let remote = try files.send(url) { done, total in
                    // A progress bar redrawn a hundred times a second helps nobody.
                    guard Date().timeIntervalSince(last) > 0.1 || done == total else { return }
                    last = Date()
                    DispatchQueue.main.async { self.transfer = .running(title: "→ \(name)", done: done, total: total) }
                }
                let summary = TransferState.summary(url: url, seconds: Date().timeIntervalSince(started))
                DispatchQueue.main.async {
                    self.transfer = .finished("\(name) → \(remote)\n\(summary)")
                    LogCapture.shared.note(L("Файлы: %@ → %@, %@", name, remote, summary))
                }
            } catch {
                DispatchQueue.main.async {
                    self.transfer = .failed(error.localizedDescription)
                    LogCapture.shared.note(L("Файлы: %@ не отправлен — %@", name, error.localizedDescription))
                }
            }
        }
    }

    /// One button: unpack the `.ipa` here, carry it in by whichever channel is
    /// available, and put it in `/Applications`. The first run also leaves the
    /// helper in the guest, so every later install finds it already there.
    func installIPA(_ url: URL) {
        guard transfer?.isRunning != true else { return }
        if let why = transferBlocker() { transfer = .failed(why); return }
        guard url.pathExtension.lowercased() == "ipa" else {
            transfer = .failed(L("Нужен файл .ipa.")); return
        }
        let name = url.lastPathComponent
        transfer = .running(title: L("→ установка %@", name), done: 0, total: 0)
        LogCapture.shared.note(L("Установка: %@", name))
        let installer = GuestInstaller(serial: serial, files: files)
        DispatchQueue.global(qos: .userInitiated).async {
            // Files picked from the Files app are lent, not given.
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let started = Date()
            // Which step is running, so the progress bar keeps saying it while
            // the bytes move. Both closures are called from this thread, one
            // after another, so the plain variable is enough.
            var phase = L("→ установка %@", name)
            do {
                var last = Date.distantPast
                let target = try installer.install(ipa: url, progress: { done, total in
                    guard Date().timeIntervalSince(last) > 0.1 || done == total else { return }
                    last = Date()
                    DispatchQueue.main.async {
                        self.transfer = .running(title: phase, done: done, total: total)
                    }
                }, note: { line in
                    phase = line
                    DispatchQueue.main.async {
                        self.transfer = .running(title: line, done: 0, total: 0)
                    }
                })
                let summary = TransferState.summary(url: url, seconds: Date().timeIntervalSince(started))
                // A warning is not a failure: the app is installed either way,
                // and saying why it will not start beats letting it look broken.
                let caveat = installer.warning.map { "\n" + $0 } ?? ""
                DispatchQueue.main.async {
                    self.transfer = .finished(L("Установлено: %@", target) + "\n" + summary + caveat)
                    LogCapture.shared.note(L("Установка: %@ → %@, %@", name, target, summary) + caveat)
                }
            } catch {
                DispatchQueue.main.async {
                    self.transfer = .failed(error.localizedDescription)
                    LogCapture.shared.note(L("Установка: %@ — %@", name, error.localizedDescription))
                }
            }
        }
    }

    func receiveFromGuest(_ path: String) {
        let remote = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !remote.isEmpty, transfer?.isRunning != true else { return }
        if let why = transferBlocker() { transfer = .failed(why); return }
        let name = (remote as NSString).lastPathComponent
        transfer = .running(title: "← \(name)", done: 0, total: 0)
        LogCapture.shared.note(L("Файлы: забираю %@ из гостя", remote))
        let files = self.files
        DispatchQueue.global(qos: .userInitiated).async {
            let started = Date()
            do {
                var last = Date.distantPast
                let saved = try files.receive(remote) { done, total in
                    guard Date().timeIntervalSince(last) > 0.1 || done == total else { return }
                    last = Date()
                    DispatchQueue.main.async { self.transfer = .running(title: "← \(name)", done: done, total: total) }
                }
                let summary = TransferState.summary(url: saved, seconds: Date().timeIntervalSince(started))
                DispatchQueue.main.async {
                    self.transfer = .finished("\(remote) → Guest/\(saved.lastPathComponent)\n\(summary)")
                    LogCapture.shared.note(L("Файлы: %@ → %@, %@", remote, saved.path, summary))
                }
            } catch {
                DispatchQueue.main.async {
                    self.transfer = .failed(error.localizedDescription)
                    LogCapture.shared.note(L("Файлы: %@ не получен — %@", remote, error.localizedDescription))
                }
            }
        }
    }

    func inspectMachine() {
        Threads.report { report in LogCapture.shared.note(report) }
        qmp.inspect { report in LogCapture.shared.note(report) }
    }

    private var bootWatch: Timer?
    private var memoryWatch: Timer?
    private var memoryLogged: UInt64 = 0
    private var memoryWarned = false

    /// A line in the log every half minute saying where CPU0 is and how many
    /// cores run: the only progress report a macOS guest gives once iBoot has
    /// handed over, since its kernel keeps quiet on the console under full
    /// security.
    func watchBoot() {
        bootWatch?.invalidate()
        let timer = Timer(timeInterval: 30, repeats: true) { [weak self] timer in
            guard let self, self.isRunning else { timer.invalidate(); return }
            self.qmp.snapshot { LogCapture.shared.note($0) }
            VMModel.publishDisplayLogs()
        }
        // Added to the main run loop by hand, in the common modes: the
        // scheduled-timer shorthand from this callback's context is what drew
        // "invalid mode 'kCFRunLoopCommonModes'" on the first device run.
        RunLoop.main.add(timer, forMode: .common)
        bootWatch = timer

        // The system kills the app outright when its footprint reaches the
        // limit, with nothing in any log to say so. A line each time the
        // footprint has grown by another 100 MB leaves the climb on record:
        // the last one before the log stops is how close it got.
        memoryWatch?.invalidate()
        memoryLogged = 0
        memoryWarned = false
        switch Threads.increasedMemoryLimitGranted() {
        case true?:  LogCapture.shared.note(L("Повышенный лимит памяти: право выдано"))
        case false?: LogCapture.shared.note(L("Повышенный лимит памяти: права нет — подпись установки его не сохранила"))
        case nil:    LogCapture.shared.note(L("Повышенный лимит памяти: проверить не удалось"))
        }
        let memory = Timer(timeInterval: 5, repeats: true) { [weak self] timer in
            guard let self, self.isRunning else { timer.invalidate(); return }
            guard let bytes = Threads.footprintBytes(),
                  bytes >= self.memoryLogged + 100 * 1_048_576 else { return }
            self.memoryLogged = bytes
            LogCapture.shared.note(L("Память: %@", Threads.memoryLine(footprint: bytes)))
            #if os(iOS)
            // Said once, and loudly: the system kills the app at the ceiling
            // without a word in any log, and this is the last chance to say so.
            if os_proc_available_memory() < 200 * 1_048_576, !self.memoryWarned {
                self.memoryWarned = true
                LogCapture.shared.note(L("ПАМЯТЬ НА ИСХОДЕ: до потолка меньше 200 МБ — iOS вот-вот закроет приложение. Уменьшите память гостя."))
            }
            #endif
            LogCapture.shared.note("  " + Threads.regionBreakdown())
        }
        RunLoop.main.add(memory, forMode: .common)
        memoryWatch = memory
    }

    /// reims-vgpu's own logs — the always-on failure channel above all, where
    /// a refused shader or pipeline says why a frame came out black — are in
    /// the process's temporary directory on iOS, which the Files app cannot
    /// reach. Copied beside the app's other logs as the machine runs, so they
    /// survive a crash too.
    static func publishDisplayLogs() {
        let fm = FileManager.default
        for name in ["reims-vgpu-fail.log", "reims-vgpu-draw.log"] {
            let source = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(name)
            guard fm.fileExists(atPath: source.path) else { continue }
            let target = VMConfig.documents.appendingPathComponent(name)
            try? fm.removeItem(at: target)
            try? fm.copyItem(at: source, to: target)
        }
        // And the shader blobs the Metal driver refused, one file each, so
        // they can be read off the device and taken apart on a desktop.
        let refused = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("reims-vgpu-mtlb")
        if let blobs = try? fm.contentsOfDirectory(atPath: refused.path), !blobs.isEmpty {
            let target = VMConfig.documents.appendingPathComponent("reims-vgpu-mtlb")
            try? fm.createDirectory(at: target, withIntermediateDirectories: true)
            for blob in blobs where !fm.fileExists(atPath: target.appendingPathComponent(blob).path) {
                try? fm.copyItem(at: refused.appendingPathComponent(blob), to: target.appendingPathComponent(blob))
            }
        }
    }

    // MARK: Input

    /// Undoes the letterboxing of the aspect-fit picture, so a finger on the
    /// screen becomes the pixel underneath it whatever the guest's resolution.
    /// Where in the guest's own pixels a touch landed.
    ///
    /// Takes the rectangle the picture actually occupies rather than the whole
    /// view: it is centred on the screen with a margin around it, and the two
    /// are no longer the same box.
    private func guestPoint(from point: CGPoint, in box: CGRect) -> CGPoint? {
        guard let fb = framebufferSize, box.width > 0, box.height > 0 else { return nil }
        let scale = CGFloat(fb.width) / box.width
        return CGPoint(x: (point.x - box.minX) * scale, y: (point.y - box.minY) * scale)
    }

    func tap(at point: CGPoint, in box: CGRect) {
        guard let fb = framebufferSize, let p = guestPoint(from: point, in: box) else { return }
        guard p.x >= 0, p.y >= 0, Int(p.x) < fb.width, Int(p.y) < fb.height else { return }
        display?.send(touch: p, pressed: true)
    }

    func release(at point: CGPoint, in box: CGRect) {
        guard let p = guestPoint(from: point, in: box) else { return }
        display?.send(touch: p, pressed: false)
    }

    /// Device buttons are wired to F1..F10 by the machine's button device.
    enum HardwareButton: String, CaseIterable {
        case power = "Питание"
        case volumeUp = "Громче"
        case volumeDown = "Тише"
        case home = "Home"

        var functionKey: UInt32 {
            switch self {
            case .power:      return 5   // Hold
            case .volumeDown: return 3
            case .volumeUp:   return 4
            case .home:       return 6   // Menu
            }
        }

        /// How long the guest needs to see it held.
        var hold: TimeInterval {
            switch self {
            case .power: return 2.5      // long enough for the power-off slider
            default:     return 0.2
            }
        }
    }

    /// Holds a device button down for as long as that button needs.
    ///
    /// All four used to be tapped for a tenth of a second, which is why they
    /// looked broken: the side button opens the power-off slider only after a
    /// long hold, and a tenth of a second of it does nothing at all.
    func press(_ button: HardwareButton) {
        guard let display else {
            LogCapture.shared.note(L("Кнопки: экран не подключён, нажатие некуда отправить."))
            return
        }
        LogCapture.shared.note(L("Кнопка %@ (F%d) на %.1f с", L(button.rawValue),
                                 Int(button.functionKey), button.hold))
        display.send(functionKey: button.functionKey, pressed: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + button.hold) {
            display.send(functionKey: button.functionKey, pressed: false)
        }
    }
}

// MARK: - Views

enum Pane: String, CaseIterable {
    case screen = "Экран"
    case terminal = "Терминал"
}

#if os(iOS)
struct RootView: View {
    @StateObject private var model = VMModel()
    @State private var pane: Pane = .screen
    @State private var fullScreen = false
    @State private var pickFile = false
    @State private var pickIPA = false
    @State private var pickDEB = false
    @State private var askPath = false
    /// Where the button sits, as a fraction of the view, so that it stays put
    /// across rotations and relaunches. Negative means it has never been moved.
    @AppStorage("menuX") private var menuX: Double = -1
    @AppStorage("menuY") private var menuY: Double = -1
    @State private var dragging: CGSize = .zero

    private static let buttonSize: CGFloat = 48

    /// Where the button sits. Kept clear of the island and the home indicator,
    /// which the screen behind it now runs underneath.
    private func menuPoint(in geo: GeometryProxy) -> CGPoint {
        let size = geo.size
        // Only far enough from the edge not to hang off it. Where the button is
        // allowed to go is the user's business; the island and the indicator are
        // taken into account for where it starts, not for where it may end up.
        let half = Self.buttonSize / 2 + 6
        let resting = CGPoint(x: menuX < 0 ? size.width - half - 10 : menuX * size.width,
                              y: menuY < 0 ? size.height - half - DeviceInsets.current.bottom - 10
                                           : menuY * size.height)
        return CGPoint(x: min(max(resting.x + dragging.width, half), size.width - half),
                       y: min(max(resting.y + dragging.height, half), size.height - half))
    }

    private func commitDrag(_ translation: CGSize, in geo: GeometryProxy) {
        let landed = menuPoint(in: geo)
        dragging = .zero
        guard geo.size.width > 0, geo.size.height > 0 else { return }
        menuX = landed.x / geo.size.width
        menuY = landed.y / geo.size.height
    }

    var body: some View {
        NavigationStack {
            Group {
                if !model.missing.isEmpty {
                    SetupView(model: model)
                        .navigationTitle("Orchard")
                        .inlineNavigationTitle()
                } else {
                    // No navigation bar: the guest's picture is nearly as tall
                    // as the phone's own screen, and a title bar was taking the
                    // room the corners need to be seen in. Everything that was
                    // in it now hangs off the one button below.
                    ZStack {
                        Color.black.ignoresSafeArea()
                        switch pane {
                        case .screen:   ScreenView(model: model, picture: model.picture, serial: model.serial,
                                                   fullScreen: $fullScreen)
                        case .terminal: TerminalView(model: model)
                        }
                    }
                    .overlay {
                        if !fullScreen {
                            GeometryReader { geo in
                                ControlMenu(model: model, pane: $pane, fullScreen: $fullScreen,
                                            pickFile: $pickFile, pickIPA: $pickIPA, pickDEB: $pickDEB,
                                            askPath: $askPath)
                                    .position(menuPoint(in: geo))
                                    // Simultaneous, so a tap still opens the
                                    // menu and only a real drag moves it.
                                    .simultaneousGesture(
                                        DragGesture(minimumDistance: 12)
                                            .onChanged { value in dragging = value.translation }
                                            .onEnded { value in commitDrag(value.translation, in: geo) }
                                    )
                                    .animation(.interactiveSpring(response: 0.3), value: dragging)
                            }
                        }
                    }
                    .toolbar(.hidden, for: .navigationBar)
                }
            }
            .modifier(GuestActions(model: model, pane: pane, pickFile: $pickFile, pickIPA: $pickIPA,
                                   pickDEB: $pickDEB, askPath: $askPath))
        }
        // An ordinary app until full screen is asked for: the phone's own status
        // bar at the top, and the home swipe doing what it always does. Full
        // screen gives the guest the whole display, status bar included.
        .statusBarHidden(fullScreen)
        // And the edges: in full screen the first swipe goes to the guest and
        // the second to the phone. The home indicator stays for that — hidden,
        // iOS ignores the deferral, and full screen used to leave the home
        // swipe a single one for exactly that reason.
        .onAppear { applyEdges() }
        .onChange(of: model.isRunning) { _ in applyEdges() }
        .onChange(of: fullScreen) { _ in applyEdges() }
    }

    /// Only while the guest is running and has the whole screen.
    private func applyEdges() {
        SystemGestures.apply(deferEdges: model.isRunning && fullScreen)
    }
}
#endif

/// What every window around the guest presents over it, on iOS and on a Mac
/// alike: the three file pickers, the question of which guest file to fetch,
/// the transfer banner, and a fresh look at JIT and the files whenever the app
/// comes back to the front.
struct GuestActions: ViewModifier {
    @ObservedObject var model: VMModel
    let pane: Pane
    @Binding var pickFile: Bool
    @Binding var pickIPA: Bool
    @Binding var pickDEB: Bool
    @Binding var askPath: Bool
    @State private var guestPath = "/var/mobile/"
    @Environment(\.scenePhase) private var scenePhase

    func body(content: Content) -> some View {
        content
            // Kept here rather than on the menu: the menu already presents the
            // settings sheet, and two sheet-like presentations on one view fight.
            .fileImporter(isPresented: $pickFile, allowedContentTypes: [.item]) { result in
                if case .success(let url) = result { model.sendToGuest(url) }
            }
            // A second importer, not a second sheet: only one of the two is ever
            // presented, so they do not fight the way two sheets would.
            .fileImporter(isPresented: $pickIPA, allowedContentTypes: [.item]) { result in
                if case .success(let url) = result { model.installIPA(url) }
            }
            // A third importer, for the same reason as the second: only one is
            // ever presented, so they do not fight the way sheets would.
            .fileImporter(isPresented: $pickDEB, allowedContentTypes: [.item]) { result in
                if case .success(let url) = result { model.installDEB(url) }
            }
            .alert(L("Забрать файл из гостя"), isPresented: $askPath) {
                TextField(L("Путь в госте"), text: $guestPath)
                    .autocorrectionDisabled()
                    .noAutocapitalization()
                Button(L("Забрать")) { model.receiveFromGuest(guestPath) }
                Button(L("Отмена"), role: .cancel) {}
            } message: {
                Text(L("Файл появится в папке Guest приложения — её видно в «Файлах»."))
            }
            .overlay(alignment: .bottom) {
                if let transfer = model.transfer {
                    TransferBanner(state: transfer) { model.transfer = nil }
                        .padding(.horizontal, 12)
                        // Above the command line, not on top of it: the terminal
                        // keeps its prompt at the bottom, and an install can run
                        // for minutes with somebody waiting to type.
                        .padding(.bottom, pane == .terminal ? 76 : 12)
                }
            }
            // Coming back from StikDebug is exactly when the answer changes.
            .onChange(of: scenePhase) { phase in
                if phase == .active {
                    model.refreshJIT()
                    model.refreshFiles()
                }
            }
    }
}

struct ControlMenu: View {
    @ObservedObject var model: VMModel
    @Binding var pane: Pane
    @Binding var fullScreen: Bool
    @Binding var pickFile: Bool
    @Binding var pickIPA: Bool
    @Binding var pickDEB: Bool
    @Binding var askPath: Bool
    @State private var showSettings = false
    @State private var confirmQuit = false
    /// How big the button is: a thumb's worth on the phone, the size of the
    /// traffic lights beside it in a Mac window's bar.
    var discSize: CGFloat = 48

    /// The terminal's own choice of what to show. Held by the same key the
    /// terminal holds it under, so the two stay in step.
    @AppStorage("terminalSource") private var sourceName = TerminalView.Source.emulator.rawValue

    var body: some View {
        Menu {
            Section(L("Вид")) {
                Picker(L("Вид"), selection: $pane) {
                    ForEach(Pane.allCases, id: \.self) { Text(L($0.rawValue)).tag($0) }
                }
                if pane == .terminal {
                    Picker(L("Источник"), selection: $sourceName) {
                        ForEach(TerminalView.Source.allCases, id: \.self) {
                            Text(L($0.rawValue)).tag($0.rawValue)
                        }
                    }
                }
            }

            Section(L("Машина")) {
                Button(L("Параметры…"), systemImage: "gearshape") {
                    #if os(macOS)
                    MacSettings.open()
                    #else
                    showSettings = true
                    #endif
                }
                Button(startTitle, systemImage: "play.fill") {
                    model.start()
                }
                .disabled(model.isRunning || model.hasRun || !model.missing.isEmpty
                          || model.needsRestore)
                Button(L("Во весь экран"), systemImage: "arrow.up.left.and.arrow.down.right") {
                    fullScreen = true
                }
                Button(L("Поднять сеть в госте"), systemImage: "network") {
                    model.fixNetwork()
                }
                .disabled(!model.isRunning)
                Button(role: .destructive) {
                    confirmQuit = true
                } label: {
                    Label(L("Выключить машину…"), systemImage: "power")
                }
                .disabled(!model.isRunning)
            }

            Section(L("Файлы")) {
                Button(L("Отправить файл в гостя…"), systemImage: "square.and.arrow.up") {
                    pickFile = true
                }
                Button(L("Забрать файл из гостя…"), systemImage: "square.and.arrow.down") {
                    askPath = true
                }
            }
            .disabled(!model.isRunning || model.transfer?.isRunning == true)

        } label: {
            // The one control on screen, so it is given some presence: a glass
            // disc that stays legible over both the guest's picture and the
            // terminal's black.
            glassDisc
        }
        #if os(macOS)
        // A Mac draws a menu as a pop-up button with an arrow unless told not
        // to; here it is the disc and nothing else, as on the phone.
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        #endif
        .sheet(isPresented: $showSettings) { SettingsView(model: model) }
        .confirmationDialog(L("Выключить машину?"), isPresented: $confirmQuit, titleVisibility: .visible) {
            Button(L("Выключить"), role: .destructive) { model.shutdown() }
            Button(L("Отмена"), role: .cancel) {}
        } message: {
            Text(L("QEMU допишет диски на файлы и завершится. Чтобы запустить машину заново, перезапустите приложение."))
        }
    }

    /// Liquid glass proper where the system provides it, and a material disc
    /// that reads much the same on anything older.
    @ViewBuilder
    private var glassDisc: some View {
        #if os(macOS)
        // In the Mac window's bar, a plain glyph like the others there.
        BarGlyph(systemName: "slider.horizontal.3")
        #else
        let face = Image(systemName: "slider.horizontal.3")
            .font(.system(size: discSize * 0.375, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: discSize, height: discSize)
        // glassEffect itself is only declared in the iOS 26 SDK — #available
        // guards it at runtime, but a toolchain built against an older SDK
        // (Xcode below 26, as CI's still is) can't even see the symbol to
        // compile this file. Gate it on the compiler too, so the same source
        // builds on both: real glass with Xcode 26, the material fallback
        // everywhere else.
        #if compiler(>=6.2)
        if #available(iOS 26.0, macOS 26.0, *) {
            // Clipped as well as shaped. While the menu opens, the glass is
            // handed to the presentation animation, and for a frame or two it
            // draws as the square it really is before the shape catches up.
            //
            // Plain glass, not `.interactive()`: interactive glass answers
            // touches itself, and the control wearing it — a menu here, buttons
            // on the credits card — loses the first tap to the glass. The press
            // animation is not worth a control that has to be pressed twice.
            face.glassEffect(.regular, in: Circle())
                .clipShape(Circle())
                .contentShape(Circle())
        } else {
            face.background(.ultraThinMaterial, in: Circle())
                .overlay(Circle().strokeBorder(.white.opacity(0.18), lineWidth: 0.5))
                .clipShape(Circle())
                .contentShape(Circle())
                .shadow(color: .black.opacity(0.35), radius: 10, y: 3)
        }
        #else
        face.background(.ultraThinMaterial, in: Circle())
            .overlay(Circle().strokeBorder(.white.opacity(0.18), lineWidth: 0.5))
            .clipShape(Circle())
            .contentShape(Circle())
            .shadow(color: .black.opacity(0.35), radius: 10, y: 3)
        #endif
        #endif
    }

    private var startTitle: String {
        if model.isRunning { return L("Запущена") }
        if model.hasRun { return L("Остановлена — перезапустите приложение") }
        if model.needsRestore { return L("Системы нет — сначала «Восстановление»") }
        return L("Запустить")
    }

}

/// The guest display. Taps become absolute pointer events, so a touch lands
/// exactly where the finger is instead of dragging a cursor around.
struct ScreenView: View {
    @ObservedObject var model: VMModel
    /// Watched here and nowhere else, so that a new frame redraws the picture
    /// and leaves the rest of the interface alone.
    @ObservedObject var picture: GuestFrame
    /// Watched for the console's flow rate, which sits beside the frame count.
    @ObservedObject var serial: SerialConsole
    @ObservedObject private var settings = Settings.shared
    @Binding var fullScreen: Bool
    /// Whether the app's own on-screen keyboard is showing. A hardware
    /// keyboard reaches the guest either way. See GuestKeyboard.swift.
    @State private var keyboard = false

    private var trackpadOn: Bool {
        #if canImport(UIKit)
        return VMConfig.macGuest && settings.trackpadMode
        #else
        return false
        #endif
    }

    /// How far the picture keeps from each edge.
    ///
    /// The same at top and bottom, and enough to clear whichever of the two
    /// system furnishings is larger — the island above, the home indicator
    /// below. Sideways it needs less, so it takes less.
    private func margins() -> (h: CGFloat, v: CGFloat) {
        #if os(macOS)
        // The window is the phone's shape already: the picture fills it.
        return (0, 0)
        #else
        guard !fullScreen else { return (0, 0) }
        // A Mac's desktop reaches the phone's edges: its picture is wider
        // than tall, so the margins only ever showed as black strips above and
        // below it, and the island and home indicator sit over its corners.
        if VMConfig.macGuest { return (0, 0) }
        let insets = DeviceInsets.current
        return (16, max(insets.top, insets.bottom, 16))
        #endif
    }

    /// The rectangle the guest's picture occupies, centred on the whole screen.
    ///
    /// Worked out here rather than left to `aspectRatio` for two reasons: the
    /// corner radius is a fraction of the picture's own width, and a touch has
    /// to be mapped back into the guest's pixels. Nothing else knows either.
    private func drawn(in view: CGSize, _ m: (h: CGFloat, v: CGFloat)) -> CGRect? {
        guard let fb = model.framebufferSize, fb.width > 0, fb.height > 0,
              view.width > 0, view.height > 0
        else { return nil }
        let free = CGSize(width: max(view.width - m.h * 2, 1),
                          height: max(view.height - m.v * 2, 1))
        let scale = min(free.width / CGFloat(fb.width), free.height / CGFloat(fb.height))
        let size = CGSize(width: CGFloat(fb.width) * scale, height: CGFloat(fb.height) * scale)
        return CGRect(x: (view.width - size.width) / 2, y: (view.height - size.height) / 2,
                      width: size.width, height: size.height)
    }

    /// The console's flow, in whichever unit keeps it to three digits.
    static func rate(_ bytes: Double) -> String {
        if bytes >= 1024 * 1024 { return L("%.1f МБ/с", bytes / (1024 * 1024)) }
        if bytes >= 1024 { return L("%.0f КБ/с", bytes / 1024) }
        return L("%.0f Б/с", bytes)
    }

    var body: some View {
        GeometryReader { geo in
            let box = drawn(in: geo.size, margins())
            // The rounded corners are an iPhone 11's screen, drawn round an
            // iPhone guest's picture. A Mac's display is square-cornered.
            let radius = settings.roundedScreen && !VMConfig.macGuest
                ? (box?.width ?? 0) * GuestBezel.radiusOverWidth : 0
            ZStack {
                Color.black
                if let frame = picture.image {
                    // At the native resolution there is nothing to interpolate;
                    // below it the picture is stretched to cover the same area,
                    // and whether that is smoothed is a matter of taste.
                    #if canImport(UIKit)
                    // Straight into a layer's contents, not through Image: an
                    // Image of a new CGImage each frame left every one of them
                    // in graphics memory — measured on the phone as 36 → 349 MB
                    // of IOAccelerator in the thirty seconds 1920×1080 frames
                    // were arriving, which is what the system killed the app
                    // for. A layer lets go of the last frame when handed the
                    // next.
                    GuestPictureLayer(frame: frame, smooth: settings.smoothUpscale, radius: radius)
                        .frame(width: box?.width, height: box?.height)
                        .position(x: box?.midX ?? geo.size.width / 2,
                                  y: box?.midY ?? geo.size.height / 2)
                    #else
                    Image(decorative: frame, scale: 1.0)
                        .resizable()
                        .interpolation(settings.smoothUpscale ? .high : .none)
                        .aspectRatio(contentMode: .fit)
                        .frame(width: box?.width, height: box?.height)
                        // Continuous, not circular: Apple's corners are
                        // squircles, and a plain arc reads as the wrong shape
                        // next to the real device.
                        .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
                        .position(x: box?.midX ?? geo.size.width / 2,
                                  y: box?.midY ?? geo.size.height / 2)
                    #endif
                    if settings.showFPS {
                        // The console rate belongs here too: when the guest is
                        // pouring kernel log into the UART, the emulated cores
                        // are formatting text instead of drawing, and the frame
                        // count on its own does not say so.
                        // In the top corner opposite the menu, inside the safe
                        // area, so the island never covers it; a dark pill keeps
                        // it readable over whatever the guest draws there.
                        let insets = DeviceInsets.current
                        Text(String(format: "%.0f FPS · %@", picture.fps, Self.rate(serial.consoleRate)))
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.85))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.black.opacity(0.45), in: Capsule())
                            .fixedSize()
                            .position(x: max(insets.left, 8) + 60,
                                      y: max(insets.top, 6) + 10)
                            .allowsHitTesting(false)
                    }
                } else {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text(placeholder)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if let box { model.tap(at: value.location, in: box) }
                    }
                    .onEnded { value in
                        if let box { model.release(at: value.location, in: box) }
                    },
                including: trackpadOn ? .subviews : .all
            )
            #if canImport(UIKit)
            // The trackpad takes the touches instead, over the whole screen:
            // the pointer moves by how far the finger travels.
            .overlay {
                if trackpadOn, let box, let fb = model.framebufferSize, box.width > 0 {
                    TrackpadSurface(trackpad: model.trackpad,
                                    size: CGSize(width: fb.width, height: fb.height),
                                    scale: CGFloat(fb.width) / box.width)
                }
            }
            // Always first responder while the machine runs, so a hardware
            // keyboard reaches the guest without the button; it never brings
            // up the system keyboard (see KeyCatcherView).
            .background(GuestKeyboard(active: .constant(VMConfig.macGuest && model.isRunning))
                .frame(width: 1, height: 1))
            .overlay(alignment: .bottom) {
                if VMConfig.macGuest, model.isRunning, keyboard {
                    OnScreenKeyboard()
                        .padding(.horizontal, 76)
                        .padding(.bottom, 8)
                }
            }
            .overlay(alignment: .bottomLeading) {
                if VMConfig.macGuest, model.isRunning {
                    Button {
                        keyboard.toggle()
                    } label: {
                        Image(systemName: keyboard ? "keyboard.chevron.compact.down" : "keyboard")
                            .font(.system(size: 17, weight: .semibold))
                            .foregroundStyle(.primary)
                            .frame(width: 48, height: 48)
                            .background(.ultraThinMaterial, in: Circle())
                            .overlay(Circle().strokeBorder(.white.opacity(0.18), lineWidth: 0.5))
                    }
                    .opacity(0.85)
                    .padding(.leading, 16)
                    .padding(.bottom, 16)
                }
            }
            #endif
            // Deliberately small and dim: it sits over the guest's picture, and
            // in full screen it is the only way back.
            .overlay(alignment: .topTrailing) {
                if fullScreen {
                    Button {
                        fullScreen = false
                    } label: {
                        Image(systemName: "arrow.down.right.and.arrow.up.left")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.primary)
                            .frame(width: 40, height: 40)
                            .background(.ultraThinMaterial, in: Circle())
                            .overlay(Circle().strokeBorder(.white.opacity(0.18), lineWidth: 0.5))
                    }
                    .opacity(0.75)
                    .padding(.top, 12)
                    .padding(.trailing, 12)
                }
            }
        }
        // Always the whole screen. A reader that respects the safe area is
        // handed the box left over after the island and the home indicator have
        // taken their share — 59 pt off the top, 34 off the bottom on an iPhone
        // 15 — and centring in that is not centring on the screen. Measuring the
        // whole thing and keeping the margin ourselves is what puts the picture
        // in the actual middle.
        .ignoresSafeArea()
    }

    private var placeholder: String {
        switch model.qemuState {
        case .idle:
            if case .unavailable(let why) = model.jit, !model.config.virtualization {
                return L("JIT недоступен — %@.\nБез него транслятор не сможет выделить буфер, и машина не запустится.", why)
            }
            return L("Откройте меню и запустите машину")
        case .running:
            switch model.displayStatus {
            case .connected(let w, let h): return L("Экран %d×%d подключён, ждём первый кадр", w, h)
            case .connecting:              return L("Машина работает, подключаемся к экрану…")
            case .failed(let why):         return L("Экран недоступен: %@", why)
            case .disconnected:            return L("Машина работает, экран ещё не слушает")
            }
        case .stopped:          return L("Машина остановлена")
        case .failed(let text): return text
        }
    }
}

/// The guest's serial console — the boot log, which stays useful while the
/// screen is still black.
struct TerminalView: View {
    @ObservedObject var model: VMModel
    /// The shell channel is its own object, so watching the model alone would
    /// miss everything it does.
    @ObservedObject private var shell: ShellChannel
    /// And so is the console: its text arrives on its own object, so a view
    /// that watched only the model would redraw for everything except the one
    /// thing it is here to show — which looked like a console that updates
    /// whenever you leave it and come back.
    @ObservedObject private var serial: SerialConsole
    @ObservedObject private var log = LogCapture.shared
    @ObservedObject private var settings = Settings.shared
    @StateObject private var screen = GuestScreen()
    /// Kept where the app keeps everything else it remembers, so that leaving
    /// for the screen and coming back does not throw the choice away.
    @AppStorage("terminalSource") private var sourceName = Source.emulator.rawValue
    @State private var command = ""
    /// Changed whenever the console should jump to its last line.
    @State private var pin = 0

    init(model: VMModel) {
        _model = ObservedObject(wrappedValue: model)
        _shell = ObservedObject(wrappedValue: model.shell)
        _serial = ObservedObject(wrappedValue: model.serial)
    }

    private var source: Source { Source(rawValue: sourceName) ?? .emulator }

    private func sendCommand() {
        guard !command.isEmpty else { return }
        switch source {
        case .shell:  shell.send(command)
        default:      serial.send(command + "\n")
        }
        command = ""
    }

    /// Whether there is anywhere to type. The console takes commands as soon as
    /// the bootstrap's bash is on it; the shell only once the guest has called
    /// back.
    private var acceptsInput: Bool {
        switch source {
        case .shell:        return shell.isUp
        case .guestConsole: return serial.interactive
        case .emulator:     return false
        }
    }

    /// The indicator at the bottom belongs to whatever is on screen.
    private var linkIsGood: Bool {
        source == .shell ? shell.isUp : serial.connected
    }

    /// Opening the pane is the request to open the channel. A failure is not
    /// retried on its own — it would go on asking the guest forever.
    private func openShellIfNeeded() {
        guard source == .shell, shell.state == .idle else { return }
        shell.connect()
    }

    enum Source: String, CaseIterable {
        case emulator = "Эмулятор"
        case shell = "Шелл"
        case guestConsole = "Лог ядра"
    }

    private var emulatorText: String {
        log.text.isEmpty
            ? L("Пока пусто. Здесь появятся сообщения эмулятора, включая причину отказа запуска.")
            : log.text
    }

    @ViewBuilder
    private func shellPane(_ fitted: CGFloat) -> some View {
        switch shell.state {
        case .up:
            TerminalTextView(text: shell.screen.content, revision: shell.screen.revision,
                             follow: settings.terminalFollow, pin: pin)
                .onAppear { shell.use(fontSize: fitted) }
                .onChange(of: fitted) { shell.use(fontSize: $0) }
        case .connecting:
            notice(L("Прошу гостя подключиться…"), busy: true, action: nil)
        case .idle:
            notice(L("Отдельный канал: гость сам звонит приложению по сети, и сюда не попадает ничего, кроме написанного шеллом. Нужны включённая сеть и bash на консоли."),
                   busy: false, action: L("Подключить"))
        case .failed(let why):
            notice(why, busy: false, action: L("Попробовать снова"))
        }
    }

    @ViewBuilder
    private func consolePane(_ fitted: CGFloat) -> some View {
        if serial.text.isEmpty {
            Text(L("Ожидание вывода консоли…"))
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .padding(8)
        } else {
            TerminalTextView(text: screen.content, revision: screen.revision,
                             follow: settings.terminalFollow, pin: pin)
                .onAppear { screen.use(fontSize: fitted) }
                .onChange(of: fitted) { screen.use(fontSize: $0) }
        }
    }

    private func emulatorPane() -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(emulatorText)
                    .font(.system(size: 10, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(8)
                    .id("log")
            }
            .onChange(of: log.text) { _ in
                if settings.terminalFollow { proxy.scrollTo("log", anchor: .bottom) }
            }
        }
    }

    private func notice(_ text: String, busy: Bool, action: String?) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            if busy { ProgressView() }
            Text(text)
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let action {
                HStack(spacing: 10) {
                    Button(action, systemImage: "bolt.horizontal") {
                        shell.connect()
                    }
                    .buttonStyle(.borderedProminent)
                    // The console is what the button above uses. This is the
                    // other way round — worth offering when the kernel log is
                    // noisy enough to get in the way.
                    Button(L("Через сеть"), systemImage: "network") {
                        shell.connectOverNetwork()
                    }
                    .buttonStyle(.bordered)
                }
                .disabled(!serial.interactive)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
    }

    var body: some View {
        ZStack(alignment: .top) {
            GeometryReader { geo in
                // The guest is certain the console is eighty columns wide — apt
                // truncates its progress line at seventy-nine and erases it with
                // a row of spaces exactly that long. So the type is sized to
                // make eighty columns fit rather than letting the grid wrap and
                // turn every drawing into nonsense.
                let fitted = min(max((geo.size.width - 16)
                                     / (CGFloat(TerminalEmulator.width) * 0.6), 5.5), 13)
                Group {
                    switch source {
                    case .shell:        shellPane(fitted)
                    case .guestConsole: consolePane(fitted)
                    case .emulator:     emulatorPane()
                    }
                }
                // Room for the two things floating over the text.
                .safeAreaInset(edge: .top) { Color.clear.frame(height: 36) }
                .safeAreaInset(edge: .bottom) { Color.clear.frame(height: acceptsInput ? 60 : 72) }
            }
            .onAppear {
                screen.rebuild(from: serial.text, hideKernel: settings.hideKernel,
                               sequence: serial.sequence)
                pin += 1
                openShellIfNeeded()
            }
            .onChange(of: sourceName) { _ in
                pin += 1
                openShellIfNeeded()
            }
            .onChange(of: serial.sequence) { seq in
                screen.feed(serial.chunk, sequence: seq)
            }
            .onChange(of: shell.state) { _ in pin += 1 }
            .onChange(of: settings.hideKernel) { on in
                screen.rebuild(from: serial.text, hideKernel: on,
                               sequence: serial.sequence)
                pin += 1
            }

            header
        }
        .overlay(alignment: .bottom) { if acceptsInput { prompt } }
    }

    /// What is being shown and whether it is alive, as one small glass pill.
    /// It replaces a whole bar of controls: the choice itself now lives in the
    /// menu, and the toggles in the settings.
    private var header: some View {
        HStack(spacing: 7) {
            Circle()
                .fill(linkIsGood ? Color.green : Color.secondary)
                .frame(width: 6, height: 6)
            Text(L(source.rawValue))
                .font(.caption.weight(.medium))
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.14), lineWidth: 0.5))
        .padding(.top, 6)
    }

    /// The command line, floating clear of the text rather than boxed in under
    /// a divider.
    private var prompt: some View {
        HStack(spacing: 8) {
            Image(systemName: "chevron.right")
                .font(.system(size: 12, weight: .bold))
                .foregroundStyle(.secondary)
            TextField(L("Команда гостю"), text: $command)
                .font(.system(size: 14, design: .monospaced))
                .autocorrectionDisabled()
                .noAutocapitalization()
                .submitLabel(.send)
                .onSubmit(sendCommand)
            if source == .shell {
                Button {
                    shell.sendControl(0x03)
                } label: {
                    Text("^C")
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            Button(action: sendCommand) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(command.isEmpty ? AnyShapeStyle(.tertiary) : AnyShapeStyle(.tint))
            }
            .buttonStyle(.plain)
            .disabled(command.isEmpty)
        }
        .padding(.leading, 14)
        .padding(.trailing, 8)
        .padding(.vertical, 7)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay(Capsule().strokeBorder(.white.opacity(0.14), lineWidth: 0.5))
        .shadow(color: .black.opacity(0.3), radius: 8, y: 2)
        // Clear of the floating menu button in the corner.
        .padding(.leading, 12)
        .padding(.trailing, 84)
        .padding(.bottom, 14)
    }
}

/// Shown until the guest images are present in the app's Documents folder.
///
/// There are two ways out of it, and the first one is new: the app can make the
/// whole kit itself out of a firmware archive, and then nothing has to be
/// copied into its folder at all. The other is the folder assembled on a
/// computer, which is how this worked before there was a restore.
/// Shown before any machine is ready to run.
///
/// There is only ever one machine right now, and no way from inside the app
/// to fetch or point at its image yet — the real picker (choosing between
/// prepared VMs) waits on the vmapple backend. This is its placeholder: it
/// says that plainly instead of walking through the iPhone restore
/// flow, which does not apply here.
struct SetupView: View {
    @ObservedObject var model: VMModel

    var body: some View {
        List {
            Section {
                Label(L("Виртуальная машина не настроена"), systemImage: "externaldrive.badge.questionmark")
                    .font(.headline)
                Text(L("Папку с VM делает scripts/utm-to-orchard.py на Маке. Её можно скопировать в «Файлы» → Orchard → OrchardVM или выбрать прямо на флешке — тогда VM грузится с неё без копирования."))
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            VMFolderSection(onChange: { model.refreshFiles() })

            if !model.missing.isEmpty {
                Section(L("Не хватает")) {
                    ForEach(model.missing, id: \.self) { item in
                        Label(item, systemImage: "xmark.circle")
                            .foregroundStyle(.red)
                    }
                }
            }

            Section {
                Button(L("Проверить снова"), systemImage: "arrow.clockwise") {
                    model.refreshFiles()
                }
            }
        }
    }
}

// MARK: - File transfer

enum TransferState: Equatable {
    case running(title: String, done: Int64, total: Int64)
    case finished(String)
    case failed(String)

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    static func size(_ bytes: Int64) -> String {
        bytes < 1 << 20
            ? L("%.0f КБ", Double(bytes) / 1024)
            : L("%.1f МБ", Double(bytes) / 1_048_576)
    }

    /// "5,0 МБ за 10,1 с · 507 КБ/с" — the rate is what tells whether the fast
    /// path was taken.
    static func summary(url: URL, seconds: TimeInterval) -> String {
        let bytes = Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        let rate = seconds > 0 ? Double(bytes) / seconds / 1024 : 0
        return L("%@ за %.1f с · %.0f КБ/с", size(bytes), seconds, rate)
    }
}

/// Sits at the bottom while a file moves, and stays with the outcome until
/// dismissed: a transfer that ended while the menu was closed should still say
/// how it ended.
struct TransferBanner: View {
    let state: TransferState
    let dismiss: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            switch state {
            case .running(let title, let done, let total):
                VStack(alignment: .leading, spacing: 4) {
                    Text(title).font(.footnote).lineLimit(1)
                    if total > 0 {
                        ProgressView(value: Double(done), total: Double(total))
                        Text(L("%@ из %@", TransferState.size(done), TransferState.size(total)))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    } else {
                        ProgressView()
                    }
                }
            case .finished(let text):
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text(text).font(.footnote)
            case .failed(let text):
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text(text).font(.footnote)
            }
            Spacer(minLength: 0)
            // Dismissable while it runs, too. The work carries on — this only
            // takes the banner off the screen — and without it a long install
            // sat on top of the terminal's command line with no way to move it,
            // which is exactly when somebody wants to ask the guest what is
            // going on.
            Button(action: dismiss) {
                Image(systemName: "xmark").font(.footnote.weight(.semibold))
            }
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

#if canImport(UIKit)
/// The guest's picture as a layer's contents, replaced frame by frame.
struct GuestPictureLayer: UIViewRepresentable {
    let frame: CGImage
    let smooth: Bool
    let radius: CGFloat

    final class View: UIView {
        override init(frame: CGRect) {
            super.init(frame: frame)
            isUserInteractionEnabled = false
            layer.contentsGravity = .resizeAspect
            // Continuous, not circular: Apple's corners are squircles.
            layer.cornerCurve = .continuous
        }
        required init?(coder: NSCoder) { fatalError("not used") }
    }

    func makeUIView(context: Context) -> View { View(frame: .zero) }

    func updateUIView(_ view: View, context: Context) {
        CATransaction.begin()
        // No cross-fade between frames: it would keep two of them alive.
        CATransaction.setDisableActions(true)
        view.layer.contents = frame
        let filter: CALayerContentsFilter = smooth ? .trilinear : .nearest
        view.layer.magnificationFilter = filter
        view.layer.minificationFilter = smooth ? .trilinear : .linear
        view.layer.cornerRadius = radius
        view.layer.masksToBounds = radius > 0
        CATransaction.commit()
    }
}
#endif
