import Foundation
#if os(iOS)
import UIKit
#endif

/// Restoring the guest from the phone, with no computer on the other end.
///
/// The stock way needs two machines: one runs the emulated iPhone, the other
/// runs `usbmuxd` and `idevicerestore` and talks to it over USB. Both ends are
/// here instead — the emulator boots the restore ramdisk, `GuestUSB` is the USB
/// host, and this drives what would have been `idevicerestore`.
///
/// The machine boots into restore mode, the guest's USB comes up, `restored`
/// answers, and `RestoreClient` carries out the restore itself — the tickets,
/// the boot objects, the NOR data and the filesystem image over ASR. What it
/// needs beyond the emulator's own kit is the firmware archive it is restoring
/// from and the forged ticket; `RESTORE.md` says where those come from.
final class RestoreSession: ObservableObject {
    enum Stage: Equatable {
        case idle
        /// The machine is coming up on the restore ramdisk.
        case booting
        /// Waiting for the guest to bring its USB port up.
        case waitingForDevice
        /// `restored` answered: the guest is ready to be restored.
        case ready(String)
        /// The restore itself, with whatever the guest is doing right now.
        case restoring(String, Double)
        case done
        case failed(String)

        var isBusy: Bool {
            switch self {
            case .idle, .failed, .ready, .done: return false
            case .booting, .waitingForDevice, .restoring: return true
            }
        }
    }

    static let shared = RestoreSession()

    @Published private(set) var stage: Stage = .idle

    private var usb: GuestUSB?

    /// Whether a restore can be started at all: the ramdisk and everything the
    /// machine needs have to be in place, and nothing else may be running.
    static var ramdisk: URL? { VMConfig.restoreRamdisk }

    /// The ramdisk the machine actually boots: the stock one with the patcher
    /// and its daemon inside it, so the filesystem patches happen in the same
    /// boot the restore does — the last step that used to need a computer.
    ///
    /// Made once and kept beside the kit; made again whenever the app is newer
    /// than the image, which is how a new patcher reaches an old kit.
    static func bootRamdisk(_ stock: URL) -> URL {
        let ours = VMConfig.dataDirectory.appendingPathComponent("Restore-inferno.dmg")
        let stamp = { (url: URL) in
            (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? Date.distantPast
        }
        if FileManager.default.fileExists(atPath: ours.path),
           let daemon = RestoreRamdisk.programs?.daemon, stamp(ours) > stamp(daemon) {
            return ours
        }
        do {
            try RestoreRamdisk.build(stock: stock, into: ours) { LogCapture.shared.note($0) }
            return ours
        } catch {
            // Worth saying out loud rather than failing the restore: the stock
            // ramdisk restores perfectly well, and what is lost is the patches
            // afterwards, which a person can still apply from a computer.
            LogCapture.shared.note(L("Рестор: патчер в RAM-диск не попал — %@", error.localizedDescription))
            return stock
        }
    }

    private func set(_ stage: Stage) {
        DispatchQueue.main.async {
            self.stage = stage
            // A restore is minutes long, and the screen locking partway
            // through suspends the app with nothing to say why it stopped —
            // one plausible reading of tonight's unexplained deaths.
            #if os(iOS)
            UIApplication.shared.isIdleTimerDisabled = stage.isBusy
            #endif
        }
    }

    /// Boots the restore ramdisk and waits for `restored`. Safe to call from the
    /// main thread; everything slow happens elsewhere.
    func start(model: VMModel) {
        guard !stage.isBusy else { return }
        guard QemuBridge.shared.state != .running else {
            set(.failed(L("Сначала выключите машину: рестор поднимает её по-своему")))
            return
        }
        guard let ramdisk = RestoreSession.ramdisk else {
            set(.failed(L("Нет RAM-диска: подготовьте набор из .ipsw")))
            return
        }
        // The machine cannot start on an incomplete folder — QEMU calls exit()
        // from inside our own process when a drive is missing. From scratch
        // this is what "Подготовить набор" makes, so say so rather than listing
        // files the person never had.
        let missing = VMConfig.missingFiles()
        if !missing.isEmpty {
            set(.failed(L("Сначала подготовьте набор — не хватает: %@",
                          missing.joined(separator: ", "))))
            return
        }

        set(.booting)
        LogCapture.shared.note(L("Рестор: RAM-диск %@", ramdisk.lastPathComponent))

        // A previous attempt leaves what it managed to write in the disk file,
        // and the restore is about to overwrite all of it anyway. Handing the
        // blocks back first is what makes room for this one.
        RestoreSession.emptyTheDisk()
        if let free = RestoreSession.freeSpace {
            LogCapture.shared.note(L("Рестор: свободно %.1f ГБ", Double(free) / 1_073_741_824))
            if free < 9 * 1_073_741_824 {
                set(.failed(L("Мало места: нужно около 9 ГБ, свободно %.1f",
                              Double(free) / 1_073_741_824)))
                return
            }
        }

        // The USB host has to be listening before the machine starts: the
        // emulator dials the socket once, as it comes up, and never again.
        let usb = GuestUSB()
        self.usb = usb
        usb.onReady = { [weak self] ready in
            guard let self else { return }
            if ready { self.talkToRestored(usb) }
        }
        do {
            try usb.listen()
        } catch {
            set(.failed(error.localizedDescription))
            return
        }

        var config = Settings.shared.config
        // Not the stock ramdisk: the one carrying the patcher, made here if
        // it is not made yet.
        config.restoreRamdiskPath = RestoreSession.bootRamdisk(ramdisk).path
        // Memory and the translation buffer are left as the person set them.
        // A gigabyte is not enough, by the way — tried on the rig, and the
        // guest's watchdog resets the machine before ASR even starts.
        // The network device would take the USB socket this host needs, and a
        // restore has nothing to do over the network anyway. Serving the port
        // to another machine would take it further still — there would be no
        // socket here at all.
        config.network = false
        config.usbExport = nil
        model.configOverride = config
        self.model = model
        model.start()
        set(.waitingForDevice)
    }

    func stop() {
        usb?.stop()
        usb = nil
        model?.configOverride = nil
        model = nil
        set(.idle)
    }

    /// Held while a restore runs, to put the machine's settings back afterwards.
    private weak var model: VMModel?

    /// Opens a channel to `restored` and asks it who it is — the same QueryType
    /// every libimobiledevice tool starts with.
    private func talkToRestored(_ usb: GuestUSB) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let channel = try usb.connect(to: .restored, timeout: 120)
                defer { channel.close() }
                let answer = try RestoreClient.request(channel, ["Request": "QueryType", "Label": "Inferno"])
                let type = answer["Type"] as? String ?? "?"
                let version = (answer["RestoreProtocolVersion"] as? NSNumber)?.intValue ?? 0
                LogCapture.shared.note(L("Рестор: гость отвечает — %@, протокол %d", type, version))
                self.set(.ready(type))
                try self.restore(over: channel, usb: usb, protocolVersion: version)
            } catch {
                LogCapture.shared.note(L("Рестор: %@", error.localizedDescription))
                self.set(.failed(error.localizedDescription))
            }
        }
    }

    /// Hands the conversation over to `RestoreClient`, once the firmware and
    /// the ticket are both on the phone.
    private func restore(over channel: GuestUSB.Channel, usb: GuestUSB,
                         protocolVersion: Int) throws {
        guard let firmware = RestoreSession.firmware else {
            LogCapture.shared.note(L("Рестор: положите .ipsw рядом с InfernoData, см. RESTORE.md"))
            return
        }
        guard let ticket = try? Data(contentsOf: RestoreSession.ticket) else {
            LogCapture.shared.note(L("Рестор: нет тикета %@, см. RESTORE.md",
                                     RestoreSession.ticket.lastPathComponent))
            return
        }

        // `RestorePrep` copies the pick into the app's own folder and points
        // this at the copy, so the scope below is usually a no-op; it is only
        // load-bearing for the older folder-assembled-on-a-computer path,
        // where nothing ever copies the archive.
        let scoped = firmware.startAccessingSecurityScopedResource()
        defer { if scoped { firmware.stopAccessingSecurityScopedResource() } }
        let ipsw = try IPSW(url: firmware)
        let identity = try ipsw.buildIdentity()
        LogCapture.shared.note(L("Рестор: прошивка %@", firmware.lastPathComponent))

        let cryptexTemplate = (try? Data(contentsOf: VMConfig.cryptexTemplate)).map { [UInt8]($0) }
        let client = RestoreClient(
            usb: usb, ipsw: ipsw, identity: identity, ticket: [UInt8](ticket),
            cryptexTemplate: cryptexTemplate,
            note: { LogCapture.shared.note($0) },
            progress: { [weak self] what, done in
                self?.set(.restoring(what, done))
                RestoreSession.noteMemory(what, done)
            })
        try client.run(on: channel, protocolVersion: protocolVersion)

        // The restore is written, and the machine is still up because the
        // emulator is holding the reset the guest asked for. That is the
        // window our daemon patches in.
        waitForPatcher()
        DispatchQueue.main.async { self.model?.shutdown() }
        set(.done)
    }

    /// Waits for the patcher in the guest, reading the console the machine
    /// writes anyway.
    ///
    /// Nothing else can be asked: the guest has no USB left by then — the
    /// restore ends with `restored` going away — and touching NVRAM or the
    /// volumes while the restore finishes is what broke two runs on the rig.
    /// The daemon says what it is doing on the console, and that is enough.
    ///
    /// An hour is generous on purpose. The patcher walks a two-gigabyte shared
    /// cache, and the phone runs the machine by translating every instruction.
    private func waitForPatcher() {
        set(.restoring(L("Патчи файловой системы"), 0.99))
        guard let handle = try? FileHandle(forReadingFrom: VMConfig.guestConsoleLog) else { return }
        defer { try? handle.close() }

        // From here on only. The console is one file across runs, and a "done"
        // left in it by the restore before this one would end the wait before
        // this patcher had started.
        _ = try? handle.seekToEnd()

        var text = ""
        var seen = Set<String>()
        let deadline = Date().addingTimeInterval(3600)
        while Date() < deadline {
            if let chunk = try? handle.readToEnd(), !chunk.isEmpty {
                text += String(decoding: chunk, as: UTF8.self)
            }
            for line in text.split(separator: "\n") where line.contains("*** PATCHER:") {
                let said = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
                if seen.insert(said).inserted {
                    LogCapture.shared.note(said.replacingOccurrences(of: "*** PATCHER: ", with: "Патчи: "))
                }
            }
            if text.contains("PATCHER: done") || text.contains("PATCHER: no shared cache") { return }
            Thread.sleep(forTimeInterval: 2)
        }
        LogCapture.shared.note(L("Патчи: не дождались — машина останавливается как есть"))
    }

    /// The firmware archive to restore from.
    ///
    /// `RestorePrep` remembers its own copy here once it has made one — see
    /// `localCopy(of:into:note:)`. Before that, or if remembering never ran,
    /// this falls back to a bookmark of wherever the person picked it, and
    /// failing that, any `.ipsw` sitting beside the emulator's files.
    static var firmware: URL? {
        firmwareLock.lock(); defer { firmwareLock.unlock() }
        // Resolved once and kept.
        //
        // Resolving a security-scoped bookmark is not free: every call takes a
        // sandbox extension, and the interface asks for this while a restore
        // runs — it redraws on every chunk of the image. Doing it each time ran
        // the process out of extensions in about a minute, and the app went
        // down mid-transfer with `sandbox_extension_consume failed: 12`.
        if let resolved = resolvedFirmware { return resolved }
        if let data = UserDefaults.standard.data(forKey: "restoreFirmware") {
            var stale = false
            #if os(macOS)
            let url = try? URL(resolvingBookmarkData: data, options: .withSecurityScope,
                               bookmarkDataIsStale: &stale)
            #else
            let url = try? URL(resolvingBookmarkData: data, bookmarkDataIsStale: &stale)
            #endif
            if let url {
                resolvedFirmware = url
                return url
            }
        }
        for directory in [VMConfig.dataDirectory, VMConfig.documents] {
            let found = (try? FileManager.default.contentsOfDirectory(at: directory,
                                                                     includingPropertiesForKeys: nil))?
                .first { $0.pathExtension.lowercased() == "ipsw" }
            if let found {
                resolvedFirmware = found
                return found
            }
        }
        return nil
    }

    /// Says how much memory the app holds, every so often, while the image
    /// moves. Guessing at this cost three runs; a number costs one line.
    private static var lastMemoryNote = Date.distantPast
    private static func noteMemory(_ what: String, _ done: Double) {
        guard Date().timeIntervalSince(lastMemoryNote) >= 15 else { return }
        lastMemoryNote = Date()
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                           / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return }
        LogCapture.shared.note(L("Рестор: %@ %.1f %%, память %.0f МБ, свободно %.1f ГБ",
                                 what, done * 100,
                                 Double(info.phys_footprint) / 1_048_576,
                                 Double(freeSpace ?? 0) / 1_073_741_824))
    }

    /// What the phone has left, as iOS counts it for something worth keeping.
    static var freeSpace: Int? {
        let values = try? VMConfig.documents.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage.map(Int.init)
    }

    /// Gives back the blocks of whatever was written before, keeping the file
    /// at its full size. A restore erases the disk in any case.
    private static func emptyTheDisk() {
        if let image = VMConfig.rootImage, image.format == "raw" {
            blank(URL(fileURLWithPath: image.path))
        }
        // The four small state disks go back to blank as well, because a
        // restore erases the device and the guide makes them blank for one.
        //
        // This is not tidiness. Builds before this one seeded them from a real
        // device, and a kit made by one of those keeps those copies — nothing
        // ever overwrites a file that is already there. A restore onto them
        // finishes and then cannot unlock its own data volume: no key, no
        // screen, and nothing in the log to say why.
        for name in ["nvram", "effaceable", "sep_nvram", "sep_ssc"] {
            blank(VMConfig.dataDirectory.appendingPathComponent(name),
                  size: RestorePrep.disks.first { $0.name == name }?.size)
        }
    }

    /// Hands a disk's blocks back, keeping the file the size it was.
    private static func blank(_ url: URL, size: Int? = nil) {
        guard FileManager.default.fileExists(atPath: url.path),
              let handle = try? FileHandle(forWritingTo: url)
        else { return }
        defer { try? handle.close() }
        let was = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int
        try? handle.truncate(atOffset: 0)
        try? handle.truncate(atOffset: UInt64(size ?? was ?? 32 << 30))
    }

    private static let firmwareLock = NSLock()
    private static var resolvedFirmware: URL?

    /// Keeps a picked file reachable after the app restarts.
    static func remember(firmware url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        firmwareLock.lock()
        resolvedFirmware = url
        firmwareLock.unlock()
        #if os(macOS)
        let data = try? url.bookmarkData(options: .withSecurityScope)
        #else
        let data = try? url.bookmarkData()
        #endif
        UserDefaults.standard.set(data, forKey: "restoreFirmware")
    }

    /// The forged AP ticket, the same file every boot afterwards needs.
    static var ticket: URL { VMConfig.dataDirectory.appendingPathComponent("root_ticket.der") }
}
