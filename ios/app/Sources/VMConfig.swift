import Foundation

/// Builds the emulator command line and reports what is missing.
///
/// The guest is macOS on QEMU's `apple-vm` machine — the one
/// Virtualization.framework runs on Apple Silicon — booted through Apple's own
/// chain, AVPBooter → iBoot → XNU. Its files are an `OrchardVM` folder in the
/// app's Documents, dropped in over the Files app; `scripts/utm-to-orchard.py`
/// makes that folder out of a UTM virtual machine.
///
/// The iPhone half of this type (`iPhoneData`, the SEP ROM, the restore
/// firmware) is what the app shell inherited from its iPhone past. It is not used to
/// start anything any more, and stays only because the restore screens still
/// compile against it; it goes when they do.
struct VMConfig {
    /// The guest is macOS. The app shell grew up around an iPhone guest and
    /// still has the code that drives one — its battery, taptic engine, status
    /// bar, packages and in-guest agent — and every one of those talks to a
    /// guest that is not there. This is what keeps them quiet.
    static let macGuest = true

    var cores: Int = 4
    var memory: String = "1536M"
    var vncPort: UInt16 = 5900
    var serialPort: UInt16 = 4555
    var qmpPort: UInt16 = 4556
    var tbSize: Int = 128
    /// The iPad's own cores instead of the translator. Set only when the
    /// kernel has already said yes — see `HVF`.
    var virtualization: Bool = false
    /// Reverse tethering over the guest's own USB port: the emulator plays the
    /// USB host, brings up the device's CDC-NCM interface and NATs through
    /// slirp. No privileges, no companion VM.
    var network: Bool = true
    /// No screen at all: nothing to encode or copy. The guest is then reachable
    /// only through its console, which is what a headless run is for.
    var headless: Bool = false
    /// Read the framebuffer where it already is instead of going through a VNC
    /// server on the loopback. The VNC path stays available: it is the one that
    /// has years of use behind it, and it is worth being able to fall back to
    /// when something looks wrong.
    var builtInDisplay: Bool = true
    /// The guest's framebuffer in pixels, and how many of them make a point.
    /// The panel is 828×1792 at two, which is an iPhone 11; halving the
    /// framebuffer and dropping the scale to one keeps the same interface over
    /// a quarter of the pixels, and the app scales the picture back up so it
    /// covers the same area of the screen.
    /// Whether the machine is given a way to be heard. The emulated sound card
    /// exists either way; this decides whether anything is on the other end.
    var audio: Bool = false
    var displayWidth: Int = 828
    var displayHeight: Int = 1792
    var displayScale: Int = 2
    /// Boot this ramdisk instead of the installed system: what a restore runs.
    /// The machine then heads for recovery by itself, so it is not told to leave
    /// it, and `GuestUSB` — not the network device — owns the USB socket.
    var restoreRamdiskPath: String?
    /// Serve the guest's USB port to another machine over VirtualHere, at this
    /// `host:port`, instead of keeping it here.
    ///
    /// The guest has one USB port and it has one host. Normally that host is on
    /// this side: the app for a restore, the emulator's own CDC-NCM device for
    /// the guest's internet. Set this and the port is served out instead — a
    /// Mac running a VirtualHere client sees a real iPhone on its own USB — and
    /// nothing on this side can have it meanwhile.
    var usbExport: String?

    /// The app's Documents on iOS, where the Files app shows it. A Mac app that
    /// is not sandboxed would be handed the user's whole Documents folder, so it
    /// keeps to a folder of its own inside it.
    static var documents: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        #if os(macOS)
        return base.appendingPathComponent("Orchard")
        #else
        return base
        #endif
    }

    static var dataDirectory: URL { documents.appendingPathComponent("iPhoneData") }

    // MARK: The macOS virtual machine

    /// The five files a macOS guest boots from, as `scripts/utm-to-orchard.py`
    /// lays them out and `scripts/run-vm.sh` uses them on a desktop.
    /// A folder picked in Settings (a USB drive, say) when there is one, the
    /// app's own `Documents/OrchardVM` otherwise; see `VMFolder`.
    static var vmDirectory: URL { VMFolder.chosen ?? documents.appendingPathComponent("OrchardVM") }
    /// The system disk, `disk.qcow2` or `disk.raw`. Only ever read: the guest
    /// writes into `overlay`. qcow2 is what the converter makes: the guest's
    /// disk is 64 GiB with ~16 in use, and a raw copy stays that small only
    /// while its holes survive, which exFAT and the Files app do not allow.
    static var disk: URL {
        let qcow = vmDirectory.appendingPathComponent("disk.qcow2")
        return FileManager.default.fileExists(atPath: qcow.path)
            ? qcow : vmDirectory.appendingPathComponent("disk.raw")
    }
    static var diskFormat: String { disk.pathExtension == "qcow2" ? "qcow2" : "raw" }
    /// A qcow2 over `disk`, holding everything the guest writes.
    ///
    /// Not optional. The machine opens the disk twice, once as a read-only
    /// flash and once as the root device, and on Darwin nothing arbitrates
    /// between the two: QEMU's `locking=auto` means no locks at all without
    /// open-file-description locks, which this kernel does not have. Through
    /// the overlay both opens of `disk` are reads, so neither can see the other
    /// change under it — and the disk stays exactly as it came, so a first boot
    /// that goes wrong costs an overlay, not the system.
    static var overlay: URL { vmDirectory.appendingPathComponent("overlay.qcow2") }
    /// The NVRAM and LocalPolicy. Personalised to this VM's ECID along with the
    /// disk, so it only ever works with the disk it came with.
    static var aux: URL { vmDirectory.appendingPathComponent("aux.img") }
    /// Apple's VM firmware, patched to run outside Apple's hypervisor.
    static var rom: URL { vmDirectory.appendingPathComponent("AVPBooter.patched.bin") }
    /// Where the ECID comes from, in the format tart and this project use.
    static var machineConfig: URL { vmDirectory.appendingPathComponent("config.json") }

    /// The VM's ECID: base64 of a binary plist `{ECID: <int>}` under `ecid`.
    ///
    /// Nil when there is no such file or no such key. The LocalPolicy on the
    /// disk is signed for exactly this number, so there is no default that
    /// could stand in for it.
    static var ecid: UInt64? {
        guard let data = try? Data(contentsOf: machineConfig),
              let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let encoded = json["ecid"] as? String,
              let blob = Data(base64Encoded: encoded),
              let plist = (try? PropertyListSerialization.propertyList(from: blob, format: nil)) as? [String: Any],
              let number = plist["ECID"] as? NSNumber
        else { return nil }
        return number.uint64Value
    }

    /// What the emulator reads from its environment for this machine, the same
    /// switches `scripts/run-vm.sh` sets.
    static let guestEnvironment: [String: String] = [
        // The stock kernel checks its own signed pointers, so pointer
        // authentication has to be real or it dies with "JOP Hash Mismatch".
        "ORCHARD_REAL_PAUTH": "1",
        // A core started by PSCI CPU_ON gets the keys the firmware would have
        // left it; without this every secondary faults on its first check.
        "ORCHARD_PAC_INHERIT": "1",
        // No REIMS_VGPU_FORCE_SCANOUT, which scripts/run-vm.sh still sets:
        // reims-vgpu no longer reads it anywhere.
    ]

    /// The scratch namespace both sides reach: the app writes bytes into this
    /// file, the guest reads the same place as a block device.
    static var transferImage: URL { dataDirectory.appendingPathComponent("xfer") }
    /// Sixteen mebibytes, and sparse, so it costs nothing until it is used. A
    /// gibibyte is not required: that floor applies only to a namespace with
    /// nstype=1, which is the root disk.
    static let transferBytes: Int64 = 16 * 1024 * 1024

    /// Creates the scratch namespace if it is not there yet.
    static func ensureTransferImage() {
        let path = transferImage.path
        guard !FileManager.default.fileExists(atPath: path) else { return }
        guard FileManager.default.createFile(atPath: path, contents: nil) else { return }
        // Truncated rather than written: the file reads as zeroes and occupies
        // only the blocks that are actually used.
        if let handle = try? FileHandle(forWritingTo: transferImage) {
            try? handle.truncate(atOffset: UInt64(transferBytes))
            try? handle.close()
        }
    }
    /// Where the emulator must chdir to before the sockets below resolve.
    static var socketDirectory: String { NSTemporaryDirectory() }
    static let usbSocketName = "orchard-usb.sock"
    /// Everything the guest prints, from the first byte, kept on disk.
    static var guestConsoleLog: URL { documents.appendingPathComponent("guest-console.log") }
    static var sepROM: URL { documents.appendingPathComponent("AppleSEPROM-Cebu-B1") }
    static var sepROMPresent: Bool { FileManager.default.fileExists(atPath: sepROM.path) }
    /// Any real device's own Cryptex1 IM4M -- iOS 16+ only, see `Cryptex1`.
    /// A Mac's own lives under
    /// `/System/Volumes/Preboot/<UUID>/cryptex1/current/apticket.*.im4m`.
    static var cryptexTemplate: URL { dataDirectory.appendingPathComponent("cryptex_template.im4m") }
    static var cryptexTemplatePresent: Bool { FileManager.default.fileExists(atPath: cryptexTemplate.path) }

    /// Whether the device disk holds a system at all.
    ///
    /// A freshly made disk is 32 GB of zeros, and booting one does not fail —
    /// the machine sits there looking for something to start, which from the
    /// outside is indistinguishable from a hang. A restore is what fills it, so
    /// the first bytes are worth a look before the machine is let go.
    ///
    /// For a macOS guest the system comes installed — the disk is one that has
    /// already booted to a desktop elsewhere — so this only guards against an
    /// empty one: a qcow2 with nothing allocated, or a raw file of zeros where
    /// a GPT should begin.
    static var systemInstalled: Bool {
        if diskFormat == "qcow2" { return qcow2HasAllocatedData(at: disk.path) }
        guard let handle = FileHandle(forReadingAtPath: disk.path) else { return false }
        defer { try? handle.close() }
        let head = handle.readData(ofLength: 64 * 1024)
        return head.contains { $0 != 0 }
    }

    /// Whether a qcow2 image has ever had anything written to it.
    ///
    /// A blank disk fresh out of `qemu-img create -f qcow2` — which is what the
    /// kit ships as `root.qcow2`, the same way a restore leaves one blank until
    /// it actually runs — has no allocated clusters at all: its L1 table (the
    /// top level of the two-level map from guest offset to host cluster) is
    /// every entry zero. Once anything is written, at least one L1 entry points
    /// at an L2 table. That is cheaper to check than trusting the file's mere
    /// existence, or its format, to mean a restore actually completed.
    private static func qcow2HasAllocatedData(at path: String) -> Bool {
        guard let handle = FileHandle(forReadingAtPath: path) else { return true }
        defer { try? handle.close() }
        guard let header = try? handle.read(upToCount: 104), header.count >= 48 else { return true }
        guard header[0..<4].elementsEqual([0x51, 0x46, 0x49, 0xFB]) else { return true }    // "QFI\xFB"

        func be32(_ at: Int) -> UInt32 { header[at..<at + 4].reduce(0) { ($0 << 8) | UInt32($1) } }
        func be64(_ at: Int) -> UInt64 { header[at..<at + 8].reduce(0) { ($0 << 8) | UInt64($1) } }

        let l1Size   = Int(be32(36))
        let l1Offset = be64(40)
        guard l1Size > 0 else { return false }    // no L1 table at all -- nothing was ever mapped

        handle.seek(toFileOffset: l1Offset)
        guard let l1Table = try? handle.read(upToCount: l1Size * 8) else { return true }
        return l1Table.contains { $0 != 0 }
    }

    /// The device image, either as the raw file from the desktop kit or as a
    /// qcow2 conversion of it. qcow2 is preferred for transfers: the raw file is
    /// 34 GB of mostly holes, and most ways of copying it onto a phone fill them in.
    static var rootImage: (path: String, format: String)? {
        let qcow = dataDirectory.appendingPathComponent("root.qcow2")
        if FileManager.default.fileExists(atPath: qcow.path) {
            return (qcow.path, "qcow2")
        }
        let raw = dataDirectory.appendingPathComponent("root")
        if FileManager.default.fileExists(atPath: raw.path) {
            return (raw.path, "raw")
        }
        return nil
    }

    /// The ramdisk a restore boots from, if one was put beside the firmware.
    ///
    /// An IPSW carries two: the smaller one erases, the larger one upgrades. The
    /// erase ramdisk is the one a fresh install needs, and the manifest names it
    /// for the identity that erases.
    static var restoreRamdisk: URL? {
        guard let manifest = buildManifest(),
              let name = manifest.path(of: "RestoreRamDisk")
        else { return nil }
        let url = dataDirectory.appendingPathComponent("Restore/" + name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// What the machine loads, taken from the IPSW's own manifest where there is
    /// one. Without it the names are the ones iOS 14.0 beta 5 uses, which is what
    /// every existing installation has.
    struct Firmware {
        var kernel: String
        var deviceTree: String
        var trustcache: String
    }

    static var firmware: Firmware {
        let fallback = Firmware(kernel: "Restore/kernelcache.release.iphone12b",
                                deviceTree: "Restore/Firmware/all_flash/DeviceTree.n104ap.im4p",
                                trustcache: "Restore/Firmware/038-44135-124.dmg.trustcache")
        guard let manifest = buildManifest() else { return fallback }
        let ramdisk = manifest.path(of: "RestoreRamDisk")
        let trustcache = manifest.path(of: "RestoreTrustCache")
            ?? ramdisk.map { "Firmware/\($0).trustcache" }
        guard let kernel = manifest.path(of: "KernelCache"),
              let tree = manifest.path(of: "DeviceTree"),
              let trustcache
        else { return fallback }
        let resolved = Firmware(kernel: "Restore/" + kernel,
                                deviceTree: "Restore/" + tree,
                                trustcache: "Restore/" + trustcache)
        // A manifest that names files nobody copied over is worse than no
        // manifest: the machine would refuse to start on a working set.
        let present = [resolved.kernel, resolved.deviceTree, resolved.trustcache].allSatisfy {
            usable(dataDirectory.appendingPathComponent($0))
        }
        return present ? resolved : fallback
    }

    /// The erase identity for the iPhone 11, out of `BuildManifest.plist`.
    private struct BuildIdentity {
        let manifest: [String: Any]
        func path(of component: String) -> String? {
            guard let entry = manifest[component] as? [String: Any],
                  let info = entry["Info"] as? [String: Any]
            else { return nil }
            return info["Path"] as? String
        }
    }

    private static var cachedManifest: (stamp: Date, identity: BuildIdentity)?
    private static let manifestLock = NSLock()

    /// The erase identity, parsed once.
    ///
    /// The file is half a megabyte, and this is asked for from view bodies —
    /// during a restore those redraw as fast as the image moves. Re-read only
    /// when the file itself changes.
    private static func buildManifest() -> BuildIdentity? {
        let url = dataDirectory.appendingPathComponent("Restore/BuildManifest.plist")
        let stamp = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
            .contentModificationDate ?? .distantPast
        manifestLock.lock()
        if let cached = cachedManifest, cached.stamp == stamp {
            manifestLock.unlock()
            return cached.identity
        }
        manifestLock.unlock()
        guard let found = parseManifest(url) else { return nil }
        manifestLock.lock()
        cachedManifest = (stamp, found)
        manifestLock.unlock()
        return found
    }

    private static func parseManifest(_ url: URL) -> BuildIdentity? {
        guard let data = try? Data(contentsOf: url),
              let plist = try? PropertyListSerialization.propertyList(from: data, options: [], format: nil),
              let root = plist as? [String: Any],
              let identities = root["BuildIdentities"] as? [[String: Any]]
        else { return nil }
        for identity in identities {
            guard let info = identity["Info"] as? [String: Any],
                  (info["DeviceClass"] as? String)?.lowercased() == "n104ap",
                  (info["Variant"] as? String)?.contains("Erase") == true,
                  let manifest = identity["Manifest"] as? [String: Any]
            else { continue }
            return BuildIdentity(manifest: manifest)
        }
        return nil
    }

    /// Everything the machine needs on disk, in the order a person should fix it.
    static let requiredFiles: [(label: String, relativePath: String)] = [
        (L("Прошивка NVMe"), "iPhoneData/firmware"),
        ("syscfg", "iPhoneData/syscfg"),
        ("ctrl_bits", "iPhoneData/ctrl_bits"),
        ("nvram", "iPhoneData/nvram"),
        ("effaceable", "iPhoneData/effaceable"),
        ("panic_log", "iPhoneData/panic_log"),
        ("SEP nvram", "iPhoneData/sep_nvram"),
        ("SEP ssc", "iPhoneData/sep_ssc"),
        (L("Тикет"), "iPhoneData/root_ticket.der"),
        (L("Прошивка SEP"), "iPhoneData/sep-firmware.n104.RELEASE.new.img4"),
        ("SEP ROM", "AppleSEPROM-Cebu-B1"),
    ]

    /// What a macOS guest needs in `OrchardVM`, in the order a person should
    /// fix it — the disk first, since everything else is personalised to it.
    static func missingFiles() -> [String] {
        let files: [(label: String, url: URL)] = [
            (L("Диск macOS (OrchardVM/disk.qcow2 или disk.raw)"), disk),
            (L("Оверлей (OrchardVM/overlay.qcow2)"), overlay),
            (L("NVRAM (OrchardVM/aux.img)"), aux),
            (L("Прошивка VM (OrchardVM/AVPBooter.patched.bin)"), rom),
            (L("Конфигурация с ECID (OrchardVM/config.json)"), machineConfig),
        ]
        var missing = files.compactMap { usable($0.url.resolvingSymlinksInPath()) ? nil : $0.label }
        // Present but unreadable is as good as absent: without the ECID the
        // machine cannot be told who it is.
        if usable(machineConfig), ecid == nil {
            missing.append(L("ECID в OrchardVM/config.json не читается"))
        }
        return missing
    }

    /// Whether a file the emulator needs is actually a file.
    ///
    /// Asking `fileExists` is not enough, and the difference is not academic:
    /// an unpacked archive can leave a *folder* named `firmware`, the check
    /// passes, Start is enabled, and then QEMU says `'file' driver requires
    /// '…/firmware' to be a regular file` and calls `exit(1)` — from inside
    /// `qemu_init`, which runs in our own process, so the whole app goes down
    /// and it looks like a crash. An empty file does the same. Reported as
    /// issue #5.
    private static func usable(_ url: URL) -> Bool {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &directory),
              !directory.boolValue
        else { return false }
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int64
        return (size ?? 0) > 0
    }

    /// The kernel's command line. A Mac can be handed another one for a single
    /// run — `open Orchard.app --args -bootArgs "…"` — which is how a boot
    /// argument is tried without a rebuild; a launch argument lives only in
    /// that process's defaults and is never saved.
    ///
    /// No `mtxspin=-1`, which the desktop kit passes. XNU caps it at 62.5 ms
    /// (`ml_init_lock_timeout`) against a default of 10 µs, and that is how
    /// long a thread waiting on a held mutex spins on its core while the owner
    /// runs elsewhere. Spinning is only ever a bet that waiting is shorter than
    /// sleeping; under emulation owners hold their locks far longer, so the
    /// bet loses and the guest's cores burn the time.
    static var bootArguments: String {
        UserDefaults.standard.string(forKey: "bootArgs")
            ?? "tlto_us=-1 agm-genuine=1 agm-authentic=1 agm-trusted=1 serial=3 wdt=-1 launchd_unsecure_cache=1 -vm_compressor_wk_sw"
    }

    /// The command line for the macOS guest.
    ///
    /// `scripts/run-vm.sh` is the reference, and every device and drive below
    /// is the same as there; its comments say what each one cost to find. What
    /// differs is what a phone cannot give: memory, cores and translation
    /// buffer come from the settings rather than a desktop's 10 GiB and 12
    /// cores, and the console and QMP are on the loopback, where the app's
    /// clients expect them.
    func arguments() -> [String] {
        // QEMU looks for its data files (VNC keymaps among them) next to the
        // binary; inside an app bundle it has to be told where they are, or it
        // reports "could not read keymap file" and exits.
        let dataDir = (Bundle.main.resourcePath ?? Bundle.main.bundlePath) + "/qemu-data"

        // Multi-threaded TCG: iOS gives applications no hypervisor. split-wx
        // maps the translation buffer twice, writable and executable, which is
        // what a debugger-enabled process may do when MAP_JIT is refused.
        let accel = "tcg,thread=multi,tb-size=\(tbSize)" + (JIT.needsSplitWX ? ",split-wx=on" : "")

        let disk = VMConfig.disk.path
        let aux = VMConfig.aux.path

        var argv = [
            "qemu-system-aarch64",
            "-L", dataDir,
            "-accel", accel,
            // missingFiles() refuses a start without a readable ECID, so the
            // zero is never what a machine is actually given.
            "-M", "apple-vm,uuid=\(VMConfig.ecid ?? 0)",
            "-smp", String(cores),
            "-m", memory,
            "-bios", VMConfig.rom.path,
            "-drive", "file=\(aux),if=pflash,format=raw",
            // Not flash: the machine takes its boot device's aux and root
            // backends from these two slots, so any format QEMU reads will do.
            "-drive", "file=\(disk),if=pflash,format=\(VMConfig.diskFormat),readonly=on",
            "-drive", "file=\(aux),if=none,id=aux,format=raw",
            "-device", "vmapple-virtio-blk-pci,variant=aux,drive=aux,share-rw=on",
            "-drive", "file=\(VMConfig.overlay.path),if=none,id=root,format=qcow2,cache=writeback,aio=threads,discard=unmap",
            "-device", "vmapple-virtio-blk-pci,variant=root,drive=root",
            // The console is logged to a file rather than only streamed: a
            // socket drops everything printed before a client attaches, and the
            // guest starts talking long before the UI can connect. Appended to,
            // so that when the app cuts an overgrown log back to nothing the
            // emulator carries on at the top instead of beyond a hole of zeros;
            // the app empties the file before each start instead.
            "-chardev", "socket,id=serial0,host=127.0.0.1,port=\(serialPort),server=on,wait=off,logfile=\(VMConfig.guestConsoleLog.path),logappend=on",
            "-serial", "chardev:serial0",
            // Lets the app ask the machine what state it is in, and stop it cleanly.
            "-qmp", "tcp:127.0.0.1:\(qmpPort),server,nowait",
        ]

        if network {
            // romfile= : no option ROM. The card's is an EFI network boot
            // image, which a macOS guest booting from its own disk never runs,
            // and the app does not ship QEMU's pc-bios: without this QEMU
            // stops at `failed to find romfile "efi-virtio.rom"`.
            argv += ["-netdev", "user,id=net0", "-device", "virtio-net-pci,netdev=net0,romfile="]
        }

        if audio {
            // A virtio-sound card whose output goes to the app through the
            // in-process backend (qemu/audio/orchardaudio.c, GuestSound.swift).
            argv += ["-audiodev", "orchard,id=snd0", "-device", "virtio-sound-pci,audiodev=snd0"]
        }

        if headless || builtInDisplay {
            // Nothing for the emulator to serve: either there is no screen at
            // all, or the app reads the framebuffer directly once the machine
            // is up (ui/orchard-embed.c).
            argv += ["-display", "none"]
        }
        else {
            argv += ["-vnc", "127.0.0.1:\(vncPort - 5900)"]
        }

        return argv
    }
}
