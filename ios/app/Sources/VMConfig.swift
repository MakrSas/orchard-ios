import Foundation

/// Builds the emulator command line and reports what is missing.
///
/// The layout mirrors the desktop kit: an `InfernoData` directory plus the SEP
/// ROM, dropped into the app's Documents folder over the Files app.
struct VMConfig {
    var cores: Int = 4          // 3 CPU cores + the SEP core
    var memory: String = "3G"
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
        return base.appendingPathComponent("Inferno")
        #else
        return base
        #endif
    }

    static var dataDirectory: URL { documents.appendingPathComponent("InfernoData") }

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
    static let usbSocketName = "inferno-usb.sock"
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
    static var systemInstalled: Bool {
        guard let image = rootImage else { return false }
        if image.format == "qcow2" { return qcow2HasAllocatedData(at: image.path) }
        guard let handle = FileHandle(forReadingAtPath: image.path) else { return false }
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
        (L("Прошивка NVMe"), "InfernoData/firmware"),
        ("syscfg", "InfernoData/syscfg"),
        ("ctrl_bits", "InfernoData/ctrl_bits"),
        ("nvram", "InfernoData/nvram"),
        ("effaceable", "InfernoData/effaceable"),
        ("panic_log", "InfernoData/panic_log"),
        ("SEP nvram", "InfernoData/sep_nvram"),
        ("SEP ssc", "InfernoData/sep_ssc"),
        (L("Тикет"), "InfernoData/root_ticket.der"),
        (L("Прошивка SEP"), "InfernoData/sep-firmware.n104.RELEASE.new.img4"),
        ("SEP ROM", "AppleSEPROM-Cebu-B1"),
    ]

    static func missingFiles() -> [String] {
        // The three the manifest names go last, since which files they are
        // depends on the version that was installed.
        let loaded = firmware
        let all = requiredFiles + [
            ("Kernelcache", "InfernoData/" + loaded.kernel),
            ("Device tree", "InfernoData/" + loaded.deviceTree),
            ("TrustCache", "InfernoData/" + loaded.trustcache),
        ]
        var missing = all.compactMap { entry -> String? in
            let url = documents.appendingPathComponent(entry.relativePath).resolvingSymlinksInPath()
            return usable(url) ? nil : entry.label
        }
        if rootImage == nil {
            missing.insert(L("Диск устройства (root.qcow2 или root)"), at: 0)
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
    /// run — `open Inferno.app --args -bootArgs "…"` — which is how a boot
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

    func arguments() -> [String] {
        let data = VMConfig.dataDirectory.path
        let sep = VMConfig.sepROM.path
        // A relative name on purpose: AF_UNIX stores the path itself, and the
        // 104-byte sun_path limit cannot hold an app container path. The
        // emulator runs with its working directory set to the socket's folder,
        // so both ends resolve this to the same file.
        let usbSocket = VMConfig.usbSocketName

        let loaded = VMConfig.firmware
        // Who gets the guest's USB port. `inferno` is the protocol this app
        // speaks: the port dials the socket below and whoever listens there is
        // the host. `virtualhere` turns it around — the emulator listens, and a
        // VirtualHere client on another machine takes the device.
        var parts = ["t8030"]
        if let usbExport {
            parts += ["usb-uplink-type=virtualhere", "usb-uplink-addr=\(usbExport)"]
        } else {
            parts += ["usb-uplink-type=inferno", "usb-uplink-addr=unix:\(usbSocket)"]
        }
        parts += [
            "trustcache=\(data)/\(loaded.trustcache)",
            "ticket=\(data)/root_ticket.der",
            "sep-fw=\(data)/sep-firmware.n104.RELEASE.new.img4",
            "sep-rom=\(sep)",
            "kaslr-off=true",
        ]
        // Which way out of recovery the machine takes, said every time rather
        // than left to whatever NVRAM happens to hold.
        //
        // An ordinary run only ever starts an installed system, and NVRAM can
        // say `auto-boot=false` — left there by a restore that did not finish.
        // Then the machine heads for recovery, wants a ramdisk nobody passed,
        // and the emulator quits with `RAM Disk required for recovery` before
        // the guest exists.
        //
        // A restore is the mirror of that, and getting it wrong is worse,
        // because nothing says so: on `auto-boot=true` the machine ignores the
        // ramdisk, boots as usual, finds no system on a blank disk, sits on
        // `Still waiting for root device` and panics a minute later in
        // `IOAESAccelerator`. Restores used to work here only because an
        // earlier unfinished one had left `auto-boot=false` behind; a kit made
        // from scratch carries a NVRAM that says otherwise.
        parts.append(restoreRamdiskPath == nil ? "boot-mode=exit_recovery" : "boot-mode=enter_recovery")
        parts += [
            "disp-width=\(displayWidth)",
            "disp-height=\(displayHeight)",
            "disp-scale=\(displayScale)",
        ]
        let machine = parts.joined(separator: ",")

        // QEMU looks for its data files (VNC keymaps among them) next to the
        // binary; inside an app bundle it has to be told where they are, or it
        // reports "could not read keymap file" and exits.
        let dataDir = (Bundle.main.resourcePath ?? Bundle.main.bundlePath) + "/qemu-data"

        // HVF where the kernel allows it: the guest's cores run on the iPad's,
        // and the emulator patches the kernel for it by itself.
        //
        // Otherwise multi-threaded TCG, since iOS gives no hypervisor access to
        // applications. Never single: the SEP and the AP cores have to move
        // together, and on one thread the SEP panics initialising its key
        // store, so the guest never boots. That used to be a setting, and it
        // only ever caught people out. split-wx maps the translation buffer
        // twice — writable and executable — which is what a debugger-enabled
        // process is allowed to do when MAP_JIT is refused.
        let accel = virtualization
            ? "hvf"
            : "tcg,thread=multi,tb-size=\(tbSize)" + (JIT.needsSplitWX ? ",split-wx=on" : "")

        var argv = [
            "qemu-system-aarch64",
            "-L", dataDir,
            "-accel", accel,
            "-M", machine,
            "-kernel", "\(data)/\(loaded.kernel)",
            "-dtb", "\(data)/\(loaded.deviceTree)",
            "-append", VMConfig.bootArguments,
            "-smp", String(cores),
            "-m", memory,
            // The console is logged to a file rather than only streamed: a
            // socket drops everything printed before a client attaches, and the
            // guest starts talking long before the UI can connect. Appended to,
            // so that when the app cuts an overgrown log back to nothing the
            // emulator carries on at the top instead of beyond a hole of zeros;
            // the app empties the file before each start instead.
            "-chardev", "socket,id=serial0,host=127.0.0.1,port=\(serialPort),server=on,wait=off,logfile=\(VMConfig.guestConsoleLog.path),logappend=on",
            "-serial", "chardev:serial0",
            // Lets the app ask the machine what state it is in.
            "-qmp", "tcp:127.0.0.1:\(qmpPort),server,nowait",
            "-drive", "file=\(data)/sep_nvram,if=pflash,format=raw",
            "-drive", "file=\(data)/sep_ssc,if=pflash,format=raw",
        ]

        // A restore boots the ramdisk from the IPSW instead of the disk. The
        // machine puts `-restore rd=md0` on the command line by itself once it
        // sees one, so nothing else changes here.
        if let ramdisk = restoreRamdiskPath {
            argv += ["-initrd", ramdisk]
            // A restore ends with the guest asking to be reset, and the machine
            // would go down with it -- taking the ramdisk, and our patcher
            // still working inside it, along. Holding the reset keeps the
            // machine up until the app stops it itself.
            argv += ["-global", "driver=apple-smc,property=hold-reset,value=on"]
        }

        if !audio {
            // Silence is asked for explicitly: with no audiodev named, the
            // machine's sound card takes the first output the build offers.
            // The global is written in its long form on purpose — the short
            // one splits the driver name at its first dot, and this driver is
            // called `apple.mca`, so `-global apple.mca.audiodev=quiet` looks
            // for a device called `apple` and is quietly dropped.
            argv += ["-audiodev", "none,id=quiet",
                     "-global", "driver=apple.mca,property=audiodev,value=quiet"]
        }

        if headless || builtInDisplay {
            // Nothing for the emulator to serve: either there is no screen at
            // all, or the app reads the framebuffer directly once the machine
            // is up.
            argv += ["-display", "none"]
        }
        else {
            argv += ["-vnc", "127.0.0.1:\(vncPort - 5900)"]
        }

        if let root = VMConfig.rootImage {
            argv += ["-drive", "file=\(root.path),format=\(root.format),if=none,id=root"]
            argv += ["-device", "nvme-ns,drive=root,bus=nvme-bus.0,nsid=1,nstype=1,logical_block_size=4096,physical_block_size=4096"]
        }

        // The remaining NVMe namespaces the machine expects, in order.
        let namespaces: [(file: String, nsid: Int, nstype: Int)] = [
            ("firmware", 2, 2),
            ("syscfg", 3, 3),
            ("ctrl_bits", 4, 4),
            ("effaceable", 6, 6),
            ("panic_log", 7, 8),
        ]
        for ns in namespaces {
            argv += ["-drive", "file=\(data)/\(ns.file),format=raw,if=none,id=\(ns.file)"]
            argv += ["-device", "nvme-ns,drive=\(ns.file),bus=nvme-bus.0,nsid=\(ns.nsid),nstype=\(ns.nstype),logical_block_size=4096,physical_block_size=4096"]
        }

        // A scratch namespace that both sides can reach: the app writes bytes
        // into the file, the guest reads them straight off the block device, and
        // nothing travels through the console or the network on the way. It is
        // attached only when the file exists, because the guest only learns of a
        // namespace if the emulator describes it in the device tree — an
        // emulator without that patch would simply ignore this one.
        //
        // cache=none is not a tuning knob here. Without it the emulator answers
        // out of the host's page cache and the guest reads what the file used to
        // hold, which looks exactly like a corrupt transfer.
        let transfer = VMConfig.dataDirectory.appendingPathComponent("xfer")
        if FileManager.default.fileExists(atPath: transfer.path) {
            argv += ["-drive", "file=\(transfer.path),format=raw,if=none,id=xfer,cache=none"]
            argv += ["-device", "nvme-ns,drive=xfer,bus=nvme-bus.0,nsid=8,nstype=2,logical_block_size=4096,physical_block_size=4096"]
        }

        // Not while the port is served to another machine: this device would be
        // a second host for a port that has one.
        if network, usbExport == nil {
            // The device listens on the same socket the machine's USB port
            // dials into, so it must be named identically.
            argv += ["-netdev", "user,id=net0"]
            argv += ["-device", "apple-ncm-host,netdev=net0,conn-addr=\(usbSocket)"]
        }

        // nvram is its own device type, not a plain namespace.
        argv += ["-drive", "file=\(data)/nvram,if=none,format=raw,id=nvram"]
        argv += ["-device", "apple-nvram,drive=nvram,bus=nvme-bus.0,nsid=5,nstype=5,id=nvram,logical_block_size=4096,physical_block_size=4096"]

        return argv
    }
}
