import Foundation

/// Everything the emulator needs, made on the phone out of what a person can
/// download onto it.
///
/// On a desktop this is a page of shell: `qemu-img create` nine times, `unzip`,
/// two Python scripts for the tickets and one `img4` call for the SEP firmware.
/// None of it needs a computer — it needs a zip reader, DER and AES, and the app
/// has all three. What a person still brings is what only they can: the
/// firmware archive, the ticket ChefKiss publishes, the SEP ROM, and the SEP key
/// off the wiki.
enum RestorePrep {
    struct Inputs {
        /// The firmware archive, as downloaded. Copied into the app's own
        /// folder once, so the rest of a restore never again depends on
        /// external media staying attached and reachable — a flash drive's
        /// bookmark surviving a multi-minute transfer is one more way for a
        /// restore to die that a phone with room to spare does not need to
        /// risk.
        var ipsw: URL
        /// `ticket.shsh2`, the ticket both forged ones are made from.
        var shsh2: URL
        /// `AppleSEPROM-Cebu-B1`, if it is not in place yet.
        var sepROM: URL?
        /// The SEP firmware key for this exact build: 96 hex digits, IV first.
        var sepKey: String
        /// Any real device's own Cryptex1 IM4M -- only iOS 16+ ever asks for
        /// one, see `Cryptex1`. Left out entirely, an iOS 14 restore never
        /// notices.
        var cryptexTemplate: URL?
    }

    enum Failure: LocalizedError {
        case cannotRead(String)
        case noSEP
        case noRoom(needed: Int, free: Int)

        var errorDescription: String? {
            switch self {
            case .cannotRead(let what): return L("Не читается: %@", what)
            case .noSEP:                return L("В прошивке нет SEP — не тот архив?")
            case .noRoom(let needed, let free):
                return L("Мало места для копии прошивки: нужно %.1f ГБ, свободно %.1f",
                        Double(needed) / 1_073_741_824, Double(free) / 1_073_741_824)
            }
        }
    }

    /// The disks the machine wants, and how big each one is.
    ///
    /// `root` is 32 GB of holes and costs nothing until a restore fills it, and
    /// every one of them starts blank, as ChefKiss's guide makes them.
    ///
    /// Four of them — `effaceable`, `nvram`, `sep_nvram` and `sep_ssc` — were
    /// once filled from a real device's copies instead, to get past a SEP panic
    /// (`sars`) partway into a restore. That cure was worse: a restore made with
    /// them finishes, but the system it leaves cannot unlock its own data
    /// volume — no `Unlock notification` in the log, no key for the volume — and
    /// never reaches a screen. Blank, the restore goes through without the panic
    /// at all, and the system comes up.
    static let disks: [(name: String, size: Int)] = [
        ("root", 32 << 30),
        ("firmware", 8 << 20),
        ("syscfg", 128 << 10),
        ("ctrl_bits", 8 << 10),
        ("nvram", 8 << 10),
        ("effaceable", 4 << 10),
        ("panic_log", 1 << 20),
        ("sep_nvram", 64 << 10),
        ("sep_ssc", 128 << 10),
    ]

    /// Makes the whole kit. Blocks; call it off the main thread.
    static func prepare(_ inputs: Inputs,
                        note: @escaping (String) -> Void,
                        progress: @escaping (Double) -> Void = { _ in }) throws {
        let data = VMConfig.dataDirectory
        try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true)

        // 1. The disks.
        for disk in disks where !FileManager.default.fileExists(atPath: data.appendingPathComponent(disk.name).path) {
            try makeBlank(data.appendingPathComponent(disk.name), size: disk.size)
            note(L("Подготовка: создан %@", disk.name))
        }

        // 2. The archive itself, brought onto the phone if it is not there
        // already. Everything past this point — unpacking here, and the ASR
        // transfer later, minutes long — reads it many times over; a flash
        // drive staying attached and its bookmark staying valid for all of
        // that is one more way to fail that local storage does not have.
        let localIPSW = try localCopy(of: inputs.ipsw, into: data, note: note)

        // 3. What the emulator loads, out of the archive. The system image and
        // the cryptexes are gigabytes it never touches — a restore reads those
        // from the archive itself.
        let ipsw = try IPSW(url: localIPSW)
        let identity = try ipsw.buildIdentity()
        let skip = bigImages(of: ipsw)
        let wanted = ipsw.members.keys.filter { !skip.contains($0) && !$0.hasSuffix("/") }
        note(L("Подготовка: распаковываю %d файлов", wanted.count))
        let restore = data.appendingPathComponent("Restore")
        for (index, path) in wanted.sorted().enumerated() {
            let destination = restore.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try ipsw.extract(path, to: destination)
            progress(Double(index + 1) / Double(wanted.count))
        }

        // 4. The tickets. Neither is signed by Apple and neither asks Apple.
        let shshScoped = inputs.shsh2.startAccessingSecurityScopedResource()
        defer { if shshScoped { inputs.shsh2.stopAccessingSecurityScopedResource() } }
        guard let shsh = try? Data(contentsOf: inputs.shsh2) else {
            throw Failure.cannotRead(inputs.shsh2.lastPathComponent)
        }
        let apTicket = try Ticket.forgeAP(shsh: shsh, identity: identity)
        try Data(apTicket).write(to: data.appendingPathComponent("root_ticket.der"))
        note(L("Подготовка: тикет AP, %d байт", apTicket.count))

        // 5. The SEP firmware, decrypted with the key off the wiki and vouched
        // for by the SEP ticket.
        let sepTicket = try Ticket.forgeSEP(shsh: shsh, identity: identity)
        guard let sepPath = IPSW.path(of: "SEP", in: identity) else { throw Failure.noSEP }
        let firmware = try SEPFirmware.rebuild(im4p: try ipsw.read(sepPath),
                                               key: try SEPFirmware.Key(hex: inputs.sepKey),
                                               ticket: sepTicket)
        try Data(firmware).write(to: data.appendingPathComponent("sep-firmware.n104.RELEASE.new.img4"))
        note(L("Подготовка: прошивка SEP, %d байт", firmware.count))

        // 6. The SEP ROM, which is a download and not something to make.
        if let rom = inputs.sepROM {
            let romScoped = rom.startAccessingSecurityScopedResource()
            defer { if romScoped { rom.stopAccessingSecurityScopedResource() } }
            let destination = VMConfig.sepROM
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.copyItem(at: rom, to: destination)
            note(L("Подготовка: SEP ROM на месте"))
        }

        // 7. The Cryptex1 template -- iOS 16+ only, and any real device's own
        // ticket will do; RestoreClient reads its own copy back out of
        // iPhoneData, same as everything else here.
        if let template = inputs.cryptexTemplate {
            let templateScoped = template.startAccessingSecurityScopedResource()
            defer { if templateScoped { template.stopAccessingSecurityScopedResource() } }
            try? FileManager.default.removeItem(at: VMConfig.cryptexTemplate)
            try FileManager.default.copyItem(at: template, to: VMConfig.cryptexTemplate)
            note(L("Подготовка: шаблон Cryptex1 на месте"))
        }

        // The restore itself reads the archive again from here on — point it
        // at the local copy instead of the pick from Files.
        RestoreSession.remember(firmware: localIPSW)

        note(L("Готово. Дальше: настройки → «Восстановление» → «Начать рестор»"))
    }

    /// Brings the archive onto the phone if it is not there already, with
    /// progress: it is gigabytes, and a silent wait that long reads as a hang.
    ///
    /// Already-local is recognized by size against the source, not merely by
    /// the name existing — a copy cut short by an earlier failed attempt is a
    /// file at that same path too, and treating it as done handed `IPSW` a
    /// truncated archive with no way to tell a person why it would not parse.
    private static func localCopy(of source: URL, into data: URL,
                                  note: @escaping (String) -> Void) throws -> URL {
        let destination = data.appendingPathComponent(source.lastPathComponent)
        if source.path == destination.path { return source }

        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }
        let size = (try? source.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0

        if let existing = (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize),
           existing == size, size > 0 {
            note(L("Подготовка: прошивка уже скопирована"))
            return destination
        }
        try? FileManager.default.removeItem(at: destination)

        if let free = RestoreSession.freeSpace, free < size + (1 << 30) {
            throw Failure.noRoom(needed: size + (1 << 30), free: free)
        }

        guard FileManager.default.createFile(atPath: destination.path, contents: nil) else {
            throw Failure.cannotRead(destination.lastPathComponent)
        }
        let reader = try FileHandle(forReadingFrom: source)
        defer { try? reader.close() }
        let writer = try FileHandle(forWritingTo: destination)
        defer { try? writer.close() }

        note(L("Подготовка: копирую прошивку, %.1f ГБ", Double(size) / 1_073_741_824))
        var copied = 0
        var lastNote = Date.distantPast
        // Without a pool of its own, every chunk's `Data` stays autoreleased
        // for as long as the surrounding GCD block runs — which is the whole
        // multi-step preparation, not just this copy. Over a ~19 GB archive in
        // 8 MB pieces that is thousands of undrained buffers, and the app was
        // hitting the jetsam ceiling partway through at a reproducible byte
        // count (a consistent percentage run to run) well before the copy
        // itself needed anywhere near that much memory.
        var finished = false
        while !finished {
            try autoreleasepool {
                guard let chunk = try reader.read(upToCount: 8 << 20), !chunk.isEmpty else {
                    finished = true
                    return
                }
                try writer.write(contentsOf: chunk)
                copied += chunk.count
                if size > 0, Date().timeIntervalSince(lastNote) >= 2 {
                    lastNote = Date()
                    note(L("Подготовка: копирую прошивку, %d %%", Int(Double(copied) / Double(size) * 100)))
                }
            }
        }
        note(L("Подготовка: прошивка скопирована"))
        return destination
    }

    /// The images a restore streams from the archive instead of unpacking.
    private static func bigImages(of ipsw: IPSW) -> Set<String> {
        guard let identity = try? ipsw.buildIdentity(),
              let manifest = identity["Manifest"] as? [String: Any]
        else { return [] }
        var skip = Set<String>()
        for (component, entry) in manifest {
            guard component == "OS" || component.hasPrefix("Cryptex1,"),
                  let info = (entry as? [String: Any])?["Info"] as? [String: Any],
                  let path = info["Path"] as? String, path.hasSuffix(".dmg")
            else { continue }
            skip.insert(path)
        }
        return skip
    }

    /// A file of the right size and nothing in it. Sparse: `root` is 32 GB on
    /// paper and a few blocks on disk until something writes to it.
    private static func makeBlank(_ url: URL, size: Int) throws {
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw Failure.cannotRead(url.lastPathComponent)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(size))
    }
}
