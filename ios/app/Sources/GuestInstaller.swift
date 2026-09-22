import Foundation

/// Installs an `.ipa` into the guest, in one go.
///
/// The kernel in the guest image is patched for `bypass code signature checks`
/// and `all binaries in trustcache`, so an app needs no signature — it only has
/// to be unpacked into `/Applications` and shown to SpringBoard with `uicache`.
/// What was missing was never the installing; it was a way to carry the
/// megabytes in. Now there are two, and this takes the better one available:
/// the scratch NVMe namespace when the emulator offers it, the USB network
/// otherwise. See `TransferNamespace` for what the fast one needs.
///
/// Everything here blocks; run it off the main thread.
final class GuestInstaller {
    enum Failure: LocalizedError {
        case noPayload
        case noApp
        case step(String, Int64)

        var errorDescription: String? {
            switch self {
            case .noPayload:
                return L("В .ipa нет папки Payload — это не приложение.")
            case .noApp:
                return L("В Payload нет ни одного .app.")
            case .step(let what, let code):
                return L("Шаг «%@» в госте вернул %d.", what, Int(code))
            }
        }
    }

    private static let stagedTar = TransferNamespace.toolsDirectory + "/install.tar"

    private let serial: SerialConsole
    private let files: GuestFiles

    /// Set when the app went in but the guest is too old to run it. Not an
    /// error: the files are installed and SpringBoard knows about them, it
    /// simply will not launch an app built for a newer iOS. Worth saying out
    /// loud, because otherwise it looks like the installer failed.
    private(set) var warning: String?

    init(serial: SerialConsole, files: GuestFiles) {
        self.serial = serial
        self.files = files
    }

    /// Unpacks, carries and installs. Returns where the app landed in the guest.
    func install(ipa: URL, progress: @escaping (Int64, Int64) -> Void,
                 note: @escaping (String) -> Void) throws -> String {
        // Unpacking happens here, on the phone: `unzip` is missing from a bare
        // bootstrap, while `tar` is on every image.
        note(L("Распаковываю .ipa…"))
        let entries = try Archive.entries(ofZip: ipa)
        guard entries.contains(where: { $0.path.hasPrefix("Payload/") }) else { throw Failure.noPayload }

        guard let appName = entries.compactMap({ entry -> String? in
            let parts = entry.path.split(separator: "/")
            guard parts.count >= 2, parts[0] == "Payload", parts[1].hasSuffix(".app") else { return nil }
            return String(parts[1])
        }).first else { throw Failure.noApp }

        // `Payload/` is dropped so the archive unpacks straight into
        // /Applications rather than into a Payload folder inside it.
        let staged = entries.map { entry -> Archive.Entry in
            var copy = entry
            copy.path = String(entry.path.dropFirst("Payload/".count))
            return copy
        }.filter { !$0.path.isEmpty }

        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("inferno-install-\(UUID().uuidString).tar")
        defer { try? FileManager.default.removeItem(at: scratch) }
        try Archive.writeTar(staged, to: scratch)

        let wanted = minimumOS(of: entries, app: appName)
        let target = "/Applications/" + appName
        return try serial.exclusive {
            let shell = GuestShell(serial: serial)
            try shell.requireAnswer("echo 1")
            _ = shell.run("mkdir -p \(TransferNamespace.toolsDirectory)")

            try files.carry(scratch, to: Self.stagedTar, plain: Self.stagedTar, shell: shell,
                            progress: progress, note: note)
            try unpack(shell, target: target, note: note)
            checkAge(shell, wanted: wanted, note: note)
            return target
        }
    }

    /// What the app says it needs, out of its own Info.plist.
    private func minimumOS(of entries: [Archive.Entry], app: String) -> Int? {
        guard let plist = entries.first(where: { $0.path == "Payload/\(app)/Info.plist" }),
              let parsed = try? PropertyListSerialization.propertyList(
                  from: plist.data, options: [], format: nil) as? [String: Any],
              let version = parsed["MinimumOSVersion"] as? String,
              let major = Int(version.split(separator: ".").first.map(String.init) ?? "")
        else { return nil }
        return major
    }

    /// Compares it with the guest's own iOS. The version is taken from the
    /// kernel rather than a plist: `uname -r` is on every image and needs no
    /// parser, and Darwin's major number runs exactly six ahead of iOS's.
    private func checkAge(_ shell: GuestShell, wanted: Int?, note: @escaping (String) -> Void) {
        guard let wanted else { return }
        guard let release = shell.text("uname -r"),
              let darwin = Int(release.split(separator: ".").first.map(String.init) ?? ""),
              darwin > 6
        else { return }
        let guestOS = darwin - 6
        guard wanted > guestOS else { return }
        warning = L("Приложению нужна iOS %d, а в госте iOS %d — оно встало, но не запустится.",
                    wanted, guestOS)
        note(warning!)
    }

    // MARK: - Putting it in place

    private func unpack(_ shell: GuestShell, target: String, note: @escaping (String) -> Void) throws {
        // One short command each: a busy guest drops bytes inside a long line,
        // and a mangled line leaves bash waiting inside an unclosed quote.
        //
        // None of them may end in a pipe. The status that comes back is the
        // last command's, and for a pipe that is `head`'s — a failed `tar`
        // would report success.
        let steps: [(String, String, Bool)] = [
            // /Applications lives on the system volume, which is mounted
            // read-only; without this the very first file fails to unpack. It
            // lasts until the guest reboots, so it is done every time.
            (L("перемонтирую корень"), "mount -uw /", true),
            (L("убираю прежнюю копию"), "rm -rf \(GuestFiles.quote(target))", false),
            (L("распаковываю"), "tar xf \(Self.stagedTar) -C /Applications", true),
            (L("права"), "chown -R root:wheel \(GuestFiles.quote(target)) && chmod -R 755 \(GuestFiles.quote(target))", true),
            // uicache complains when the guest is older than the app's
            // MinimumOSVersion. The files are in place either way, so that is
            // not a reason to call the install failed.
            (L("показываю SpringBoard"), "/usr/bin/uicache -p \(GuestFiles.quote(target))", false),
            (L("прибираю"), "rm -f \(Self.stagedTar)", false),
        ]

        for (label, command, critical) in steps {
            note(label + "…")
            let status = shell.run(command, timeout: 600) ?? -1
            if critical, status != 0 { throw Failure.step(label, status) }
        }

        guard shell.number("test -d \(GuestFiles.quote(target)) && echo 1 || echo 0") == 1 else {
            throw Failure.step(L("проверка"), 1)
        }
    }
}
