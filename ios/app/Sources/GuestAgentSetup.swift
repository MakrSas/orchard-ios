import Foundation

/// Puts the guest agent into the guest and starts it, once per guest lifetime.
///
/// The agent rides in over the console, packed, in base64 lines — the same way
/// the status bar helper does, and for the same reasons: it needs no network
/// (which a fresh guest keeps putting down) and no other helper (which would
/// need carrying in first). It is bigger than the status bar helper, so it is
/// sent in chunks each checked on its own, and a chunk that lost bytes is sent
/// again rather than the whole file.
///
/// After it is in place, `agent install` edits the launchd cache so the agent
/// starts on every future boot, and spawns a copy for this boot without waiting
/// for one. From then on the app reaches it over the namespace.
///
/// None of this is required. When there is no scratch namespace, no `ldid` at
/// build time (so no bundled agent), or the install does not take, this returns
/// nil and the app goes on through the console exactly as before.
///
/// Everything here blocks; run it off the main thread.
enum GuestAgentSetup {
    enum Outcome {
        case installed(GuestAgent)
        /// `retry` is true when the reason is only that the guest is not ready
        /// yet (no shell on the console, a delivery that lost bytes), and false
        /// when there is no point trying again this session (no namespace, no
        /// bundled agent).
        case unavailable(String, retry: Bool)
    }

    /// A packed agent to use instead of the bundled one. Set only by the rig
    /// harness, which has no app bundle to read from; nil in the app.
    static var packedOverride: Data?

    /// The bundled agent, packed, or nil when the build had no `ldid` to sign it.
    private static var bundled: Data? {
        if let packedOverride { return packedOverride }
        let url = (Bundle.main.resourceURL ?? Bundle.main.bundleURL).appendingPathComponent("guest-tools/agent.gz")
        return try? Data(contentsOf: url)
    }

    /// Returns a live client, installing the agent first if need be.
    static func bringUp(serial: SerialConsole, note: @escaping (String) -> Void) -> Outcome {
        // Already there and answering — the common case after the first boot.
        if let agent = GuestAgent.discover() {
            note(L("Агент уже в госте."))
            return .installed(agent)
        }

        // No scratch namespace means no channel to the agent at all; the app
        // stays on the console. Told apart from "packed helper missing" so the
        // log says which it was. Neither is worth retrying this session.
        guard fileExists(VMConfig.transferImage) else {
            return .unavailable(L("нет namespace xfer — агент не используется"), retry: false)
        }
        guard let packed = bundled else {
            return .unavailable(L("агент не собран (нет ldid) — работаю через консоль"), retry: false)
        }

        // The first install rides in over the console, so there has to be a
        // shell answering on it. Early in a boot there is not yet; rather than
        // sit through a delivery's worth of 60-second timeouts, this asks once
        // and defers.
        //
        // Asked, not inferred. `serial.interactive` only says the socket is
        // connected, which is not the same as bash being on the other end — and
        // it is published on the main queue, so anything without a run loop
        // never sees it turn true at all.
        let answers = serial.exclusive { GuestShell(serial: serial).number("echo 1", timeout: 20) == 1 }
        guard answers else {
            return .unavailable(L("консоль гостя ещё не готова — попробую позже"), retry: true)
        }

        var sum = PosixChecksum()
        packed.withUnsafeBytes { sum.update($0) }
        let remote = "\(TransferNamespace.toolsDirectory)/agent-\(String(format: "%08x", sum.value))"

        let delivered: Bool = serial.exclusive {
            let shell = GuestShell(serial: serial)
            // The binary carries a checksum in its name, so a build already
            // delivered is found rather than sent again — and never overwritten,
            // which the kernel would answer with `Killed: 9`.
            if shell.number("test -x \(remote) && echo 1 || echo 0") == 1 { return true }
            return deliver(packed, to: remote, crc: sum.value, shell: shell, note: note)
        }
        guard delivered else { return .unavailable(L("агента не удалось занести в гостя"), retry: true) }

        note(L("Ставлю агента (launchd)…"))
        let installed: Bool = serial.exclusive {
            let shell = GuestShell(serial: serial)
            // The install writes to the system volume, which is read-only after
            // a boot; the same remount the package repair uses.
            shell.line("mount -uw /", timeout: 120)
            // Generous: the install reads, edits and rewrites the ~1.5 MB
            // launchd cache, and parsing that on the emulated cores is not fast.
            let said = shell.text("\(remote) install 2>&1 | tail -1", timeout: 300) ?? ""
            note(L("Агент: %@", said))
            return said.contains("AGENT-INSTALL OK")
        }
        guard installed else { return .unavailable(L("agent install не удался"), retry: true) }

        // The install spawned a serving copy; give it a moment to open the
        // device, then reach it over the namespace.
        for _ in 0..<10 {
            if let agent = GuestAgent.discover() {
                note(L("Агент поднялся."))
                return .installed(agent)
            }
            Thread.sleep(forTimeInterval: 1)
        }
        return .unavailable(L("агент установлен, но не отвечает по namespace"), retry: true)
    }

    // MARK: - Console delivery

    /// Pours the packed agent into the guest as base64, a chunk at a time, and
    /// checks each chunk before moving on. Returns true when the whole file is
    /// in place and its checksum agrees.
    private static func deliver(_ packed: Data, to remote: String, crc: UInt32,
                                shell: GuestShell, note: @escaping (String) -> Void) -> Bool {
        note(L("Заношу агента в гостя (один раз)…"))
        let directory = TransferNamespace.toolsDirectory
        let staging = "\(directory)/agent.b64"
        let text = Array(packed.base64EncodedString().utf8)

        // Lines of 76: wider ones stop arriving whole on a busy guest. Sent in
        // groups so a lost byte costs one group, not the file; each group's
        // running byte count is checked before the next.
        let lineWidth = 76
        let linesPerGroup = 48

        for attempt in 0..<3 {
            guard shell.line("mkdir -p \(directory) && : > \(staging)", timeout: 60) == 0 else { continue }
            var ok = true
            var expected = 0
            var start = 0
            while start < text.count {
                var group = ""
                var lines = 0
                while start < text.count, lines < linesPerGroup {
                    let end = min(start + lineWidth, text.count)
                    group += String(decoding: text[start..<end], as: UTF8.self) + "\n"
                    expected += end - start
                    start = end
                    lines += 1
                }
                // `printf %s` of a here-free single argument: no quoting games,
                // and the newline count is our own. The group ends with a check
                // that the file is exactly as long as it should be so far.
                guard writeGroup(group, to: staging, shell: shell) else { ok = false; break }
                if shell.number("wc -c < \(staging)") != Int64(expected) { ok = false; break }
            }
            guard ok else {
                shell.reset()
                if attempt < 2 { Thread.sleep(forTimeInterval: 1) }
                continue
            }

            // GNU base64 wants -d, BSD -D; the image has GNU but a bare one may
            // not. The checksum confirms every byte before anything is run.
            let decode = "{ base64 -d \(staging) 2>/dev/null || base64 -D -i \(staging); }"
            let line = shell.text("\(decode) | cksum", timeout: 120)
            let got = line?.split(separator: " ").first.flatMap { UInt32($0) }
            guard got == crc else {
                note(L("Агент: контрольная сумма не сошлась, повтор %d", attempt + 1))
                shell.reset()
                if attempt < 2 { Thread.sleep(forTimeInterval: 1) }
                continue
            }

            guard shell.line("\(decode) | gunzip -c > \(remote) && chmod +x \(remote) && rm -f \(staging)",
                             timeout: 120) == 0
            else { continue }
            return true
        }
        return false
    }

    /// Writes one group of base64 lines, each as its own short command so the
    /// console never holds much of ours at once.
    private static func writeGroup(_ group: String, to staging: String, shell: GuestShell) -> Bool {
        for line in group.split(separator: "\n") {
            guard shell.line("printf %s \(line) >> \(staging)", timeout: 60) == 0 else { return false }
        }
        return true
    }

    private static func fileExists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }
}
