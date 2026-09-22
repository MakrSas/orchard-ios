import Darwin
import Foundation

/// Moves files between the phone and the guest — the fast way, over the USB
/// network, with the guest's console carrying nothing but the command.
///
/// The guest reaches this app at 10.0.2.2: slirp, which runs inside this very
/// process, turns that address into 127.0.0.1 (libslirp `socket.c`). So the app
/// listens on the loopback, tells the guest's shell to connect there through
/// bash's `/dev/tcp`, and `cat` on the far end copies the bytes as they are. On
/// the test rig five megabytes went across in ten seconds. The console alone
/// manages a few kilobytes a second: it has no flow control, so every two
/// kilobytes need a round trip to make sure none were lost.
///
/// The commands are kept short on purpose. When the guest is busy the console
/// drops bytes even inside a single line, and a long command arrives mangled —
/// bash is then left inside an unclosed quote and swallows whatever comes next.
///
/// Everything here blocks; run it off the main thread.
final class GuestFiles {
    enum Failure: LocalizedError {
        case networkOff
        case networkDown
        case noShell
        case destination
        case notFound(String)
        case noConnection
        case mismatch(String)
        case io(String)

        var errorDescription: String? {
            switch self {
            case .networkOff:
                return L("Включите «Интернет через USB» в параметрах: файлы идут по той же сети.")
            case .networkDown:
                return L("Сеть в госте не поднялась. Нажмите «Поднять сеть в госте» и попробуйте снова.")
            case .noShell:
                return L("Шелл гостя не отвечает. Передача файлов работает только с бутстрапом, где на консоли сидит bash.")
            case .destination:
                return L("Не удалось подготовить папку для файлов в госте: он слишком занят. Попробуйте ещё раз.")
            case .notFound(let path):
                return L("Нет такого файла в госте: %@", path)
            case .noConnection:
                return L("Гость не подключился к приложению: команда не дошла или сеть не работает.")
            case .mismatch(let detail):
                return L("Файл не сошёлся по контрольной сумме (%@).", detail)
            case .io(let detail):
                return L("Ошибка ввода-вывода: %@", detail)
            }
        }
    }

    /// Where received files land: the app's Documents, so they show up in Files.
    static var inbox: URL { VMConfig.documents.appendingPathComponent("Guest") }
    /// The guest's name for this app, as slirp presents it.
    private static let hostAddress = "10.0.2.2"

    private let serial: SerialConsole
    private let linkUp: () -> Bool
    private let bringNetworkUp: () -> Void
    /// The guest agent, when there is one.
    ///
    /// It owns the scratch namespace — it polls the device continuously — so the
    /// `nsio` helper cannot read and write that same device at the same time
    /// without corrupting the agent's control regions. That does not mean giving
    /// up the fast channel: the agent carries files itself, through a data
    /// region kept clear of those headers, and that is the path taken here when
    /// the destination is a plain path. The network is only what is left when
    /// neither channel can be had.
    private let agent: () -> GuestAgent?

    init(serial: SerialConsole, linkUp: @escaping () -> Bool, bringNetworkUp: @escaping () -> Void,
         agent: @escaping () -> GuestAgent? = { nil }) {
        self.serial = serial
        self.linkUp = linkUp
        self.bringNetworkUp = bringNetworkUp
        self.agent = agent
    }

    // MARK: - To the guest

    /// Returns the path the file ended up at inside the guest.
    ///
    /// The network is no longer demanded up front: with the helper already in
    /// the guest the bytes ride the namespace, and that works whether or not
    /// the link is up. It is asked for only when the slow path is the one left.
    func send(_ file: URL, progress: @escaping (Int64, Int64) -> Void) throws -> String {
        try serial.exclusive { try sendExclusively(file, progress: progress) }
    }

    /// Moves a local file to an exact path in the guest by the fastest channel
    /// there is, falling back to the network. Shared by the file menu and the
    /// `.ipa` installer, so both get the same speed and the same checking.
    /// `remote` is a shell expression — a quoted path, or one built around a
    /// variable the guest holds. `plain` is the same destination as an ordinary
    /// filesystem path, when the caller knows it: the agent talks to the guest
    /// off the console and has no shell to expand anything for it.
    func carry(_ file: URL, to remote: String, plain: String? = nil, shell: GuestShell,
               progress: @escaping (Int64, Int64) -> Void,
               note: @escaping (String) -> Void) throws {
        // The folder has to exist before anything is poured into it. When it did
        // not, the guest's `cat` failed on opening the file and closed the
        // socket, and this end saw only "Broken pipe" — a true statement about
        // the socket that says nothing about the cause. One short command costs
        // nothing and removes the whole class of confusion.
        _ = shell.run("mkdir -p \"$(dirname \(remote))\"")

        // The agent's own channel: the same namespace, minus the console.
        if let agent = agent(), let plain {
            note(L("Канал: агент, NVMe."))
            try agent.send(file, to: plain, progress: progress)
            return
        }

        if let fast = fastChannel(shell, note: note) {
            note(L("Канал: NVMe, %@.", fast.device))
            try fast.send(file, to: remote, shell: shell, progress: progress)
            return
        }
        note(L("Канал: USB-сеть."))
        try requireNetwork()
        try stream(file, to: remote, shell: shell, progress: progress)
    }

    /// The fast channel, or nil when the emulator does not offer one. Looked up
    /// once per transfer: finding it costs a couple of short commands.
    private func fastChannel(_ shell: GuestShell, note: @escaping (String) -> Void) -> TransferNamespace? {
        // The agent, when it is up, is the one reading and writing this device.
        // Its helper cannot share it — whatever the agent could not carry goes
        // over the network instead.
        if agent() != nil { return nil }
        return TransferNamespace.discover(shell: shell, deliver: { local, path in
            // The helper itself can only come in the slow way — it is what makes
            // the fast way possible.
            try self.requireNetwork()
            try self.stream(local, to: Self.quote(path), shell: shell, progress: { _, _ in })
        }, note: note)
    }

    private func sendExclusively(_ file: URL, progress: @escaping (Int64, Int64) -> Void) throws -> String {
        let shell = GuestShell(serial: serial)
        try openDestination(shell)

        let name = Self.safeName(file.lastPathComponent)
        // Built around the guest's own variable rather than spelled out: the
        // folder it points at has spaces in its name, and a path with spaces is
        // one more thing to get wrong on a console that drops bytes.
        let remote = "\"$F/Inferno/\"" + Self.quote(name)
        // The agent needs the destination spelled out, so `$F` is asked for once
        // and expanded here. It holds spaces, which is fine off the console.
        let plain = shell.text("echo \"$F/Inferno\"").map { $0 + "/" + name }
        try carry(file, to: remote, plain: plain, shell: shell, progress: progress, note: { _ in })
        // Root wrote it; the phone's own user has to be able to open it.
        _ = shell.line("chown -R mobile:mobile \"$F/Inferno\" 2>/dev/null", timeout: 60)
        return shell.text("echo \"$F/Inferno\"").map { $0 + "/" + name } ?? name
    }

    /// The transfer itself. `remote` is already a shell expression — a quoted
    /// path, or one built around a variable the guest holds.
    private func stream(_ file: URL, to remote: String, shell: GuestShell,
                        progress: @escaping (Int64, Int64) -> Void) throws {
        let handle: FileHandle
        do { handle = try FileHandle(forReadingFrom: file) }
        catch { throw Failure.io(error.localizedDescription) }
        defer { try? handle.close() }
        let total = Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)

        let listener = try LoopbackListener()
        // `remote` is already a shell expression, quotes and all. Quoting it a
        // second time turned the path into a filename with quotes in it, and
        // the file was never written at all.
        shell.send("exec 3<>/dev/tcp/\(Self.hostAddress)/\(listener.port);cat<&3>\(remote);exec 3<&-\n")
        guard let conn = listener.accept(timeout: 60) else {
            shell.reset()
            throw Failure.noConnection
        }
        defer { Darwin.close(conn) }

        var sum = PosixChecksum()
        var sent: Int64 = 0
        while true {
            let chunk: Data
            do { chunk = try handle.read(upToCount: 64 * 1024) ?? Data() }
            catch { throw Failure.io(error.localizedDescription) }
            if chunk.isEmpty { break }
            try chunk.withUnsafeBytes { raw in
                sum.update(raw)
                var offset = 0
                while offset < raw.count {
                    let n = Darwin.send(conn, raw.baseAddress! + offset, raw.count - offset, 0)
                    if n <= 0 { throw Failure.io(String(cString: strerror(errno))) }
                    offset += n
                }
            }
            sent += Int64(chunk.count)
            progress(sent, max(total, sent))
        }

        // The guest closes its end once `cat` has read everything; only then
        // is the file complete and worth checking.
        shutdown(conn, SHUT_WR)
        var scratch = [UInt8](repeating: 0, count: 4096)
        while Darwin.recv(conn, &scratch, scratch.count, 0) > 0 {}

        try verify(shell, path: remote, sum)
    }

    /// Points the guest's `$F` at somewhere its own Files app will look.
    ///
    /// A file dropped in `/var/mobile/Inferno` is invisible from inside the
    /// guest: the Files app shows "On My iPhone" out of the local provider's
    /// own container, which lives under an app group whose name is a UUID —
    /// different on every device, so it has to be asked for rather than known.
    ///
    /// The path is left in a shell variable instead of being carried back and
    /// forth. It contains spaces, and the console is not the place for those.
    /// If the glob matches nothing the variable holds a path that does not
    /// exist, which the test below catches.
    private func openDestination(_ shell: GuestShell) throws {
        var heard = false
        for attempt in 0..<2 {
            if prepareDestination(shell, heard: &heard) { return }
            if attempt == 0 { shell.reset() }
        }
        // Silence and a wrong answer are different faults, and used to be
        // reported as the same one. Nothing at all means there is no shell on
        // the console; an answer we did not want means the guest is there and
        // the folder is not.
        throw heard ? Failure.destination : Failure.noShell
    }

    /// One pass of the preparation, true when the folder is there at the end.
    ///
    /// Every line waits for the one before it. The `grep` walks every app group
    /// on the phone and on a loaded guest that takes a while; whatever is sent
    /// meanwhile only sits in the terminal, where the console drops bytes from
    /// it — and a Ctrl-C meant to clear one mangled line throws away the rest.
    /// That is how `$F` used to end up unset and the folder never made, with
    /// the retry then asking about `/Inferno` and being told, honestly, no.
    ///
    /// So a failed check repeats the whole preparation rather than the check:
    /// after a reset the variables are as likely to be missing as wrong.
    private func prepareDestination(_ shell: GuestShell, heard: inout Bool) -> Bool {
        // The storage folder itself may not exist yet — the Files app creates it
        // the first time something is saved locally, and on a fresh guest that
        // has never happened. The group container is always there, though, and
        // says what it belongs to in its own metadata, so that is what is looked
        // for. Failing everything, the old place, which at least works.
        //
        // The `grep` walks every app group on the phone, which is the slow one;
        // the rest are instant, and are given room only in case the guest is
        // busy when they arrive.
        let steps: [(String, TimeInterval)] = [
            ("A=/var/mobile/Containers/Shared/AppGroup", 30),
            ("G=$(grep -l LocalStorage $A/*/.com.apple*.plist 2>/dev/null|head -1)", 180),
            ("[ -n \"$G\" ] && F=\"${G%/*}/File Provider Storage\" || F=/var/mobile", 30),
            ("mkdir -p \"$F/Inferno\"", 60),
        ]
        for (command, timeout) in steps {
            guard shell.line(command, timeout: timeout) != nil else { return false }
            heard = true
        }
        guard let answer = shell.number("test -d \"$F/Inferno\" && echo 1 || echo 0") else { return false }
        heard = true
        return answer == 1
    }

    // MARK: - From the guest

    /// Returns where the file was saved on the phone.
    func receive(_ remote: String, progress: @escaping (Int64, Int64) -> Void) throws -> URL {
        try serial.exclusive { try receiveExclusively(remote, progress: progress) }
    }

    private func receiveExclusively(_ remote: String, progress: @escaping (Int64, Int64) -> Void) throws -> URL {
        let shell = GuestShell(serial: serial)
        let exists = try shell.requireAnswer("test -f \(quote(remote)) && echo 1 || echo 0", accepting: [0, 1])
        guard exists == 1 else { throw Failure.notFound(remote) }
        let total = shell.number("wc -c < \(quote(remote))") ?? 0

        let fm = FileManager.default
        try? fm.createDirectory(at: Self.inbox, withIntermediateDirectories: true)
        let destination = Self.freeName(for: (remote as NSString).lastPathComponent, in: Self.inbox)
        let partial = destination.appendingPathExtension("part")
        fm.createFile(atPath: partial.path, contents: nil)
        let out: FileHandle
        do { out = try FileHandle(forWritingTo: partial) }
        catch { throw Failure.io(error.localizedDescription) }

        // The agent reads the file and lays it in the data region itself.
        if let agent = agent() {
            do {
                try agent.receive(remote, to: partial, progress: progress)
                try? out.close()
                do { try fm.moveItem(at: partial, to: destination) }
                catch { throw Failure.io(error.localizedDescription) }
                return destination
            } catch {
                try? out.close()
                try? fm.removeItem(at: partial)
                throw error
            }
        }

        // The namespace carries it whole, and checks it, without the console or
        // the network being involved in the bytes at all.
        if let fast = fastChannel(shell, note: { _ in }) {
            do {
                try fast.receive(quote(remote), to: partial, shell: shell, progress: progress)
                try? out.close()
                do { try fm.moveItem(at: partial, to: destination) }
                catch { throw Failure.io(error.localizedDescription) }
                return destination
            } catch {
                try? out.close()
                try? fm.removeItem(at: partial)
                throw error
            }
        }

        try requireNetwork()
        let listener = try LoopbackListener()
        shell.send("cat \(quote(remote))>/dev/tcp/\(Self.hostAddress)/\(listener.port)\n")
        guard let conn = listener.accept(timeout: 60) else {
            try? out.close()
            try? fm.removeItem(at: partial)
            shell.reset()
            throw Failure.noConnection
        }

        var sum = PosixChecksum()
        var got: Int64 = 0
        var buffer = [UInt8](repeating: 0, count: 64 * 1024)
        do {
            defer { Darwin.close(conn) }
            while true {
                let n = buffer.withUnsafeMutableBytes { Darwin.recv(conn, $0.baseAddress, $0.count, 0) }
                if n < 0 { throw Failure.io(String(cString: strerror(errno))) }
                if n == 0 { break }
                try buffer.withUnsafeBytes { raw in
                    let piece = UnsafeRawBufferPointer(rebasing: raw[0..<n])
                    sum.update(piece)
                    do { try out.write(contentsOf: Data(piece)) }
                    catch { throw Failure.io(error.localizedDescription) }
                }
                got += Int64(n)
                progress(got, max(total, got))
            }
            try out.close()
            try verify(shell, path: quote(remote), sum)
        } catch {
            try? out.close()
            try? fm.removeItem(at: partial)
            throw error
        }

        do { try fm.moveItem(at: partial, to: destination) }
        catch { throw Failure.io(error.localizedDescription) }
        return destination
    }

    // MARK: - Helpers

    private func requireNetwork() throws {
        if linkUp() { return }
        // The emulator already tried everything the USB side allows; what is
        // left is asking the guest to configure its interface itself.
        bringNetworkUp()
        let deadline = Date().addingTimeInterval(25)
        while Date() < deadline {
            if linkUp() { return }
            Thread.sleep(forTimeInterval: 1)
        }
        throw Failure.networkDown
    }

    /// `path` is already quoted for the guest's shell — it may be a plain
    /// quoted string, or one built around a variable the guest holds.
    private func verify(_ shell: GuestShell, path: String, _ sum: PosixChecksum) throws {
        // cksum walks the whole file, and the guest is not fast.
        let slack = 60 + Double(sum.length) / 200_000
        let size = shell.number("wc -c < \(path)", timeout: slack)
        let crc = shell.number("cksum < \(path) | cut -d' ' -f1", timeout: slack)
        guard size == sum.length, crc == Int64(sum.value) else {
            throw Failure.mismatch(L("в госте %@ Б, у нас %d Б", size.map(String.init) ?? "?", sum.length))
        }
    }

    private func quote(_ path: String) -> String { Self.quote(path) }

    /// Quotes a path for the guest's shell without putting a single non-ASCII
    /// byte on the command line.
    ///
    /// The console hands bytes above 0x7F to bash in a way that breaks the line:
    /// a command with a Cyrillic file name never completes, and bash is left
    /// waiting inside it. `$'…'` with octal escapes spells the same bytes in
    /// plain ASCII, so the name survives exactly while the line stays safe.
    static func quote(_ path: String) -> String {
        let bytes = Array(path.utf8)
        if bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7F }) {
            return "'" + path.replacingOccurrences(of: "'", with: "'\\''") + "'"
        }
        var out = "$'"
        for byte in bytes {
            if byte >= 0x20, byte < 0x7F, byte != UInt8(ascii: "'"), byte != UInt8(ascii: "\\") {
                out.append(Character(Unicode.Scalar(byte)))
            } else {
                out += String(format: "\\%03o", byte)
            }
        }
        return out + "'"
    }

    /// A name that cannot break a command line: no slashes, nothing a terminal
    /// would act on.
    static func safeName(_ name: String) -> String {
        let cleaned = String(name.unicodeScalars.filter { $0.value >= 0x20 && $0 != "/" && $0.value != 0x7F })
        return cleaned.isEmpty ? "file" : cleaned
    }

    /// Keeps earlier files: `a.txt`, then `a 2.txt`, like Files itself does.
    static func freeName(for name: String, in directory: URL) -> URL {
        let safe = safeName(name)
        let base = (safe as NSString).deletingPathExtension
        let ext = (safe as NSString).pathExtension
        var candidate = directory.appendingPathComponent(safe)
        var index = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            let numbered = ext.isEmpty ? "\(base) \(index)" : "\(base) \(index).\(ext)"
            candidate = directory.appendingPathComponent(numbered)
            index += 1
        }
        return candidate
    }
}

/// The guest's shell as a request/response channel: commands go in through the
/// console, answers are picked out of what it prints by a marker only the guest
/// can produce. The marker is assembled from shell variables, so the echo of the
/// command shows `$v$t` while the answer shows `VAL1a2b` — the two can never be
/// mistaken for each other.
final class GuestShell {
    private let serial: SerialConsole
    private let lock = NSLock()
    private var buffer = Data()
    private let arrived = DispatchSemaphore(value: 0)
    private var token: UUID?

    init(serial: SerialConsole) {
        self.serial = serial
        token = serial.tap { [weak self] data in self?.append(data) }
    }

    deinit {
        if let token { serial.untap(token) }
    }

    private func append(_ data: Data) {
        lock.lock()
        buffer.append(data)
        if buffer.count > 1 << 20 { buffer.removeFirst(buffer.count - (1 << 19)) }
        lock.unlock()
        arrived.signal()
    }

    func send(_ text: String) { serial.send(text) }

    /// Waits for the marker and a newline after it; returns what follows it.
    func wait(for marker: String, timeout: TimeInterval) -> String? {
        let needle = Data(marker.utf8)
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            lock.lock()
            if let found = buffer.range(of: needle),
               let newline = buffer[found.upperBound...].firstIndex(of: 0x0A) {
                let rest = String(decoding: buffer[found.upperBound..<newline], as: UTF8.self)
                buffer.removeSubrange(buffer.startIndex...newline)
                lock.unlock()
                return rest.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            lock.unlock()
            let left = deadline.timeIntervalSinceNow
            if left <= 0 { return nil }
            _ = arrived.wait(timeout: .now() + min(left, 0.5))
        }
    }

    /// A line printed by the guest, read off the marker's own line.
    func text(_ expression: String, timeout: TimeInterval = 20) -> String? {
        let tag = String(format: "%04x", UInt16.random(in: 0...UInt16.max))
        send("v=VAL; t=\(tag); echo \"$v$t $(\(expression))\"\n")
        return wait(for: "VAL" + tag, timeout: timeout)
    }

    /// A number computed in the guest, read off the marker's own line — never
    /// out of free output, which carries the prompt and kernel lines with
    /// numbers of their own.
    func number(_ expression: String, timeout: TimeInterval = 30) -> Int64? {
        let tag = String(format: "%04x", UInt16.random(in: 0...UInt16.max))
        send("v=VAL; t=\(tag); echo \"$v$t $(\(expression))\"\n")
        guard let rest = wait(for: "VAL" + tag, timeout: timeout),
              let first = rest.split(separator: " ").first
        else { return nil }
        return Int64(first)
    }

    /// Runs a command and hands back its exit status, with the output thrown
    /// away. The command must not end in a pipe: the status would then be the
    /// last stage's, and `head` succeeds however badly the command before it
    /// failed.
    func run(_ command: String, timeout: TimeInterval = 120) -> Int64? {
        number("{ \(command) ;} >/dev/null 2>&1; echo $?", timeout: timeout)
    }

    /// Runs a line in the shell itself and hands back its exit status.
    ///
    /// `run` cannot do this: its command substitution is a child shell, so a
    /// variable set there dies with it. Here the marker rides on the same line
    /// as the work, which is what makes the waiting safe — nothing of ours is
    /// left sitting in the terminal while the guest is busy, and the console
    /// loses bytes only from what sits there.
    @discardableResult
    func line(_ command: String, timeout: TimeInterval = 120) -> Int64? {
        let tag = String(format: "%04x", UInt16.random(in: 0...UInt16.max))
        // The status is caught before anything else runs, or it would be the
        // status of the assignment right after it — which is always success.
        send("\(command); s=$?; v=VAL; t=\(tag); echo \"$v$t $s\"\n")
        guard let rest = wait(for: "VAL" + tag, timeout: timeout),
              let first = rest.split(separator: " ").first
        else { return nil }
        return Int64(first)
    }

    /// Runs a check that must answer with one of `accepting`; one retry after
    /// clearing a line that lost bytes on the way in.
    @discardableResult
    func requireAnswer(_ expression: String, accepting: Set<Int64> = [1]) throws -> Int64 {
        for attempt in 0..<2 {
            if let answer = number(expression), accepting.contains(answer) { return answer }
            if attempt == 0 { reset() }
        }
        throw GuestFiles.Failure.noShell
    }

    /// Clears a line mangled by lost bytes, or an unclosed quote left behind by
    /// one. Ctrl-C, never Ctrl-D: the latter would end the shell, and the bash
    /// daemon on older images has no KeepAlive to bring it back.
    func reset() {
        send("\u{03}")
        Thread.sleep(forTimeInterval: 0.5)
        send("\n")
        Thread.sleep(forTimeInterval: 0.5)
    }
}

/// A one-connection listener on the loopback, on a port the kernel picks.
final class LoopbackListener {
    let fd: Int32
    let port: UInt16

    init() throws {
        let s = socket(AF_INET, SOCK_STREAM, 0)
        guard s >= 0 else { throw GuestFiles.Failure.io("socket(): \(String(cString: strerror(errno)))") }
        var one: Int32 = 1
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, socklen_t(MemoryLayout<Int32>.size))

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = 0
        addr.sin_addr.s_addr = INADDR_LOOPBACK.bigEndian
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(s, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, listen(s, 1) == 0 else {
            let reason = String(cString: strerror(errno))
            Darwin.close(s)
            throw GuestFiles.Failure.io("bind/listen: \(reason)")
        }

        var actual = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        _ = withUnsafeMutablePointer(to: &actual) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(s, $0, &length) }
        }
        fd = s
        port = UInt16(bigEndian: actual.sin_port)
    }

    deinit { Darwin.close(fd) }

    func accept(timeout: TimeInterval) -> Int32? {
        var probe = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
        guard poll(&probe, 1, Int32(timeout * 1000)) > 0 else { return nil }
        let conn = Darwin.accept(fd, nil, nil)
        guard conn >= 0 else { return nil }
        var one: Int32 = 1
        setsockopt(conn, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
        var tv = timeval(tv_sec: 120, tv_usec: 0)
        setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(conn, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        return conn
    }
}

/// The number `cksum` prints in the guest: CRC-32/CKSUM with the length folded
/// in at the end. The guest has neither md5 nor shasum, so this is the check
/// both sides can compute.
struct PosixChecksum {
    private static let table: [UInt32] = (0..<256).map { index in
        var crc = UInt32(index) << 24
        for _ in 0..<8 { crc = (crc & 0x8000_0000) != 0 ? (crc << 1) ^ 0x04C1_1DB7 : crc << 1 }
        return crc
    }

    private var crc: UInt32 = 0
    private(set) var length: Int64 = 0

    mutating func update(_ bytes: UnsafeRawBufferPointer) {
        var value = crc
        for byte in bytes {
            value = (value << 8) ^ Self.table[Int(((value >> 24) ^ UInt32(byte)) & 0xFF)]
        }
        crc = value
        length += Int64(bytes.count)
    }

    var value: UInt32 {
        var value = crc
        var remaining = length
        while remaining > 0 {
            value = (value << 8) ^ Self.table[Int(((value >> 24) ^ UInt32(remaining & 0xFF)) & 0xFF)]
            remaining >>= 8
        }
        return ~value
    }
}
