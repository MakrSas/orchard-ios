import Combine
import Darwin
import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// The guest's shell, separated from the kernel's chatter.
///
/// One console shared by the kernel and the shell means every answer comes back
/// interleaved with driver messages, and no amount of pattern matching makes
/// that separation honest — a message in an unfamiliar shape gets through, and
/// one that lands in the middle of a line takes the line with it.
///
/// There are two ways out, and the channel takes whichever is available.
///
/// **Over the network.** Slirp runs inside this very process and turns 10.0.2.2
/// into its own loopback, and bash opens TCP sockets by itself through
/// `/dev/tcp`. The console is used once, to ask the guest to call back, and from
/// then on the shell has a socket to itself. This is the better channel — real
/// interactivity, no per-line cost — and the same path already carries files.
///
/// **Over the console.** When the guest has no working link, the console is all
/// there is, so the shell marks where its answer begins and ends. Between those
/// two marks is the command's output; outside them nothing is shown at all, so
/// the kernel's endless chatter never reaches the pane.
///
/// Tagging every line was tried first and was too slow to use: it put a bash
/// `read` loop in front of the output, and `read` takes one byte per system
/// call, which an emulated processor feels. The marks cost two `echo`s per
/// command and let the output travel straight from the command to the console.
/// What the kernel manages to squeeze into the window between them is still
/// sifted by the pattern filter — that part remains a guess.
///
/// Like the console, this is a main-thread object: reading happens on threads of
/// its own, but everything it changes is changed after hopping back.
final class ShellChannel: ObservableObject {
    enum State: Equatable {
        case idle
        case connecting
        case up(Transport)
        case failed(String)
    }

    enum Transport: Equatable {
        case network
        case console
    }

    @Published private(set) var state: State = .idle

    /// The terminal this channel draws into. Its changes are republished here so
    /// that a view watching the channel sees them without watching both.
    let screen = GuestScreen()

    private let serial: SerialConsole
    private let linkUp: () -> Bool

    private var sock: Sock?
    private var forward: AnyCancellable?
    /// Tells a reader whose channel has been replaced to stop talking.
    private var generation = 0

    // The console transport's state.
    private var ear: UUID?
    private var marker = ""
    private var inside = false
    private var pending = Data()
    private let filter = KernelFilter()

    /// The guest's name for this app, as slirp presents it.
    private static let host = "10.0.2.2"

    init(serial: SerialConsole, linkUp: @escaping () -> Bool) {
        self.serial = serial
        self.linkUp = linkUp
        forward = screen.objectWillChange.sink { [weak self] in self?.objectWillChange.send() }
    }

    var isUp: Bool { if case .up = state { return true }; return false }

    var transport: Transport? { if case .up(let t) = state { return t }; return nil }

    func use(fontSize: CGFloat) { screen.use(fontSize: fontSize) }

    /// Opens the best channel the guest can manage right now.
    ///
    /// The network is not waited for and the guest is not asked to bring it up:
    /// something else in the app already watches the link and does the asking,
    /// and a second voice only produced a screenful of `ipconfig` lines.
    /// Always over the console.
    ///
    /// The network path is still here and still works, but it cannot be relied
    /// on: it needs the guest to have an address and to call back, and when
    /// either does not happen there is no shell at all — which is how this
    /// looked from the outside, as a channel that simply never came up. The
    /// console is there from the moment the bootstrap's bash is, and that is
    /// worth more than not sharing it with the kernel log.
    func connect() {
        guard state != .connecting, !isUp else { return }

        state = .connecting
        generation += 1
        let mine = generation
        release()
        screen.reset()
        waitForConsole(mine, attempt: 0)
    }

    /// Waits for bash to appear on the console instead of giving up on it.
    ///
    /// On a phone the guest can be minutes from power-on to a shell, and the
    /// old behaviour — one look, then an error — meant the pane stayed empty
    /// for the rest of the session unless somebody pressed a button.
    private func waitForConsole(_ mine: Int, attempt: Int) {
        guard mine == generation else { return }
        if serial.interactive {
            openConsole(mine, note: nil)
            return
        }
        guard attempt < 150 else {
            state = .failed(L("Шелл гостя не отвечает: на консоли должен сидеть bash из бутстрапа."))
            return
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in
            self?.waitForConsole(mine, attempt: attempt + 1)
        }
    }

    /// Over the network instead: no sharing with the kernel log, at the price
    /// of needing the guest to have an address and to call back.
    func connectOverNetwork() {
        guard serial.interactive else {
            state = .failed(L("Шелл гостя не отвечает: на консоли должен сидеть bash из бутстрапа."))
            return
        }
        state = .connecting
        generation += 1
        release()
        screen.reset()
        openNetwork(generation)
    }


    // MARK: - Over the network

    private func openNetwork(_ mine: Int) {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            let listener: LoopbackListener
            do {
                listener = try LoopbackListener()
            } catch {
                self.report(mine, error.localizedDescription)
                return
            }

            // What the console says while we wait is worth listening to: bash
            // reports a refusal at once, and hearing it beats sitting out the
            // whole timeout.
            let complaint = Complaint()
            let listening = self.serial.tap { data in complaint.consider(data) }

            // `exec` on the inner bash so the subshell does not linger; the whole
            // thing in the background so the console gets its prompt back. Kept
            // short on purpose: the console has no flow control, and a busy guest
            // drops bytes inside a single line.
            self.serial.send("(exec 3<>/dev/tcp/\(Self.host)/\(listener.port);exec bash -i<&3>&3 2>&3)&\n")

            var accepted: Int32?
            let deadline = Date().addingTimeInterval(30)
            while Date() < deadline {
                if let fd = listener.accept(timeout: 0.5) { accepted = fd; break }
                if complaint.heard != nil { break }
                if self.generation != mine { self.serial.untap(listening); return }
            }
            self.serial.untap(listening)

            guard let descriptor = accepted else {
                // The console is still there, and it can carry a shell too.
                let why = complaint.heard ?? L("Гость не позвонил обратно.")
                DispatchQueue.main.async {
                    guard self.generation == mine else { return }
                    self.openConsole(mine, note: why + " " + L("Перехожу на консоль."))
                }
                return
            }

            let sock = Sock(adopting: descriptor)
            // The shell may say nothing for minutes on end; a receive timeout
            // would take the first quiet spell for the end of the stream.
            sock.waitIndefinitely()

            DispatchQueue.main.async {
                guard self.generation == mine else { sock.close(); return }
                self.sock = sock
                self.state = .up(.network)
            }

            while let data = sock.readSome() {
                let text = String(decoding: data, as: UTF8.self)
                DispatchQueue.main.async {
                    guard self.generation == mine else { return }
                    self.screen.append(text)
                }
            }

            DispatchQueue.main.async {
                guard self.generation == mine else { return }
                self.sock = nil
                self.state = .failed(L("Гость закрыл канал."))
            }
        }
    }

    // MARK: - Over the console

    /// Teaches the guest two marks, then checks that it learned them.
    ///
    /// The marks are assembled out of shell variables on purpose. The console
    /// shell echoes back everything typed at it, so a command carrying the mark
    /// literally would announce the mark twice — once in the echo, once for
    /// real — and the echo would open a window that was never meant to open.
    /// Written as `$s` and `$e`, the echo shows the names and only the shell's
    /// own output ever shows the values.
    private func openConsole(_ mine: Int, note: String?) {
        let tag = String(format: "%06x", UInt32.random(in: 0...0xFF_FFFF))
        marker = "Q\(tag)"
        inside = false
        pending = Data()
        filter.reset()

        ear = serial.tap { [weak self] data in
            DispatchQueue.main.async {
                guard let self, self.generation == mine else { return }
                self.absorb(data)
            }
        }

        // One conversation: the two lines only mean anything together.
        let mark = marker
        DispatchQueue.global(qos: .userInitiated).async {
            // One line, and it carries its own marker. Two lines could be
            // separated by somebody else writing to the console between them,
            // and a marker kept in a variable is lost the moment the guest's
            // bash restarts — after which every command printed nothing and the
            // pane looked dead.
            self.serial.exclusive {
                self.serial.send("m=\(mark); echo \"S$m\"; echo ok; echo \"E$m\"\n")
            }
            // The clock starts when the probe is actually sent, not when it is
            // queued. Somebody else can hold the console for a minute — the
            // repair at startup does — and a timeout measured from here used to
            // expire before this shell had said a word.
            DispatchQueue.main.asyncAfter(deadline: .now() + 20) { [weak self] in
                guard let self, self.generation == mine, self.state == .connecting else { return }
                self.release()
                self.state = .failed(L("Шелл не отозвался на проверку. Похоже, на консоли не bash."))
            }
        }

        if let note { LogCapture.shared.note(L("Шелл: %@", note)) }
    }

    /// Shows what lies between the marks, and nothing else.
    private func absorb(_ data: Data) {
        pending.append(data)
        if pending.count > 1 << 20 { pending.removeFirst(pending.count - (1 << 19)) }

        while let end = pending.firstIndex(of: 0x0A) {
            let line = pending[pending.startIndex..<end]
            pending.removeSubrange(pending.startIndex...end)

            let text = String(decoding: line, as: UTF8.self)
            if text.contains("S" + marker) { inside = true; continue }
            if let mark = text.range(of: "E" + marker) {
                // The status rides on the end marker's own line, which is the
                // only place it can be read without a prompt to look at.
                let tail = text[mark.upperBound...].trimmingCharacters(in: .whitespaces)
                inside = false
                finished(code: Int(tail))
                continue
            }
            guard inside else { continue }
            // The app's own machinery shares this console — the package repair,
            // the agent install — and the tty echoes whatever it writes the
            // moment it is written, even while this pane's command is still
            // running. Those echoes land between our marks and are not what the
            // user asked for. Everything that machinery sends carries the same
            // marker assembly, and its answers carry the marker itself, so both
            // are recognisable and neither belongs here.
            if text.contains("v=VAL; t=") { continue }
            if text.range(of: "VAL[0-9a-f]{4}", options: .regularExpression) != nil { continue }
            // The kernel can still write into the window; that much is filtered
            // the old way, by the shape of what it writes.
            screen.append(filter.process(text + "\r\n"))
        }
    }

    private func finished(code: Int?) {
        if state == .connecting {
            state = .up(.console)
            screen.append("\u{1B}[32m" + L("Канал по консоли открыт: показывается только вывод команд.")
                          + "\u{1B}[0m\r\n")
            return
        }
        // Without a prompt of its own, the pane gave no sign that a command had
        // ended — and on a guest this slow, "still running" and "finished with
        // nothing to say" look exactly alike.
        guard let code else { return }
        screen.append("\u{1B}[\(code == 0 ? "90" : "31")m" + L("— готово, код %d", code) + "\u{1B}[0m\r\n")
    }

    // MARK: - Talking

    /// Sends a command, and shows it.
    ///
    /// Over the network nothing comes back on its own: bash turns off line
    /// editing when its input is not a terminal, and with it the echo. Over the
    /// console the shell's echo is not marked, so it is dropped along with the
    /// kernel's lines. Either way the app has to print what was typed.
    func send(_ command: String) {
        switch transport {
        case .network:
            guard let sock else { return }
            screen.append(command + "\r\n")
            DispatchQueue.global(qos: .userInitiated).async {
                _ = sock.write(Array((command + "\n").utf8))
            }
        case .console:
            screen.append("\u{1B}[36m# \u{1B}[0m" + command + "\r\n")
            // The marker is set on this line too, for the same reason: nothing
            // is remembered between commands.
            serial.send("m=\(marker); echo \"S$m\"; { \(command) ;} 2>&1; r=$?; echo \"E$m $r\"\n")
        case nil:
            break
        }
    }

    /// Ctrl-C — over the console it goes to whatever the console shell is doing,
    /// which is the same command.
    func sendControl(_ byte: UInt8) {
        switch transport {
        case .network:
            guard let sock else { return }
            DispatchQueue.global(qos: .userInitiated).async { _ = sock.write([byte]) }
        case .console:
            serial.send(String(UnicodeScalar(byte)))
        case nil:
            break
        }
    }

    func disconnect() {
        generation += 1
        release()
        state = .idle
    }

    private func release() {
        sock?.close()
        sock = nil
        if let ear { serial.untap(ear) }
        ear = nil
        pending = Data()
    }

    private func report(_ mine: Int, _ reason: String) {
        DispatchQueue.main.async {
            guard self.generation == mine else { return }
            self.state = .failed(reason)
        }
    }
}

/// Listens to the console for the reasons a call back cannot be made.
private final class Complaint {
    private let lock = NSLock()
    private var found: String?

    /// Whatever bash said, translated, or nil while it has said nothing.
    var heard: String? {
        lock.lock()
        defer { lock.unlock() }
        return found
    }

    func consider(_ data: Data) {
        let text = String(decoding: data, as: UTF8.self)
        let reason: String
        if text.contains("Network is unreachable") || text.contains("No route to host") {
            reason = L("Сети в госте нет: bash ответил «Network is unreachable».")
        } else if text.contains("Connection refused") {
            reason = L("Гость дозвонился, но соединение отвергнуто.")
        } else if text.contains("/dev/tcp") && text.contains("No such file") {
            reason = L("В этом bash нет поддержки /dev/tcp.")
        } else {
            return
        }
        lock.lock()
        if found == nil { found = reason }
        lock.unlock()
    }
}
