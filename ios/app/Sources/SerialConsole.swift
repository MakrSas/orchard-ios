import Foundation

/// Shows the guest's serial console — the boot log, and the first place to look
/// when the screen stays black.
///
/// The emulator writes it to a file rather than only offering a socket: a socket
/// throws away everything printed before a client attaches, and the guest starts
/// talking long before the interface can connect. Reading the file means the log
/// always starts at the first byte, and survives reconnects.
final class SerialConsole: ObservableObject {
    @Published private(set) var text: String = ""
    @Published private(set) var connected = false

    /// The bytes that arrived last, and a number that counts them.
    ///
    /// The terminal is fed as the console speaks rather than rebuilt from the
    /// whole log every half second: escape codes only mean anything in order,
    /// and replaying a quarter of a megabyte four times a second to learn that
    /// would be silly. The counter lets the view tell a chunk it has already
    /// seen from a new one.
    @Published private(set) var chunk: String = ""
    @Published private(set) var sequence: Int = 0
    /// Whether the console has already been reported as silent, so that it is
    /// said once rather than twice a second forever.
    private var quiet = false

    /// Bytes a second the guest is pouring into the console, for the counter
    /// under the screen.
    @Published private(set) var consoleRate: Double = 0
    private let rateLock = NSLock()
    private var rateBytes = 0
    private var rateSince = Date()

    private var handle: FileHandle?
    private var timer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "inferno.serial")
    /// A long boot produces a lot; keep the tail.
    private let limit = 256 * 1024
    /// How much of the emulator's console log is allowed to sit on disk.
    private let logLimit: Int64 = 64 * 1024 * 1024
    private var lastTrim = Date.distantPast

    private var url: URL { VMConfig.guestConsoleLog }

    /// The console is a socket as well as a file: the file keeps the history
    /// from the very first byte, the socket carries what we type back. With the
    /// jailbreak bootstrap installed a bash daemon sits on /dev/console, so this
    /// is a real shell rather than a log.
    private var input: Sock?
    private var attached = false
    private var stopping = false
    @Published private(set) var interactive = false
    /// Commands arrive from the terminal and from the file transfer at once;
    /// two writes interleaved byte by byte would be a command neither meant.
    private let sendLock = NSLock()

    /// Held for a whole conversation, not for a single write.
    ///
    /// Locking each write was not enough. A file transfer is a sequence — set a
    /// variable, make a directory, ask whether it is there — and the network
    /// watchdog put `ipconfig set en0 DHCP` between two of those lines. Nothing
    /// was garbled, and the sequence still fell apart, because the guest ran
    /// somebody else's command in the middle of ours.
    private let conversation = NSLock()

    /// Runs a sequence of commands with the console to itself. Blocks; never
    /// call it from the main thread.
    @discardableResult
    func exclusive<T>(_ body: () throws -> T) rethrows -> T {
        conversation.lock()
        defer { conversation.unlock() }
        return try body()
    }

    /// The same, for a caller with nothing to say if the console is busy —
    /// the watchdog would rather skip a poke than break a transfer.
    @discardableResult
    func ifFree(_ body: () -> Void) -> Bool {
        guard conversation.try() else { return false }
        defer { conversation.unlock() }
        body()
        return true
    }

    /// How many times the guest has fallen over since the app started.
    ///
    /// A kernel panic ends every conversation on this console at once, and
    /// whoever was waiting for a package to finish would otherwise sit out the
    /// whole timeout — half an hour — before finding out that there is nobody
    /// left to answer. The guest says so plainly on its way down; this counts
    /// the times it has, and the waiters compare the count against the one they
    /// started with.
    private let deathLock = NSLock()
    private var deaths = 0
    private var deathWhy = ""
    private var lastDeath = Date.distantPast
    /// The tail of the previous read, so a line split across two of them is
    /// still recognised. Nothing looked for here is longer than one line.
    private var deathTail = ""

    var guestDeaths: Int {
        deathLock.lock()
        defer { deathLock.unlock() }
        return deaths
    }

    var guestDeathReason: String {
        deathLock.lock()
        defer { deathLock.unlock() }
        return deathWhy
    }

    /// Called on the main queue when the guest falls over, for whoever has to
    /// set it up again once it comes back.
    var onGuestDeath: (() -> Void)?

    /// What a dying guest writes. `initproc exited` is a launchd that could not
    /// go on — which is what installing a hooking runtime does to this image.
    private static let deathMarks = ["panic(cpu ", "wdog panic", "initproc exited"]

    private func watchForDeath(_ piece: String) {
        let text = deathTail + piece
        deathTail = String(text.suffix(120))
        // By isNewline, not by "\n": the console ends its lines with "\r\n",
        // which Swift holds as one character that is not "\n". Split by "\n",
        // a whole read was one line, and the reason given for a panic was
        // whatever the read began with — never the panic itself.
        guard let line = text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline).first(where: { line in
            SerialConsole.deathMarks.contains { line.contains($0) }
        }) else { return }

        // A panic is a page of text, and the guest prints the whole of it again
        // on the next boot. One report per fall is what a waiter needs.
        guard Date().timeIntervalSince(lastDeath) > 60 else { return }
        lastDeath = Date()

        deathLock.lock()
        deaths += 1
        deathWhy = String(line.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160))
        deathLock.unlock()
        DispatchQueue.main.async { self.onGuestDeath?() }
    }

    /// Everyone who wants the console's output as it arrives. The file transfer
    /// picks its answers out of it.
    private let tapsLock = NSLock()
    private var taps: [UUID: (Data) -> Void] = [:]

    func tap(_ handler: @escaping (Data) -> Void) -> UUID {
        let id = UUID()
        tapsLock.lock()
        taps[id] = handler
        tapsLock.unlock()
        return id
    }

    func untap(_ id: UUID) {
        tapsLock.lock()
        taps[id] = nil
        tapsLock.unlock()
    }

    private func deliver(_ data: Data) {
        count(data.count)
        tapsLock.lock()
        let handlers = Array(taps.values)
        tapsLock.unlock()
        handlers.forEach { $0(data) }
    }

    /// Keeps the console's flow rate, published once a second.
    ///
    /// It belongs next to the frame counter. A guest that has been up for a
    /// while can pour tens of megabytes a second of kernel log down this pipe —
    /// every line of it formatted by the emulated cores, which are the same
    /// cores that have to draw the screen. When that number is large, nothing
    /// else about the frame rate is worth reading.
    private func count(_ bytes: Int) {
        rateLock.lock()
        rateBytes += bytes
        let elapsed = Date().timeIntervalSince(rateSince)
        guard elapsed >= 1 else { rateLock.unlock(); return }
        let rate = Double(rateBytes) / elapsed
        rateBytes = 0
        rateSince = Date()
        rateLock.unlock()
        DispatchQueue.main.async { self.consoleRate = rate }
    }

    /// Keeps a reader on the console socket for as long as the machine runs.
    ///
    /// Reconnecting matters more than it looks. Somebody has to take what the
    /// emulator writes; when this loop ended — a read error, a socket timeout —
    /// nothing did, the emulator's buffer filled, and everything behind it
    /// stopped with it. So the loop is a loop: if the read ends and the machine
    /// has not been asked to stop, it connects again.
    func attachInput(port: UInt16) {
        guard !attached else { return }
        attached = true
        Thread.detachNewThread { [weak self] in
            while self?.stopping == false { self?.drain(port: port) }
        }
    }

    private func drain(port: UInt16) {
        autoreleasepool {
            let sock = Sock()
            if sock.connect(port: port) != nil {
                Thread.sleep(forTimeInterval: 2)
                return
            }
            // The console can stay silent for minutes. With the socket's usual
            // twenty-second receive timeout this took the first quiet spell for
            // the end of the stream and stopped for good — after which nobody
            // saw the guest's answers, and the file transfer reported a dead
            // shell while the shell was answering on screen.
            sock.waitIndefinitely()
            input = sock
            DispatchQueue.main.async { self.interactive = true }
            // What is drained goes to anyone waiting for an answer.
            while input != nil, let data = sock.readSome() { deliver(data) }

            sock.close()
            if input === sock { input = nil }
            DispatchQueue.main.async { self.interactive = false }
            if !stopping { Thread.sleep(forTimeInterval: 1) }
        }
    }

    func send(_ text: String) {
        sendLock.lock()
        defer { sendLock.unlock() }
        guard let input else { return }
        _ = input.write(Array(text.utf8))
    }

    /// Starts following the log, waiting for the file to appear if necessary.
    func follow() {
        guard timer == nil else { return }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        // Often enough that output does not arrive in visible steps. A read that
        // finds nothing costs one syscall, and the work behind a read that finds
        // something is the same however it is divided up.
        timer.schedule(deadline: .now(), repeating: .milliseconds(120))
        timer.setEventHandler { [weak self] in self?.poll() }
        self.timer = timer
        timer.resume()
    }

    func stop() {
        stopping = true
        input?.close()
        input = nil
        DispatchQueue.main.async { self.interactive = false }
        timer?.cancel()
        timer = nil
        try? handle?.close()
        handle = nil
        DispatchQueue.main.async { self.connected = false }
    }

    func clear() {
        text = ""
    }

    /// Keeps the emulator's own console log from eating the phone.
    ///
    /// The log is `-chardev …,logfile=` — the emulator writes it, nobody
    /// rotates it. A guest in its first ten minutes pours tens of megabytes a
    /// second down the console: measured on the rig, four gigabytes before it
    /// settles, and on the phone that lands in the app's Documents.
    ///
    /// Cutting the file to nothing is enough: the emulator appends, so what it
    /// writes next lands at the top of a short file, and `poll` sees that the
    /// file is now shorter than where it stopped reading. The emulator used to
    /// write on from its old offset and leave a hole of zeros in front, and a
    /// log sent in was hundreds of megabytes of nothing.
    private func trimIfHuge() {
        guard Date().timeIntervalSince(lastTrim) >= 5 else { return }
        lastTrim = Date()
        var info = stat()
        guard stat(url.path, &info) == 0 else { return }
        guard Int64(info.st_blocks) * 512 > logLimit else { return }
        _ = truncate(url.path, 0)
    }

    private func poll() {
        trimIfHuge()
        if handle == nil {
            guard FileManager.default.fileExists(atPath: url.path),
                  let opened = try? FileHandle(forReadingFrom: url)
            else { return }
            handle = opened
            DispatchQueue.main.async { self.connected = true }
        }

        guard let handle else { return }
        // Cut back by `trimIfHuge`, or emptied for a new start: reading goes on
        // from the top, instead of waiting past the end of a file that will not
        // grow that far again.
        var info = stat()
        if fstat(handle.fileDescriptor, &info) == 0, let offset = try? handle.offset(), UInt64(info.st_size) < offset {
            try? handle.seek(toOffset: 0)
        }
        guard let bytes = try? handle.readToEnd(), !bytes.isEmpty else {
            // A shell prompt carries no newline after it, so whoever reads this
            // cannot tell a half-written line from a finished one until the
            // console goes quiet. Saying so once, when it does, is what makes
            // the prompt appear.
            if !quiet {
                quiet = true
                DispatchQueue.main.async {
                    self.chunk = ""
                    self.sequence += 1
                }
            }
            return
        }
        quiet = false

        let piece = String(decoding: bytes, as: UTF8.self)
        watchForDeath(piece)
        DispatchQueue.main.async {
            self.text += piece
            if self.text.count > self.limit {
                self.text = String(self.text.suffix(self.limit))
            }
            self.chunk = piece
            self.sequence += 1
        }
    }
}
