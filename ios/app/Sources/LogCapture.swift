import Foundation

/// Captures the emulator's stdout/stderr.
///
/// QEMU reports fatal configuration problems to stderr and then calls exit(),
/// which inside an app looks like a crash with no explanation. Redirecting both
/// streams into a pipe means the reason is on screen — and, because it is also
/// written to a file, still readable after a restart.
///
/// Every line carries the seconds since the app started. A log is read after
/// the fact, and most of what it is asked is how soon one thing followed
/// another: a machine that resets every few seconds and one that resets once an
/// hour print the same line. A line the same as the one before it is counted
/// rather than written again; the count follows when something else comes,
/// after a quiet moment, or every hundred.
final class LogCapture: ObservableObject {
    static let shared = LogCapture()

    @Published private(set) var text: String = ""

    private let pipe = Pipe()
    private var started = false
    private let limit = 128 * 1024
    private var fileHandle: FileHandle?

    /// Lines are cut, stamped and counted here, in the order they arrive.
    private let queue = DispatchQueue(label: "inferno.log")
    private let startTime = Date()
    /// What the pipe delivered after its last newline, waiting for the rest.
    private var partial = ""
    private var lastLine: String?
    private var repeats = 0
    private var flushGeneration = 0

    /// Printed by SwiftUI into the same stderr, and nothing to do with the
    /// machine.
    private static let noise = ["=== AttributeGraph: cycle detected"]

    /// Kept local rather than routed through `VMConfig.documents` so this file
    /// has no dependency on the rest of the app — the JIT probe target links
    /// it on its own, without VMConfig.
    private static var documents: URL {
        let base = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        #if os(macOS)
        return base.appendingPathComponent("Inferno")
        #else
        return base
        #endif
    }

    var logFileURL: URL {
        LogCapture.documents.appendingPathComponent("emulator.log")
    }

    /// The log from the run before this one.
    ///
    /// The interesting run is almost always the one that just died, and the
    /// app is started again to find out why — which used to overwrite the only
    /// copy of the evidence. Every crash report we were sent was from a session
    /// where nothing had happened yet.
    var previousLogFileURL: URL {
        LogCapture.documents.appendingPathComponent("emulator.prev.log")
    }

    func start() {
        guard !started else { return }
        started = true

        try? FileManager.default.removeItem(at: previousLogFileURL)
        try? FileManager.default.moveItem(at: logFileURL, to: previousLogFileURL)
        // Keep a copy on disk: the process may die before the UI updates.
        FileManager.default.createFile(atPath: logFileURL.path, contents: nil)
        fileHandle = try? FileHandle(forWritingTo: logFileURL)

        setvbuf(stdout, nil, _IOLBF, 0)
        setvbuf(stderr, nil, _IONBF, 0)
        dup2(pipe.fileHandleForWriting.fileDescriptor, STDOUT_FILENO)
        dup2(pipe.fileHandleForWriting.fileDescriptor, STDERR_FILENO)

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard let self, !data.isEmpty else { return }
            let chunk = String(decoding: data, as: UTF8.self)
            self.queue.async { self.take(chunk) }
        }
    }

    /// A note is whole lines of its own, and goes straight out rather than
    /// through the pipe's buffer: a note that arrived while the emulator was
    /// halfway through a line was spliced into the middle of it.
    func note(_ line: String) {
        queue.async {
            for piece in line.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline) {
                self.emit(String(piece))
            }
        }
    }

    /// The machine the app runs on, for the head of the log: which phone or
    /// Mac, which system, how much memory. A report without it leaves the
    /// first question about it unanswered.
    func noteDevice() {
        #if os(macOS)
        let (modelKey, system) = ("hw.model", "macOS")
        #else
        let (modelKey, system) = ("hw.machine", "iOS")
        #endif
        let info = ProcessInfo.processInfo
        let version = info.operatingSystemVersion
        var release = "\(version.majorVersion).\(version.minorVersion)"
        if version.patchVersion > 0 { release += ".\(version.patchVersion)" }
        // Numbers and the build rather than operatingSystemVersionString, which
        // the system words in the language it picked for the app.
        note(L("Устройство: %@, %@ %@ (%@), память %.1f ГБ", LogCapture.sysctlString(modelKey), system, release,
               LogCapture.sysctlString("kern.osversion"), Double(info.physicalMemory) / 1_073_741_824))
    }

    private static func sysctlString(_ name: String) -> String {
        var size = 0
        sysctlbyname(name, nil, &size, nil, 0)
        var value = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname(name, &value, &size, nil, 0)
        return String(cString: value)
    }

    /// Cuts what the pipe delivered into whole lines and passes each one on.
    private func take(_ chunk: String) {
        partial += chunk
        while let newline = partial.firstIndex(where: \.isNewline) {
            let line = String(partial[..<newline])
            partial.removeSubrange(...newline)
            emit(line)
        }
    }

    private func emit(_ line: String) {
        if LogCapture.noise.contains(where: { line.hasPrefix($0) }) { return }
        if line == lastLine {
            // A run of empty lines is one empty line, with nothing to count.
            if line.isEmpty { return }
            repeats += 1
            // Counted out every hundred too, so that a line repeating without a
            // pause until the app is killed does not look like it came once.
            if repeats >= 100 { flushRepeats() }
            else { scheduleFlush() }
            return
        }
        flushRepeats()
        lastLine = line
        write(line)
    }

    /// The count of a repeated line is written when a different line comes,
    /// or after a quiet moment, so that a line repeating as the app dies is
    /// still accounted for.
    private func scheduleFlush() {
        flushGeneration += 1
        let generation = flushGeneration
        queue.asyncAfter(deadline: .now() + 2) { [weak self] in
            guard let self, self.flushGeneration == generation else { return }
            self.flushRepeats()
        }
    }

    private func flushRepeats() {
        guard repeats > 0, let lastLine else { return }
        // Once more is shorter written out than counted.
        if repeats == 1 { write(lastLine) }
        else { write(L("(повторилось ещё: %d)", repeats)) }
        repeats = 0
    }

    private func write(_ line: String) {
        let stamped = String(format: "[%7.2f] ", Date().timeIntervalSince(startTime)) + line + "\n"
        try? fileHandle?.write(contentsOf: Data(stamped.utf8))
        DispatchQueue.main.async {
            self.text += stamped
            if self.text.count > self.limit {
                self.text = String(self.text.suffix(self.limit))
            }
        }
    }
}
