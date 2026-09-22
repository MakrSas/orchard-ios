import Foundation

/// Talks to the emulator's QMP socket.
///
/// This is the only way to ask the running machine what it is actually doing —
/// whether the vCPUs are executing or the machine is sitting stopped — which
/// guesswork from the outside cannot answer.
final class QMPClient {
    private let port: UInt16

    init(port: UInt16 = 4556) {
        self.port = port
    }

    /// Runs a short sequence of queries and reports the answers as text.
    func inspect(_ completion: @escaping (String) -> Void) {
        Thread.detachNewThread { [port] in
            let sock = Sock()
            defer { sock.close() }

            if let problem = sock.connect(port: port) {
                return completion("QMP: \(problem)")
            }

            var lines: [String] = ["QMP:"]

            // Greeting banner arrives unsolicited.
            guard let greeting = sock.readSome() else {
                return completion(L("QMP: сокет открыт, но приветствия нет"))
            }
            lines.append("greeting → " + String(decoding: greeting, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines).prefix(200))

            for command in ["qmp_capabilities", "query-status", "query-cpus-fast", "query-vnc"] {
                guard sock.write(Array("{\"execute\":\"\(command)\"}\n".utf8)) else {
                    lines.append(L("%@ → не удалось отправить", command))
                    break
                }
                guard let reply = sock.readSome(max: 64 * 1024) else {
                    lines.append(L("%@ → нет ответа", command))
                    break
                }
                if command != "qmp_capabilities" {
                    lines.append("\(command) → " + String(decoding: reply, as: UTF8.self)
                        .trimmingCharacters(in: .whitespacesAndNewlines).prefix(700))
                }
            }

            completion(lines.joined(separator: "\n  "))
        }
    }

    /// Where the guest is, in one line: CPU0's program counter and exception
    /// level, and how many cores are running rather than halted.
    ///
    /// For a guest whose console goes quiet at the kernel handoff — a macOS
    /// kernel under full security ignores `serial=3` — this is what tells a
    /// boot that is moving from one that is stuck. Early XNU runs at EL1 in
    /// the kernel's own addresses (0xfffffe…), iBoot does not, and the
    /// kernel's bring-up of the secondary cores shows as cores leaving halt.
    func snapshot(_ completion: @escaping (String) -> Void) {
        Thread.detachNewThread { [port] in
            let sock = Sock()
            defer { sock.close() }
            // The same steps, in the same order, as `inspect`, which is known
            // to get answers; each failure says which step it was.
            if let problem = sock.connect(port: port) { return completion("CPU: \(problem)") }
            guard sock.readSome() != nil else { return completion(L("CPU: QMP не прислал приветствие")) }
            guard sock.write(Array("{\"execute\":\"qmp_capabilities\"}\n".utf8)),
                  sock.readSome() != nil
            else { return completion(L("CPU: QMP не принял qmp_capabilities")) }

            func hmp(_ line: String) -> String? {
                let command = "{\"execute\":\"human-monitor-command\",\"arguments\":{\"command-line\":\"\(line)\"}}\n"
                guard sock.write(Array(command.utf8)) else { return nil }
                // A reply is one line; it may arrive in pieces, so read on
                // until the newline that ends it.
                var reply = Data()
                while !reply.contains(UInt8(ascii: "\n")) {
                    guard let chunk = sock.readSome(max: 64 * 1024) else { return nil }
                    reply.append(chunk)
                }
                let line = reply.prefix { $0 != UInt8(ascii: "\n") }
                guard let object = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any]
                else { return nil }
                return object["return"] as? String
            }

            var parts: [String] = []
            if let regs = hmp("info registers") {
                let pc = regs.range(of: #"PC=[0-9a-f]+"#, options: .regularExpression).map { String(regs[$0]) } ?? "PC=?"
                let el = regs.range(of: #"EL[0-3][th]"#, options: .regularExpression).map { String(regs[$0]) } ?? "EL?"
                let place = pc.dropFirst(3).hasPrefix("fffffe") ? L("ядро XNU") : L("не в ядре")
                parts.append("CPU0 \(pc) \(el) (\(place))")
            } else {
                parts.append(L("info registers не ответил"))
            }
            if let cpus = hmp("info cpus") {
                let lines = cpus.split(separator: "\n").filter { $0.contains("CPU #") }
                let halted = lines.filter { $0.contains("(halted)") }.count
                parts.append(L("ядер в работе: %d из %d", lines.count - halted, lines.count))
            }
            completion(parts.joined(separator: " · "))
        }
    }

    /// Asks the machine to shut down.
    ///
    /// This is the only safe way to stop it. `quit` unwinds QEMU's main loop,
    /// which flushes the block layer to the files. Killing the process instead
    /// leaves qcow2 metadata half-written, and the guest's disk then disagrees
    /// with the Secure Enclave's replay counters — a mismatch the guest answers
    /// with a SEP panic on the next boot.
    func quit(_ completion: @escaping (String) -> Void) {
        Thread.detachNewThread { [port] in
            let sock = Sock()
            defer { sock.close() }

            if let problem = sock.connect(port: port) {
                return completion(L("Выключение: %@", problem))
            }
            guard sock.readSome() != nil else {
                return completion(L("Выключение: приветствия от QMP нет"))
            }
            guard sock.write(Array("{\"execute\":\"qmp_capabilities\"}\n".utf8)),
                  sock.readSome() != nil,
                  sock.write(Array("{\"execute\":\"quit\"}\n".utf8))
            else {
                return completion(L("Выключение: команда не ушла"))
            }
            // The reply may never arrive: QEMU acts on quit and exits.
            _ = sock.readSome()
            completion(L("Выключение: команда отправлена, машина сбрасывает диски на файлы"))
        }
    }
}
