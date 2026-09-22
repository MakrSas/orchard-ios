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
