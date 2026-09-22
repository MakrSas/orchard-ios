import SwiftUI

/// Standalone JIT smoke test.
///
/// Orchard's QEMU dylib for iOS does not exist yet — building it means
/// cross-compiling the vmapple-patched tree and deciding what replaces
/// reims-vgpu's Vulkan path. This app answers a narrower question first: does
/// the same MAP_JIT / self-trace / mirror-mapping dance that Inferno-iOS uses
/// (see JIT.swift, copied unmodified from Inferno-iOS) actually get executable
/// memory on this phone, under this signing method. It links only JIT.swift,
/// LogCapture.swift and L10n.swift from ../app/Sources — nothing about the
/// emulator itself.
@main
struct JITProbeApp: App {
    init() {
        LogCapture.shared.start()
        LogCapture.shared.noteDevice()
        JIT.prepare()
    }

    var body: some Scene {
        WindowGroup {
            JITProbeView()
                .preferredColorScheme(.dark)
        }
    }
}

struct JITProbeView: View {
    @State private var status: JIT.Availability = JIT.status
    @State private var report: String = ""
    @ObservedObject private var log = LogCapture.shared

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    statusBadge

                    Button("Обновить проверку (без исполнения)") {
                        status = JIT.prepare()
                        report = JIT.diagnose(includeExecution: false)
                    }
                    .buttonStyle(.borderedProminent)

                    Text("Кнопки \"выполнить сгенерированный код\" больше нет: оба варианта (RWX-в-одном-mmap и RW→RX через mprotect на одном и том же отображении) вешали телефон намертво под StikDebug — трассировщик перехватывает fault и не возобновляет поток. Судя по комментарию в needsSplitWX, настоящий QEMU при недоступном MAP_JIT использует не переключение прав на одном mmap, а два разных отображения одной физической памяти — mprotect-переключение тут просто не тот путь. Реальное доказательство, что JIT работает — то, что сам Inferno у вас уже крутит его без зависаний.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    if !report.isEmpty {
                        GroupBox("Отчёт") {
                            Text(report)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }

                    GroupBox("Лог") {
                        Text(log.text)
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding()
            }
            .navigationTitle("Orchard — проверка JIT")
        }
        .onAppear {
            status = JIT.status
            report = JIT.diagnose(includeExecution: false)
        }
    }

    private var statusBadge: some View {
        HStack {
            Circle()
                .fill(status.isAvailable ? .green : .red)
                .frame(width: 12, height: 12)
            switch status {
            case .available(let via):
                Text("JIT доступен: \(via)")
            case .unavailable(let why):
                Text("JIT недоступен: \(why)")
            }
        }
        .font(.headline)
    }
}
