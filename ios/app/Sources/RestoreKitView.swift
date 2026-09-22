import SwiftUI
#if os(iOS)
import UIKit
#endif

/// Making the kit out of a firmware archive, as a set of form sections.
///
/// The same rows are wanted in two places — at first launch, where there is
/// nothing in the app's folder at all, and in the restore screen later — so
/// they live here rather than twice over. Drop it into a `List` or a `Form`.
struct RestoreKitSetup: View {
    /// Called once the kit is made, so the host can look at the folder again.
    var onReady: () -> Void = {}

    @State private var firmware: URL? = RestoreSession.firmware
    @State private var ticket: URL?
    @State private var rom: URL?
    @State private var cryptexTemplate: URL?
    @AppStorage("sepKey") private var sepKey = ""
    @State private var preparing = false
    @State private var progress = ""

    private var keyIsGood: Bool { (try? SEPFirmware.Key(hex: sepKey)) != nil }

    private var ready: Bool {
        firmware != nil && ticket != nil && keyIsGood
            && (rom != nil || VMConfig.sepROMPresent)
    }

    var body: some View {
        Section {
            KitRow(title: L("Прошивка .ipsw"),
                   example: "iPhone11,8,iPhone12,1_14.0_18A5351d_Restore.ipsw",
                   source: L("Apple: ссылку на свою версию ищите на ipsw.me для iPhone12,1"),
                   chosen: firmware?.lastPathComponent) { url in
                RestoreSession.remember(firmware: url)
                firmware = url
            }

            KitRow(title: L("Тикет"),
                   example: "ticket.shsh2",
                   source: L("ChefKiss, страница File Setup"),
                   chosen: ticket?.lastPathComponent) { ticket = $0 }

            KitRow(title: L("ПЗУ SEP"),
                   example: "AppleSEPROM-Cebu-B1",
                   source: L("securerom.fun"),
                   chosen: rom?.lastPathComponent
                       ?? (VMConfig.sepROMPresent ? L("уже в папке приложения") : nil)) { rom = $0 }

            // Only iOS 16+ ever asks for this; left empty, an iOS 14 restore
            // never notices, and a 16+ one fails with a clear message instead
            // of hanging.
            KitRow(title: L("Шаблон Cryptex1 (только iOS 16+)"),
                   example: "apticket.im4m",
                   source: L("любой ваш собственный тикет Cryptex1 — форма и подпись, не содержимое"),
                   chosen: cryptexTemplate?.lastPathComponent
                       ?? (VMConfig.cryptexTemplatePresent ? L("уже в папке приложения") : nil)) { cryptexTemplate = $0 }

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text(L("Ключ SEP"))
                    Spacer()
                    if keyIsGood {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                    } else if !sepKey.isEmpty {
                        Text(L("%d из 96", sepKey.filter(\.isHexDigit).count))
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                TextField("a1b2c3… (96 цифр: IV и ключ подряд)", text: $sepKey)
                    .font(.footnote.monospaced())
                    .autocorrectionDisabled()
                    .noAutocapitalization()
                Text(L("The Apple Wiki, страница Keys:<кодовое имя> <сборка> (iPhone12,1) — SEPFirmwareIV и сразу за ним SEPFirmwareKey"))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Button(L("Подготовить набор")) { prepare() }
                .disabled(preparing || !ready)
            if !progress.isEmpty {
                Text(progress)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        } header: {
            Text(L("Подготовка"))
        } footer: {
            Text(L("Ничего из этого приложение не скачивает — файлы приносите сами; подробности в RESTORE.md. Скачанное Safari лежит в «Файлах» → «На iPhone» → «Загрузки»."))
        }
    }

    /// One file to pick, saying plainly what it is called, where it comes from,
    /// and whether it has been picked yet.
    private struct KitRow: View {
        let title: String
        let example: String
        let source: String
        let chosen: String?
        let picked: (URL) -> Void

        /// Each row presents its own picker.
        ///
        /// One importer for the whole section does not work: presentation
        /// modifiers on a `Section` are ignored, and the button then does
        /// nothing at all. Every row is a real view and can present.
        @State private var picking = false

        var body: some View {
            Button { picking = true } label: {
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(title)
                        Spacer()
                        if chosen != nil {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        } else {
                            Text(L("выбрать"))
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Text(chosen ?? example)
                        .font(.footnote.monospaced())
                        .foregroundStyle(chosen == nil ? .secondary : .primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(source)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                }
                .contentShape(Rectangle())
            }
            // Not `.plain`: in a list row that one swallows the first tap.
            .buttonStyle(.borderless)
            .foregroundStyle(.primary)
            .fileImporter(isPresented: $picking, allowedContentTypes: [.item]) { result in
                if case .success(let url) = result { picked(url) }
            }
        }
    }

    private func prepare() {
        guard let firmware, let ticket else { return }
        preparing = true
        progress = L("Готовлю…")
        // Copying the firmware alone can run minutes; the screen locking
        // partway through suspends the app with nothing in the log to say why
        // it stopped.
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = true
        #endif
        let inputs = RestorePrep.Inputs(ipsw: firmware, shsh2: ticket, sepROM: rom, sepKey: sepKey,
                                        cryptexTemplate: cryptexTemplate)
        DispatchQueue.global(qos: .userInitiated).async {
            var made = false
            do {
                try RestorePrep.prepare(inputs, note: { line in
                    LogCapture.shared.note(line)
                    DispatchQueue.main.async { progress = line }
                })
                made = true
            } catch {
                LogCapture.shared.note(L("Подготовка: %@", error.localizedDescription))
                DispatchQueue.main.async { progress = error.localizedDescription }
            }
            DispatchQueue.main.async {
                preparing = false
                #if os(iOS)
                UIApplication.shared.isIdleTimerDisabled = false
                #endif
                if made { onReady() }
            }
        }
    }
}
