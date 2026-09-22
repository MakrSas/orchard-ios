import SwiftUI

/// The catalogue screen: an icon, a name, and a button on the right — the shape
/// people already know from the App Store, stocked with what this guest can
/// actually run. See `AppCatalog.swift` for why that is not the App Store itself.
struct CatalogView: View {
    @ObservedObject var model: VMModel
    @StateObject private var store = CatalogStore()
    @StateObject private var icons = IconStore()
    @Environment(\.dismiss) private var dismiss

    @State private var query = ""
    @State private var sources = false
    @State private var chosen: CatalogApp?
    /// Off by default: sources rarely declare a minimum iOS, so hiding
    /// everything that did not say would hide most of the shelf.
    @State private var onlyFitting = false

    private var shown: [CatalogApp] {
        var all = store.apps
        if onlyFitting { all = all.filter { $0.best?.fit == .yes } }
        guard !query.isEmpty else { return all }
        let needle = query.lowercased()
        return all.filter { $0.name.lowercased().contains(needle) || $0.id.lowercased().contains(needle) }
    }

    var body: some View {
        NavigationStack {
            List {
                if case .loading(let who) = store.state {
                    HStack(spacing: 10) {
                        ProgressView()
                        Text(L("Читаю %@…", who)).foregroundStyle(.secondary)
                    }
                }
                if case .failed(let why) = store.state {
                    Text(why).font(.footnote).foregroundStyle(.orange)
                }

                ForEach(shown) { app in
                    row(app)
                        .contentShape(Rectangle())
                        .onTapGesture { chosen = app }
                }

                if store.apps.isEmpty, store.state == .idle {
                    Text(L("Пусто. Перечитайте источники из меню «⋯»."))
                        .foregroundStyle(.secondary)
                }
            }
            .listStyle(.plain)
            .searchable(text: $query, prompt: L("Поиск приложения"))
            .refreshable { await store.refresh() }
            .navigationTitle(L("Каталог приложений"))
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(L("Закрыть")) { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Menu {
                        Toggle(L("Только для iOS 14"), isOn: $onlyFitting)
                        Button(L("Источники…"), systemImage: "list.bullet") { sources = true }
                        // Pulling the list down does this on a phone; a Mac has no such gesture.
                        Button(L("Перечитать источники"), systemImage: "tray.and.arrow.down") {
                            Task { await store.refresh() }
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(isPresented: $sources) { CatalogSourcesView(store: store) }
            .sheet(item: $chosen) { app in CatalogDetail(app: app, icons: icons) { install(app) } }
            .task { if store.apps.isEmpty { await store.refresh() } }
        }
        .sheetSize()
    }

    private func row(_ app: CatalogApp) -> some View {
        HStack(spacing: 12) {
            icon(app, size: 48)
            VStack(alignment: .leading, spacing: 2) {
                Text(app.name).font(.body).lineLimit(1)
                if !app.summary.isEmpty {
                    Text(app.summary).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                fitness(app)
            }
            Spacer(minLength: 8)
            Button { install(app) } label: {
                Text(L("Взять"))
                    .font(.caption.weight(.bold))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 6)
                    .background(.tint.opacity(0.15), in: Capsule())
            }
            // Borderless, not plain: the row around this button carries a tap
            // of its own, and a plain button inside a row hands the first tap
            // to that — the button then only works on the second press.
            .buttonStyle(.borderless)
            .disabled(!model.isRunning || model.transfer?.isRunning == true)
        }
        .padding(.vertical, 4)
        .onAppear { icons.load(app.iconURL) }
    }

    /// Said plainly on every row, because "it installed and then did nothing"
    /// is the worst way to find out.
    @ViewBuilder
    private func fitness(_ app: CatalogApp) -> some View {
        let version = app.best ?? app.newest
        switch version?.fit {
        case .yes:
            Text(L("iOS %@ и новее · %@", version?.minOS ?? "14", version?.version ?? ""))
                .font(.caption2).foregroundStyle(.green)
        case .unknown:
            Text(L("iOS не указана · %@", version?.version ?? ""))
                .font(.caption2).foregroundStyle(.secondary)
        default:
            Text(L("Нужна iOS %@ — в этом госте не пойдёт", version?.minOS ?? "?"))
                .font(.caption2).foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private func icon(_ app: CatalogApp, size: CGFloat) -> some View {
        if let image = icons.images[app.iconURL?.absoluteString ?? ""] {
            Image(platformImage: image)
                .resizable().scaledToFill()
                .frame(width: size, height: size)
                .clipShape(RoundedRectangle(cornerRadius: size * 0.22, style: .continuous))
        } else {
            RoundedRectangle(cornerRadius: size * 0.22, style: .continuous)
                .fill(.quaternary)
                .frame(width: size, height: size)
                .overlay(Image(systemName: "app").foregroundStyle(.secondary))
        }
    }

    private func install(_ app: CatalogApp) {
        guard let version = app.best ?? app.newest else { return }
        dismiss()
        model.installCatalogApp(name: app.name, url: version.downloadURL, size: version.size)
    }
}

/// What one app says about itself, before taking it.
private struct CatalogDetail: View {
    let app: CatalogApp
    @ObservedObject var icons: IconStore
    let install: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 14) {
                        if let image = icons.images[app.iconURL?.absoluteString ?? ""] {
                            Image(platformImage: image).resizable().scaledToFill()
                                .frame(width: 64, height: 64)
                                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                        VStack(alignment: .leading, spacing: 3) {
                            Text(app.name).font(.headline)
                            if !app.developer.isEmpty {
                                Text(app.developer).font(.caption).foregroundStyle(.secondary)
                            }
                            Text(app.source).font(.caption2).foregroundStyle(.tertiary)
                        }
                    }
                }

                if let version = app.best ?? app.newest {
                    Section(L("Версия")) {
                        LabeledContent(L("Версия"), value: version.version)
                        if version.size > 0 {
                            LabeledContent(L("Размер"), value: TransferState.size(version.size))
                        }
                        LabeledContent(L("Минимальная iOS"), value: version.minOS ?? L("не указана"))
                        if !version.notes.isEmpty {
                            Text(version.notes).font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                }

                Section {
                    Button(L("Установить в гостя"), systemImage: "arrow.down.app") {
                        dismiss()
                        install()
                    }
                } footer: {
                    Text(L("Приложение скачается на телефон и уедет в гостя тем же путём, что и «Установить .ipa»."))
                }
            }
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button(L("Готово")) { dismiss() } }
            }
        }
        .sheetSize(idealWidth: 500, idealHeight: 540)
    }
}

/// The sources, editable the way the package repositories are.
private struct CatalogSourcesView: View {
    @ObservedObject var store: CatalogStore
    @Environment(\.dismiss) private var dismiss
    @State private var adding = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        TextField(L("https://адрес/источника.json"), text: $adding)
                            .autocorrectionDisabled()
                            .noAutocapitalization()
                            .urlKeyboard()
                        Button(L("Добавить")) {
                            let url = adding.trimmingCharacters(in: .whitespaces)
                            guard !url.isEmpty, !store.sources.contains(where: { $0.url == url }) else { return }
                            store.sources.append(AppSource(url: url))
                            adding = ""
                        }
                        .disabled(adding.isEmpty)
                    }
                } footer: {
                    Text(L("Источники в формате AltStore: обычный JSON со списком приложений. Приложений из App Store тут быть не может — они зашифрованы и в эмуляторе не запускаются."))
                }

                Section(L("Источники")) {
                    ForEach(store.sources) { source in
                        Text(source.url).font(.callout)
                    }
                    .onDelete { store.sources.remove(atOffsets: $0) }
                }
            }
            .navigationTitle(L("Источники"))
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button(L("Готово")) { dismiss() } }
            }
        }
        .sheetSize(idealWidth: 500, idealHeight: 460)
    }
}

/// Icons, fetched once and kept while the screen is open.
@MainActor
final class IconStore: ObservableObject {
    @Published private(set) var images: [String: PlatformImage] = [:]
    private var inFlight: Set<String> = []

    func load(_ url: URL?) {
        guard let url else { return }
        let key = url.absoluteString
        guard images[key] == nil, !inFlight.contains(key) else { return }
        inFlight.insert(key)
        Task { [weak self] in
            var request = URLRequest(url: url)
            request.cachePolicy = .returnCacheDataElseLoad
            request.timeoutInterval = 20
            let image = (try? await URLSession.shared.data(for: request)).flatMap { PlatformImage(data: $0.0) }
            await MainActor.run {
                guard let self else { return }
                self.inFlight.remove(key)
                if let image { self.images[key] = image }
            }
        }
    }
}
