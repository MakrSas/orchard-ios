import SwiftUI

/// A package manager that runs on the phone rather than in the guest.
///
/// Cydia inside the guest works, but barely: it blocks its own main thread
/// while dpkg unpacks, and at a few frames a second that is long enough for
/// iOS to kill it — which also kills whatever was installing. Here the
/// browsing and the downloading happen on the phone, where the network is real
/// and nothing is watching a clock, and only the finished `.deb` goes in.
struct PackagesView: View {
    @ObservedObject var model: VMModel
    @StateObject private var store = RepoStore()
    @Environment(\.dismiss) private var dismiss

    @State private var query = ""
    @State private var sources = false
    @State private var complaint: String?
    /// What the guest already has, by package name. Read once when the screen
    /// opens: the answer costs a file transfer, and it does not change behind
    /// our back while the list is being read.
    @State private var installed: [String: String] = [:]
    @State private var onlyInstalled = false
    @State private var chosen: RepoPackage?

    /// Repositories hold tens of thousands of packages and a list that long is
    /// slower to draw than it is to scroll. Until something is typed, this is a
    /// window onto the index, not the whole of it.
    private var shown: [RepoPackage] {
        var all = store.packages
        if onlyInstalled { all = all.filter { installed[$0.id] != nil } }
        guard !query.isEmpty else { return Array(all.prefix(200)) }
        let needle = query.lowercased()
        return all.filter {
            $0.name.lowercased().contains(needle) || $0.id.lowercased().contains(needle)
        }
        .prefix(200)
        .map { $0 }
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
                if let complaint {
                    Text(complaint).font(.footnote).foregroundStyle(.red)
                }

                ForEach(shown) { package in
                    Button { chosen = package } label: { row(package) }
                        .buttonStyle(.plain)
                        .disabled(model.transfer?.isRunning == true)
                }

                if store.packages.isEmpty, store.state == .idle {
                    Text(L("Пусто. Перечитайте источники из меню «⋯»."))
                        .foregroundStyle(.secondary)
                }
            }
            .listStyle(.plain)
            .searchable(text: $query, prompt: L("Поиск пакета"))
            .refreshable { await store.refresh() }
            .navigationTitle(L("Менеджер пакетов"))
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(L("Закрыть")) { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Menu {
                        Toggle(L("Только установленные"), isOn: $onlyInstalled)
                        Button(L("Источники…"), systemImage: "list.bullet") { sources = true }
                        // Pulling the list down does this on a phone; a Mac has no such gesture.
                        Button(L("Перечитать источники"), systemImage: "tray.and.arrow.down") {
                            Task { await store.refresh() }
                        }
                        Button(L("Перечитать установленное"), systemImage: "arrow.clockwise") {
                            Task { installed = await model.installedPackages() }
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(isPresented: $sources) { SourcesView(store: store) }
            .confirmationDialog(chosen?.name ?? "", isPresented: Binding(
                get: { chosen != nil }, set: { if !$0 { chosen = nil } }), titleVisibility: .visible)
            {
                if let package = chosen {
                    let have = installed[package.id]
                    // A package that panics the guest still gets a button, at
                    // the far end of a warning: this is somebody's own machine,
                    // and the answer to «I know, do it» is not a grey button.
                    if package.refusal != nil {
                        Button(L("Всё равно установить"), role: .destructive) { install(package) }
                    }
                    else {
                        Button(have == nil ? L("Установить") : L("Переустановить")) { install(package) }
                    }
                    if have != nil {
                        Button(L("Удалить"), role: .destructive) {
                            dismiss()
                            model.removePackage(package.id)
                        }
                    }
                }
            } message: {
                if let package = chosen {
                    let version = installed[package.id].map { L("Установлен %@, в источнике %@", $0, package.version) }
                        ?? L("Версия %@", package.version)
                    Text(package.refusal.map { version + "\n\n" + L("Гостю нельзя: %@", $0) } ?? version)
                }
            }
            .task {
                if store.packages.isEmpty { await store.refresh() }
                if installed.isEmpty { installed = await model.installedPackages() }
            }
        }
        .sheetSize()
    }

    private func row(_ package: RepoPackage) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline) {
                if package.refusal != nil {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                if installed[package.id] != nil {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
                Text(package.name).font(.body)
                Spacer()
                Text(installed[package.id] ?? package.version)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !package.summary.isEmpty {
                Text(package.summary).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Text(package.repo.host ?? "").font(.caption2).foregroundStyle(.tertiary)
        }
        .padding(.vertical, 2)
    }

    /// Hands the work to the model and gets out of the way: the download and
    /// the install are shown by the same banner as every other transfer, so
    /// this screen has nothing left to wait for.
    private func install(_ package: RepoPackage) {
        guard let url = package.url else { return }
        dismiss()
        model.installFromRepo(name: package.name, id: package.id, version: package.version,
                              url: url, size: package.size)
    }
}

/// The list of repositories, which is the only state this screen keeps.
private struct SourcesView: View {
    @ObservedObject var store: RepoStore
    @Environment(\.dismiss) private var dismiss
    @State private var adding = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        TextField(L("https://адрес.репозитория/"), text: $adding)
                            .autocorrectionDisabled()
                            .noAutocapitalization()
                            .urlKeyboard()
                        Button(L("Добавить")) {
                            let url = adding.trimmingCharacters(in: .whitespaces)
                            guard !url.isEmpty, !store.repos.contains(where: { $0.url == url }) else { return }
                            store.repos.append(Repo(url: url))
                            adding = ""
                        }
                        .disabled(adding.isEmpty)
                    }
                } footer: {
                    Text(L("Читаются указатели `Packages` и `Packages.gz`. Источники на `.bz2` или `.zst` не поддерживаются: распаковщиков для них в iOS нет."))
                }

                Section(L("Источники")) {
                    ForEach(store.repos) { repo in
                        Text(repo.url).font(.callout)
                    }
                    .onDelete { store.repos.remove(atOffsets: $0) }
                }
            }
            .navigationTitle(L("Источники"))
            .inlineNavigationTitle()
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(L("Готово")) { dismiss() }
                }
            }
        }
        .sheetSize(idealWidth: 500, idealHeight: 460)
    }
}
