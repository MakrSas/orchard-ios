import Foundation

/// A catalogue of apps that can actually be installed into the guest.
///
/// Not an App Store, and it cannot be one. An `.ipa` bought from Apple carries
/// its main binary encrypted with FairPlay, and that is undone at install time
/// by the real device's own hardware against the account that bought it — which
/// is why downgraders like PancakeStore ask iOS itself to install, on the phone
/// they run on, rather than handing anybody a file. An emulated guest has none
/// of that, so an App Store `.ipa` would install and die on launch.
///
/// What does work is everything distributed outside the App Store: emulators,
/// utilities, jailbreak tools, open-source apps. Those ship unencrypted —
/// measured, not assumed: DolphiniOS from OatmealDome's source has `cryptid 0`,
/// no `SC_Info`, and `MinimumOSVersion 14.0`. That is the shelf this stocks.
///
/// The format is AltStore's, which two different generations of sources use at
/// once — checked against five live ones: some carry only the modern
/// `versions` array, some only the older flat fields, and some both. Both are
/// read here, because dropping either would empty half the catalogue.
struct CatalogVersion: Hashable {
    let version: String
    let date: String
    let size: Int64
    let downloadURL: URL
    let minOS: String?
    let maxOS: String?
    let notes: String

    /// Whether this build says it runs on the guest's iOS 14.
    ///
    /// Three answers, not two. Plenty of sources never declare `minOSVersion`
    /// at all — Taurine, the jailbreak written *for* iOS 14, is one of them —
    /// so "did not say" must not be treated as "will not run", or the filter
    /// would hide exactly the things worth having.
    enum Fit { case yes, no, unknown }

    var fit: Fit {
        guard let minOS, let low = CatalogVersion.major(minOS) else { return .unknown }
        if low > 14 { return .no }
        if let maxOS, let high = CatalogVersion.major(maxOS), high < 14 { return .no }
        return .yes
    }

    private static func major(_ text: String) -> Int? {
        Int(text.split(separator: ".").first.map(String.init) ?? "")
    }
}

struct CatalogApp: Identifiable, Hashable {
    let id: String
    let name: String
    let developer: String
    let summary: String
    let iconURL: URL?
    let versions: [CatalogVersion]
    let source: String

    /// The newest build that admits to running on iOS 14, or the newest that
    /// says nothing. Nil when every version rules the guest out.
    var best: CatalogVersion? {
        versions.first { $0.fit == .yes } ?? versions.first { $0.fit == .unknown }
    }

    var newest: CatalogVersion? { versions.first }
}

/// One source, kept as a plain address the way the package repositories are.
struct AppSource: Identifiable, Hashable, Codable {
    var url: String
    var id: String { url }
}

@MainActor
final class CatalogStore: ObservableObject {
    enum State: Equatable {
        case idle
        case loading(String)
        case failed(String)
    }

    @Published private(set) var apps: [CatalogApp] = []
    @Published private(set) var state: State = .idle
    @Published var sources: [AppSource] = CatalogStore.remembered {
        didSet { CatalogStore.remember(sources) }
    }

    /// Sources that were actually fetched and parsed while this was written, and
    /// that carry something an iOS 14 guest can run. Deliberately short: a list
    /// of dead addresses is worse than no list.
    static let defaults = [
        // DolphiniOS — its public beta declares iOS 14.0 outright.
        "https://altstore.oatmealdome.me",
        // Taurine — the jailbreak for 14.0–14.3.
        "https://taurine.app/altstore/taurinestore.json",
        // iSH — a Linux shell that runs on old iOS.
        "https://ish.app/altstore.json",
        // Provenance — emulators.
        "https://provenance-emu.com/apps.json",
    ]

    private static let key = "appSources"

    private static var remembered: [AppSource] {
        guard let data = UserDefaults.standard.data(forKey: key),
              let list = try? JSONDecoder().decode([AppSource].self, from: data)
        else { return defaults.map { AppSource(url: $0) } }
        return list
    }

    private static func remember(_ list: [AppSource]) {
        guard let data = try? JSONEncoder().encode(list) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }

    func refresh() async {
        var found: [CatalogApp] = []
        var complaints: [String] = []

        for source in sources {
            guard let url = URL(string: source.url) else { continue }
            state = .loading(url.host ?? source.url)
            do {
                found += try await CatalogStore.fetch(url)
            } catch {
                complaints.append("\(url.host ?? source.url): \(error.localizedDescription)")
            }
        }

        // One entry per bundle id; the name decides the order.
        var byID: [String: CatalogApp] = [:]
        for app in found where byID[app.id] == nil { byID[app.id] = app }
        apps = byID.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        state = complaints.isEmpty ? .idle : .failed(complaints.joined(separator: "\n"))
    }

    enum Failure: LocalizedError {
        case notASource

        var errorDescription: String? { L("это не источник приложений (нет списка apps)") }
    }

    private static func fetch(_ url: URL) async throws -> [CatalogApp] {
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw Failure.notASource }
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let list = root["apps"] as? [[String: Any]]
        else { throw Failure.notASource }
        let name = (root["name"] as? String) ?? (url.host ?? "")
        return list.compactMap { parse($0, source: name) }
    }

    private static func parse(_ item: [String: Any], source: String) -> CatalogApp? {
        guard let id = item["bundleIdentifier"] as? String, !id.isEmpty,
              let name = item["name"] as? String
        else { return nil }

        var versions: [CatalogVersion] = []
        if let list = item["versions"] as? [[String: Any]] {
            versions = list.compactMap { version($0) }
        }
        // The older shape keeps the newest build's fields on the app itself.
        // Sources that publish both agree, so this only fills in for the ones
        // that publish nothing else.
        if versions.isEmpty {
            versions = [version(item, fallbackDate: item["versionDate"] as? String,
                                fallbackNotes: item["versionDescription"] as? String)].compactMap { $0 }
        }
        guard !versions.isEmpty else { return nil }

        let summary = (item["subtitle"] as? String)
            ?? (item["localizedDescription"] as? String)?
                .split(separator: "\n").first.map(String.init)
            ?? ""
        return CatalogApp(
            id: id,
            name: name,
            developer: (item["developerName"] as? String) ?? "",
            summary: summary,
            iconURL: (item["iconURL"] as? String).flatMap { URL(string: $0) },
            versions: versions,
            source: source)
    }

    private static func version(_ item: [String: Any], fallbackDate: String? = nil,
                                fallbackNotes: String? = nil) -> CatalogVersion? {
        guard let link = item["downloadURL"] as? String, let url = URL(string: link) else { return nil }
        let size: Int64
        if let number = item["size"] as? NSNumber { size = number.int64Value }
        else if let text = item["size"] as? String { size = Int64(text) ?? 0 }
        else { size = 0 }
        return CatalogVersion(
            version: (item["version"] as? String) ?? "",
            date: (item["date"] as? String) ?? fallbackDate ?? "",
            size: size,
            downloadURL: url,
            minOS: item["minOSVersion"] as? String,
            maxOS: item["maxOSVersion"] as? String,
            notes: (item["localizedDescription"] as? String) ?? fallbackNotes ?? "")
    }
}
