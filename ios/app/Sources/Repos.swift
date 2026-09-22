import Compression
import Foundation

/// A package index, read straight from a jailbreak repository.
///
/// Cydia in the guest cannot be trusted with this work: it blocks its own main
/// thread while dpkg runs, and a guest this slow keeps it blocked long enough
/// for iOS to kill it. The phone, on the other hand, has a real network and
/// nothing watching a clock — so the browsing and the downloading happen here,
/// and only the finished `.deb` is handed to the guest.
struct RepoPackage: Identifiable, Hashable {
    let id: String
    let name: String
    let version: String
    let section: String
    let summary: String
    let author: String
    /// Where the `.deb` lives, relative to the repository's own address.
    let path: String
    let size: Int64
    let repo: URL

    var url: URL? { URL(string: path, relativeTo: repo)?.absoluteURL }
}

/// Packages that take the guest down with them, and why.
///
/// Every hooking runtime works the same way: its library has to be loaded into
/// processes that did not ask for it, launchd first of all. This guest cannot
/// grant that. AMFI is enforcing, nothing patches amfid, and the page carrying
/// the injected code fails validation the moment it is touched — launchd takes
/// a SIGBUS, and a launchd that exits is not an error but a kernel panic:
/// `initproc exited -- exit reason namespace 2 subcode 0xa`, then the watchdog
/// reboots the machine. Measured on 12 September 2026 with mobilesubstrate
/// 0.9.7113: the guest panicked in the middle of the install.
///
/// So these are refused rather than warned about. The tweaks that depend on
/// them are not: without a runtime they are inert files, which is dull but
/// harmless.
extension RepoPackage {
    var refusal: String? { RepoPackage.refusals[id] }

    private static var hookingRuntime: String {
        L("Внедряется в launchd, а гость этого не умеет: launchd падает, ядро уходит в панику.")
    }

    private static let refusals: [String: String] = [
        "mobilesubstrate": hookingRuntime,
        "com.saurik.substrate.safemode": hookingRuntime,
        "com.ex.substitute": hookingRuntime,
        "com.ex.libsubstitute": hookingRuntime,
        "org.coolstar.libhooker": hookingRuntime,
        "org.coolstar.libhooker-utils": hookingRuntime,
        "ellekit": hookingRuntime,
        "com.ex.substitute-safemode": hookingRuntime,
    ]
}

/// One repository and what the last fetch had to say about it.
struct Repo: Identifiable, Hashable, Codable {
    var url: String
    var id: String { url }

    var address: URL? {
        // Repositories are named by a directory, and a missing slash turns the
        // last component into a sibling. An address that already points inside
        // a `dists` tree is cut back to the repository itself: that is what the
        // packages' own paths are relative to.
        var text = url
        if let cut = text.range(of: "/dists/") { text = String(text[..<cut.lowerBound]) }
        return URL(string: text.hasSuffix("/") ? text : text + "/")
    }
}

@MainActor
final class RepoStore: ObservableObject {
    enum State: Equatable {
        case idle
        case loading(String)
        case failed(String)
    }

    @Published private(set) var packages: [RepoPackage] = []
    @Published private(set) var state: State = .idle
    @Published var repos: [Repo] = RepoStore.remembered {
        didSet { RepoStore.remember(repos) }
    }

    /// The repositories a fresh bootstrap comes with, so the list is useful
    /// before anyone types an address.
    static let defaults = [
        "https://apt.bingner.com/",
        // Plain HTTP on purpose: its certificate has expired, so HTTPS fails
        // outright. iOS refuses HTTP unless told otherwise, and Info.plist
        // makes an exception for this one host.
        "http://apt.thebigboss.org/repofiles/cydia/",
        "https://repo.chariz.com/",
        "https://havoc.app/",
        // Filza lives nowhere else.
        "https://tigisoftware.com/cydia/",
    ]

    private static let key = "repositories"

    private static var remembered: [Repo] {
        guard let data = UserDefaults.standard.data(forKey: key),
              let list = try? JSONDecoder().decode([Repo].self, from: data)
        else { return defaults.map { Repo(url: $0) } }
        return list
    }

    private static func remember(_ list: [Repo]) {
        guard let data = try? JSONEncoder().encode(list) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }

    func refresh() async {
        var found: [RepoPackage] = []
        var complaints: [String] = []

        for repo in repos {
            guard let address = repo.address else { continue }
            state = .loading(address.host ?? repo.url)
            do {
                found += try await RepoStore.fetch(address)
            }
            catch {
                complaints.append("\(address.host ?? repo.url): \(error.localizedDescription)")
            }
        }

        // Newest version of each package wins, and the name decides the order —
        // the indexes come in whatever order the repository felt like.
        var newest: [String: RepoPackage] = [:]
        for package in found {
            if let have = newest[package.id], have.version >= package.version { continue }
            newest[package.id] = package
        }
        packages = newest.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        state = complaints.isEmpty ? .idle : .failed(complaints.joined(separator: "\n"))
    }

    /// Where an index can be, relative to the repository's own address.
    ///
    /// Two layouts and three encodings. The flat one — an index in the root —
    /// is what most tweak repositories use. The older ones are proper Debian
    /// archives with a `dists` tree, and their `stable`/`ios` names have to be
    /// guessed at, because nothing at the root says which they use. `.bz2`
    /// comes first: BigBoss publishes nothing else. `.zst`, which a few of the
    /// newest use, is left alone — iOS has no decompressor for it.
    private static let indexPaths = [
        "Packages.bz2", "Packages.gz", "Packages",
        "dists/stable/main/binary-iphoneos-arm/Packages.bz2",
        "dists/stable/main/binary-iphoneos-arm/Packages.gz",
        "dists/ios/1443.00/main/binary-iphoneos-arm/Packages.bz2",
        "dists/ios/1443.00/main/binary-iphoneos-arm/Packages.gz",
    ]

    /// Reads one repository's index.
    ///
    /// The packages' own `Filename` is relative to the repository's address,
    /// never to wherever the index turned out to live.
    private static func fetch(_ repo: URL) async throws -> [RepoPackage] {
        // Why a request never got an answer at all. When not one of the paths
        // was answered, this is the real reason — a refused connection, a
        // certificate, iOS blocking plain HTTP — and "no index" would hide it.
        var unanswered: Error?
        var answered = false
        for name in indexPaths {
            guard let url = URL(string: name, relativeTo: repo) else { continue }
            var request = URLRequest(url: url)
            request.setValue("Telesphoreo APT-HTTP/1.0.592", forHTTPHeaderField: "User-Agent")
            request.setValue("iPhone12,1", forHTTPHeaderField: "X-Machine")
            request.setValue("14.0", forHTTPHeaderField: "X-Firmware")
            request.timeoutInterval = 30

            let data: Data
            let response: URLResponse
            do { (data, response) = try await URLSession.shared.data(for: request) }
            catch { unanswered = error; continue }
            answered = true
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { continue }

            let text: Data
            if name.hasSuffix(".bz2") {
                guard let unpacked = Bzip2.decompress(data) else { continue }
                text = unpacked
            }
            else if name.hasSuffix(".gz") {
                guard let unpacked = Gzip.inflate(data) else { continue }
                text = unpacked
            }
            else {
                text = data
            }
            return parse(String(decoding: text, as: UTF8.self), repo: repo)
        }
        if !answered, let unanswered { throw unanswered }
        throw Failure.noIndex
    }

    enum Failure: LocalizedError {
        case noIndex

        var errorDescription: String? {
            L("нет читаемого указателя пакетов (Packages, Packages.gz, Packages.bz2)")
        }
    }

    /// Debian's control format: paragraphs of `Key: value`, separated by blank
    /// lines, with continuation lines indented.
    private static func parse(_ text: String, repo: URL) -> [RepoPackage] {
        var out: [RepoPackage] = []
        var field: [String: String] = [:]

        func flush() {
            defer { field = [:] }
            guard let id = field["package"], let path = field["filename"], !id.isEmpty else { return }
            out.append(RepoPackage(
                id: id,
                name: field["name"] ?? id,
                version: field["version"] ?? "",
                section: field["section"] ?? "",
                summary: field["description"]?.split(separator: "\n").first.map(String.init) ?? "",
                author: field["author"] ?? field["maintainer"] ?? "",
                path: path,
                size: Int64(field["size"] ?? "") ?? 0,
                repo: repo))
        }

        var lastKey = ""
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.isEmpty { flush(); lastKey = ""; continue }
            if line.hasPrefix(" ") || line.hasPrefix("\t") {
                if !lastKey.isEmpty { field[lastKey, default: ""] += "\n" + line.trimmingCharacters(in: .whitespaces) }
                continue
            }
            guard let colon = line.firstIndex(of: ":") else { continue }
            lastKey = line[line.startIndex..<colon].lowercased()
            field[lastKey] = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
        }
        flush()
        return out
    }
}

/// bzip2, borrowed from the system.
///
/// iOS ships libbz2 but does not declare it, so the one function needed is
/// looked up by hand. The older repositories — bingner's and BigBoss's among
/// them — publish their index only as `.bz2`, and without this they simply do
/// not open.
enum Bzip2 {
    private typealias Decompress = @convention(c) (
        UnsafeMutablePointer<CChar>?, UnsafeMutablePointer<UInt32>?,
        UnsafeMutablePointer<CChar>?, UInt32, Int32, Int32) -> Int32

    static func decompress(_ data: Data) -> Data? {
        guard data.count > 4, data[0] == 0x42, data[1] == 0x5A, data[2] == 0x68 else { return nil }
        guard let library = dlopen("/usr/lib/libbz2.1.0.dylib", RTLD_LAZY) ?? dlopen("libbz2.1.0.dylib", RTLD_LAZY),
              let symbol = dlsym(library, "BZ2_bzBuffToBuffDecompress")
        else { return nil }

        let call = unsafeBitCast(symbol, to: Decompress.self)
        // An index compresses well; when the guess is short bzip2 says so and
        // the room is doubled rather than given up on.
        var capacity = max(data.count * 12, 1 << 20)
        for _ in 0..<5 {
            var written = UInt32(capacity)
            var out = Data(count: capacity)
            let status: Int32 = out.withUnsafeMutableBytes { target in
                data.withUnsafeBytes { source in
                    call(target.baseAddress?.assumingMemoryBound(to: CChar.self), &written,
                         UnsafeMutablePointer(mutating: source.baseAddress?.assumingMemoryBound(to: CChar.self)),
                         UInt32(data.count), 0, 0)
                }
            }
            if status == 0 { return out.prefix(Int(written)) }
            guard status == -8 else { return nil }    // BZ_OUTBUFF_FULL
            capacity *= 2
        }
        return nil
    }
}

/// Just enough gzip to read a package index.
///
/// The Compression framework speaks raw deflate, not gzip, so the header comes
/// off by hand. Everything a repository sends is a plain stream — no
/// multi-member files, no dictionaries.
enum Gzip {
    static func inflate(_ data: Data) -> Data? {
        guard data.count > 18, data[0] == 0x1F, data[1] == 0x8B, data[2] == 0x08 else { return nil }

        let flags = data[3]
        var offset = 10
        if flags & 0x04 != 0 {    // extra field
            guard data.count > offset + 2 else { return nil }
            offset += 2 + Int(data[offset]) + Int(data[offset + 1]) << 8
        }
        if flags & 0x08 != 0 {    // original name
            while offset < data.count, data[offset] != 0 { offset += 1 }
            offset += 1
        }
        if flags & 0x10 != 0 {    // comment
            while offset < data.count, data[offset] != 0 { offset += 1 }
            offset += 1
        }
        if flags & 0x02 != 0 { offset += 2 }    // header checksum
        guard offset < data.count - 8 else { return nil }

        let body = data.subdata(in: offset..<(data.count - 8))
        // The last four bytes of a gzip stream are the uncompressed length, and
        // an index is far too big to guess at.
        let size = data.suffix(4).enumerated().reduce(Int(0)) { $0 | Int($1.element) << (8 * $1.offset) }
        return raw(body, hint: max(size, body.count * 8))
    }

    private static func raw(_ data: Data, hint: Int) -> Data? {
        var out = Data()
        let capacity = max(hint, 1 << 16)
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
        defer { buffer.deallocate() }

        let written = data.withUnsafeBytes { source -> Int in
            guard let base = source.bindMemory(to: UInt8.self).baseAddress else { return 0 }
            return compression_decode_buffer(buffer, capacity, base, data.count, nil, COMPRESSION_ZLIB)
        }
        guard written > 0 else { return nil }
        out.append(buffer, count: written)
        return out
    }
}
