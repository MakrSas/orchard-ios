import Foundation
import SwiftUI

/// Picks where the VM is read from: a folder anywhere the Files picker
/// reaches, or back to the app's own `Documents/OrchardVM`.
struct VMFolderSection: View {
    var onChange: () -> Void = {}
    @State private var picking = false
    @State private var failure: String?
    /// Redraws the row after a change; `VMFolder` itself is not observable.
    @State private var shown = VMFolder.displayName

    var body: some View {
        Section {
            LabeledContent(L("Папка VM"), value: shown ?? "Documents/OrchardVM")
            Button(L("Выбрать папку…"), systemImage: "folder") { picking = true }
            if shown != nil {
                Button(L("Вернуть Documents/OrchardVM"), systemImage: "arrow.uturn.backward") {
                    VMFolder.reset()
                    shown = VMFolder.displayName
                    onChange()
                }
            }
            if let failure {
                Text(failure).foregroundStyle(.red)
            }
        } header: {
            Text(L("Где лежит VM"))
        } footer: {
            Text(L("С флешки VM грузится напрямую, пока она подключена. Оверлей пишется туда же, поэтому флешка должна быть доступна на запись; USB-SSD грузит заметно быстрее обычной флешки."))
        }
        .fileImporter(isPresented: $picking, allowedContentTypes: [.folder]) { result in
            switch result {
            case .success(let url):
                do {
                    try VMFolder.choose(url)
                    failure = nil
                } catch {
                    failure = L("Нет доступа к папке: %@", error.localizedDescription)
                }
            case .failure(let error):
                failure = error.localizedDescription
            }
            shown = VMFolder.displayName
            onChange()
        }
    }
}

/// Where the macOS VM's files are.
///
/// By default the app's own `Documents/OrchardVM`. A folder picked in the Files
/// picker can be used in place — a USB drive plugged into the phone, iCloud
/// Drive, another app's storage — so a 15 GB disk does not have to be copied
/// in first.
///
/// Such a folder lives outside the sandbox, and the picker's answer is only a
/// grant to open it. That grant is kept as a bookmark so it survives relaunch,
/// and opened once per process: the emulator runs inside this process and
/// opens the disk, overlay and firmware with plain `open()` as the machine
/// starts, and it writes the overlay for as long as it runs. So access is
/// started on first use and never stopped — the process ends before the
/// machine could need it to.
enum VMFolder {
    private static let bookmarkKey = "vmFolderBookmark"
    private static let lock = NSLock()
    /// The resolved folder with access started, once it has been asked for.
    private static var opened: URL?
    private static var resolved = false

    #if os(macOS)
    private static let bookmarkOptions: URL.BookmarkCreationOptions = .withSecurityScope
    private static let resolveOptions: URL.BookmarkResolutionOptions = .withSecurityScope
    #else
    private static let bookmarkOptions: URL.BookmarkCreationOptions = []
    private static let resolveOptions: URL.BookmarkResolutionOptions = []
    #endif

    /// The folder that was picked, opened for this process; nil if none was,
    /// or if it can no longer be reached (the drive is not plugged in).
    static var chosen: URL? {
        lock.lock()
        defer { lock.unlock() }
        if !resolved {
            resolved = true
            opened = open()
        }
        return opened
    }

    /// The name to show for where the VM is read from.
    static var displayName: String? {
        guard UserDefaults.standard.data(forKey: bookmarkKey) != nil else { return nil }
        return chosen?.lastPathComponent ?? L("недоступна")
    }

    /// Takes the folder the picker returned and keeps it for next time.
    static func choose(_ url: URL) throws {
        guard url.startAccessingSecurityScopedResource() else {
            throw CocoaError(.fileReadNoPermission)
        }
        let bookmark = try url.bookmarkData(options: bookmarkOptions,
                                            includingResourceValuesForKeys: nil,
                                            relativeTo: nil)
        UserDefaults.standard.set(bookmark, forKey: bookmarkKey)
        lock.lock()
        opened?.stopAccessingSecurityScopedResource()
        opened = url
        resolved = true
        lock.unlock()
        LogCapture.shared.note(L("Папка VM: %@", url.path))
    }

    /// Back to `Documents/OrchardVM`.
    static func reset() {
        UserDefaults.standard.removeObject(forKey: bookmarkKey)
        lock.lock()
        opened?.stopAccessingSecurityScopedResource()
        opened = nil
        resolved = true
        lock.unlock()
    }

    private static func open() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data, options: resolveOptions,
                                 relativeTo: nil, bookmarkDataIsStale: &stale)
        else {
            LogCapture.shared.note(L("Папка VM недоступна: закладка не разрешается"))
            return nil
        }
        guard url.startAccessingSecurityScopedResource() else {
            LogCapture.shared.note(L("Папка VM недоступна: нет доступа к %@", url.path))
            return nil
        }
        // A stale bookmark still resolved; renew it while access is open, so a
        // moved or remounted folder keeps being found.
        if stale, let renewed = try? url.bookmarkData(options: bookmarkOptions,
                                                      includingResourceValuesForKeys: nil,
                                                      relativeTo: nil) {
            UserDefaults.standard.set(renewed, forKey: bookmarkKey)
        }
        return url
    }
}
