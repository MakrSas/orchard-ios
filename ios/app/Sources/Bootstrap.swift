import Foundation

/// Prepares the app's Documents folder on first launch.
///
/// iOS hides an app's folder in the Files app while its Documents directory is
/// empty, even with UIFileSharingEnabled set. Creating the directory tree (and
/// leaving a note in it) makes the folder appear, and gives the guest images an
/// obvious place to land.
enum Bootstrap {
    static func prepareDocuments() {
        let fm = FileManager.default
        let documents = VMConfig.documents

        // iPhoneData only for now: it is where the scratch/xfer namespace
        // below lives. The rest of the iPhone restore tree (SEP firmware,
        // kernelcache, DeviceTree…) doesn't apply to a macOS guest and isn't
        // created here anymore.
        let url = documents.appendingPathComponent("iPhoneData")
        if !fm.fileExists(atPath: url.path) {
            try? fm.createDirectory(at: url, withIntermediateDirectories: true)
        }

        // The scratch namespace for fast transfers. Made here so it exists
        // before the emulator's command line is built; the machine only picks
        // it up when the file is already there.
        VMConfig.ensureTransferImage()

        let readme = documents.appendingPathComponent(L("КУДА КЛАСТЬ ФАЙЛЫ.txt"))
        if !fm.fileExists(atPath: readme.path) {
            try? note.data(using: .utf8)?.write(to: readme)
        }
    }

    private static let note = """
    Orchard — файлы гостевой системы
    =================================

    Бэкенд (сборка QEMU под iOS с поддержкой vmapple) ещё не готов, так что
    класть сюда пока нечего — этот файл появляется заранее, только чтобы
    папка приложения была видна в «Файлах».

    Когда бэкенд будет готов, сюда лягут те же три файла, что и на хосте
    (см. orchard/README.md): disk.raw, aux.img, config.json — образ macOS,
    его NVRAM и ECID, а не прошивка SEP/kernelcache, как для iPhone.

    JIT включайте через StikDebug ДО запуска машины — без него транслятор не
    сможет сделать буфер трансляций исполняемым.

    Этот файл можно удалить.
    """
}
