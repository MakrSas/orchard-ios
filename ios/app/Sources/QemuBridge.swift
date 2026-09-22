import Foundation

/// Loads the Inferno emulator library and drives its lifecycle.
///
/// iOS cannot fork/exec, so the emulator runs inside this process: the dylib is
/// dlopen'd and `qemu_init` / `qemu_main_loop` / `qemu_cleanup` are called on a
/// dedicated thread. `qemu_init` returns with the BQL held and `qemu_main_loop`
/// expects to be entered that way, so both must run on the same thread.
final class QemuBridge {
    typealias InitFn = @convention(c) (Int32, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> Void
    typealias MainLoopFn = @convention(c) () -> Int32
    typealias CleanupFn = @convention(c) (Int32) -> Void

    enum State: Equatable {
        case idle
        case running
        case stopped(Int32)
        case failed(String)
    }

    static let shared = QemuBridge()

    private(set) var state: State = .idle
    private var handle: UnsafeMutableRawPointer?
    private var thread: Thread?
    /// Called on the emulator's own thread once the machine exists and while
    /// its lock is still held — the only moment a display listener may be
    /// registered from outside.
    var afterInit: (() -> Void)?

    /// Emulator lifecycle changes, delivered on the main queue.
    var onStateChange: ((State) -> Void)?

    /// Switches the emulator reads from the environment rather than from the
    /// command line. They are not machine properties — they choose between two
    /// ways of doing the same thing, and exist so the two can be compared on
    /// the device, where the frame rate is the only honest measurement. Set
    /// before `start`; read once, as the machine comes up.
    var environment: [String: String] = [:]

    private func set(_ new: State) {
        DispatchQueue.main.async {
            self.state = new
            self.onStateChange?(new)
        }
    }

    /// The dylib ships inside the app bundle's Frameworks directory.
    private static func libraryPath() -> String? {
        let names = ["libqemu-aarch64-softmmu.dylib"]
        var candidates: [String] = []
        if let frameworks = Bundle.main.privateFrameworksPath {
            candidates += names.map { frameworks + "/" + $0 }
        }
        candidates += names.map { Bundle.main.bundlePath + "/" + $0 }
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }

    /// Looks up one of the emulator's exported entry points. Available only
    /// once the library is loaded, which is to say once the machine has started.
    func symbol(_ name: String) -> UnsafeMutableRawPointer? {
        guard let handle else { return nil }
        return dlsym(handle, name)
    }

    func start(arguments: [String]) {
        guard state != .running else { return }

        guard let path = QemuBridge.libraryPath() else {
            set(.failed("libqemu-aarch64-softmmu.dylib not found in the app bundle"))
            return
        }

        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = String(cString: dlerror() ?? UnsafeMutablePointer(mutating: ("unknown" as NSString).utf8String!))
            set(.failed("dlopen failed: \(reason)"))
            return
        }
        self.handle = handle

        guard let initSym = dlsym(handle, "qemu_init"),
              let loopSym = dlsym(handle, "qemu_main_loop"),
              let cleanupSym = dlsym(handle, "qemu_cleanup")
        else {
            set(.failed("the library does not export qemu_init/qemu_main_loop/qemu_cleanup"))
            return
        }

        LogCapture.shared.note(L("Библиотека загружена: %@", path))
        LogCapture.shared.note(L("Аргументы:\n  ") + arguments.joined(separator: " "))

        let qemuInit = unsafeBitCast(initSym, to: InitFn.self)
        let qemuMainLoop = unsafeBitCast(loopSym, to: MainLoopFn.self)
        let qemuCleanup = unsafeBitCast(cleanupSym, to: CleanupFn.self)

        // AF_UNIX paths are capped at 104 bytes, which an app container path
        // exceeds on its own. Both ends of the USB link therefore use a bare
        // file name resolved against this directory.
        if !FileManager.default.changeCurrentDirectoryPath(VMConfig.socketDirectory) {
            LogCapture.shared.note(L("Не удалось перейти в %@ — USB-сокет может не подняться",
                                 VMConfig.socketDirectory))
        }

        let argv = arguments
        let env = environment
        let thread = Thread {
            for (name, value) in env {
                setenv(name, value, 1)
                LogCapture.shared.note("\(name)=\(value)")
            }
            // Build a C argv that stays alive for the whole run.
            var cargs: [UnsafeMutablePointer<CChar>?] = argv.map { strdup($0) }
            cargs.append(nil)

            self.set(.running)
            cargs.withUnsafeMutableBufferPointer { buf in
                qemuInit(Int32(argv.count), buf.baseAddress)
            }
            // qemu_init returns holding the big lock and qemu_main_loop expects
            // to be entered that way, so this gap is the one safe place to
            // attach to the machine's display.
            self.afterInit?()
            let status = qemuMainLoop()
            qemuCleanup(status)
            // Let go of the big lock, as QEMU's own main does after cleanup.
            // qemu_init handed it to this thread, and a thread that ends holding
            // it leaves it held for good: on a Mac, quitting then calls exit(),
            // QEMU's exit notifiers ask for the lock first thing, and the app
            // hangs instead of closing.
            if let unlock = dlsym(handle, "bql_unlock"),
               let locked = dlsym(handle, "bql_locked"),
               unsafeBitCast(locked, to: (@convention(c) () -> Bool).self)() {
                unsafeBitCast(unlock, to: (@convention(c) () -> Void).self)()
            }
            self.set(.stopped(status))
        }
        // The translation buffer and device emulation want room to breathe.
        thread.stackSize = 4 * 1024 * 1024
        thread.name = "inferno.qemu"
        thread.qualityOfService = .userInitiated
        self.thread = thread
        thread.start()
    }
}
