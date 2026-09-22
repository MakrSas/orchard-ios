import Foundation
import Darwin

/// Makes the process able to allocate executable memory.
///
/// TCG needs a JIT buffer, and on iOS `mmap(MAP_JIT)` is refused unless the
/// process carries the `dynamic-codesigning` entitlement or the kernel considers
/// it debugged. There are two ways to be considered debugged:
///
///  * an external debugger is attached — this is what StikDebug does;
///  * the process traces itself with `PT_TRACE_ME`, which the kernel treats the
///    same way, since debugging implies the right to patch code.
///
/// The self-trace is attempted only when no debugger is present: calling it
/// while one is attached fails and would leave a misleading error behind.
enum JIT {
    private static let PT_TRACE_ME: Int32 = 0
    private static let MAP_JIT_FLAG: Int32 = 0x800

    enum Availability: Equatable {
        case available(via: String)
        case unavailable(String)

        var isAvailable: Bool {
            if case .available = self { return true }
            return false
        }
    }

    private(set) static var status: Availability = .unavailable(L("не проверялось"))

    /// True when the translator has to use two mappings of the same memory —
    /// one writable, one executable — because MAP_JIT is refused. This is the
    /// normal situation for a sideloaded app whose JIT comes from a debugger
    /// rather than from the dynamic-codesigning entitlement.
    private(set) static var needsSplitWX = true

    /// True when a debugger (StikDebug, Xcode, …) is attached to us.
    static func isBeingDebugged() -> Bool {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        let rc = sysctl(&mib, u_int(mib.count), &info, &size, nil, 0)
        guard rc == 0 else { return false }
        return (info.kp_proc.p_flag & P_TRACED) != 0
    }

    /// `ptrace` is not declared in the iOS SDK, so reach it through the dynamic
    /// linker rather than hard-coding a syscall number.
    private static func selfTrace() -> Bool {
        typealias PtraceFn = @convention(c) (Int32, pid_t, UnsafeMutableRawPointer?, Int32) -> Int32
        guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "ptrace") else { return false }
        let ptrace = unsafeBitCast(sym, to: PtraceFn.self)
        return ptrace(PT_TRACE_ME, 0, nil, 0) >= 0
    }

    /// Allocates one page the same way TCG will, to find out for certain.
    private static func canAllocateExecutable() -> Bool {
        let size = Int(getpagesize())
        let addr = mmap(nil, size, PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT_FLAG, -1, 0)
        guard addr != MAP_FAILED else { return false }
        munmap(addr, size)
        return true
    }

    /// What a debugger-enabled process is actually allowed: write to a page,
    /// then hand it execute permission. QEMU builds its mirror mapping on top
    /// of exactly this.
    private static func canMirrorMap() -> Bool {
        let size = Int(getpagesize())
        let addr = mmap(nil, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        guard addr != MAP_FAILED, let page = addr else { return false }
        defer { munmap(page, size) }
        return mprotect(page, size, PROT_READ | PROT_EXEC) == 0
    }

    /// Writes a real function into a page and calls it.
    ///
    /// Allocation succeeding proves nothing: arm64 enforces W^X in hardware, so
    /// a mapping can accept PROT_EXEC and still fault on the first instruction.
    /// The only trustworthy answer comes from executing.
    private static func canExecute(_ name: String,
                                   prot: Int32,
                                   flags: Int32,
                                   thenProtect: Int32? = nil) -> String {
        typealias Fn = @convention(c) () -> Int32
        let size = Int(getpagesize())

        let addr = mmap(nil, size, prot, flags, -1, 0)
        guard addr != MAP_FAILED, let page = addr else {
            return L("%@: mmap отказал (%@)", name, String(cString: strerror(errno)))
        }
        defer { munmap(page, size) }

        // mov w0, #42 ; ret
        var code: [UInt32] = [0x5280_0540, 0xD65F_03C0]
        memcpy(page, &code, code.count * 4)

        if let target = thenProtect, mprotect(page, size, target) != 0 {
            return L("%@: mprotect отказал (%@)", name, String(cString: strerror(errno)))
        }

        // Instructions written as data are invisible to the instruction cache.
        if let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "sys_icache_invalidate") {
            typealias Flush = @convention(c) (UnsafeMutableRawPointer, Int) -> Void
            unsafeBitCast(sym, to: Flush.self)(page, size)
        }

        let fn = unsafeBitCast(page, to: Fn.self)
        let result = fn()
        return result == 42 ? L("%@: ВЫПОЛНЯЕТСЯ", name) : L("%@: вернул %d, ожидалось 42", name, Int(result))
    }

    /// Tries every way of getting executable memory and reports which ones the
    /// kernel allows. MAP_JIT and plain RWX are granted by different rules —
    /// the entitlement versus being debugged — so knowing which one works tells
    /// us how the emulator should allocate its buffer.
    /// `includeExecution` actually runs generated code. That is the decisive
    /// test, but a page that cannot be executed takes the process down with it,
    /// so it stays behind an explicit request.
    static func diagnose(includeExecution: Bool = false) -> String {
        let size = Int(getpagesize())
        var lines: [String] = []

        lines.append(L("отлаживается: %@", isBeingDebugged() ? L("да") : L("нет")))

        func attempt(_ name: String, prot: Int32, flags: Int32, thenExec: Bool = false) {
            let addr = mmap(nil, size, prot, flags, -1, 0)
            if addr == MAP_FAILED {
                lines.append(L("%@: нет (%@)", name, String(cString: strerror(errno))))
                return
            }
            if thenExec {
                let rc = mprotect(addr, size, PROT_READ | PROT_WRITE | PROT_EXEC)
                lines.append(L("%@: %@", name, rc == 0 ? L("да")
                                        : L("нет (mprotect: %@)", String(cString: strerror(errno)))))
            } else {
                lines.append(L("%@: да", name))
            }
            munmap(addr, size)
        }

        let anon = MAP_PRIVATE | MAP_ANONYMOUS
        attempt("MAP_JIT + PROT_NONE", prot: PROT_NONE, flags: anon | MAP_JIT_FLAG)
        attempt("MAP_JIT + RW", prot: PROT_READ | PROT_WRITE, flags: anon | MAP_JIT_FLAG)
        attempt(L("RWX без MAP_JIT"), prot: PROT_READ | PROT_WRITE | PROT_EXEC, flags: anon)
        attempt("RW → mprotect RWX", prot: PROT_READ | PROT_WRITE, flags: anon, thenExec: true)

        if includeExecution {
            // Only the RW→RX (mirrored/split-WX) case is tried here. The other
            // theoretical case — mmap'ing PROT_READ|WRITE|EXEC together without
            // MAP_JIT — is not a path the translator ever takes (QEMU either
            // uses MAP_JIT with write-protect toggling, or this split-WX
            // mprotect toggling; never a single combined-prot mapping), and
            // attempting to execute on it is exactly the case the doc comment
            // above warns about: on a device with a tracer attached (StikDebug),
            // the resulting hardware fault stops the thread for the tracer
            // instead of crashing the process, which looks like the app
            // hanging forever rather than reporting a result. Confirmed by
            // hitting that hang while testing this file on-device.
            lines.append(canExecute(L("исполнение RW→RX"), prot: PROT_READ | PROT_WRITE, flags: anon,
                                    thenProtect: PROT_READ | PROT_EXEC))
        }

        let report = lines.joined(separator: "\n  ")
        LogCapture.shared.note(L("Проверка исполняемой памяти:\n  ") + report)
        return report
    }

    private static var selfTraced = false

    /// Only on iOS. A Mac gives MAP_JIT to any process without the hardened
    /// runtime, and PT_TRACE_ME there hands the process to its parent as a
    /// tracee, which turns its signals into stops.
    #if os(iOS)
    private static let maySelfTrace = true
    #else
    private static let maySelfTrace = false
    #endif

    /// Re-runs the probe. Cheap, and safe to call repeatedly.
    ///
    /// It has to be repeatable: StikDebug attaches *after* the app is already
    /// running, so a single check at launch would report a stale "no" for the
    /// rest of the session.
    @discardableResult
    static func prepare() -> Availability {
        let previous = status

        // MAP_JIT is the only meaningful test. It is granted for exactly two
        // reasons — the JIT entitlement, or the process being debugged — which
        // is precisely the condition the translator needs. A plain RWX mapping
        // is NOT a substitute: mmap accepts it and the hardware still refuses
        // to execute the page, which looks like a hang rather than an error.
        if canAllocateExecutable() {
            // The entitlement route: the translator can keep one RWX buffer.
            needsSplitWX = false
            status = .available(via: isBeingDebugged() ? L("MAP_JIT, отладчик") : "MAP_JIT")
        } else if canMirrorMap() {
            // The debugger route: MAP_JIT stays refused, but permissions can be
            // changed, which is all the mirror mapping needs.
            needsSplitWX = true
            status = .available(via: L("зеркальное отображение (split-wx)"))
        } else if !selfTraced, Self.maySelfTrace, selfTrace() {
            selfTraced = true
            if canAllocateExecutable() {
                needsSplitWX = false
                status = .available(via: "ptrace(PT_TRACE_ME)")
            } else if canMirrorMap() {
                needsSplitWX = true
                status = .available(via: L("ptrace + зеркальное отображение"))
            } else {
                status = .unavailable(L("самотрассировка прошла, но исполняемой памяти всё равно нет"))
            }
        } else {
            status = .unavailable(L("включите JIT (StikDebug) и вернитесь в приложение"))
        }

        if status != previous {
            switch status {
            case .available(let how):    LogCapture.shared.note(L("JIT: доступен (%@)", how))
            case .unavailable(let why):  LogCapture.shared.note(L("JIT: НЕДОСТУПЕН — %@", why))
            }
        }
        return status
    }
}
