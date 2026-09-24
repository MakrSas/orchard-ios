import Foundation

/// The guest's disk as the emulator sees it: whether macOS flushes its write
/// cache, and whether any I/O failed. A corrupted overlay (the guest stuck in
/// Recovery with no disk) leaves no other trace, so these go in the log
/// whenever they change. Also flushes the block layer on request, for when
/// the app may be ended without QEMU closing its files.
final class DiskWatch {
    static let shared = DiskWatch()

    private typealias StatsFn = @convention(c) (UnsafeMutablePointer<UInt64>, UnsafeMutablePointer<UInt64>,
                                                UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>) -> Void
    private typealias FlushFn = @convention(c) () -> Void
    private var timer: Timer?
    private var last: (UInt64, UInt64, Int32) = (0, 0, -2)

    func start() {
        guard timer == nil, QemuBridge.shared.symbol("orchard_blk_stats") != nil else { return }
        let t = Timer(timeInterval: 30, repeats: true) { [weak self] _ in self?.report() }
        RunLoop.main.add(t, forMode: .common)
        timer = t
    }

    private func report() {
        guard let sym = QemuBridge.shared.symbol("orchard_blk_stats") else { return }
        var flushes: UInt64 = 0, errors: UInt64 = 0
        var lastErrno: Int32 = 0, wce: Int32 = -1
        unsafeBitCast(sym, to: StatsFn.self)(&flushes, &errors, &lastErrno, &wce)
        guard (flushes, errors, wce) != last else { return }
        let firstReport = last.2 == -2
        let newFlushes = flushes - (firstReport ? 0 : last.0)
        last = (flushes, errors, wce)
        let cache = wce == 1 ? L("с кешем записи (гость сам сбрасывает)")
            : wce == 0 ? L("без кеша записи (каждая запись сразу на диск)") : L("кеш записи ещё не согласован")
        var line = L("Диск: %@; сбросов кеша от гостя: +%@ (всего %@)", cache, String(newFlushes), String(flushes))
        if errors > 0 {
            line += L("; ОШИБКИ ввода-вывода: %@, последняя errno %@ (%@)", String(errors), String(lastErrno),
                      String(cString: strerror(lastErrno)))
        }
        LogCapture.shared.note(line)
    }

    /// Writes qcow2's cached tables and the data under them to the files.
    func flush(reason: String) {
        guard let sym = QemuBridge.shared.symbol("orchard_block_flush") else { return }
        let started = Date()
        unsafeBitCast(sym, to: FlushFn.self)()
        LogCapture.shared.note(L("Диск: записал кеш в файлы (%@), %@ мс", reason,
                                 String(Int(Date().timeIntervalSince(started) * 1000))))
    }
}
