import Foundation
import Darwin
#if os(iOS)
import os
#endif

/// Reports what every thread in the process is doing.
///
/// When the emulator stops responding there are two very different causes —
/// vCPUs burning CPU with a stuck main loop, or nothing running at all — and
/// from the outside they look identical. Per-thread CPU time tells them apart.
enum Threads {
    struct Snapshot {
        var total: Int
        var running: Int
        var cpuSeconds: Double
    }

    static func snapshot() -> Snapshot {
        var list: thread_act_array_t?
        var count: mach_msg_type_number_t = 0
        guard task_threads(mach_task_self_, &list, &count) == KERN_SUCCESS, let list else {
            return Snapshot(total: 0, running: 0, cpuSeconds: 0)
        }
        defer {
            vm_deallocate(mach_task_self_,
                          vm_address_t(UInt(bitPattern: list)),
                          vm_size_t(Int(count) * MemoryLayout<thread_t>.size))
        }

        var running = 0
        var seconds = 0.0
        for i in 0..<Int(count) {
            var info = thread_basic_info()
            // THREAD_BASIC_INFO_COUNT is a macro the Swift importer drops.
            var size = mach_msg_type_number_t(MemoryLayout<thread_basic_info>.size / MemoryLayout<integer_t>.size)
            let rc = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                    thread_info(list[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &size)
                }
            }
            guard rc == KERN_SUCCESS else { continue }
            if info.run_state == TH_STATE_RUNNING { running += 1 }
            seconds += Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1e6
            seconds += Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1e6
        }
        return Snapshot(total: Int(count), running: running, cpuSeconds: seconds)
    }

    /// Two samples a few seconds apart: the difference is what matters.
    static func report(over interval: TimeInterval = 3, _ completion: @escaping (String) -> Void) {
        Thread.detachNewThread {
            let first = snapshot()
            Thread.sleep(forTimeInterval: interval)
            let second = snapshot()

            let burned = second.cpuSeconds - first.cpuSeconds
            let cores = burned / interval
            let console = consoleSize()

            let verdict: String
            if cores < 0.05 {
                verdict = L("процессор простаивает — гость не исполняется")
            } else {
                verdict = String(format: L("загружено ~%.1f ядра — гость исполняется"), cores)
            }

            let threads = L("Потоки: %d, из них выполняются %d", second.total, second.running)
            let spent = L("за %d с сожжено %@ с процессорного времени", Int(interval), String(format: "%.2f", burned))
            completion("""
            \(threads)
              \(spent)
              \(verdict)
              \(L("память приложения: %@", Threads.footprint()))
              guest-console.log: \(console)
            """)
        }
    }

    /// What the system counts against this app.
    ///
    /// The number that matters on a phone: a restore moves gigabytes through
    /// the guest's disk, and when this climbs into the app's limit the system
    /// kills the process outright — from outside it looks like the transfer
    /// simply stopped.
    private static func footprint() -> String {
        guard let bytes = footprintBytes() else { return L("неизвестно") }
        return memoryLine(footprint: bytes)
    }

    /// The footprint and, on iOS, how far it is from the limit the system will
    /// kill the process at. The limit is the number that decides whether the
    /// guest's memory fits, and it is not fixed: the increased-memory-limit
    /// entitlement raises it only when whatever signed the app kept it.
    static func memoryLine(footprint bytes: UInt64) -> String {
        let used = String(format: "%.0f МБ", Double(bytes) / 1_048_576)
        #if os(iOS)
        let left = UInt64(os_proc_available_memory())
        return L("%@, до потолка %@ (потолок ~%@)", used,
                 String(format: "%.0f МБ", Double(left) / 1_048_576),
                 String(format: "%.0f МБ", Double(bytes + left) / 1_048_576))
        #else
        return used
        #endif
    }

    /// Where the footprint is: the regions holding the most, with their size
    /// and VM tag, largest first. The guest's RAM is the one region as big as
    /// the machine's memory, the translation buffer the one as big as its
    /// setting; what is left is the emulator, the graphics and the app.
    ///
    /// Counted as dirty plus compressed pages, which is what the footprint
    /// charges for private memory.
    static func regionBreakdown(top: Int = 6) -> String {
        struct Region { var size: UInt64; var held: UInt64; var tag: UInt32 }
        var regions: [Region] = []
        var address: vm_address_t = 0
        var depth: natural_t = 0
        let page = UInt64(vm_page_size)
        while true {
            var size: vm_size_t = 0
            var info = vm_region_submap_info_data_64_t()
            var count = mach_msg_type_number_t(MemoryLayout<vm_region_submap_info_data_64_t>.size
                                               / MemoryLayout<natural_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: Int32.self, capacity: Int(count)) {
                    vm_region_recurse_64(mach_task_self_, &address, &size, &depth, $0, &count)
                }
            }
            guard kr == KERN_SUCCESS else { break }
            if info.is_submap != 0 { depth += 1; continue }
            let held = (UInt64(info.pages_dirtied) + UInt64(info.pages_swapped_out)) * page
            if held > 0 { regions.append(Region(size: UInt64(size), held: held, tag: info.user_tag)) }
            address += size
        }
        let mb = { (bytes: UInt64) in String(format: "%.0f", Double(bytes) / 1_048_576) }
        var byTag: [UInt32: UInt64] = [:]
        for r in regions { byTag[r.tag, default: 0] += r.held }
        let biggest = regions.sorted { $0.held > $1.held }.prefix(top)
            .map { "\(mb($0.held))/\(mb($0.size)) МБ tag \($0.tag)" }
        let tags = byTag.sorted { $0.value > $1.value }.prefix(top)
            .map { "tag \($0.key): \(mb($0.value))" }
        return L("области (занято/размер): %@; по тегам, МБ: %@",
                 biggest.joined(separator: ", "), tags.joined(separator: ", "))
    }

    /// Whether the entitlement that raises the memory limit reached this
    /// process. The app asks for it; whatever signed the install decides.
    static func increasedMemoryLimitGranted() -> Bool? {
        typealias CreateFn = @convention(c) (CFAllocator?) -> Unmanaged<AnyObject>?
        typealias CopyFn = @convention(c) (AnyObject, CFString, UnsafeMutablePointer<Unmanaged<CFError>?>?) -> Unmanaged<AnyObject>?
        guard let handle = dlopen("/System/Library/Frameworks/Security.framework/Security", RTLD_NOW),
              let create = dlsym(handle, "SecTaskCreateFromSelf"),
              let copy = dlsym(handle, "SecTaskCopyValueForEntitlement"),
              let task = unsafeBitCast(create, to: CreateFn.self)(nil)?.takeRetainedValue()
        else { return nil }
        let value = unsafeBitCast(copy, to: CopyFn.self)(
            task, "com.apple.developer.kernel.increased-memory-limit" as CFString, nil)?.takeRetainedValue()
        return (value as? Bool) ?? false
    }

    /// What the system counts against this app, in bytes.
    static func footprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                           / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return info.phys_footprint
    }

    private static func consoleSize() -> String {
        let path = VMConfig.guestConsoleLog.path
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int
        else { return L("файла нет") }
        return L("%d байт", size)
    }
}
