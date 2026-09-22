import Foundation
import Darwin

/// Finds the thread that is burning CPU and reports where it is executing.
///
/// One core pinned with nothing else happening has two very different causes:
/// the guest spinning inside translated code, or the emulator itself stuck in a
/// loop. The program counter separates them — `dladdr` names the function when
/// the address belongs to a loaded image, and returns nothing when it points
/// into a JIT buffer.
enum Sampler {
    private static func cpuTime(_ thread: thread_t) -> Double? {
        var info = thread_basic_info()
        var size = mach_msg_type_number_t(MemoryLayout<thread_basic_info>.size / MemoryLayout<integer_t>.size)
        let rc = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                thread_info(thread, thread_flavor_t(THREAD_BASIC_INFO), $0, &size)
            }
        }
        guard rc == KERN_SUCCESS else { return nil }
        return Double(info.user_time.seconds) + Double(info.user_time.microseconds) / 1e6
             + Double(info.system_time.seconds) + Double(info.system_time.microseconds) / 1e6
    }

    private static func programCounter(_ thread: thread_t) -> UInt64? {
        var state = arm_thread_state64_t()
        var count = mach_msg_type_number_t(MemoryLayout<arm_thread_state64_t>.size / MemoryLayout<natural_t>.size)
        let rc = withUnsafeMutablePointer(to: &state) {
            $0.withMemoryRebound(to: natural_t.self, capacity: Int(count)) {
                thread_get_state(thread, ARM_THREAD_STATE64, $0, &count)
            }
        }
        guard rc == KERN_SUCCESS else { return nil }
        return state.__pc
    }

    private static func describe(_ pc: UInt64) -> String {
        var info = Dl_info()
        guard dladdr(UnsafeRawPointer(bitPattern: UInt(pc)), &info) != 0 else {
            return String(format: L("0x%llx — вне загруженных образов (код, сгенерированный транслятором)"), pc)
        }
        let image = info.dli_fname.map { String(cString: $0) } ?? "?"
        let symbol = info.dli_sname.map { String(cString: $0) } ?? "?"
        let offset = pc - UInt64(UInt(bitPattern: info.dli_saddr))
        return String(format: "0x%llx — %@ +%llu  (%@)", pc, symbol, offset,
                      (image as NSString).lastPathComponent)
    }

    /// Samples the busiest thread a few times: a moving PC means it is running
    /// a loop, a fixed one means it is wedged on a single instruction.
    static func report(_ completion: @escaping (String) -> Void) {
        Thread.detachNewThread {
            guard let busiest = findBusiest() else {
                return completion(L("Пробник: не удалось определить занятый поток"))
            }

            var lines = [L("Где крутится поток (%d замеров):", busiest.samples)]
            var seen = Set<UInt64>()
            for pc in busiest.pcs {
                seen.insert(pc)
                lines.append("  " + describe(pc))
            }
            lines.append(seen.count == 1
                ? L("  адрес не меняется — поток стоит на одной инструкции")
                : L("  адрес меняется — поток исполняет цикл"))
            completion(lines.joined(separator: "\n"))
        }
    }

    private static func findBusiest() -> (samples: Int, pcs: [UInt64])? {
        func threads() -> [thread_t] {
            var list: thread_act_array_t?
            var count: mach_msg_type_number_t = 0
            guard task_threads(mach_task_self_, &list, &count) == KERN_SUCCESS, let list else { return [] }
            defer {
                vm_deallocate(mach_task_self_,
                              vm_address_t(UInt(bitPattern: list)),
                              vm_size_t(Int(count) * MemoryLayout<thread_t>.size))
            }
            return (0..<Int(count)).map { list[$0] }
        }

        let me = mach_thread_self()
        let before = threads().compactMap { t -> (thread_t, Double)? in
            guard t != me, let c = cpuTime(t) else { return nil }
            return (t, c)
        }
        Thread.sleep(forTimeInterval: 1)

        var best: (thread_t, Double)?
        for (t, was) in before {
            guard let now = cpuTime(t) else { continue }
            let delta = now - was
            if delta > (best?.1 ?? 0.2) { best = (t, delta) }
        }
        guard let (target, _) = best else { return nil }

        // Four snapshots, briefly suspending so the register state is coherent.
        var pcs: [UInt64] = []
        for _ in 0..<4 {
            if thread_suspend(target) == KERN_SUCCESS {
                if let pc = programCounter(target) { pcs.append(pc) }
                thread_resume(target)
            }
            Thread.sleep(forTimeInterval: 0.2)
        }
        return pcs.isEmpty ? nil : (pcs.count, pcs)
    }
}
