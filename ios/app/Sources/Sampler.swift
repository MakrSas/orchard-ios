import Foundation
import Darwin

/// Finds the threads that are burning CPU and reports where they execute.
///
/// The program counter says what the time is spent on — `dladdr` names the
/// function when the address belongs to a loaded image, and returns nothing
/// when it points into the JIT buffer, which is the guest's own code.
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

    /// A profile of the emulator's busy threads: every thread that burned a
    /// fifth of a core or more is sampled every 10 ms for four seconds, and
    /// the samples are counted by function. The share is of all samples, so
    /// two busy vCPU threads each contribute half.
    ///
    /// The four samples this used to take named a function or two and could
    /// not say how much of the time went there; a few hundred can, which is
    /// what deciding where to speed the emulator up needs.
    static func report(_ completion: @escaping (String) -> Void) {
        Thread.detachNewThread {
            let busy = busyThreads()
            guard !busy.isEmpty else {
                return completion(L("Пробник: не удалось определить занятый поток"))
            }
            let tcgBefore = tcgStats()
            let started = Date()
            var bySymbol: [String: Int] = [:]
            // Per thread as well, for the ones that are not vCPUs: the graphics
            // drain has a profile of its own that the combined one drowns.
            var byThread: [thread_t: [String: Int]] = [:]
            var threadTotal: [thread_t: Int] = [:]
            var byGroup: [String: Int] = [:]
            var total = 0
            let deadline = Date().addingTimeInterval(4)
            while Date() < deadline {
                for thread in busy {
                    guard thread_suspend(thread) == KERN_SUCCESS else { continue }
                    let pc = programCounter(thread)
                    thread_resume(thread)
                    guard let pc else { continue }
                    let name = symbol(pc)
                    bySymbol[name, default: 0] += 1
                    byThread[thread, default: [:]][name, default: 0] += 1
                    threadTotal[thread, default: 0] += 1
                    byGroup[group(name), default: 0] += 1
                    total += 1
                }
                Thread.sleep(forTimeInterval: 0.01)
            }
            guard total > 0 else { return completion(L("Пробник: не удалось определить занятый поток")) }
            let percent = { (n: Int) in String(format: "%.1f%%", Double(n) * 100 / Double(total)) }
            var lines = [L("Профиль: %d замеров, потоков: %d", total, busy.count)]
            lines.append("  " + byGroup.sorted { $0.value > $1.value }
                .map { "\($0.key) \(percent($0.value))" }.joined(separator: " · "))
            for (name, n) in bySymbol.sorted(by: { $0.value > $1.value }).prefix(30) {
                lines.append("  \(percent(n))  \(name)")
            }
            for thread in busy {
                let label = threadName(thread)
                guard let counts = byThread[thread], let n = threadTotal[thread], n > 0,
                      !label.hasPrefix("CPU") else { continue }
                lines.append("  " + L("Поток %@: %d замеров", label, n))
                for (name, c) in counts.sorted(by: { $0.value > $1.value }).prefix(15) {
                    lines.append(String(format: "    %.1f%%  ", Double(c) * 100 / Double(n)) + name)
                }
            }
            if let before = tcgBefore, let after = tcgStats() {
                lines.append(tcgLine(before, after, Date().timeIntervalSince(started)))
            }
            completion(lines.joined(separator: "\n"))
        }
    }

    private typealias TCGStatsFn = @convention(c) (UnsafeMutablePointer<UInt64>, Int) -> Void

    /// The emulator's running totals (see orchard_tcg_stats in cpu-exec.c).
    private static func tcgStats() -> [UInt64]? {
        guard let sym = QemuBridge.shared.symbol("orchard_tcg_stats") else { return nil }
        var v = [UInt64](repeating: 0, count: 13)
        v.withUnsafeMutableBufferPointer { unsafeBitCast(sym, to: TCGStatsFn.self)($0.baseAddress!, 13) }
        return v
    }

    /// Per-second rates over the sampling window: how often the jump cache
    /// missed, and how often something emptied a cache or stopped both cores.
    private static func tcgLine(_ a: [UInt64], _ b: [UInt64], _ secs: Double) -> String {
        let d = zip(a, b).map { Double($1 &- $0) / max(secs, 0.1) }
        let miss = d[0] > 0 ? d[1] * 100 / d[0] : 0
        func k(_ x: Double) -> String { x >= 10000 ? String(format: "%.0fk", x / 1000) : String(format: "%.0f", x) }
        return "  TCG/s: " + [
            "lookups \(k(d[0])) (miss \(String(format: "%.1f%%", miss)): empty \(k(d[10])), other pc \(k(d[11])), same pc \(k(d[12])))",
            "translated \(k(d[2]))", "exceptions \(k(d[3]))",
            "jc flushes \(k(d[4]))", "invalidated \(k(d[5]))", "tb_flush \(k(d[6]))",
            "stop-all \(k(d[7]))", "tlb full \(k(d[8]))", "tlb part \(k(d[9]))",
        ].joined(separator: " · ")
    }

    /// A thread's pthread name, or its port number when it has none.
    private static func threadName(_ thread: thread_t) -> String {
        guard let p = pthread_from_mach_thread_np(thread) else { return "#\(thread)" }
        var buf = [CChar](repeating: 0, count: 64)
        pthread_getname_np(p, &buf, buf.count)
        let name = String(cString: buf)
        return name.isEmpty ? "#\(thread)" : name
    }

    /// The function a PC is in, or the translator's buffer when it is in none.
    private static func symbol(_ pc: UInt64) -> String {
        var info = Dl_info()
        guard dladdr(UnsafeRawPointer(bitPattern: UInt(pc)), &info) != 0,
              let name = info.dli_sname
        else { return L("[код гостя, переведённый JIT]") }
        return String(cString: name)
    }

    /// Where a function belongs, by the prefixes QEMU's own sources use.
    private static func group(_ name: String) -> String {
        if name.hasPrefix("[") { return L("код гостя") }
        let groups: [(String, [String])] = [
            ("PAC", ["pauth_", "helper_pac", "helper_aut", "helper_xpac", "qemu_xxhash", "aa64_va_parameters"]),
            (L("адреса/TLB"), ["get_phys_addr", "probe_access", "tlb_", "cputlb", "arm_ldq_ptw", "arm_ldl_ptw",
                               "S1_ptw", "ptw_", "helper_le_", "helper_be_", "helper_ld", "helper_st",
                               "get_S1prot", "regime_", "arm_cpu_tlb_fill", "do_ld", "do_st", "mmu_lookup"]),
            (L("поиск блоков"), ["helper_lookup_tb_ptr", "tb_lookup", "tb_htable", "qht_", "curr_cflags"]),
            (L("трансляция"), ["tcg_", "gen_", "translator_", "disas_", "aarch64_tr_", "tb_gen_code",
                               "sys_icache_invalidate", "tb_flush", "tb_invalidate", "tb_phys"]),
            (L("ожидание"), ["__psynch", "__semwait", "mach_msg", "__ulock", "qemu_cond", "qemu_mutex", "qemu_sem",
                             "pthread_", "__select", "poll", "kevent"]),
            (L("криптография гостя"), ["helper_crypto"]),
            ("reims-vgpu", ["reims_vgpu", "_ZN10reims_vgpu", "_ZN", "_R"]),
        ]
        for (label, prefixes) in groups where prefixes.contains(where: { name.hasPrefix($0) }) {
            return label
        }
        return L("прочее")
    }

    private static func busyThreads() -> [thread_t] {
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
        return before.compactMap { t, was in
            guard let now = cpuTime(t), now - was > 0.2 else { return nil }
            return t
        }
    }
}
