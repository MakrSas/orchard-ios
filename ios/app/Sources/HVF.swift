import Foundation
import Darwin

/// Hardware virtualisation, on the iPads whose kernel still has it.
///
/// iPadOS up to 16.3.1 on M1 and M2 keeps Apple's hypervisor in the kernel —
/// 16.4 took it out — but ships no Hypervisor.framework, and the kernel lets in
/// only a process signed with `com.apple.private.hypervisor`, which a
/// sideloading certificate cannot give. So this needs a build that bundles a
/// reimplementation of the framework (utmapp/Hypervisor) and an install that
/// keeps private entitlements: TrollStore, or a jailbreak.
///
/// Under HVF the guest's cores run on the iPad's own, and the translator — and
/// with it the need for JIT — is gone. Everywhere else the emulator must not
/// even be asked: QEMU exits the process when HVF refuses it, and here the
/// process is the app. So the kernel is asked first, with the same call the
/// framework makes before anything else.
enum HVF {
    enum Probe: Equatable {
        /// The kernel answered, to this process: HVF can be used.
        case available
        /// This build carries no hypervisor framework.
        case notInBuild
        /// The kernel has the hypervisor but this process lacks the entitlement.
        case denied
        /// No hypervisor here — another device, or iPadOS 16.4 or later.
        case unsupported(UInt64)
    }

    private static let HV_CALL_VM_GET_CAPABILITIES: UInt32 = 0
    private static let HV_CALL_VM_DESTROY: UInt32 = 2
    private static let HV_DENIED: UInt64 = 0xfae9_4007

    #if os(macOS)
    /// Every Mac this runs on has the framework: it is part of the system.
    static var isInBuild: Bool { true }

    /// On a Mac the system's own framework answers, and the entitlement is the
    /// public `com.apple.security.hypervisor`. The check is the same as on the
    /// iPad and for the same reason: HV_DENIED appears only when a VM is
    /// created, and a refused VM would end the process with QEMU's exit(1).
    static let probe: Probe = {
        guard let handle = dlopen("/System/Library/Frameworks/Hypervisor.framework/Hypervisor", RTLD_NOW | RTLD_LOCAL),
              let createSymbol = dlsym(handle, "hv_vm_create"),
              let destroySymbol = dlsym(handle, "hv_vm_destroy")
        else { return .notInBuild }
        var supported = 0
        var length = MemoryLayout<Int>.size
        guard sysctlbyname("kern.hv_support", &supported, &length, nil, 0) == 0, supported != 0 else {
            return .unsupported(0)
        }
        let create = unsafeBitCast(createSymbol, to: (@convention(c) (OpaquePointer?) -> Int32).self)
        let destroy = unsafeBitCast(destroySymbol, to: (@convention(c) () -> Int32).self)
        let status = UInt64(UInt32(bitPattern: create(nil)))
        if status == HV_DENIED { return .denied }
        guard status == 0 else { return .unsupported(status) }
        _ = destroy()
        return .available
    }()
    #else
    /// Whether the framework is in the bundle at all, without calling it.
    static var isInBuild: Bool {
        guard let frameworks = Bundle.main.privateFrameworksPath else { return false }
        return FileManager.default.fileExists(atPath: frameworks + "/Hypervisor.framework/Hypervisor")
    }

    /// Asked once: the answer cannot change while the app runs.
    static let probe: Probe = {
        guard let frameworks = Bundle.main.privateFrameworksPath,
              let handle = dlopen(frameworks + "/Hypervisor.framework/Hypervisor", RTLD_NOW | RTLD_LOCAL),
              let trapSymbol = dlsym(handle, "hv_trap"),
              let createSymbol = dlsym(handle, "hv_vm_create")
        else { return .notInBuild }

        typealias TrapFn = @convention(c) (UInt32, UnsafeMutableRawPointer?) -> UInt64
        typealias CreateFn = @convention(c) (OpaquePointer?) -> Int32
        let trap = unsafeBitCast(trapSymbol, to: TrapFn.self)
        let create = unsafeBitCast(createSymbol, to: CreateFn.self)

        // Whether the kernel has a hypervisor at all. The kernel copies its
        // capabilities out into this, so it has to be real memory; they take a
        // few hundred bytes on every XNU that has them. A mach trap, not a BSD
        // system call: where there is no hypervisor the slot is an invalid trap
        // that returns an error, rather than a signal that would kill the app.
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: 4096, alignment: 16)
        buffer.initializeMemory(as: UInt8.self, repeating: 0, count: 4096)
        defer { buffer.deallocate() }
        let capabilities = trap(HV_CALL_VM_GET_CAPABILITIES, buffer)
        guard capabilities == 0 else { return .unsupported(capabilities) }

        // Whether this process may use it. The capabilities answer to anyone —
        // checked on a Mac, where they come back 0 with no entitlement at all;
        // the entitlement is checked when a VM is created. So one is created
        // and destroyed straight away, which leaves the process free to create
        // the emulator's own: also checked on a Mac, create-destroy-create.
        let status = UInt64(UInt32(bitPattern: create(nil)))
        if status == HV_DENIED { return .denied }
        guard status == 0 else { return .unsupported(status) }
        _ = trap(HV_CALL_VM_DESTROY, nil)
        return .available
    }()
    #endif

    /// One line for the log and the diagnostics screen.
    static var description: String {
        switch probe {
        case .available:
            return L("HVF: доступна")
        case .notInBuild:
            return L("HVF: в этой сборке нет")
        #if os(macOS)
        case .denied:
            return L("HVF: у приложения нет права com.apple.security.hypervisor")
        case .unsupported(let status):
            return L("HVF: этот Mac не поддерживает (0x%llx)", status)
        #else
        case .denied:
            return L("HVF: нет права com.apple.private.hypervisor — установите через TrollStore")
        case .unsupported(let status):
            return L("HVF: устройство не поддерживает (0x%llx) — нужен iPad на M1/M2 с iPadOS до 16.3.1", status)
        #endif
        }
    }
}
