//  MetallibProbe.swift
//
//  Answers one question: does an iOS Metal driver accept a metallib that
//  Apple's *macOS* Metal compiler produced?
//
//  That is the open question for running the arm64 macOS guest's graphics on
//  an iPhone. The guest hands reims-vgpu already-compiled MTLB blobs; the host
//  passes them straight to `newLibraryWithData:`. On macOS that is a library
//  built for macOS being loaded by macOS. On iOS it is a library built for
//  macOS being loaded by iOS, and nothing but a device settles whether the
//  driver's loader minds.
//
//  The probe is deliberately layered, because "it didn't work" has several
//  distinguishable causes and they imply very different amounts of work:
//
//    1. load      — does `makeLibrary(data:)` return a library at all?
//    2. functions — does it expose the function names it should?
//    3. pipeline  — does a pipeline state actually build from one?
//
//  Only (3) means the shader would really run. A yes at (1) and a no at (3)
//  would say the loader is fine and the *code* is not, which is a different
//  problem from the one we are asking about.
//
//  It also runs each macOS sample a second time with the platform byte
//  rewritten to the iOS value (see `MetallibHeader`). If the original is
//  refused and the rewritten one loads, the fallback path is a two-byte edit
//  rather than a recompile, and that is worth knowing before spending a night
//  on it.

import Foundation
import Metal

// MARK: - Header

/// The first 24 bytes of an MTLB container, to the extent this probe needs them.
///
/// Recovered by surveying ~900 metallibs shipped on this machine (macOS system
/// frameworks, the iPhoneOS SDK, and the iOS simulator runtimes) and reading
/// off which bytes covary with the platform they were built for. It is an
/// inference from samples, **not** a documented Apple layout: treat a parse
/// that disagrees with a real file as the parse being wrong.
///
/// Byte 0x0B separated the three platforms cleanly with no overlap, and byte
/// 0x05's high bit tracked it:
///
///     platform        [0x0B]   [0x05] & 0x80
///     macOS           0x81     0x80
///     iOS (device)    0x82     0x00
///     iOS Simulator   0x87     0x00
struct MetallibHeader {
    static let magic = Data([0x4D, 0x54, 0x4C, 0x42])  // "MTLB"

    /// Offset of the byte that separated macOS / iOS / iOS-Simulator samples.
    static let platformByteOffset = 0x0B
    /// Offset of the byte whose high bit tracked the same split.
    static let flagsByteOffset = 0x05

    enum Platform: UInt8 {
        case macOS = 0x81
        case iOS = 0x82
        case iOSSimulator = 0x87

        var name: String {
            switch self {
            case .macOS: return "macOS"
            case .iOS: return "iOS"
            case .iOSSimulator: return "iOS-Simulator"
            }
        }
    }

    var platformByte: UInt8
    var flagsByte: UInt8
    var languageVersion: UInt8
    var osMajor: UInt16
    var osMinor: UInt16

    var platform: Platform? { Platform(rawValue: platformByte) }

    var summary: String {
        let p = platform?.name ?? String(format: "unknown(0x%02X)", platformByte)
        return "\(p) os=\(osMajor).\(osMinor) airLang=\(languageVersion)"
    }

    init?(_ data: Data) {
        guard data.count >= 24, data.prefix(4) == MetallibHeader.magic else { return nil }
        func u8(_ i: Int) -> UInt8 { data[data.startIndex + i] }
        func u16(_ i: Int) -> UInt16 { UInt16(u8(i)) | (UInt16(u8(i + 1)) << 8) }
        platformByte = u8(MetallibHeader.platformByteOffset)
        flagsByte = u8(MetallibHeader.flagsByteOffset)
        languageVersion = u8(0x08)
        osMajor = u16(0x0C)
        osMinor = u16(0x0E)
    }

    /// The same blob with its platform stamp rewritten to `target`.
    ///
    /// Nothing here recomputes a checksum, because no checksum field has been
    /// identified — if the container has one, this will be rejected and that
    /// rejection is itself the answer.
    static func restamped(_ data: Data, to target: Platform) -> Data {
        var out = data
        out[out.startIndex + platformByteOffset] = target.rawValue
        let flags = out[out.startIndex + flagsByteOffset]
        out[out.startIndex + flagsByteOffset] = target == .macOS ? (flags | 0x80) : (flags & 0x7F)
        return out
    }
}

/// Split a possibly-fat (`0xCAFEBABE`) metallib archive into its MTLB members.
///
/// Apple ships plenty of metallibs as fat archives; handing one of those
/// straight to `makeLibrary(data:)` would fail for a reason that has nothing
/// to do with the platform question.
func metallibSlices(_ data: Data) -> [Data] {
    if data.prefix(4) == MetallibHeader.magic { return [data] }
    guard data.count >= 8, data.prefix(4) == Data([0xCA, 0xFE, 0xBA, 0xBE]) else { return [] }
    func be32(_ i: Int) -> UInt32 {
        var v: UInt32 = 0
        for k in 0..<4 { v = (v << 8) | UInt32(data[data.startIndex + i + k]) }
        return v
    }
    let count = Int(be32(4))
    var out: [Data] = []
    for i in 0..<count {
        let entry = 8 + i * 20
        guard data.count >= entry + 20 else { break }
        let off = Int(be32(entry + 8)), size = Int(be32(entry + 12))
        guard off >= 0, size > 0, off + size <= data.count else { continue }
        let slice = data.subdata(in: (data.startIndex + off)..<(data.startIndex + off + size))
        if slice.prefix(4) == MetallibHeader.magic { out.append(slice) }
    }
    return out
}

// MARK: - Result

struct ProbeResult {
    var label: String
    var header: String
    var loaded: Bool
    var error: String?
    var functionNames: [String] = []
    var pipelineBuilt: Bool?
    var pipelineError: String?

    /// The line that answers the question this probe exists for.
    var verdict: String {
        if !loaded { return "REFUSED at load" }
        if let ok = pipelineBuilt { return ok ? "LOADED + PIPELINE OK" : "LOADED, pipeline FAILED" }
        return functionNames.isEmpty ? "LOADED, no functions" : "LOADED (\(functionNames.count) functions)"
    }

    var detail: String {
        var s = "\(label)\n  header:  \(header)\n  verdict: \(verdict)"
        if let e = error { s += "\n  error:   \(e)" }
        if !functionNames.isEmpty {
            s += "\n  funcs:   \(functionNames.prefix(6).joined(separator: ", "))"
            if functionNames.count > 6 { s += " (+\(functionNames.count - 6) more)" }
        }
        if let pe = pipelineError { s += "\n  pipeErr: \(pe)" }
        return s
    }
}

// MARK: - Probe

enum MetallibProbe {

    /// The kernel in this directory's `probe.metal`. A pipeline is only ever
    /// built from a function with this name — see `probe(_:label:data:)`.
    static let probeKernelName = "probe_add"

    /// Describe the GPU this is running on, including the two capabilities
    /// that were open questions for the port.
    static func describeDevice(_ device: MTLDevice) -> String {
        var lines = ["device:  \(device.name)"]
        var families: [String] = []
        let candidates: [(String, MTLGPUFamily)] = [
            ("apple6", .apple6), ("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9),
        ]
        for (name, fam) in candidates where device.supportsFamily(fam) { families.append(name) }
        lines.append("families: \(families.isEmpty ? "none of apple6-9" : families.joined(separator: " "))")
        // BC is the format family a macOS guest can send and pre-A14 silicon
        // lacks. Ask the device rather than infer it from the family table.
        lines.append("BC texture compression: \(device.supportsBCTextureCompression)")
        lines.append("argument buffers tier:  \(device.argumentBuffersSupport == .tier2 ? "2" : "1")")
        lines.append("max buffer length:      \(device.maxBufferLength)")
        lines.append("unified memory:         \(device.hasUnifiedMemory)")
        return lines.joined(separator: "\n")
    }

    /// Try one blob end to end.
    static func probe(_ device: MTLDevice, label: String, data: Data) -> ProbeResult {
        let header = MetallibHeader(data)?.summary ?? "not an MTLB container"
        var result = ProbeResult(label: label, header: header, loaded: false)

        let dispatchData = data.withUnsafeBytes { DispatchData(bytes: $0) }
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(data: dispatchData as __DispatchData)
        } catch {
            result.error = "\(error)"
            return result
        }
        result.loaded = true
        result.functionNames = library.functionNames

        // Loading is not running. Build a pipeline so a "yes" means the shader
        // would actually execute, not merely that the container parsed.
        //
        // Only for `probeKernelName` — the kernel in this directory's
        // probe.metal — and never for a harvested system library. Metal
        // answers questions about a function it does not like with
        // `abort()`, not with a thrown error: reading `functionType` on a
        // function out of Apple's own `default.metallib` kills the process
        // with "type is not a valid MTLFunctionType", which would take the
        // load result down with it. The load answer is the one this probe
        // exists for, so nothing is risked to get the pipeline bonus.
        guard library.functionNames.contains(MetallibProbe.probeKernelName),
              let function = library.makeFunction(name: MetallibProbe.probeKernelName)
        else { return result }
        do {
            _ = try device.makeComputePipelineState(function: function)
            result.pipelineBuilt = true
        } catch {
            result.pipelineBuilt = false
            result.pipelineError = "\(error)"
        }
        return result
    }

    /// Every bundled sample, each macOS one also retried restamped as iOS.
    static func runAll() -> (summary: String, results: [ProbeResult]) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return ("MTLCreateSystemDefaultDevice() returned nil — no Metal here.", [])
        }
        var results: [ProbeResult] = []

        // `.mtlbsample`, not `.metallib`: a fat metallib's 0xCAFEBABE is also
        // Mach-O's FAT_MAGIC, and sideloaders walking the bundle for binaries
        // to sign refuse the install over it. See the README.
        let urls = (Bundle.main.urls(forResourcesWithExtension: "mtlbsample", subdirectory: nil) ?? [])
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        if urls.isEmpty {
            return (describeDevice(device) + "\n\nNo .mtlbsample resources bundled. Run fetch-samples.sh.", [])
        }

        for url in urls {
            guard let raw = try? Data(contentsOf: url) else { continue }
            let name = url.lastPathComponent
            for (i, slice) in metallibSlices(raw).enumerated() {
                let tag = metallibSlices(raw).count > 1 ? "\(name)[\(i)]" : name
                let header = MetallibHeader(slice)
                results.append(probe(device, label: tag, data: slice))

                // Only a macOS-stamped blob has anything to learn from a
                // restamp; an iOS one is already the control.
                if header?.platform == .macOS {
                    let patched = MetallibHeader.restamped(slice, to: .iOS)
                    results.append(probe(device, label: "\(tag)  [platform byte -> iOS]", data: patched))
                }
            }
        }
        return (describeDevice(device), results)
    }

    /// One block of text, for a console or a screen.
    static func report() -> String {
        let (summary, results) = runAll()
        var out = ["=== Metal metallib cross-platform probe ===", "", summary, ""]
        for r in results { out.append(r.detail); out.append("") }
        if !results.isEmpty {
            let macOSOriginals = results.filter { !$0.label.contains("->") && $0.header.hasPrefix("macOS") }
            let restamped = results.filter { $0.label.contains("->") }
            out.append("--- bottom line ---")
            if let anyMac = macOSOriginals.first {
                out.append(anyMac.loaded
                    ? "A macOS-built metallib LOADS on this device as-is."
                    : "A macOS-built metallib is REFUSED on this device as-is.")
            }
            if let anyPatched = restamped.first {
                out.append(anyPatched.loaded
                    ? "Rewriting the platform byte to iOS makes it load."
                    : "Rewriting the platform byte to iOS does NOT make it load.")
            }
        }
        return out.joined(separator: "\n")
    }
}
