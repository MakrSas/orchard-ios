//  Same probe, driven from the command line on the host.
//
//  It exists so the probe can be checked before a phone is involved, and the
//  first thing it turned up is worth stating: on an M1 Mac, an *iOS*-stamped
//  metallib loads with every function intact. macOS's loader does not enforce
//  the platform byte. Restamping in either direction also loads, so the
//  container carries no checksum over that byte.
//
//  That is evidence about the macOS loader and not about the iOS one — the
//  direction we need is the opposite one, and only a device answers it. But
//  it does mean a refusal on iOS would be a deliberate asymmetry rather than
//  a format incompatibility. Run with no arguments to use Samples/.

import Foundation
import Metal

var paths = Array(CommandLine.arguments.dropFirst())
if paths.isEmpty {
    let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("Samples")
    paths = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
        .filter { $0.hasSuffix(".mtlbsample") }
        .sorted()
        .map { dir.appendingPathComponent($0).path }
}

guard let device = MTLCreateSystemDefaultDevice() else {
    print("no Metal device"); exit(1)
}
print("=== Metal metallib cross-platform probe (host CLI) ===\n")
print(MetallibProbe.describeDevice(device)); print("")

for path in paths {
    guard let raw = try? Data(contentsOf: URL(fileURLWithPath: path)) else { continue }
    let name = (path as NSString).lastPathComponent
    let parts = metallibSlices(raw)
    for (i, slice) in parts.enumerated() {
        let tag = parts.count > 1 ? "\(name)[\(i)]" : name
        let header = MetallibHeader(slice)
        print(MetallibProbe.probe(device, label: tag, data: slice).detail); print("")
        if header?.platform == .macOS {
            let patched = MetallibHeader.restamped(slice, to: .iOS)
            print(MetallibProbe.probe(device, label: "\(tag)  [platform byte -> iOS]", data: patched).detail)
            print("")
        }
        if header?.platform == .iOS {
            let patched = MetallibHeader.restamped(slice, to: .macOS)
            print(MetallibProbe.probe(device, label: "\(tag)  [platform byte -> macOS]", data: patched).detail)
            print("")
        }
        if parts.count > 3 && i >= 2 { print("… \(parts.count - 3) more slices not shown\n"); break }
    }
}
