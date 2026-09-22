import Darwin
import Foundation

/// The fast way in and out of the guest: a scratch NVMe namespace.
///
/// The app writes bytes into the namespace's backing file and the guest reads
/// the same place as a block device — nothing goes through the console or the
/// network, so this is megabytes a second rather than hundreds of kilobytes.
///
/// Three things have to line up, and each fails in its own quiet way:
///
/// 1. **The emulator has to describe the namespace in the device tree.** The
///    guest never enumerates the controller; it takes its namespace list from
///    the tree and says so — `Obtained N namespaces from DT`. An emulator
///    without that patch simply shows no extra device, and everything here
///    politely gives up so the caller can use the network instead.
/// 2. **The guest's shell cannot open block devices.** `dd if=/dev/rdisk2`
///    answers `Operation not permitted` even to root, while `/sbin/fsck_hfs`
///    reads the very same device. It is the binary's entitlements that decide,
///    so a small signed helper is carried in the app bundle and put into the
///    guest once.
/// 3. **The helper must land under a name the kernel has not seen before.**
///    Writing a new binary over a path that was already signed gets it killed
///    on sight — `Killed: 9`, before `main()`. The name therefore carries a
///    checksum of the binary, and a new build simply gets a new name.
///
/// Everything here blocks; run it off the main thread.
final class TransferNamespace {
    /// Where the app keeps its own things inside the guest. On the data volume,
    /// so it survives the guest rebooting.
    static let toolsDirectory = "/var/mobile/.inferno"
    private static let chunkFile = toolsDirectory + "/chunk.bin"

    let device: String
    let helper: String
    let image: URL
    let capacity: Int64

    private init(device: String, helper: String, image: URL, capacity: Int64) {
        self.device = device
        self.helper = helper
        self.image = image
        self.capacity = capacity
    }

    // MARK: - Finding it

    /// Returns the channel when it can be had. Nil is not a failure — it means
    /// this transfer goes over the network.
    ///
    /// `deliver` carries a local file to a path in the guest by whatever slow
    /// means already work; it is used once, for the helper itself.
    static func discover(shell: GuestShell,
                         deliver: (URL, String) throws -> Void,
                         note: (String) -> Void) -> TransferNamespace? {
        let image = VMConfig.transferImage
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: image.path),
              let capacity = (attributes[.size] as? NSNumber)?.int64Value, capacity > 0
        else { return nil }

        // Cheap first: if the guest shows no device beyond its own disk and the
        // firmware namespace, the emulator has not described ours and there is
        // nothing to set up.
        guard let listing = shell.text("ls /dev/rdisk2 2>/dev/null | head -1"), listing.contains("rdisk") else {
            return nil
        }

        guard let helper = try? ensureHelper(shell, deliver: deliver, note: note) else { return nil }

        // Which /dev/rdiskN it is depends on how many namespaces the guest chose
        // to expose, so it is found by size rather than by number: ours is the
        // only one exactly this big.
        for index in 1...5 {
            let candidate = "/dev/rdisk\(index)"
            guard let line = shell.text("\(helper) size \(candidate) 2>/dev/null", timeout: 30),
                  let bytes = line.split(separator: "=").last.flatMap({ Int64($0) }),
                  bytes == capacity
            else { continue }
            return TransferNamespace(device: candidate, helper: helper, image: image, capacity: capacity)
        }
        return nil
    }

    /// Puts the helper into the guest once, and finds it already there after.
    private static func ensureHelper(_ shell: GuestShell,
                                     deliver: (URL, String) throws -> Void,
                                     note: (String) -> Void) throws -> String {
        let bundled = (Bundle.main.resourceURL ?? Bundle.main.bundleURL).appendingPathComponent("guest-tools/nsio")
        guard let data = try? Data(contentsOf: bundled) else {
            throw GuestFiles.Failure.io(L("помощника нет в приложении"))
        }
        var sum = PosixChecksum()
        data.withUnsafeBytes { sum.update($0) }
        let remote = "\(toolsDirectory)/nsio-\(String(format: "%08x", sum.value))"

        if shell.number("test -x \(remote) && echo 1 || echo 0") == 1 { return remote }

        note(L("Ставлю помощника в гостя — это один раз…"))
        _ = shell.run("mkdir -p \(toolsDirectory)")

        // The helper rides in over the network, which is the only channel there
        // is until it arrives. On a freshly booted guest the interface is not up
        // yet, and a single failed attempt used to cost the fast channel for the
        // whole transfer — so the link is waited for, and the send is retried.
        waitForLink(shell)
        var lastError: Error?
        for attempt in 0..<3 {
            do {
                try deliver(bundled, remote)
                lastError = nil
                break
            } catch {
                lastError = error
                shell.reset()
                if attempt < 2 { Thread.sleep(forTimeInterval: 3) }
            }
        }
        if let lastError { throw lastError }

        guard shell.run("chmod +x \(remote)") == 0 else {
            throw GuestFiles.Failure.io(L("помощника не удалось сделать исполняемым"))
        }
        return remote
    }

    /// Waits for the guest to have an address of its own, asking for one first.
    ///
    /// The guest brings its end up, asks by DHCP and then puts it back down
    /// again; `ipconfig set en0 DHCP` from inside is what actually settles it.
    private static func waitForLink(_ shell: GuestShell) {
        if let address = shell.text("/usr/sbin/ipconfig getifaddr en0 2>/dev/null"), address.hasPrefix("10.") {
            return
        }
        _ = shell.run("/usr/sbin/ipconfig set en0 DHCP", timeout: 90)
        for _ in 0..<20 {
            if let address = shell.text("/usr/sbin/ipconfig getifaddr en0 2>/dev/null"), address.hasPrefix("10.") {
                return
            }
            Thread.sleep(forTimeInterval: 1.5)
        }
    }

    // MARK: - Carrying bytes

    /// Sends a local file to `remote`, which is already a shell expression.
    func send(_ file: URL, to remote: String, shell: GuestShell,
              progress: (Int64, Int64) -> Void) throws {
        let source = try FileHandle(forReadingFrom: file)
        defer { try? source.close() }
        let total = Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)

        var sum = PosixChecksum()
        var sent: Int64 = 0
        var first = true
        while sent < total {
            let want = Int(min(capacity, total - sent))
            guard let piece = try source.read(upToCount: want), !piece.isEmpty else { break }
            piece.withUnsafeBytes { sum.update($0) }
            try writeWindow(piece)

            // `>` for the first window and `>>` after, so the file is assembled
            // in one pass without a separate command to create it.
            let append = first ? ">" : ">>"
            let command = "\(helper) read \(device) 0 \(piece.count) \(Self.chunkFile)"
                + " && cat \(Self.chunkFile) \(append) \(remote)"
            let status = shell.run(command, timeout: 600) ?? -1
            guard status == 0 else { throw GuestFiles.Failure.io(L("гость не прочитал носитель (%d)", Int(status))) }

            sent += Int64(piece.count)
            first = false
            progress(sent, total)
        }
        _ = shell.run("rm -f \(Self.chunkFile)")
        try verify(remote, shell: shell, length: sum.length, crc: sum.value)
    }

    /// Pulls `remote` out of the guest into a local file.
    func receive(_ remote: String, to local: URL, shell: GuestShell,
                 progress: (Int64, Int64) -> Void) throws {
        let total = shell.number("wc -c < \(remote)") ?? 0
        guard total > 0 else { throw GuestFiles.Failure.notFound(remote) }

        FileManager.default.createFile(atPath: local.path, contents: nil)
        let sink = try FileHandle(forWritingTo: local)
        defer { try? sink.close() }

        var sum = PosixChecksum()
        var got: Int64 = 0
        var index = 0
        while got < total {
            let take = Int(min(capacity, total - got))
            // The guest cuts one window out of the file and lays it on the
            // device; we read the same place back.
            let command = "dd if=\(remote) of=\(Self.chunkFile) bs=\(capacity) skip=\(index) count=1 2>/dev/null"
                + " && \(helper) write \(device) 0 \(Self.chunkFile)"
            let status = shell.run(command, timeout: 600) ?? -1
            guard status == 0 else { throw GuestFiles.Failure.io(L("гость не записал носитель (%d)", Int(status))) }

            let piece = try readWindow(take)
            piece.withUnsafeBytes { sum.update($0) }
            try sink.write(contentsOf: piece)

            got += Int64(take)
            index += 1
            progress(got, total)
        }
        _ = shell.run("rm -f \(Self.chunkFile)")
        try verify(remote, shell: shell, length: sum.length, crc: sum.value)
    }

    // MARK: - The namespace itself

    private func writeWindow(_ piece: Data) throws {
        let sink = try FileHandle(forWritingTo: image)
        defer { try? sink.close() }
        try sink.seek(toOffset: 0)
        try sink.write(contentsOf: piece)
        // Forced out, not merely written: the emulator reads the file again on
        // the guest's next request, and anything still sitting in a buffer would
        // be served as the file's previous contents.
        fsync(sink.fileDescriptor)
    }

    private func readWindow(_ length: Int) throws -> Data {
        let source = try FileHandle(forReadingFrom: image)
        defer { try? source.close() }
        try source.seek(toOffset: 0)
        return try source.read(upToCount: length) ?? Data()
    }

    /// Both sides count the same way, so a wrong byte anywhere shows up here.
    private func verify(_ remote: String, shell: GuestShell, length: Int64, crc: UInt32) throws {
        let slack = 60 + Double(length) / 200_000
        let size = shell.number("wc -c < \(remote)", timeout: slack)
        let theirs = shell.number("cksum < \(remote) | cut -d' ' -f1", timeout: slack)
        guard size == length, theirs == Int64(crc) else {
            throw GuestFiles.Failure.mismatch(
                L("в госте %@ Б, у нас %d Б", size.map(String.init) ?? "?", length))
        }
    }
}
