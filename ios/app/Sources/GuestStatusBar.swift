#if canImport(CoreTelephony)
import CoreTelephony
#endif
import Foundation
import Network

/// Draws a network into the guest's status bar.
///
/// The machine has neither a modem nor a Wi-Fi chip, so on its own the guest
/// shows four grey dots and no Wi-Fi, however well connected the phone is. The
/// `sbnet` helper in `guest-tools` hands SpringBoard a status bar override —
/// the mechanism behind `simctl status_bar` — and SpringBoard draws from that
/// instead. What it can draw, and why it needs what it needs, is written down
/// in `netlab/sbnet.m`.
///
/// Only a picture: the guest's apps still see no Wi-Fi and no carrier, and its
/// traffic still goes over the USB link.
enum GuestStatusBar {
    enum Mode: String, CaseIterable {
        /// The guest's status bar is left as the guest makes it.
        case off
        /// Wi-Fi or cellular, and which generation, as the phone has it.
        case phone
        /// Whatever the settings say.
        case custom

        var title: String {
            switch self {
            case .off:    return L("Выкл.")
            case .phone:  return L("Как на телефоне")
            case .custom: return L("Своё")
            }
        }
    }

    /// The labels the guest has pictures for. The numbers are iOS 14's own;
    /// 9 to 11 draw a question mark and anything above falls back to 4G.
    enum Network: Int, CaseIterable {
        case g = 0, e = 1, threeG = 2, fourG = 3, lte = 4, oneX = 7, fiveGE = 8

        var title: String {
            switch self {
            case .g:      return "G"
            case .e:      return "E"
            case .threeG: return "3G"
            case .fourG:  return "4G"
            case .lte:    return "LTE"
            case .oneX:   return "1x"
            case .fiveGE: return "5G E"
            }
        }
    }

    /// One complete state of the status bar.
    struct Look: Equatable {
        var cellularBars: Int
        var network: Network
        var carrier: String
        var wifi: Bool
        var wifiBars: Int
        /// Only whether the slot is drawn at all. Its own half of the bar —
        /// bars, carrier, network type — is deliberately not offered: the
        /// `secondary*` fields were tried and the guest kept drawing the slot as
        /// "no service" whatever they said, so a setting for them would have
        /// promised what it could not deliver.
        var secondSIM: Bool
        var vpn: Bool
        var airplane: Bool

        /// The helper's command line. Always from a clean slate, so what is
        /// drawn is exactly this and nothing left over from before.
        var arguments: String {
            // Items 4, 6 and 9 are the signal, the service and the data
            // network; the cellular entry is built out of all three.
            // `serviceContentType` 1 would mean "no service" and bring the dots
            // back, so it is pinned to 0.
            var items = [4, 6, 9]
            if secondSIM { items.append(7) }
            if vpn { items.append(29) }
            if airplane { items.append(3) }
            var args = ["-z", "-e", items.map(String.init).joined(separator: ","), "-s", "0",
                        "-g", String(min(max(cellularBars, 0), 4)),
                        "-c", Self.shellWord(carrier)]
            // One slot shows either the generation or the Wi-Fi glyph: type 5
            // is Wi-Fi, and its strength has a field of its own.
            if wifi {
                args += ["-n", "5", "-w", String(min(max(wifiBars, 0), 3))]
            } else {
                args += ["-n", String(network.rawValue)]
            }
            return args.joined(separator: " ")
        }

        /// The same look as the agent wants it: a plain object it turns into the
        /// override structure itself. No shell quoting, because nothing here
        /// touches a shell.
        var json: [String: Any] {
            [
                "cellularBars": min(max(cellularBars, 0), 4),
                "network": network.rawValue,
                "carrier": carrier,
                "wifi": wifi,
                "wifiBars": min(max(wifiBars, 0), 3),
                "secondSIM": secondSIM,
                "vpn": vpn,
                "airplane": airplane,
            ]
        }

        /// A carrier name the console can carry: plain characters, quoted.
        private static func shellWord(_ text: String) -> String {
            let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " _-."))
            let clean = String(text.unicodeScalars.filter { $0.isASCII && allowed.contains($0) }.prefix(40))
            return "'" + (clean.isEmpty ? "-" : clean) + "'"
        }
    }

    /// What the settings ask for right now; nil when the guest is to be left
    /// alone.
    static func requested(settings: Settings = .shared, phone: PhoneNetwork = .shared) -> Look? {
        let mode = Mode(rawValue: settings.statusBarMode) ?? .off
        switch mode {
        case .off:
            return nil
        case .phone:
            // The strength is not the phone's: iOS gives an app no signal level
            // at all, so it is shown full, like the connection it stands for.
            return Look(cellularBars: 4, network: phone.radio, carrier: settings.statusBarCarrier,
                        wifi: phone.onWiFi, wifiBars: 3, secondSIM: false,
                        vpn: false, airplane: false)
        case .custom:
            return Look(cellularBars: settings.statusBarBars,
                        network: Network(rawValue: settings.statusBarNetwork) ?? .lte,
                        carrier: settings.statusBarCarrier,
                        wifi: settings.statusBarWifi, wifiBars: settings.statusBarWifiBars,
                        secondSIM: settings.statusBarSecondSIM,
                        vpn: settings.statusBarVPN,
                        airplane: settings.statusBarAirplane)
        }
    }

    enum Failure: LocalizedError {
        case missingHelper
        case delivery(String)
        case silent

        var errorDescription: String? {
            switch self {
            case .missingHelper:    return L("помощника строки состояния нет в приложении")
            case .delivery(let at): return L("помощник не доехал до гостя (%@)", at)
            case .silent:           return L("консоль гостя не ответила")
            }
        }
    }

    /// Draws `look`, or takes the override back when it is nil. Blocks, and
    /// wants the console to itself: call it inside `serial.exclusive` or
    /// `serial.ifFree`, off the main thread.
    ///
    /// A command that gets no answer is interrupted before giving up. Left
    /// alone, it keeps bash busy, and every conversation after it — the shell
    /// pane, the packages, this — types into a console nobody reads.
    static func apply(_ look: Look?, shell: GuestShell) throws {
        let helper = try ensureHelper(shell)
        let arguments = look?.arguments ?? "-z"
        guard shell.line("\(helper) \(arguments) >/dev/null 2>&1", timeout: 60) != nil else {
            shell.reset()
            throw Failure.silent
        }
    }

    // MARK: - The helper

    /// Puts the helper into the guest once, and finds it already there after.
    ///
    /// Over the console rather than by the file transfer: that one needs the
    /// USB network, which is exactly what the guest likes to put down, and the
    /// helper is three kilobytes packed. The name carries a checksum, because
    /// the guest's kernel kills a binary written over a path it has already
    /// seen signed.
    private static func ensureHelper(_ shell: GuestShell) throws -> String {
        let bundled = (Bundle.main.resourceURL ?? Bundle.main.bundleURL).appendingPathComponent("guest-tools/sbnet.gz")
        guard let packed = try? Data(contentsOf: bundled) else { throw Failure.missingHelper }

        var sum = PosixChecksum()
        packed.withUnsafeBytes { sum.update($0) }
        let directory = TransferNamespace.toolsDirectory
        let remote = "\(directory)/sbnet-\(String(format: "%08x", sum.value))"
        switch shell.number("test -x \(remote) && echo 1 || echo 0") {
        case 1?: return remote
        case 0?: break
        default:
            // No answer at all: pouring three kilobytes into that would only
            // make it worse.
            shell.reset()
            throw Failure.silent
        }

        // Seventy-six characters a line: longer lines stop arriving whole on a
        // busy guest, and every line waits for its answer before the next.
        let staging = "\(directory)/sbnet.b64"
        guard shell.line("mkdir -p \(directory) && : > \(staging)", timeout: 60) == 0 else {
            throw Failure.delivery("mkdir")
        }
        let text = Array(packed.base64EncodedString().utf8)
        for start in stride(from: 0, to: text.count, by: 76) {
            let piece = String(decoding: text[start..<min(start + 76, text.count)], as: UTF8.self)
            guard shell.line("printf %s \(piece) >> \(staging)", timeout: 60) == 0 else {
                throw Failure.delivery("printf")
            }
        }

        // GNU base64 on this image, BSD on a bare one; the checksum says
        // whether every byte made it before anything is run.
        let decode = "{ base64 -d \(staging) 2>/dev/null || base64 -D -i \(staging); }"
        guard let line = shell.text("\(decode) | cksum", timeout: 60),
              let crc = line.split(separator: " ").first.flatMap({ UInt32($0) }), crc == sum.value
        else { throw Failure.delivery("cksum") }

        guard shell.line("\(decode) | gunzip -c > \(remote) && chmod +x \(remote) && rm -f \(staging)",
                         timeout: 60) == 0
        else { throw Failure.delivery("gunzip") }
        return remote
    }
}

/// The phone's own connection, as far as an app is told: Wi-Fi or cellular,
/// and the cellular generation. Main-thread object.
final class PhoneNetwork {
    static let shared = PhoneNetwork()

    private(set) var onWiFi = false
    private(set) var radio: GuestStatusBar.Network = .lte
    /// Called on the main thread whenever either of the above changes.
    var onChange: (() -> Void)?

    private var monitor: NWPathMonitor?
    #if os(iOS)
    private let telephony = CTTelephonyNetworkInfo()
    #endif

    func start() {
        guard monitor == nil else { return }
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] path in
            #if os(iOS)
            let wifi = path.status == .satisfied && path.usesInterfaceType(.wifi)
            #else
            // A Mac is never on a cellular network, and the guest has no glyph
            // for a cable: any connection at all is drawn as Wi-Fi.
            let wifi = path.status == .satisfied
            #endif
            DispatchQueue.main.async { self?.update(wifi: wifi) }
        }
        monitor.start(queue: DispatchQueue(label: "inferno.phone-network"))
        self.monitor = monitor
        update(wifi: onWiFi)
    }

    private func update(wifi: Bool) {
        #if os(iOS)
        let radio = Self.generation(telephony.serviceCurrentRadioAccessTechnology?.values.first)
        #else
        let radio = self.radio
        #endif
        guard wifi != onWiFi || radio != self.radio else { return }
        onWiFi = wifi
        self.radio = radio
        onChange?()
    }

    #if os(iOS)
    /// The nearest label the guest can draw for a radio technology.
    private static func generation(_ technology: String?) -> GuestStatusBar.Network {
        switch technology {
        case CTRadioAccessTechnologyNR?, CTRadioAccessTechnologyNRNSA?: return .fiveGE
        case CTRadioAccessTechnologyLTE?:                              return .lte
        case CTRadioAccessTechnologyWCDMA?, CTRadioAccessTechnologyHSDPA?, CTRadioAccessTechnologyHSUPA?,
             CTRadioAccessTechnologyCDMAEVDORev0?, CTRadioAccessTechnologyCDMAEVDORevA?,
             CTRadioAccessTechnologyCDMAEVDORevB?, CTRadioAccessTechnologyeHRPD?:
            return .threeG
        case CTRadioAccessTechnologyEdge?:     return .e
        case CTRadioAccessTechnologyGPRS?:     return .g
        case CTRadioAccessTechnologyCDMA1x?:   return .oneX
        default:                               return .lte
        }
    }
    #endif
}
