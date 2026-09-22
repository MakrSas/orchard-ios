import Foundation
#if canImport(UIKit)
import UIKit
#else
import IOKit.ps
#endif

/// Shows the guest the phone's own battery.
///
/// The machine's SMC used to answer 69 % and "on battery" whatever the phone
/// was doing: the figure was written into the emulator. The library now takes
/// the real one through `inferno_battery_set`, tells the guest the power state
/// changed, and the guest's battery driver reads it within a few seconds rather
/// than on its own twenty-second poll.
///
/// The library keeps what it was last told outside the machine, so a value sent
/// before the machine is up is the one it boots with, and a guest reboot does
/// not lose it.
///
/// The charge goes over in steps of five, because that is all an app on iOS 27
/// is given, and every other way round was tried on the phone:
/// - `UIDevice.batteryLevel` is rounded to five (45 with the status bar at 44).
/// - IOKit's `IOPSCopyPowerSourcesInfo` answers, rounded the same way (75, 80);
///   `IOPSGetPercentRemaining` is refused with kIOReturnNotPrivileged.
/// - The status bar's own data is empty in an app's process, visible status bar
///   or not: `+[UIStatusBarServer getStatusBarData]` holds zeros and an empty
///   clock, and `-[UIStatusBarManager createLocalStatusBar]` returns nil.
///
/// A Mac has no such limit: IOKit gives the capacity as it is, and a Mac without
/// a battery at all gives nothing, which leaves the guest its own figure.
final class HostBattery {
    private typealias SetFn = @convention(c) (Int32, Bool, Bool) -> Void

    static let shared = HostBattery()

    private var setFn: SetFn?
    private var observers: [NSObjectProtocol] = []
    private var timer: Timer?
    private var reported: Reading?

    private struct Reading: Equatable {
        var percent: Int32
        var external: Bool
        var charging: Bool
    }

    /// Starts following the battery, or just reports it again if already
    /// following. Main thread only, like UIDevice itself.
    func start(bridge: QemuBridge = .shared) {
        if timer != nil {
            report(force: true)
            return
        }
        guard let symbol = bridge.symbol("inferno_battery_set") else {
            LogCapture.shared.note(L("Батарея: в этой сборке библиотеки её нет, гость видит 69 %."))
            return
        }
        setFn = unsafeBitCast(symbol, to: SetFn.self)

        #if os(iOS)
        // Without this the level reads -1 and the state .unknown.
        UIDevice.current.isBatteryMonitoringEnabled = true
        let center = NotificationCenter.default
        for name in [UIDevice.batteryLevelDidChangeNotification, UIDevice.batteryStateDidChangeNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                self?.report(force: false)
            })
        }
        #endif
        // UIKit sends its level notifications at most once a minute; the timer
        // is for whatever they leave out.
        let timer = Timer(timeInterval: 30, repeats: true) { [weak self] _ in self?.report(force: false) }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
        report(force: true)
    }

    private func report(force: Bool) {
        guard let setFn, let reading = read() else { return }
        // Sent every time, cheap as it is; only the log waits for a change.
        setFn(reading.percent, reading.external, reading.charging)
        let changed = reading != reported
        reported = reading
        guard force || changed else { return }
        let state = reading.charging ? L("заряжается") : reading.external ? L("от сети") : L("от батареи")
        LogCapture.shared.note(L("Батарея гостя: %d %%, %@", Int(reading.percent), state))
    }

    /// Nil where there is no battery to speak of — the simulator reports
    /// exactly that, and the guest is better off keeping its own number.
    private func read() -> Reading? {
        let settings = Settings.shared
        guard settings.guestBatteryReal else {
            let charging = settings.guestBatteryCharging
            return Reading(percent: Int32(settings.guestBatteryPercent.rounded()),
                           external: charging, charging: charging)
        }

        #if os(macOS)
        return Self.macBattery()
        #else
        let device = UIDevice.current
        guard device.batteryLevel >= 0 else { return nil }
        let percent = Int32((device.batteryLevel * 100).rounded())
        switch device.batteryState {
        case .charging:  return Reading(percent: percent, external: true, charging: true)
        // Full on a cable still counts as charging. A real iPhone keeps the bolt
        // at 100 %, but the guest lights it from CHSC alone, so reporting "in,
        // not charging" took the bolt away exactly when the phone was full.
        case .full:      return Reading(percent: percent, external: true, charging: true)
        case .unplugged: return Reading(percent: percent, external: false, charging: false)
        case .unknown:   return nil
        @unknown default: return nil
        }
        #endif
    }

    #if os(macOS)
    /// The internal battery as IOKit describes it: current and maximum capacity,
    /// whether the Mac is on its adapter, and whether it is charging. A charged
    /// battery on the adapter counts as charging, for the same reason as on the
    /// phone: the guest lights its bolt from that alone.
    private static func macBattery() -> Reading? {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef]
        else { return nil }
        for source in list {
            guard let description = IOPSGetPowerSourceDescription(info, source)?.takeUnretainedValue()
                    as? [String: Any],
                  description[kIOPSTypeKey] as? String == kIOPSInternalBatteryType,
                  let current = description[kIOPSCurrentCapacityKey] as? Int,
                  let maximum = description[kIOPSMaxCapacityKey] as? Int, maximum > 0
            else { continue }
            let external = description[kIOPSPowerSourceStateKey] as? String == kIOPSACPowerValue
            let charging = description[kIOPSIsChargingKey] as? Bool ?? false
            let charged = description[kIOPSIsChargedKey] as? Bool ?? false
            let percent = Int32((Double(current) * 100 / Double(maximum)).rounded())
            return Reading(percent: percent, external: external, charging: charging || (external && charged))
        }
        return nil
    }
    #endif
}
