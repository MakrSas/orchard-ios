import Foundation

/// Gives the guest the phone's time zone.
///
/// The guest's clock is right — the machine keeps its RTC in UTC — but its zone
/// is whatever the image was made with, US/Pacific on images built by ChefKiss's
/// guide, so the time on its screen is hours out from the phone's. iOS keeps the
/// system zone as one symlink, `/var/db/timezone/localtime`, pointing into
/// zoneinfo. It lives on the data volume, so changing it needs no remount and
/// outlasts a reboot, and nothing has to be told about it: checked on the rig,
/// a booted SpringBoard shows the new zone from its next minute on.
enum GuestTimeZone {
    static let link = "/var/db/timezone/localtime"

    /// The phone's zone by name, or nil when the name is not one a shell line
    /// can carry as it is.
    static var phone: String? {
        // The system zone is cached per process; a change of zone on the phone
        // would otherwise go unseen until the app restarts.
        NSTimeZone.resetSystemTimeZone()
        let name = TimeZone.current.identifier
        // It goes into a shell command unquoted, so only what zone names are
        // made of: letters, digits, `/`, `_`, `+` and `-`.
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/_+-")
        guard !name.isEmpty, !name.hasPrefix("-"), !name.hasPrefix("/"), !name.contains(".."),
              name.unicodeScalars.allSatisfy({ allowed.contains($0) })
        else { return nil }
        return name
    }

    /// One line for the guest's shell: points the link at the zone and prints
    /// where the link now points — or `NOZONE` when this guest's zoneinfo has
    /// no such zone, which an older image can lack. Kept short, because the
    /// console loses bytes inside long lines.
    static func command(for zone: String) -> String {
        let target = "/var/db/timezone/zoneinfo/\(zone)"
        return "if [ -e \(target) ]; then ln -sfn \(target) \(link) && readlink \(link); else echo NOZONE; fi"
    }

    enum Outcome { case set, missing, unexpected }

    /// What the guest's answer to `command(for:)` means.
    static func outcome(of answer: String, zone: String) -> Outcome {
        let said = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        if said.hasSuffix("/zoneinfo/\(zone)") { return .set }
        if said.hasSuffix("NOZONE") { return .missing }
        return .unexpected
    }
}
