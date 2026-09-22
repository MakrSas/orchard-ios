import Foundation

/// Forging the tickets a restore needs.
///
/// The versions this emulator runs are not signed by Apple any more, so the
/// ticket that vouches for them is made rather than fetched: a ticket Apple did
/// sign once is taken apart, the digests inside it are replaced with the ones
/// from the firmware being installed, and it is put back together. The emulated
/// device's SEP ROM is patched to accept it — that is the whole trick, and it is
/// why nothing here talks to Apple.
///
/// This is ChefKiss's `create_apticket.py` and `create_septicket.py`, in Swift,
/// so the phone can do it without a computer. It matches them byte for byte.
enum Ticket {
    enum Failure: LocalizedError {
        case noTicket
        case shape(String)
        case noDigest(String)

        var errorDescription: String? {
            switch self {
            case .noTicket:          return L("В ticket.shsh2 нет ApImg4Ticket")
            case .shape(let what):   return L("Тикет устроен не так, как ожидалось: %@", what)
            case .noDigest(let key): return L("В манифесте нет дайджеста для %@", key)
            }
        }
    }

    /// Which manifest entry each four-character code in the ticket stands for.
    static let apComponents: [String: String] = [
        "rosi": "OS",
        "krnl": "KernelCache",
        "dtre": "DeviceTree",
        "rdtr": "RestoreDeviceTree",
        "trst": "StaticTrustCache",
        "rtsc": "RestoreTrustCache",
        "mtfw": "Multitouch",
        "anef": "ANE",
        "aopf": "AOP",
        "avef": "AVE",
        "hpas": "Ap,HapticAssets",
        "acfw": "AudioCodecFirmware",
        "gfxf": "GFX",
        "ispf": "ISP",
        "lphp": "LeapHaptics",
        "pmpf": "PMP",
        "siof": "SIO",
        "sepi": "SEP",
        "rsep": "RestoreSEP",
        "rdsk": "RestoreRamDisk",
        "isys": "SystemVolume",
        "msys": "SystemVolumeCanonicalMetadata",
    ]

    /// The SEP's own ticket vouches for fewer things.
    static let sepComponents: [String: String] = [
        "rosi": "OS",
        "krnl": "KernelCache",
        "dtre": "DeviceTree",
        "rdtr": "RestoreDeviceTree",
        "trst": "StaticTrustCache",
        "rtsc": "RestoreTrustCache",
        "sepi": "SEP",
        "rsep": "RestoreSEP",
    ]

    /// Codes whose entries have to stop matching anything: the name inside the
    /// component is reversed, which leaves a well-formed ticket that no longer
    /// claims those firmwares.
    private static let corrupted: Set<String> = ["rfta", "ftap", "rfts", "ftsp"]

    /// The ECID the emulated device reports, and the nonce the SEP expects with
    /// it. Both are the machine's, not Apple's.
    static let ecid: UInt64 = 0x1122_3344_5566_7788
    private static let nonce: [UInt8] = Array(repeating: [0xFE, 0xED, 0xFA, 0xCE], count: 5).flatMap { $0 }

    // MARK: - The two tickets

    static func forgeAP(shsh: Data, identity: [String: Any]) throws -> [UInt8] {
        let ticket = try root(of: shsh)
        try rewrite(components(of: ticket), with: identity, map: apComponents, chipID: nil)
        return DER.encode(ticket)
    }

    static func forgeSEP(shsh: Data, identity: [String: Any], chipID: UInt64 = 0x8030) throws -> [UInt8] {
        let ticket = try root(of: shsh)
        try rewrite(components(of: ticket), with: identity, map: sepComponents, chipID: chipID)

        // The same edits again, inside the certificate that carries a copy of
        // the manifest: extension number four holds a DER SET of the very same
        // components. The device checks that copy, so it has to agree — but its
        // ECID and nonce are left alone.
        let extensionValue = try certificateManifest(of: ticket)
        let inner = try DER.parse(extensionValue.value)
        guard let items = inner.children else { throw Failure.shape("certificate manifest") }
        try rewrite(items, with: identity, map: sepComponents, chipID: chipID, touchDeviceIdentity: false)
        extensionValue.value = DER.encode(inner)

        return DER.encode(ticket)
    }

    // MARK: - Walking the ticket

    private static func root(of shsh: Data) throws -> DER.Node {
        guard let plist = try PropertyListSerialization.propertyList(from: shsh, options: [], format: nil)
                as? [String: Any],
              let ticket = plist["ApImg4Ticket"] as? Data
        else { throw Failure.noTicket }
        let node = try DER.parse([UInt8](ticket))
        guard node.children?.first?.text == "IM4M" else { throw Failure.shape("not an IM4M") }
        return node
    }

    /// `IM4M ::= SEQUENCE { "IM4M", version, SET { MANB { SEQUENCE { "MANB", SET { component… } } } }, … }`
    private static func components(of ticket: DER.Node) -> [DER.Node] {
        guard let top = ticket.children, top.count > 2,
              let manb = top[2].children?.first,
              let body = manb.children?.first?.children, body.count > 1,
              let items = body[1].children
        else { return [] }
        return items
    }

    /// One field of a component: `'DGST' { SEQUENCE { "DGST", value } }`, and it
    /// is the value node that is returned, to be written over in place.
    private static func field(_ name: String, of component: DER.Node) -> DER.Node? {
        guard let entries = component.children?.first?.children, entries.count > 1,
              let fields = entries[1].children
        else { return nil }
        for field in fields where field.fourCC == name {
            if let pair = field.children?.first?.children, pair.count > 1 { return pair[1] }
        }
        return nil
    }

    private static func rewrite(_ components: [DER.Node], with identity: [String: Any],
                                map: [String: String], chipID: UInt64?,
                                touchDeviceIdentity: Bool = true) throws {
        let manifest = identity["Manifest"] as? [String: Any] ?? [:]
        for component in components {
            guard let code = component.fourCC else { continue }

            if let key = map[code] {
                // A component the firmware does not carry keeps whatever the
                // signed ticket said about it: an installation that never has
                // that firmware never asks about it either. iOS 16 has no
                // SystemVolumeCanonicalMetadata, for one.
                guard let entry = manifest[key] as? [String: Any],
                      let digest = entry["Digest"] as? Data,
                      let value = field("DGST", of: component)
                else { continue }
                value.identifier = [0x04]
                value.value = [UInt8](digest)
                continue
            }

            if corrupted.contains(code) {
                // The name inside the component, not the tag around it: that is
                // what the device matches on.
                guard let name = component.children?.first?.children?.first else { continue }
                name.value = name.value.reversed()
                continue
            }

            if code == "MANP", let chipID {
                if let chip = field("CHIP", of: component) {
                    let encoded = DER.integer(chipID)
                    chip.identifier = encoded.identifier
                    chip.value = encoded.value
                }
                if touchDeviceIdentity {
                    if let value = field("ECID", of: component) {
                        let encoded = DER.integer(ecid)
                        value.identifier = encoded.identifier
                        value.value = encoded.value
                    }
                    if let value = field("snon", of: component) {
                        value.identifier = [0x04]
                        value.value = nonce
                    }
                }
            }
        }
    }

    /// The fifth extension of the first certificate, where a second copy of the
    /// manifest lives.
    private static func certificateManifest(of ticket: DER.Node) throws -> DER.Node {
        guard let top = ticket.children, top.count > 4,
              let certificate = top[4].children?.first,
              let tbs = certificate.children?.first,
              let tbsFields = tbs.children
        else { throw Failure.shape("certificate") }

        // Extensions ride in an explicit [3] on the TBS certificate.
        guard let extensions = tbsFields.first(where: { $0.identifier == [0xA3] })?.children?.first?.children,
              extensions.count > 4,
              let fields = extensions[4].children
        else { throw Failure.shape("extensions") }

        // Extension ::= SEQUENCE { extnID, critical DEFAULT FALSE, extnValue }
        guard let value = fields.last, value.identifier == [0x04] else {
            throw Failure.shape("extnValue")
        }
        return value
    }
}
