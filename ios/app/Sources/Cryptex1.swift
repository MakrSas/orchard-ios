import Foundation

/// Forging the Cryptex1 ticket iOS 16+ asks for mid-restore.
///
/// From iOS 16 on, the OS ships in Cryptex1 disk images (the system and app
/// "cryptexes"), and each needs its own ticket -- requested by the device in
/// the middle of the restore, not handed over up front like the AP ticket.
/// `restored` asks for `FirmwareUpdaterData` with `MessageArgUpdaterName` =
/// "Cryptex1", which on a real restore turns into a TSS request to Apple; an
/// unsigned build gets nothing back (ChefKiss issue #305).
///
/// This is `netlab/tssd.py`, in Swift, so the phone can do it without a
/// computer: a real device's own Cryptex1 ticket is taken apart, its digests
/// are replaced with this build's, and its identity fields are patched to
/// match -- the signature underneath is left exactly as it was, unchecked by
/// the emulator's own kernel patches, same trick as `Ticket.forgeAP`.
enum Cryptex1 {
    enum Failure: LocalizedError {
        case shape(String)

        var errorDescription: String? {
            switch self {
            case .shape(let what): return L("Шаблон Cryptex1 устроен не так, как ожидалось: %@", what)
            }
        }
    }

    /// Cryptex component four-character code -> build identity manifest key.
    private static let components: [String: String] = [
        "caos": "Cryptex1,AppOS",
        "casy": "Cryptex1,AppVolume",
        "trca": "Cryptex1,AppTrustCache",
        "csos": "Cryptex1,SystemOS",
        "cssy": "Cryptex1,SystemVolume",
        "trcs": "Cryptex1,SystemTrustCache",
        "cros": "Cryptex1,RosettaOS",
        "crsy": "Cryptex1,RosettaVolume",
        "trcr": "Cryptex1,RosettaTrustCache",
    ]

    /// Forges a Cryptex1 IM4M for this build identity, out of any real
    /// device's own Cryptex1 ticket. The template's shape and signature ride
    /// along untouched -- every field that says whose ticket this is, and
    /// what it vouches for, gets overwritten.
    static func forge(identity: [String: Any], template: [UInt8]) throws -> [UInt8] {
        let ticket = try DER.parse(template)
        guard ticket.children?.first?.text == "IM4M" else { throw Failure.shape("not an IM4M") }
        guard let top = ticket.children, top.count > 2,
              let manb = top[2].children?.first,
              let manbBody = manb.children?.first?.children, manbBody.count > 1
        else { throw Failure.shape("no MANB") }
        let payloadSet = manbBody[1]
        guard let items = payloadSet.children,
              let manp = items.first(where: { $0.fourCC == "MANP" })
        else { throw Failure.shape("no MANP") }

        try patchManp(manp, with: identity)

        // The template's own cryptex components are some other build's --
        // thrown away and replaced with one entry per component this build
        // actually carries, digest straight out of its own manifest.
        let manifest = identity["Manifest"] as? [String: Any] ?? [:]
        var newItems = [manp]
        for (fourcc, key) in components {
            guard let entry = manifest[key] as? [String: Any], let digest = entry["Digest"] as? Data else { continue }
            newItems.append(component(fourcc, digest: [UInt8](digest)))
        }
        payloadSet.children = newItems

        return DER.encode(ticket)
    }

    /// A cryptex component: `fourcc { SEQUENCE { fourcc, SET { DGST { SEQUENCE { "DGST", digest } } } } }`.
    private static func component(_ fourcc: String, digest: [UInt8]) -> DER.Node {
        DER.tagged(fourcc, [DER.ia5(fourcc), DER.set([DER.tagged("DGST", [DER.ia5("DGST"), DER.octets(digest)])])])
    }

    /// Patches the identity fields inside the template's own MANP in place --
    /// same fields ChefKiss's own signing server rewrites, same reasoning:
    /// `fchp`/`clas`/`type`/`styp` say which chip and cryptex class this
    /// ticket is for, and `pave`/`vnum` carry the build's own version string.
    private static func patchManp(_ manp: DER.Node, with identity: [String: Any]) throws {
        guard let fields = manp.children?.first?.children, fields.count > 1,
              let props = fields[1].children
        else { throw Failure.shape("MANP payload") }

        func hex(_ key: String, default def: UInt64) -> UInt64 {
            guard let raw = identity[key] else { return def }
            if let n = raw as? NSNumber { return n.uint64Value }
            if let s = raw as? String {
                let trimmed = s.hasPrefix("0x") ? String(s.dropFirst(2)) : s
                return UInt64(trimmed, radix: 16) ?? def
            }
            return def
        }
        let overrides: [String: UInt64] = [
            "fchp": hex("Cryptex1,ChipID", default: 0xFF10),
            "clas": hex("Cryptex1,ProductClass", default: 0xF1),
            "type": hex("Cryptex1,Type", default: 1),
            "styp": hex("Cryptex1,SubType", default: 1),
        ]
        let version = identity["Cryptex1,Version"] as? String

        for prop in props {
            guard let pair = prop.children?.first?.children, pair.count > 1,
                  let name = pair[0].text
            else { continue }
            if let value = overrides[name] {
                let encoded = DER.integer(value)
                pair[1].identifier = encoded.identifier
                pair[1].value = encoded.value
            } else if (name == "pave" || name == "vnum"), let version {
                pair[1].identifier = [0x04]
                pair[1].value = Array(version.utf8)
            }
        }
    }
}
