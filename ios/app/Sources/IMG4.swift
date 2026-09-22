import Foundation

/// Apple's IMG4 containers, and just enough DER to work on them.
///
/// Everything the restore hands the device is one of these: the kernel, the
/// device tree, the ramdisk, the SEP firmware, the ticket itself. A component
/// arrives from the IPSW as an `IM4P` — a type, a version and a payload — and
/// goes to the device as an `IMG4`, which is that payload plus the ticket
/// (`IM4M`) that vouches for it. Personalising a component is exactly that
/// wrapping, and forging a ticket is editing the digests inside one.
///
/// The DER here is deliberately small: a tree of tag/value nodes that survives a
/// decode/encode round trip byte for byte, so anything not understood is carried
/// through untouched. That matters — a ticket holds fields nobody outside Apple
/// can name, and they have to come out the way they went in.
enum DER {
    /// One tag-length-value node. Constructed nodes keep their children;
    /// primitive ones keep their bytes.
    final class Node {
        /// The identifier octets, verbatim: the class, the constructed bit and
        /// the tag number, which for Apple's private tags is a four-character
        /// code like `MANB` and does not fit in one byte.
        var identifier: [UInt8]
        var children: [Node]?
        var value: [UInt8]

        init(identifier: [UInt8], children: [Node]) {
            self.identifier = identifier
            self.children = children
            self.value = []
        }

        init(identifier: [UInt8], value: [UInt8]) {
            self.identifier = identifier
            self.children = nil
            self.value = value
        }

        var isConstructed: Bool { identifier[0] & 0x20 != 0 }
        var tagClass: UInt8 { identifier[0] & 0xC0 }

        /// The tag number, for the single-byte and the long forms alike.
        var tagNumber: UInt64 {
            if identifier[0] & 0x1F != 0x1F { return UInt64(identifier[0] & 0x1F) }
            var number: UInt64 = 0
            for byte in identifier.dropFirst() {
                number = (number << 7) | UInt64(byte & 0x7F)
            }
            return number
        }

        /// A private tag read as the four characters Apple writes it with.
        var fourCC: String? {
            guard tagClass == 0xC0 else { return nil }
            let number = tagNumber
            guard number > 0xFFFF_FF else { return nil }
            let bytes = (0..<4).reversed().map { UInt8((number >> (8 * UInt64($0))) & 0xFF) }
            guard bytes.allSatisfy({ $0 >= 0x20 && $0 < 0x7F }) else { return nil }
            return String(decoding: bytes, as: UTF8.self)
        }

        /// The text of an IA5String or similar, when that is what this is.
        var text: String? {
            guard children == nil else { return nil }
            return String(decoding: value, as: UTF8.self)
        }

        /// The immediate child under this four-character code.
        func child(_ name: String) -> Node? {
            children?.first { $0.fourCC == name }
        }

        /// The first node with this four-character code anywhere below, depth
        /// first. Convenient for a look, too vague to build on: anything that
        /// edits a ticket walks the shape it expects instead.
        func firstTagged(_ name: String) -> Node? {
            if fourCC == name { return self }
            for child in children ?? [] {
                if let found = child.firstTagged(name) { return found }
            }
            return nil
        }
    }

    enum Failure: LocalizedError {
        case malformed(String)

        var errorDescription: String? {
            switch self {
            case .malformed(let what): return L("DER: %@", what)
            }
        }
    }

    // MARK: - Decoding

    static func parse(_ bytes: [UInt8]) throws -> Node {
        var at = 0
        let node = try parseOne(bytes, &at)
        return node
    }

    static func parseAll(_ bytes: [UInt8]) throws -> [Node] {
        var at = 0
        var nodes: [Node] = []
        while at < bytes.count {
            nodes.append(try parseOne(bytes, &at))
        }
        return nodes
    }

    private static func parseOne(_ bytes: [UInt8], _ at: inout Int) throws -> Node {
        guard at < bytes.count else { throw Failure.malformed("ran out of bytes") }
        var identifier = [bytes[at]]
        at += 1
        if identifier[0] & 0x1F == 0x1F {
            // A long-form tag: seven bits per byte, the top bit says "more".
            while at < bytes.count {
                let byte = bytes[at]
                identifier.append(byte)
                at += 1
                if byte & 0x80 == 0 { break }
            }
        }

        guard at < bytes.count else { throw Failure.malformed("no length") }
        var length = Int(bytes[at])
        at += 1
        if length & 0x80 != 0 {
            let count = length & 0x7F
            guard count > 0, count <= 8, at + count <= bytes.count else {
                throw Failure.malformed("bad length")
            }
            length = 0
            for _ in 0..<count {
                length = (length << 8) | Int(bytes[at])
                at += 1
            }
        }
        guard at + length <= bytes.count else { throw Failure.malformed("length past the end") }
        let body = Array(bytes[at..<at + length])
        at += length

        if identifier[0] & 0x20 != 0 {
            var inner = 0
            var children: [Node] = []
            while inner < body.count {
                children.append(try parseOne(body, &inner))
            }
            return Node(identifier: identifier, children: children)
        }
        return Node(identifier: identifier, value: body)
    }

    // MARK: - Encoding

    static func encode(_ node: Node) -> [UInt8] {
        let body: [UInt8]
        if let children = node.children {
            body = children.flatMap { encode($0) }
        } else {
            body = node.value
        }
        return node.identifier + length(body.count) + body
    }

    private static func length(_ count: Int) -> [UInt8] {
        if count < 0x80 { return [UInt8(count)] }
        var bytes: [UInt8] = []
        var left = count
        while left > 0 {
            bytes.insert(UInt8(left & 0xFF), at: 0)
            left >>= 8
        }
        return [UInt8(0x80 | bytes.count)] + bytes
    }

    // MARK: - Building

    static func sequence(_ children: [Node]) -> Node { Node(identifier: [0x30], children: children) }
    static func set(_ children: [Node]) -> Node { Node(identifier: [0x31], children: children) }
    static func ia5(_ text: String) -> Node { Node(identifier: [0x16], value: Array(text.utf8)) }
    static func octets(_ bytes: [UInt8]) -> Node { Node(identifier: [0x04], value: bytes) }

    static func integer(_ value: UInt64) -> Node {
        var bytes: [UInt8] = []
        var left = value
        repeat {
            bytes.insert(UInt8(left & 0xFF), at: 0)
            left >>= 8
        } while left > 0
        if bytes[0] & 0x80 != 0 { bytes.insert(0, at: 0) }
        return Node(identifier: [0x02], value: bytes)
    }

    /// A node under one of Apple's private four-character tags -- explicit
    /// tagging, the way every field in a ticket is actually built: the tag
    /// wraps exactly one child, a plain SEQUENCE, and that SEQUENCE is what
    /// holds `children` (typically the name and the value). Confirmed against
    /// a real Cryptex1 IM4M byte for byte -- every `MANP`, `CHIP`, `caos`,
    /// `DGST`, ... field nests this way, one SEQUENCE deep under its tag.
    static func tagged(_ name: String, _ children: [Node]) -> Node {
        var number: UInt64 = 0
        for byte in Array(name.utf8) { number = (number << 8) | UInt64(byte) }
        var identifier: [UInt8] = [0xE0 | 0x1F]          // private, constructed, long form
        var seven: [UInt8] = []
        var left = number
        repeat {
            seven.insert(UInt8(left & 0x7F), at: 0)
            left >>= 7
        } while left > 0
        for index in 0..<seven.count where index < seven.count - 1 { seven[index] |= 0x80 }
        identifier += seven
        return Node(identifier: identifier, children: [sequence(children)])
    }
}

/// An IM4P out of an IPSW, and the IMG4 the device wants instead.
struct IMG4 {
    enum Failure: LocalizedError {
        case notIM4P(String)
        case notTicket

        var errorDescription: String? {
            switch self {
            case .notIM4P(let what): return L("Это не IM4P: %@", what)
            case .notTicket:         return L("В файле нет тикета IM4M")
            }
        }
    }

    /// `IM4P ::= SEQUENCE { "IM4P", type, version, payload, … }`
    struct Payload {
        var type: String
        var version: String
        var data: [UInt8]
        /// Whatever followed the payload — keybags and the properties newer
        /// builds carry. Kept so a repacked file looks like the original.
        var extras: [DER.Node]
    }

    static func readIM4P(_ bytes: [UInt8]) throws -> Payload {
        let root = try DER.parse(bytes)
        guard let children = root.children, children.count >= 4,
              children[0].text == "IM4P"
        else { throw Failure.notIM4P(L("нет заголовка")) }
        return Payload(type: children[1].text ?? "",
                       version: children[2].text ?? "",
                       data: children[3].value,
                       extras: Array(children.dropFirst(4)))
    }

    /// Wraps a payload and a ticket into what the device is given: this is what
    /// "personalising" a component means.
    static func make(payload: Payload, ticket: [UInt8]) throws -> [UInt8] {
        let im4p = DER.sequence([
            DER.ia5("IM4P"),
            DER.ia5(payload.type),
            DER.ia5(payload.version),
            DER.octets(payload.data),
        ] + payload.extras)

        let ticketNode = try DER.parse(ticket)
        guard ticketNode.children?.first?.text == "IM4M" else { throw Failure.notTicket }

        // The ticket rides in an explicit [0], as the IMG4 definition says.
        let wrapped = DER.Node(identifier: [0xA0], children: [ticketNode])
        return DER.encode(DER.sequence([DER.ia5("IMG4"), im4p, wrapped]))
    }

    /// Rebuilds an IM4P around new payload bytes — what repacking the decrypted
    /// SEP firmware comes down to.
    static func makeIM4P(type: String, version: String, data: [UInt8]) -> [UInt8] {
        DER.encode(DER.sequence([
            DER.ia5("IM4P"),
            DER.ia5(type),
            DER.ia5(version),
            DER.octets(data),
        ]))
    }
}
