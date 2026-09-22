import SwiftUI

/// A logo SF Symbols does not have, drawn rather than shipped as a picture.
///
/// A bitmap would have to live in the asset catalogue, and this build compiles
/// that catalogue only when there is an app icon to compile with it — a build
/// without one would lose the logos too. An outline needs nothing: it is in the
/// binary, it stays sharp at any size, and it takes whatever colour it is given.
///
/// The outline is kept in the form the artwork is published in, SVG path data,
/// normalised into a 1×1 box so the shape can be asked for any size. Only
/// moves, lines, cubic curves and closes occur in it, and the scanner below
/// knows exactly that much.

struct Brandmark: Shape {
    let outline: String

    /// GitHub's mark, from the symbol Apple ships the template for — the
    /// semibold weight, which is what the buttons around it are set in.
    static let github = Brandmark(outline: """
        M0.5 0.0082C0.2235 0.0082 0 0.2327 0 0.5104C0 0.7324 0.1432 0.9203 0.3419
        0.9868C0.3667 0.9918 0.3758 0.976 0.3758 0.9627C0.3758 0.9511 0.375 0.9112 0.375
        0.8696C0.2359 0.8995 0.207 0.8097 0.207 0.8097C0.1846 0.7515 0.1515 0.7366 0.1515
        0.7366C0.106 0.7058 0.1548 0.7058 0.1548 0.7058C0.2053 0.7091 0.2318 0.7573 0.2318
        0.7573C0.2765 0.8338 0.3485 0.8122 0.3775 0.7989C0.3816 0.7665 0.3949 0.744 0.4089
        0.7316C0.298 0.7199 0.1813 0.6767 0.1813 0.4838C0.1813 0.4289 0.2012 0.384 0.2326
        0.3491C0.2277 0.3366 0.2103 0.2851 0.2376 0.216C0.2376 0.216 0.2798 0.2027 0.375
        0.2676C0.4158 0.2566 0.4578 0.251 0.5 0.251C0.5422 0.251 0.5853 0.2568 0.625
        0.2676C0.7202 0.2027 0.7624 0.216 0.7624 0.216C0.7898 0.2851 0.7724 0.3366 0.7674
        0.3491C0.7997 0.384 0.8187 0.4289 0.8187 0.4838C0.8187 0.6767 0.702 0.7191 0.5902
        0.7316C0.6085 0.7474 0.6242 0.7773 0.6242 0.8247C0.6242 0.892 0.6234 0.9461 0.6234
        0.9627C0.6234 0.976 0.6325 0.9918 0.6573 0.9868C0.856 0.9203 0.9992 0.7324 0.9992
        0.5104C1 0.2327 0.7757 0.0082 0.5 0.0082Z
        """)

    /// X's mark. The slit through the back stroke is part of the logo, not a
    /// mistake in the tracing — the mark on x.com has it too. It is a hole, so
    /// the winding of that subpath has to stay as it is.
    static let x = Brandmark(outline: "M0.7876 0.048L0.9409 0.048L0.6059 0.4309L1 0.952L0.6914 0.952L0.4497 0.636L0.1732 0.952L0.0197 0.952L0.3781 0.5424L0 0.048L0.3164 0.048L0.5349 0.3369ZM0.7337 0.8602L0.8187 0.8602L0.2702 0.135L0.1791 0.135Z")

    func path(in rect: CGRect) -> Path {
        // The marks are square. A frame that is not gets the mark centred in
        // it rather than a stretched one.
        let side = min(rect.width, rect.height)
        let left = rect.midX - side / 2
        let top = rect.midY - side / 2

        var path = Path()
        var cursor = CGPoint.zero       // where the pen is, in the 1×1 box
        var opened = CGPoint.zero       // where the current subpath began

        func place(_ point: CGPoint) -> CGPoint {
            CGPoint(x: left + point.x * side, y: top + point.y * side)
        }
        func point(_ x: CGFloat, _ y: CGFloat, relative: Bool) -> CGPoint {
            relative ? CGPoint(x: cursor.x + x, y: cursor.y + y) : CGPoint(x: x, y: y)
        }

        /// One command and the numbers that followed it. Extra numbers repeat
        /// the command, which is what the data does instead of writing the
        /// letter again.
        func run(_ letter: Character, _ values: [CGFloat]) {
            let step: Int
            switch letter {
            case "M", "m", "L", "l": step = 2
            case "H", "h", "V", "v": step = 1
            case "C", "c":           step = 6
            case "Z", "z":
                path.closeSubpath()
                cursor = opened
                return
            default: return
            }

            var letter = letter
            var index = 0
            while index + step <= values.count {
                let value = Array(values[index ..< index + step])
                switch letter {
                case "M", "m":
                    let end = point(value[0], value[1], relative: letter == "m")
                    path.move(to: place(end))
                    cursor = end
                    opened = end
                    // A move given more than one pair draws lines to the rest.
                    letter = letter == "m" ? "l" : "L"
                case "L", "l":
                    let end = point(value[0], value[1], relative: letter == "l")
                    path.addLine(to: place(end))
                    cursor = end
                case "H", "h":
                    let end = CGPoint(x: letter == "h" ? cursor.x + value[0] : value[0], y: cursor.y)
                    path.addLine(to: place(end))
                    cursor = end
                case "V", "v":
                    let end = CGPoint(x: cursor.x, y: letter == "v" ? cursor.y + value[0] : value[0])
                    path.addLine(to: place(end))
                    cursor = end
                default:
                    let relative = letter == "c"
                    let first = point(value[0], value[1], relative: relative)
                    let second = point(value[2], value[3], relative: relative)
                    let end = point(value[4], value[5], relative: relative)
                    path.addCurve(to: place(end), control1: place(first), control2: place(second))
                    cursor = end
                }
                index += step
            }
        }

        var letter: Character?
        var values: [CGFloat] = []
        for token in Self.scan(outline) {
            switch token {
            case .number(let value):
                values.append(value)
            case .command(let next):
                if let letter { run(letter, values) }
                letter = next
                values = []
            }
        }
        if let letter { run(letter, values) }
        return path
    }

    private enum Token {
        case command(Character)
        case number(CGFloat)
    }

    /// Splits path data into letters and numbers.
    ///
    /// Not a split on whitespace: the grammar lets numbers run together, where
    /// a minus sign or a second decimal point is all that starts the next one.
    private static func scan(_ data: String) -> [Token] {
        var tokens: [Token] = []
        var number = ""

        func finish() {
            if let value = Double(number) { tokens.append(.number(CGFloat(value))) }
            number = ""
        }

        for character in data {
            switch character {
            case "0" ... "9":
                number.append(character)
            case ".":
                if number.contains(".") { finish() }
                number.append(character)
            case "-", "+":
                // A sign belongs to the exponent it follows, and starts a new
                // number anywhere else.
                if number.hasSuffix("e") || number.hasSuffix("E") {
                    number.append(character)
                } else {
                    finish()
                    number.append(character)
                }
            case "e" where !number.isEmpty, "E" where !number.isEmpty:
                number.append(character)
            case "a" ... "z", "A" ... "Z":
                finish()
                tokens.append(.command(character))
            default:
                finish()
            }
        }
        finish()
        return tokens
    }
}
