import SwiftUI

/// A small terminal.
///
/// The guest's console is a real terminal and the programs on it behave like it:
/// `apt` redraws its progress line with a carriage return, `neofetch` prints an
/// eighteen-line logo and then walks the cursor back up to write the details
/// beside it. Dropping the escape codes leaves the words but ruins the shape, so
/// these are interpreted instead, into a grid the view can draw.
///
/// Eighty columns, because that is what the guest believes: `apt` truncates its
/// progress line at seventy-nine characters and erases it with a row of spaces
/// exactly that wide.
final class TerminalEmulator {
    struct Cell {
        var ch: Character = " "
        /// ANSI colour index, or -1 for the terminal's own default.
        var fg: Int16 = -1
        var bg: Int16 = -1
        var bold = false
    }

    /// What the guest believes, and what the view sizes its type against.
    static let width = 80

    let columns: Int
    private let scrollback: Int

    private(set) var lines: [[Cell]] = [[]]

    private var row = 0
    private var col = 0
    /// Where the current screen starts. Only `ESC[2J` moves it: a terminal with
    /// scrollback pushes the old screen up rather than losing it.
    private var screenTop = 0
    private var savedRow = 0
    private var savedCol = 0

    private var fg: Int16 = -1
    private var bg: Int16 = -1
    private var bold = false

    private enum State { case ground, esc, csi, osc, swallow }
    private var state: State = .ground
    private var params = ""

    init(columns: Int = TerminalEmulator.width, scrollback: Int = 500) {
        self.columns = columns
        self.scrollback = scrollback
    }

    func reset() {
        lines = [[]]
        row = 0; col = 0; screenTop = 0; savedRow = 0; savedCol = 0
        fg = -1; bg = -1; bold = false
        state = .ground; params = ""
    }

    /// Fed a scalar at a time rather than a `Character`, because Swift joins a
    /// carriage return and the line feed after it into one grapheme cluster
    /// whose `asciiValue` is a plain line feed — which would quietly lose every
    /// carriage return in a stream built entirely out of them.
    func feed(_ text: String) {
        for scalar in text.unicodeScalars { step(scalar) }
        trim()
    }

    // MARK: - Parsing

    private func step(_ ch: Unicode.Scalar) {
        switch state {
        case .swallow:
            state = .ground

        case .ground:
            switch ch {
            case "\u{1B}": state = .esc
            case "\r":     col = 0
            case "\n":     lineFeed()
            case "\u{8}":  col = max(col - 1, 0)
            case "\t":     col = min((col / 8 + 1) * 8, columns - 1)
            default:
                // Anything else in the control range has no meaning here.
                if ch.value < 0x20 || ch.value == 0x7F { return }
                put(ch)
            }

        case .esc:
            switch ch {
            case "[": params = ""; state = .csi
            case "]": params = ""; state = .osc
            case "7": savedRow = row; savedCol = col; state = .ground
            case "8": row = savedRow; col = savedCol; state = .ground
            case "M": row = max(row - 1, 0); state = .ground
            case "D": lineFeed(); state = .ground
            case "E": lineFeed(); col = 0; state = .ground
            // A charset selector carries one more byte that means nothing to us.
            case "(", ")", "*", "+", "#", "%": state = .swallow
            default: state = .ground
            }

        case .csi:
            if ch.value >= 0x40, ch.value <= 0x7E {
                csi(Character(ch))
                state = .ground
            } else {
                params.unicodeScalars.append(ch)
                if params.count > 48 { state = .ground }
            }

        case .osc:
            // Ends at BEL, or at ESC \ — where the backslash still has to go.
            if ch == "\u{7}" { state = .ground }
            else if ch == "\u{1B}" { state = .swallow }
        }
    }

    private func csi(_ final: Character) {
        let priv = params.hasPrefix("?") || params.hasPrefix(">") || params.hasPrefix("=")
        let body = priv ? String(params.dropFirst()) : params
        let nums = body.split(separator: ";", omittingEmptySubsequences: false).map { Int($0) ?? 0 }
        /// A missing or zero parameter means one.
        func arg(_ i: Int) -> Int {
            guard i < nums.count, nums[i] > 0 else { return 1 }
            return nums[i]
        }
        let mode = nums.first ?? 0

        switch final {
        case "A": row = max(row - arg(0), 0)
        case "B": row += arg(0); ensure(row)
        case "C": col = min(col + arg(0), columns - 1)
        case "D": col = max(col - arg(0), 0)
        case "E": row += arg(0); col = 0; ensure(row)
        case "F": row = max(row - arg(0), 0); col = 0
        case "G", "`": col = min(arg(0) - 1, columns - 1)
        case "H", "f":
            row = screenTop + (arg(0) - 1)
            col = min(nums.count > 1 ? max(nums[1], 1) - 1 : 0, columns - 1)
            ensure(row)
        case "J": eraseDisplay(mode)
        case "K": eraseLine(mode)
        case "X": eraseCells(arg(0))
        case "P": deleteCells(arg(0))
        case "@": insertCells(arg(0))
        case "m": if !priv { sgr(nums.isEmpty ? [0] : nums) }
        case "s": savedRow = row; savedCol = col
        case "u": row = savedRow; col = savedCol
        default: break
        }
    }

    private func sgr(_ nums: [Int]) {
        var i = 0
        while i < nums.count {
            let n = nums[i]
            switch n {
            case 0:         fg = -1; bg = -1; bold = false
            case 1:         bold = true
            case 22:        bold = false
            case 30...37:   fg = Int16(n - 30)
            case 39:        fg = -1
            case 40...47:   bg = Int16(n - 40)
            case 49:        bg = -1
            case 90...97:   fg = Int16(n - 90 + 8)
            case 100...107: bg = Int16(n - 100 + 8)
            case 38, 48:
                if i + 2 < nums.count, nums[i + 1] == 5 {
                    let c = Int16(clamping: nums[i + 2])
                    if n == 38 { fg = c } else { bg = c }
                    i += 2
                } else if i + 4 < nums.count, nums[i + 1] == 2 {
                    let c = TerminalEmulator.cube(nums[i + 2], nums[i + 3], nums[i + 4])
                    if n == 38 { fg = c } else { bg = c }
                    i += 4
                }
            default: break
            }
            i += 1
        }
    }

    // MARK: - The grid

    private func ensure(_ target: Int) {
        while lines.count <= target { lines.append([]) }
    }

    private func lineFeed() {
        row += 1
        ensure(row)
    }

    private func put(_ ch: Unicode.Scalar) {
        // An accent belongs to the letter before it, not to a cell of its own.
        if ch.properties.generalCategory == .nonspacingMark, col > 0,
           row < lines.count, col - 1 < lines[row].count {
            var scalars = String.UnicodeScalarView(lines[row][col - 1].ch.unicodeScalars)
            scalars.append(ch)
            lines[row][col - 1].ch = Character(String(scalars))
            return
        }
        if col >= columns {
            col = 0
            lineFeed()
        }
        ensure(row)
        while lines[row].count <= col { lines[row].append(Cell()) }
        lines[row][col] = Cell(ch: Character(ch), fg: fg, bg: bg, bold: bold)
        col += 1
    }

    private func eraseLine(_ mode: Int) {
        ensure(row)
        switch mode {
        case 0:
            if lines[row].count > col { lines[row].removeSubrange(col...) }
        case 1:
            for i in 0..<min(col + 1, lines[row].count) { lines[row][i] = Cell() }
        default:
            lines[row] = []
        }
    }

    private func eraseDisplay(_ mode: Int) {
        switch mode {
        case 0:
            eraseLine(0)
            if row + 1 < lines.count {
                for i in (row + 1)..<lines.count { lines[i] = [] }
            }
        case 1:
            eraseLine(1)
            if screenTop < row {
                for i in screenTop..<row { lines[i] = [] }
            }
        default:
            // `clear` — start a fresh screen below what is already there.
            screenTop = lines.count
            row = screenTop
            col = 0
            ensure(row)
        }
    }

    private func eraseCells(_ n: Int) {
        ensure(row)
        for i in col..<min(col + n, lines[row].count) { lines[row][i] = Cell() }
    }

    private func deleteCells(_ n: Int) {
        ensure(row)
        guard col < lines[row].count else { return }
        lines[row].removeSubrange(col..<min(col + n, lines[row].count))
    }

    private func insertCells(_ n: Int) {
        ensure(row)
        while lines[row].count < col { lines[row].append(Cell()) }
        lines[row].insert(contentsOf: Array(repeating: Cell(), count: n), at: col)
    }

    /// Keeps the buffer bounded, moving every cursor that points into it.
    private func trim() {
        guard lines.count > scrollback else { return }
        let drop = lines.count - scrollback
        lines.removeFirst(drop)
        row = max(row - drop, 0)
        screenTop = max(screenTop - drop, 0)
        savedRow = max(savedRow - drop, 0)
    }

    // MARK: - Drawing

    /// The lines to draw, trailing blanks removed — carrying them costs a great
    /// many attribute runs and shows nothing.
    func visible(maxLines: Int) -> ArraySlice<[Cell]> {
        lines[max(0, lines.count - maxLines)...]
    }

    /// The xterm palette, as plain numbers so that the same file builds for the
    /// stand on the Mac as well as for the phone. Colour 0 is lifted off black:
    /// the console's own background is already dark, and true black would simply
    /// vanish. `nil` means the terminal's default.
    static func rgb(_ index: Int16, bold: Bool) -> (Double, Double, Double)? {
        var i = index
        if i < 0 { return nil }
        if bold && i < 8 { i += 8 }

        switch i {
        case 0:  return (0.42, 0.42, 0.42)
        case 1:  return (0.80, 0.25, 0.25)
        case 2:  return (0.35, 0.68, 0.35)
        case 3:  return (0.76, 0.62, 0.18)
        case 4:  return (0.31, 0.56, 0.90)
        case 5:  return (0.72, 0.40, 0.80)
        case 6:  return (0.20, 0.64, 0.62)
        case 7:  return (0.72, 0.72, 0.72)
        case 8:  return (0.52, 0.52, 0.52)
        case 9:  return (1.00, 0.44, 0.42)
        case 10: return (0.44, 0.86, 0.47)
        case 11: return (0.93, 0.82, 0.35)
        case 12: return (0.50, 0.72, 1.00)
        case 13: return (0.88, 0.56, 0.92)
        case 14: return (0.42, 0.85, 0.85)
        case 15: return (0.95, 0.95, 0.95)
        case 16...231:
            let n = Int(i) - 16
            let levels: [Double] = [0, 0.373, 0.529, 0.686, 0.843, 1.0]
            return (levels[n / 36], levels[(n / 6) % 6], levels[n % 6])
        case 232...255:
            let v = (Double(i) - 232) / 23 * 0.85 + 0.10
            return (v, v, v)
        default:
            return nil
        }
    }

    /// Truecolor mapped onto the nearest cube entry — close enough, and it keeps
    /// every colour a single small number.
    private static func cube(_ r: Int, _ g: Int, _ b: Int) -> Int16 {
        func level(_ v: Int) -> Int { min(max(v, 0), 255) * 5 / 255 }
        return Int16(16 + 36 * level(r) + 6 * level(g) + level(b))
    }
}

/// Cuts the kernel's own chatter out of the console stream.
///
/// The guest has one console for everything: the kernel prints to it while the
/// shell writes to it, so an answer arrives torn apart by driver messages — and
/// not only between lines. A message can land in the middle of one, right after
/// an escape sequence:
///
///     ESC[0m static IOReturn AppleMobileFileIntegrityUserClient::isCdhash…
///
/// so a line is examined as a whole and cut at the point the message starts,
/// with a carriage return in its place: that is what the kernel's own line
/// ending did anyway, and whatever was on the line gets overwritten next.
final class KernelFilter {
    /// Boot timestamp, `file.c:123:`, and the C++ method names the drivers log
    /// under. The last one may appear anywhere in a line, the others only start
    /// one.
    /// Every shape a driver message was seen to take in a captured console, and
    /// each one may appear anywhere in a line, not only at its start.
    private static let anywhere = try? NSRegularExpression(pattern: [
        // A C++ method, with whatever return type was printed in front of it:
        //   static IOReturn AppleMobileFileIntegrityUserClient::isCdhashInTrust…
        "(?:(?:static|virtual|inline)\\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\\s+)?"
            + "[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_~]",
        // A plain C signature, printed whole before the message:
        //   int gl_rec_write(struct gl_ctx *const, …, size_t): Writing record…
        "(?:(?:static|virtual|inline|const|unsigned)\\s+)*[A-Za-z_][A-Za-z0-9_]*\\s+\\**"
            + "[A-Za-z_][A-Za-z0-9_]*\\([^)]{0,200}\\)\\s*(?:const\\s*)?:",
        // A source position: apfs.c:123: or tx_flush:1075:
        "\\b[A-Za-z_][A-Za-z0-9_]*(?:\\.[ch](?:pp)?)?:[0-9]+:",
        // The names things log under.
        "\\b(?:AssertMacros|AIDLog|IOMFB|IOAccessory|ACMRM|ACMKernel|RTBuddy|SEP|NAND|ANS2?)\\s*:",
        "\\bApple[A-Z][A-Za-z0-9_]*\\s*:",
        "\\b(?:apfs_[a-z_]+|nx_[a-z_]+|spaceman_[a-z_]+|obj_[a-z_]+|tx_flush|handle_mount|dev_init)\\b",
        // syslog, once launchd is up.
        "<(?:Notice|Error|Warning|Debug|Info|Critical)>:",
        "\\[effaceable",
    ].joined(separator: "|"))

    /// Shapes that only ever start a line: the boot timestamp and the date
    /// stamp launchd puts in front of its own notices.
    private static let atStart = try? NSRegularExpression(pattern: [
        "^[0-9]{6}\\.[0-9]{6} ",
        "^[A-Z][a-z]{2} [A-Z][a-z]{2} +[0-9]{1,2} [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4} ",
    ].joined(separator: "|"))

    private static let prefixes = [
        "Apple", "IOAccessory", "IOMFB", "AIDLog", "ACMRM", "ACMKernel", "apfs_", "nx_",
        "spaceman_", "tx_flush", "handle_mount", "dev_init", "SEP ", "RTBuddy", "[effaceable",
        "AssertMacros", "AppleSEP", "PMU", "hfs:", "nvme",
    ]

    /// The line being collected, and how much of it has already gone out. Held
    /// as scalars for the same reason the terminal reads them: a carriage return
    /// and the line feed after it are one `Character`, and every line here ends
    /// with exactly that pair.
    private var pending: [Unicode.Scalar] = []
    private var emitted = 0
    /// What `pending` held at the end of the previous chunk. A line that has not
    /// grown since then is one the guest has stopped writing — a prompt — so it
    /// is safe to show without waiting for a newline that may never come.
    private var settled = 0

    func reset() {
        pending = []
        emitted = 0
        settled = -1
    }

    func process(_ chunk: String) -> String {
        var out = ""
        for ch in chunk.unicodeScalars {
            pending.append(ch)
            if ch == "\n" {
                out += decide()
                pending = []
                emitted = 0
                settled = -1
            }
        }
        if !pending.isEmpty && pending.count == settled && emitted < pending.count {
            out += KernelFilter.text(pending[emitted...])
            emitted = pending.count
        }
        settled = pending.count
        return out
    }

    /// Lets go of a half-finished line — used when the whole log is replayed and
    /// there is no later chunk to settle against.
    func flush() -> String {
        guard emitted < pending.count else { return "" }
        let rest = KernelFilter.text(pending[emitted...])
        emitted = pending.count
        return rest
    }

    /// Decides what of a finished line may be shown.
    ///
    /// What replaces the cut message matters more than it looks. The kernel's
    /// own line ending really did move the cursor down, so keeping the row is
    /// the faithful thing — but `apt` redraws one progress line hundreds of
    /// times, and the kernel interrupts most of those, which would leave a
    /// hundred stale `0% [Working]` rows behind. So the row is wiped and reused
    /// instead, except when it ends in a shell prompt: that one is worth a row
    /// of its own, being the only sign the shell is there at all.
    private func decide() -> String {
        guard let cut = KernelFilter.kernelStart(pending) else {
            return KernelFilter.text(pending[emitted...])
        }
        let end = max(cut, emitted)
        guard end > 0 else { return "" }
        let ours = KernelFilter.text(pending[emitted..<max(emitted, cut)])
        return ours + (KernelFilter.endsInPrompt(pending[0..<end]) ? "\r\n" : "\r\u{1B}[K")
    }

    private static func endsInPrompt(_ scalars: ArraySlice<Unicode.Scalar>) -> Bool {
        let tail = Array(scalars.suffix(2))
        guard tail.count == 2, tail[1] == " " else { return false }
        return tail[0] == "#" || tail[0] == "$" || tail[0] == "%" || tail[0] == ">"
    }

    private static func text(_ scalars: ArraySlice<Unicode.Scalar>) -> String {
        String(String.UnicodeScalarView(scalars))
    }

    /// Where the kernel's message begins in this line, as an index into the raw
    /// characters, or nil if the line is the guest's own output.
    private static func kernelStart(_ raw: [Unicode.Scalar]) -> Int? {
        let (plain, map) = strip(raw)
        guard !plain.isEmpty else { return nil }
        let text = String(String.UnicodeScalarView(plain))

        for prefix in prefixes where text.hasPrefix(prefix) { return 0 }

        let full = NSRange(text.startIndex..., in: text)
        if let re = atStart, re.firstMatch(in: text, options: [], range: full) != nil { return 0 }
        if let re = anywhere, let m = re.firstMatch(in: text, options: [], range: full) {
            guard let r = Range(m.range, in: text) else { return nil }
            let offset = text.unicodeScalars.distance(from: text.unicodeScalars.startIndex,
                                                      to: r.lowerBound)
            return offset < map.count ? map[offset] : nil
        }
        return nil
    }

    /// Escape sequences carry no words, but they do shift every position, so the
    /// plain text is kept alongside a map back into the original.
    private static func strip(_ raw: [Unicode.Scalar]) -> (plain: [Unicode.Scalar], map: [Int]) {
        func isFinal(_ s: Unicode.Scalar) -> Bool { s.value >= 0x40 && s.value <= 0x7E }
        var plain: [Unicode.Scalar] = []
        var map: [Int] = []
        var i = 0
        while i < raw.count {
            guard raw[i] == "\u{1B}" else {
                if raw[i] != "\n" && raw[i] != "\r" {
                    plain.append(raw[i])
                    map.append(i)
                }
                i += 1
                continue
            }
            var j = i + 1
            if j < raw.count, raw[j] == "[" {
                j += 1
                while j < raw.count, !isFinal(raw[j]) { j += 1 }
                if j < raw.count { j += 1 }
            } else if j < raw.count, raw[j] == "]" {
                while j < raw.count, raw[j] != "\u{7}" { j += 1 }
                if j < raw.count { j += 1 }
            } else {
                j = min(j + 1, raw.count)
            }
            i = j
        }
        return (plain, map)
    }
}

#if canImport(UIKit)
import UIKit

typealias PlatformFont = UIFont
typealias PlatformColor = UIColor
#else
import AppKit

typealias PlatformFont = NSFont
typealias PlatformColor = NSColor
#endif

extension TerminalEmulator {
    /// Draws the grid for the text view. Runs of one colour are joined so the result
    /// carries as few attribute runs as it can — that count, not the character
    /// count, is what the text view's layout pass costs.
    func render(font: PlatformFont, maxLines: Int = 240) -> NSAttributedString {
        let out = NSMutableAttributedString()
        let rows = visible(maxLines: maxLines)

        for (offset, row) in rows.enumerated() {
            var line = row
            while let last = line.last, last.ch == " ", last.bg < 0 { line.removeLast() }

            var text = ""
            var runFg: Int16 = -1
            var runBg: Int16 = -1
            var runBold = false

            func flush() {
                guard !text.isEmpty else { return }
                var attrs: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: TerminalEmulator.uiColor(runFg, bold: runBold) ?? TerminalEmulator.defaultText,
                ]
                if let bg = TerminalEmulator.uiColor(runBg, bold: false) {
                    attrs[.backgroundColor] = bg
                }
                out.append(NSAttributedString(string: text, attributes: attrs))
                text = ""
            }

            for cell in line {
                if cell.fg != runFg || cell.bg != runBg || cell.bold != runBold {
                    flush()
                    runFg = cell.fg; runBg = cell.bg; runBold = cell.bold
                }
                text.append(cell.ch)
            }
            flush()

            if offset < rows.count - 1 {
                out.append(NSAttributedString(string: "\n", attributes: [.font: font]))
            }
        }
        return out
    }

    static func uiColor(_ index: Int16, bold: Bool) -> PlatformColor? {
        guard let (r, g, b) = rgb(index, bold: bold) else { return nil }
        return PlatformColor(red: r, green: g, blue: b, alpha: 1)
    }

    /// Text in no particular colour: whatever the system uses for labels.
    static var defaultText: PlatformColor {
        #if canImport(UIKit)
        .label
        #else
        .labelColor
        #endif
    }
}

/// What the guest console pane shows: the emulator, the filter in front of it,
/// and the finished text for the view.
///
/// Everything here is touched from the main thread only — by the view, and by
/// whoever feeds it after hopping there. It is not an actor: the terminal it
/// wraps is a plain mutable grid, and the hop has to happen anyway before the
/// published text can reach SwiftUI.
final class GuestScreen: ObservableObject {
    @Published private(set) var content = NSAttributedString()
    /// Bumped on every change — cheaper for the view to watch than the text.
    @Published private(set) var revision = 0

    private let term = TerminalEmulator()
    private let filter = KernelFilter()
    private var hideKernel = true
    private var consumed = -1
    private var font = PlatformFont.monospacedSystemFont(ofSize: 8, weight: .regular)

    /// The view knows how wide it is; the terminal does not. Eighty columns have
    /// to fit, so the size comes from there.
    func use(fontSize: CGFloat) {
        guard abs(font.pointSize - fontSize) > 0.01 else { return }
        font = .monospacedSystemFont(ofSize: fontSize, weight: .regular)
        publish()
    }

    /// Replays the whole log — on first appearance, and whenever the kernel
    /// filter is switched, since that changes what every past line should have
    /// looked like.
    func rebuild(from text: String, hideKernel: Bool, sequence: Int) {
        self.hideKernel = hideKernel
        term.reset()
        filter.reset()
        term.feed(hideKernel ? filter.process(text) + filter.flush() : text)
        consumed = sequence
        publish()
    }

    /// Starts over — used when a channel is opened afresh.
    func reset() {
        term.reset()
        filter.reset()
        consumed = -1
        publish()
    }

    /// Text from a channel that needs no filtering and carries no history to
    /// replay: whatever arrives is simply what the far end wrote.
    func append(_ text: String) {
        guard !text.isEmpty else { return }
        term.feed(text)
        publish()
    }

    /// An empty chunk is not nothing: it says the console has fallen silent,
    /// which is what lets the filter let go of a line with no newline on it yet.
    func feed(_ chunk: String, sequence: Int) {
        guard sequence > consumed else { return }
        consumed = sequence
        let text = hideKernel ? filter.process(chunk) : chunk
        guard !text.isEmpty else { return }
        term.feed(text)
        publish()
    }

    private func publish() {
        content = term.render(font: font)
        revision += 1
    }
}

#if canImport(UIKit)
/// The console is drawn by UIKit rather than by `Text`.
///
/// SwiftUI builds a `Text` layout from scratch whenever its string changes, and
/// this string changes several times a second with tens of thousands of
/// characters in it — which is how a pane of text ended up feeling slower than
/// the emulated screen beside it. A text view keeps its layout, and it can be
/// told where to sit, which is what puts the console at its last line instead of
/// its first.
struct TerminalTextView: UIViewRepresentable {
    let text: NSAttributedString
    let revision: Int
    let follow: Bool
    /// Changed by the caller to demand a jump to the bottom — on appearing, and
    /// whenever the pane is switched back to.
    let pin: Int

    final class Coordinator {
        var revision = -1
        var pin = -1
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> UITextView {
        let view = UITextView()
        view.isEditable = false
        view.isSelectable = true
        view.backgroundColor = .clear
        view.textContainerInset = UIEdgeInsets(top: 8, left: 8, bottom: 8, right: 8)
        view.textContainer.lineFragmentPadding = 0
        view.textContainer.lineBreakMode = .byClipping
        view.showsHorizontalScrollIndicator = false
        view.alwaysBounceVertical = true
        return view
    }

    func updateUIView(_ view: UITextView, context: Context) {
        let changed = context.coordinator.revision != revision
        if changed {
            context.coordinator.revision = revision
            view.attributedText = text
        }
        let demanded = context.coordinator.pin != pin
        if demanded { context.coordinator.pin = pin }

        guard (changed && follow) || demanded else { return }
        // The new text has not been laid out yet, so the height to scroll to
        // does not exist until the next turn of the run loop.
        DispatchQueue.main.async {
            let bottom = view.contentSize.height - view.bounds.height + view.adjustedContentInset.bottom
            guard bottom > 0 else { return }
            view.setContentOffset(CGPoint(x: 0, y: bottom), animated: false)
        }
    }
}
#else
/// The console on a Mac: the same reasoning as on iOS — a text view keeps its
/// layout where `Text` rebuilds it — in AppKit's shape, a text view inside a
/// scroll view.
struct TerminalTextView: NSViewRepresentable {
    let text: NSAttributedString
    let revision: Int
    let follow: Bool
    /// Changed by the caller to demand a jump to the bottom — on appearing, and
    /// whenever the pane is switched back to.
    let pin: Int

    final class Coordinator {
        var revision = -1
        var pin = -1
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.drawsBackground = false
        scroll.hasHorizontalScroller = false
        if let view = scroll.documentView as? NSTextView {
            view.isEditable = false
            view.isSelectable = true
            view.drawsBackground = false
            view.textContainerInset = NSSize(width: 8, height: 8)
            view.textContainer?.lineFragmentPadding = 0
            view.textContainer?.lineBreakMode = .byClipping
        }
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        guard let view = scroll.documentView as? NSTextView else { return }
        let changed = context.coordinator.revision != revision
        if changed {
            context.coordinator.revision = revision
            view.textStorage?.setAttributedString(text)
        }
        let demanded = context.coordinator.pin != pin
        if demanded { context.coordinator.pin = pin }

        guard (changed && follow) || demanded else { return }
        DispatchQueue.main.async { view.scrollToEndOfDocument(nil) }
    }
}
#endif
