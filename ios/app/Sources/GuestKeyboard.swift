#if canImport(UIKit)
import SwiftUI
import UIKit

/// Keys for the guest, as USB HID usages — what the machine's USB keyboard
/// speaks, and what iOS itself reports for a hardware key.
///
/// The iPhone guest this app shell was written for had no keyboard at all; a
/// macOS guest cannot get past its login window without one.
final class GuestKeys {
    static let shared = GuestKeys()

    /// Set while the app's own text field is open: the key catcher must not
    /// take focus back from it on every redraw of the guest's picture.
    static var composing = false

    private typealias KeyFn = @convention(c) (UInt32, Bool) -> Void
    /// Looked up each time rather than cached: the library is loaded only once
    /// the machine starts, and a key pressed before then goes nowhere.
    private var keyFn: KeyFn? {
        QemuBridge.shared.symbol("orchard_input_key_hid").map { unsafeBitCast($0, to: KeyFn.self) }
    }

    static let leftShift: UInt32 = 0xE1
    static let leftCommand: UInt32 = 0xE3
    static let returnKey: UInt32 = 0x28
    static let escape: UInt32 = 0x29
    static let backspace: UInt32 = 0x2A
    static let tab: UInt32 = 0x2B
    static let space: UInt32 = 0x2C
    static let right: UInt32 = 0x4F
    static let left: UInt32 = 0x50
    static let down: UInt32 = 0x51
    static let up: UInt32 = 0x52

    /// Keys go out one at a time, in the order they were pressed, off the
    /// main thread: each call waits for the emulator's lock.
    private let outbox = DispatchQueue(label: "guest.keys", qos: .userInteractive)

    /// A hardware key's press or release, as it happens: the timing between
    /// the two is the person's own, and is passed on as it is.
    func send(_ usage: UInt32, down: Bool) {
        outbox.async { [self] in keyFn?(usage, down) }
    }

    private typealias TapFn = @convention(c) (UInt32, UInt32) -> Void
    private var tapFn: TapFn? {
        QemuBridge.shared.symbol("orchard_input_key_tap").map { unsafeBitCast($0, to: TapFn.self) }
    }

    /// One whole keystroke, with Shift or Command held around it.
    ///
    /// Handed to the emulator in one call, so the press and the release reach
    /// the guest's keyboard queue together. Sent as separate events, each one
    /// waited for the emulator's lock on its own, and a busy guest saw the key
    /// held long enough to repeat it: every letter and every backspace twice.
    func type(_ usage: UInt32, shift: Bool, command: Bool = false) {
        let mods = (shift ? 1 : 0) | (command ? 2 : 0) as UInt32
        outbox.async { [self] in
            if let tap = tapFn {
                tap(usage, mods)
            } else {
                // An emulator library from before the tap existed.
                if command { keyFn?(Self.leftCommand, true) }
                if shift { keyFn?(Self.leftShift, true) }
                keyFn?(usage, true)
                keyFn?(usage, false)
                if shift { keyFn?(Self.leftShift, false) }
                if command { keyFn?(Self.leftCommand, false) }
            }
            // Between keystrokes, not inside one: the guest drains its
            // keyboard queue slowly, and a burst would outrun it.
            usleep(30_000)
        }
    }

    /// Types out a whole string, one keystroke at a time, `gap` apart.
    ///
    /// For text written in the app's own field: nothing reaches the guest
    /// until the person is done, and then every character is one whole
    /// keystroke, spaced far enough apart that a guest emulated at a frame or
    /// two a second has read each before the next arrives. Characters with no
    /// key on a US layout are skipped.
    func type(text: String, gap: TimeInterval = 0.15) {
        for character in text {
            guard let key = Self.usage(for: character) else { continue }
            type(key.usage, shift: key.shift)
            outbox.async { usleep(useconds_t(gap * 1_000_000)) }
        }
    }

    /// A character from the on-screen keyboard as the key that makes it on a
    /// US layout, and whether Shift is needed. The guest decides what that key
    /// means — whichever input source it has selected — so this is the one
    /// layout the table can be written against; characters it has no key for
    /// are dropped.
    static func usage(for character: Character) -> (usage: UInt32, shift: Bool)? {
        if let ascii = character.asciiValue {
            switch ascii {
            case UInt8(ascii: "a")...UInt8(ascii: "z"):
                return (0x04 + UInt32(ascii - UInt8(ascii: "a")), false)
            case UInt8(ascii: "A")...UInt8(ascii: "Z"):
                return (0x04 + UInt32(ascii - UInt8(ascii: "A")), true)
            case UInt8(ascii: "1")...UInt8(ascii: "9"):
                return (0x1E + UInt32(ascii - UInt8(ascii: "1")), false)
            case UInt8(ascii: "0"):
                return (0x27, false)
            default:
                break
            }
        }
        return punctuation[character]
    }

    private static let punctuation: [Character: (usage: UInt32, shift: Bool)] = [
        "\n": (0x28, false), "\r": (0x28, false), "\t": (0x2B, false), " ": (0x2C, false),
        "!": (0x1E, true), "@": (0x1F, true), "#": (0x20, true), "$": (0x21, true),
        "%": (0x22, true), "^": (0x23, true), "&": (0x24, true), "*": (0x25, true),
        "(": (0x26, true), ")": (0x27, true),
        "-": (0x2D, false), "_": (0x2D, true), "=": (0x2E, false), "+": (0x2E, true),
        "[": (0x2F, false), "{": (0x2F, true), "]": (0x30, false), "}": (0x30, true),
        "\\": (0x31, false), "|": (0x31, true), ";": (0x33, false), ":": (0x33, true),
        "'": (0x34, false), "\"": (0x34, true), "`": (0x35, false), "~": (0x35, true),
        ",": (0x36, false), "<": (0x36, true), ".": (0x37, false), ">": (0x37, true),
        "/": (0x38, false), "?": (0x38, true),
        // What iOS substitutes when smart punctuation slips through anyway.
        "\u{2018}": (0x34, false), "\u{2019}": (0x34, false),
        "\u{201C}": (0x34, true), "\u{201D}": (0x34, true),
    ]
}

/// An invisible view that takes a hardware keyboard's keys while it is first
/// responder.
///
/// A hardware key is passed on as the HID usage iOS reports for it, down and
/// up as they happen — so Shift, arrows, Command and held keys all work, and
/// the guest's own layout applies.
///
/// The system's on-screen keyboard never comes up: `inputView` is an empty
/// view. It lives in this process and costs tens of megabytes the first time
/// it is shown — measured on the phone as what tipped a guest already near the
/// memory limit over it. `OnScreenKeyboard` is drawn instead.
final class KeyCatcherView: UIView, UIKeyInput {
    override var canBecomeFirstResponder: Bool { true }
    var hasText: Bool { true }

    private let noKeyboard = UIView(frame: .zero)
    override var inputView: UIView? { noKeyboard }

    // Everything that would rewrite what was typed before the guest sees it.
    var autocorrectionType: UITextAutocorrectionType = .no
    var autocapitalizationType: UITextAutocapitalizationType = .none
    var spellCheckingType: UITextSpellCheckingType = .no
    var smartQuotesType: UITextSmartQuotesType = .no
    var smartDashesType: UITextSmartDashesType = .no
    var smartInsertDeleteType: UITextSmartInsertDeleteType = .no
    var keyboardType: UIKeyboardType = .asciiCapable
    var returnKeyType: UIReturnKeyType = .default

    func insertText(_ text: String) {
        for character in text {
            guard let key = GuestKeys.usage(for: character) else { continue }
            GuestKeys.shared.type(key.usage, shift: key.shift)
        }
    }

    func deleteBackward() {
        GuestKeys.shared.type(GuestKeys.backspace, shift: false)
    }

    override func pressesBegan(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        if !forward(presses, down: true) { super.pressesBegan(presses, with: event) }
    }

    override func pressesEnded(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        if !forward(presses, down: false) { super.pressesEnded(presses, with: event) }
    }

    override func pressesCancelled(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        if !forward(presses, down: false) { super.pressesCancelled(presses, with: event) }
    }

    /// Handled presses are not passed up, which is also what keeps UIKit from
    /// turning the same key into `insertText` and typing it twice.
    private func forward(_ presses: Set<UIPress>, down: Bool) -> Bool {
        var handled = false
        for press in presses {
            guard let key = press.key else { continue }
            GuestKeys.shared.send(UInt32(key.keyCode.rawValue), down: down)
            handled = true
        }
        return handled
    }
}

/// Puts a `KeyCatcherView` in the hierarchy and makes it first responder while
/// `active` is true.
struct GuestKeyboard: UIViewRepresentable {
    @Binding var active: Bool

    func makeUIView(context: Context) -> KeyCatcherView { KeyCatcherView() }

    func updateUIView(_ view: KeyCatcherView, context: Context) {
        // After this layout pass: a view not yet in a window cannot become
        // first responder.
        DispatchQueue.main.async {
            let wanted = active && !GuestKeys.composing
            if wanted, !view.isFirstResponder { view.becomeFirstResponder() }
            if !wanted, view.isFirstResponder { view.resignFirstResponder() }
        }
    }
}

/// The guest's keyboard, drawn by the app: a US layout, letters and two pages
/// of symbols, plus the keys a Mac needs that a phone keyboard lacks — Tab,
/// Escape, Command and the arrows.
///
/// Plain views on a dark ground, no blur: it sits over a picture that is being
/// redrawn, and a material behind every key would be composited on every frame.
struct OnScreenKeyboard: View {
    private enum Page { case letters, numbers, symbols }
    private enum Key: Hashable {
        case text(String), backspace, enter, shift, tab, escape, command, space
        case page(Page), left, right, up, down, compose
    }

    @State private var page = Page.letters
    /// The app's own text field, for typing a whole line at once.
    @State private var composing = false
    @State private var composed = ""
    @State private var shift = false
    /// Held for the next key only, like Shift: Command then C is a copy.
    @State private var command = false

    private var rows: [[Key]] {
        switch page {
        case .letters:
            let letters = ["qwertyuiop", "asdfghjkl", "zxcvbnm"].map { row in
                row.map { Key.text(shift ? String($0).uppercased() : String($0)) }
            }
            return [[.escape] + letters[0] + [.backspace],
                    [.tab] + letters[1] + [.enter],
                    [.shift] + letters[2] + [.text(","), .text("."), .up],
                    [.page(.numbers), .compose, .command, .space, .left, .down, .right]]
        case .numbers:
            return [[.escape] + "1234567890".map { .text(String($0)) } + [.backspace],
                    [.tab] + ["-", "/", ":", ";", "(", ")", "$", "&", "@", "\""].map { .text($0) } + [.enter],
                    [.page(.symbols)] + [".", ",", "?", "!", "'", "`"].map { .text($0) } + [.up],
                    [.page(.letters), .compose, .command, .space, .left, .down, .right]]
        case .symbols:
            return [[.escape] + ["[", "]", "{", "}", "#", "%", "^", "*", "+", "="].map { .text($0) } + [.backspace],
                    [.tab] + ["_", "\\", "|", "~", "<", ">"].map { .text($0) } + [.enter],
                    [.page(.numbers)] + [".", ",", "?", "!", "'", "`"].map { .text($0) } + [.up],
                    [.page(.letters), .compose, .command, .space, .left, .down, .right]]
        }
    }

    var body: some View {
        VStack(spacing: 6) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                HStack(spacing: 5) {
                    ForEach(Array(row.enumerated()), id: \.offset) { _, key in
                        cap(key)
                    }
                }
            }
        }
        .padding(6)
        .background(Color(white: 0.08).opacity(0.94), in: RoundedRectangle(cornerRadius: 12))
        .onChange(of: composing) { open in GuestKeys.composing = open }
        .alert(L("Ввести строку"), isPresented: $composing) {
            TextField(L("Текст"), text: $composed)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button(L("Ввести")) { send(composed, enter: false) }
            Button(L("Ввести и нажать ⏎")) { send(composed, enter: true) }
            Button(L("Отмена"), role: .cancel) { composed = "" }
        } message: {
            Text(L("Текст уйдёт гостю целиком, по одной клавише, с паузами. Годится и для пароля."))
        }
    }

    private func send(_ text: String, enter: Bool) {
        GuestKeys.shared.type(text: text)
        if enter { GuestKeys.shared.type(GuestKeys.returnKey, shift: false) }
        composed = ""
    }

    @ViewBuilder
    private func cap(_ key: Key) -> some View {
        let lit = (key == .shift && shift) || (key == .command && command)
        Button { press(key) } label: {
            label(key)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(lit ? Color.black : Color.white)
                .frame(minWidth: width(key), maxWidth: .infinity, minHeight: 34)
                .background(lit ? Color.white : Color(white: special(key) ? 0.22 : 0.32),
                            in: RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
    }

    private func width(_ key: Key) -> CGFloat {
        switch key {
        case .text: return 30
        case .space: return 160
        case .backspace, .enter, .tab, .shift, .page: return 46
        default: return 36
        }
    }

    private func special(_ key: Key) -> Bool {
        if case .text = key { return false }
        return key != .space
    }

    @ViewBuilder
    private func label(_ key: Key) -> some View {
        switch key {
        case .text(let t): Text(t)
        case .backspace: Image(systemName: "delete.left")
        case .enter: Image(systemName: "return")
        case .shift: Image(systemName: shift ? "shift.fill" : "shift")
        case .tab: Text("tab").font(.system(size: 13))
        case .escape: Text("esc").font(.system(size: 13))
        case .command: Image(systemName: "command")
        case .space: Text(" ")
        case .page(let p): Text(p == .letters ? "ABC" : p == .numbers ? "123" : "#+=").font(.system(size: 13))
        case .left: Image(systemName: "arrow.left")
        case .right: Image(systemName: "arrow.right")
        case .up: Image(systemName: "arrow.up")
        case .down: Image(systemName: "arrow.down")
        case .compose: Image(systemName: "text.cursor")
        }
    }

    private func press(_ key: Key) {
        let keys = GuestKeys.shared
        switch key {
        case .shift: shift.toggle(); return
        case .command: command.toggle(); return
        case .page(let p): page = p; return
        case .compose: composing = true; return
        default: break
        }
        let cmd = command
        let plain = { (usage: UInt32) in keys.type(usage, shift: false, command: cmd) }
        switch key {
        case .text(let t):
            for character in t {
                if let k = GuestKeys.usage(for: character) { keys.type(k.usage, shift: k.shift, command: cmd) }
            }
        case .backspace: plain(GuestKeys.backspace)
        case .enter: plain(GuestKeys.returnKey)
        case .tab: plain(GuestKeys.tab)
        case .escape: plain(GuestKeys.escape)
        case .space: plain(GuestKeys.space)
        case .left: plain(GuestKeys.left)
        case .right: plain(GuestKeys.right)
        case .up: plain(GuestKeys.up)
        case .down: plain(GuestKeys.down)
        case .shift, .command, .page, .compose: break
        }
        command = false
        // One-shot, as on the phone's own keyboard.
        if shift, case .text = key { shift = false }
    }
}
#endif
