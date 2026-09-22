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

    private typealias KeyFn = @convention(c) (UInt32, Bool) -> Void
    /// Looked up each time rather than cached: the library is loaded only once
    /// the machine starts, and a key pressed before then goes nowhere.
    private var keyFn: KeyFn? {
        QemuBridge.shared.symbol("inferno_input_key_hid").map { unsafeBitCast($0, to: KeyFn.self) }
    }

    static let leftShift: UInt32 = 0xE1
    static let backspace: UInt32 = 0x2A

    func send(_ usage: UInt32, down: Bool) {
        keyFn?(usage, down)
    }

    /// One whole keystroke, with Shift held around it when the character needs it.
    func type(_ usage: UInt32, shift: Bool) {
        if shift { send(Self.leftShift, down: true) }
        send(usage, down: true)
        send(usage, down: false)
        if shift { send(Self.leftShift, down: false) }
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

/// An invisible view that takes the keyboard: the on-screen one while it is
/// first responder, and a hardware keyboard's keys whenever it is.
///
/// A hardware key is passed on as the HID usage iOS reports for it, down and
/// up as they happen — so Shift, arrows, Command and held keys all work, and
/// the guest's own layout applies. The on-screen keyboard only ever produces
/// text, which is typed out through the US table in `GuestKeys`.
final class KeyCatcherView: UIView, UIKeyInput {
    override var canBecomeFirstResponder: Bool { true }
    var hasText: Bool { true }

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
            if active, !view.isFirstResponder { view.becomeFirstResponder() }
            if !active, view.isFirstResponder { view.resignFirstResponder() }
        }
    }
}
#endif
