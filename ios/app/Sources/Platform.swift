import SwiftUI

#if canImport(UIKit)
import UIKit

typealias PlatformImage = UIImage
#else
import AppKit

typealias PlatformImage = NSImage
#endif

/// The few places where iOS and macOS spell the same thing differently.
///
/// The app is one set of sources for both. Everything that is not the same on
/// both — the window around the guest, the battery, the text view — lives in
/// its own file per platform; this is only the small change of spelling that
/// would otherwise be an `#if` in the middle of every screen.
extension Image {
    init(platformImage: PlatformImage) {
        #if canImport(UIKit)
        self.init(uiImage: platformImage)
        #else
        self.init(nsImage: platformImage)
        #endif
    }
}

extension View {
    /// The compact title of an iOS navigation bar. A Mac window has no such
    /// bar, so there it is nothing.
    @ViewBuilder
    func inlineNavigationTitle() -> some View {
        #if os(iOS)
        navigationBarTitleDisplayMode(.inline)
        #else
        self
        #endif
    }

    /// The keyboard with `/` and `.` on it. A Mac has only the one keyboard.
    @ViewBuilder
    func urlKeyboard() -> some View {
        #if os(iOS)
        keyboardType(.URL)
        #else
        self
        #endif
    }

    /// Paths and host names are typed as they are. Only iOS capitalises.
    @ViewBuilder
    func noAutocapitalization() -> some View {
        #if os(iOS)
        textInputAutocapitalization(.never)
        #else
        self
        #endif
    }

    /// A sheet's size on a Mac. There a sheet is a window fitted to what it
    /// holds, and a list has no height of its own to fit: the package manager
    /// came up as its toolbar and nothing under it. An iOS sheet takes the
    /// screen, so there it is nothing.
    @ViewBuilder
    func sheetSize(idealWidth: CGFloat = 560, idealHeight: CGFloat = 620) -> some View {
        #if os(macOS)
        frame(minWidth: 420, idealWidth: idealWidth, minHeight: 360, idealHeight: idealHeight)
        #else
        self
        #endif
    }

    /// A sheet laid out for a phone's screen, held at a phone's width on a Mac:
    /// a card whose picture and name are drawn for one looks stretched across
    /// the width a Mac sheet would give it. As tall as the window allows.
    @ViewBuilder
    func phoneSheetSize() -> some View {
        #if os(macOS)
        frame(minWidth: 414, idealWidth: 414, maxWidth: 414, minHeight: 360, idealHeight: 896)
        #else
        self
        #endif
    }
}

/// How far the device's own furniture reaches into the screen — the island
/// above, the home indicator below. A Mac window has none.
struct ScreenInsets {
    var top: CGFloat = 0
    var bottom: CGFloat = 0
}
