#if os(macOS)
import AppKit
import SwiftUI

/// The Mac app, built the way iPhone Mirroring is.
///
/// At rest the window is nothing but the guest's screen, with the device's own
/// rounded corners and a hairline around them. Bring the pointer to the top of
/// it and the window chrome comes up around the screen: a dark frame, and above
/// the screen a bar with the traffic lights on the left and, on the right, the
/// phone's menu and a button for the settings. Leave, and it goes again.
///
/// Taken apart, iPhone Mirroring does this with a window whose shape is the
/// device (SwiftUI's windowContentShape, which is not public), a titlebar
/// container of its own holding the standard window buttons, a chrome that is
/// shown and hidden on a timer, and a content aspect ratio with the chrome as
/// insets. None of those SwiftUI modifiers can be called from outside Apple, so
/// the same result is built from AppKit here: a borderless window, which has no
/// system frame to disagree with the device's corners, drawn entirely by the
/// content, the standard buttons placed in the bar, and the aspect ratio kept
/// by the window's delegate.
///
/// One window and one model. A second window would be a second emulator, in
/// this process, where there is room for one.
@main
struct InfernoMacApp: App {
    @NSApplicationDelegateAdaptor(MacAppDelegate.self) private var delegate

    init() {
        Bootstrap.prepareDocuments()
        // Start capturing before anything can fail, so the reason is on screen.
        LogCapture.shared.start()
        LogCapture.shared.note(L("Сборка приложения: %@", BuildInfo.stamp))
        LogCapture.shared.noteDevice()
        JIT.prepare()
    }

    var body: some Scene {
        // The app needs a scene to exist. The settings are not in it: a Settings
        // scene is opened by showSettingsWindow:, which since macOS 14 does
        // nothing when sent from a window SwiftUI did not create — and the
        // phone's window is AppKit's. They get a window of their own instead,
        // on the same ⌘,.
        SwiftUI.Settings { EmptyView() }
        .commands {
            CommandGroup(replacing: .appSettings) {
                Button(L("Параметры…")) { MacSettings.open() }
                    .keyboardShortcut(",")
            }
            // A second window would be a second emulator.
            CommandGroup(replacing: .newItem) {}
            CommandGroup(after: .toolbar) {
                Button(L("Реальный размер")) { MacAppDelegate.window?.scale(to: 1) }
                    .keyboardShortcut("0")
                Button(L("Крупнее")) { MacAppDelegate.window?.scale(by: 1.1) }
                    .keyboardShortcut("+")
                Button(L("Мельче")) { MacAppDelegate.window?.scale(by: 1 / 1.1) }
                    .keyboardShortcut("-")
                Divider()
            }
        }
    }
}

extension VMModel {
    /// The Mac app's one model, shared by the window and the settings.
    static let mac = VMModel()

    /// Keeps the Mac from napping the app while its machine runs.
    ///
    /// An app whose window is covered or on another space gets App Nap: its
    /// timers are coalesced and its threads throttled, and the guest inside it
    /// slows down with them — which also skews any frame count taken while the
    /// window is out of sight.
    private static var awake: NSObjectProtocol?

    static func holdAwake(_ on: Bool) {
        if on, awake == nil {
            awake = ProcessInfo.processInfo.beginActivity(
                options: [.userInitiatedAllowingIdleSystemSleep, .latencyCritical],
                reason: "The guest machine is running")
        } else if !on, let activity = awake {
            ProcessInfo.processInfo.endActivity(activity)
            awake = nil
        }
    }
}

enum MacSettings {
    private static var window: NSWindow?

    /// Brings up the settings window, making it the first time.
    static func open() {
        if window == nil {
            let settings = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 600),
                                    styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                                    backing: .buffered, defer: false)
            settings.title = L("Параметры")
            settings.toolbarStyle = .unified
            settings.isReleasedWhenClosed = false
            let hosting = NSHostingView(rootView: MacSettingsView(model: VMModel.mac)
                .preferredColorScheme(.dark))
            // The split view's sidebar and each page's title belong in the
            // window's toolbar, which a hosting view only fills when asked.
            hosting.sceneBridgingOptions = .all
            settings.contentView = hosting
            settings.contentMinSize = NSSize(width: 680, height: 480)
            settings.center()
            window = settings
        }
        NSApp.activate()
        window?.makeKeyAndOrderFront(nil)
    }
}

final class MacAppDelegate: NSObject, NSApplicationDelegate {
    static var window: PhoneWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let window = PhoneWindow(model: VMModel.mac)
        Self.window = window
        window.scale(to: 1, animate: false)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate()
        // For testing from a script: `open Inferno.app --args -autostart YES`
        // starts the machine without anyone reaching for the menu. A launch
        // argument lives only in this process's defaults and is never saved.
        if UserDefaults.standard.bool(forKey: "autostart") { VMModel.mac.start() }
        // Likewise `-openSettings YES`, to look at the settings window from a script.
        if UserDefaults.standard.bool(forKey: "openSettings") { MacSettings.open() }
    }

    /// Quitting under a running machine would stop QEMU in the middle of a
    /// write, as swiping the app away does on the phone. So the machine is told
    /// to quit first — QMP `quit` flushes the disks — and the app follows once
    /// it has stopped, or after a few seconds if it does not.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        let model = VMModel.mac
        guard model.isRunning else { return .terminateNow }
        model.shutdown()
        let deadline = Date().addingTimeInterval(8)
        func wait() {
            if !model.isRunning || Date() >= deadline {
                NSApp.reply(toApplicationShouldTerminate: true)
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { wait() }
            }
        }
        wait()
        return .terminateLater
    }
}

// MARK: - The window

/// How the chrome sits around the device's screen, in points.
enum PhoneChrome {
    /// The corners of an ordinary window on this system, for the top of the
    /// chrome — asked of a titled window rather than guessed, since each macOS
    /// draws its own. iPhone Mirroring does the same: macOS corners above, the
    /// device's own below.
    static let windowCornerRadius: CGFloat = {
        let probe = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 200, height: 200),
                             styleMask: [.titled, .closable, .miniaturizable, .resizable],
                             backing: .buffered, defer: true)
        let selector = NSSelectorFromString("_cornerRadius")
        guard probe.responds(to: selector), let method = class_getInstanceMethod(NSWindow.self, selector) else {
            return 16
        }
        let read = unsafeBitCast(method_getImplementation(method),
                                 to: (@convention(c) (AnyObject, Selector) -> Double).self)
        let radius = read(probe, selector)
        return radius > 0 ? CGFloat(radius) : 16
    }()
    /// The bar above the screen, which holds the traffic lights.
    static let top: CGFloat = 40
    /// The frame at the sides and below.
    static let side: CGFloat = 10
    /// The smallest width the screen may be shrunk to.
    static let minimumScreenWidth: CGFloat = 220
}

/// A borderless window that behaves like a titled one: it becomes key, it can
/// be closed, minimised and taken full screen, and it keeps the device's
/// proportions when it is resized.
final class PhoneWindow: NSWindow, NSWindowDelegate {
    private let chromeState = ChromeState()

    init(model: VMModel) {
        let screen = PhoneWindow.screenSize(for: Settings.shared.panel)
        super.init(contentRect: NSRect(origin: .zero, size: PhoneWindow.frameSize(forScreen: screen)),
                   styleMask: [.borderless, .closable, .miniaturizable, .resizable],
                   backing: .buffered, defer: false)
        title = "Inferno"
        isOpaque = false
        backgroundColor = .clear
        hasShadow = true
        isMovableByWindowBackground = false
        isReleasedWhenClosed = false
        collectionBehavior = [.fullScreenPrimary, .managed]
        delegate = self
        minSize = PhoneWindow.frameSize(forScreen: CGSize(width: PhoneChrome.minimumScreenWidth,
                                                           height: PhoneChrome.minimumScreenWidth * screen.height / screen.width))
        let hosting = FirstClickHostingView(rootView: MacRootView(model: model, chrome: chromeState))
        hosting.wantsLayer = true
        hosting.layer?.backgroundColor = .clear
        contentView = hosting
    }

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    /// The bar's red and yellow buttons, and ⌘W and ⌘M. NSWindow's own
    /// performClose: and performMiniaturize: act only for a window with those
    /// buttons in its title bar; a borderless window has none, and they did
    /// nothing at all. So the window does what the buttons mean itself.
    override func performClose(_ sender: Any?) { close() }
    override func performMiniaturize(_ sender: Any?) { miniaturize(sender) }

    /// ⌘W is greyed out in a borderless window for the same reason.
    override func validateUserInterfaceItem(_ item: NSValidatedUserInterfaceItem) -> Bool {
        item.action == #selector(performClose(_:)) || super.validateUserInterfaceItem(item)
    }

    /// The guest's screen in points: its framebuffer at the panel's own scale.
    static func screenSize(for panel: String) -> CGSize {
        let pixels = (GuestPanel(rawValue: panel) ?? .iphone11).pixels
        return CGSize(width: CGFloat(pixels.width) / CGFloat(pixels.scale),
                      height: CGFloat(pixels.height) / CGFloat(pixels.scale))
    }

    static func frameSize(forScreen screen: CGSize) -> CGSize {
        CGSize(width: screen.width + PhoneChrome.side * 2,
               height: screen.height + PhoneChrome.top + PhoneChrome.side)
    }

    /// The panel's proportions, taken from the settings each time.
    private var aspect: CGFloat {
        let screen = PhoneWindow.screenSize(for: Settings.shared.panel)
        return screen.height / screen.width
    }

    /// Resizing keeps the screen's proportions; the chrome is added around it.
    /// Not in full screen, where the frame is the display's and the screen is
    /// fitted inside it instead: held to the proportions there, the window grew
    /// taller than the display and the picture was cut off.
    func windowWillResize(_ sender: NSWindow, to frameSize: NSSize) -> NSSize {
        guard !styleMask.contains(.fullScreen), !chromeState.fullScreen else { return frameSize }
        let width = max(frameSize.width - PhoneChrome.side * 2, PhoneChrome.minimumScreenWidth)
        return PhoneWindow.frameSize(forScreen: CGSize(width: width, height: width * aspect))
    }

    func windowDidResize(_ notification: Notification) { invalidateShadow() }

    /// The window is the app: closing it quits, and the quit goes through
    /// applicationShouldTerminate, which lets the machine write its disks. Even
    /// with the settings still open — they would otherwise be left holding a
    /// machine that no window shows.
    func windowWillClose(_ notification: Notification) {
        DispatchQueue.main.async { NSApp.terminate(nil) }
    }

    func window(_ window: NSWindow, willUseFullScreenContentSize proposedSize: NSSize) -> NSSize { proposedSize }
    func windowWillEnterFullScreen(_ notification: Notification) { chromeState.fullScreen = true }
    func windowDidExitFullScreen(_ notification: Notification) {
        chromeState.fullScreen = false
        invalidateShadow()
    }

    /// Scales the screen relative to its actual size in points, keeping the
    /// window's top edge where it is, and never past the display.
    func scale(to factor: CGFloat, animate: Bool = true) {
        let screen = PhoneWindow.screenSize(for: Settings.shared.panel)
        resizeScreen(toWidth: screen.width * factor, animate: animate)
    }

    func scale(by factor: CGFloat) {
        resizeScreen(toWidth: (frame.width - PhoneChrome.side * 2) * factor)
    }

    func resizeScreen(toWidth requested: CGFloat, animate: Bool = true) {
        guard !styleMask.contains(.fullScreen) else { return }
        let available = (screen ?? NSScreen.main)?.visibleFrame.size ?? CGSize(width: 1440, height: 900)
        // A margin above and below, so the whole window is in view with the Dock.
        let widestByHeight = (available.height - 24 - PhoneChrome.top - PhoneChrome.side) / aspect
        let width = min(max(requested, PhoneChrome.minimumScreenWidth), widestByHeight, available.width)
        let size = PhoneWindow.frameSize(forScreen: CGSize(width: width, height: width * aspect))
        var rect = frame
        rect.origin.y += rect.height - size.height
        rect.size = size
        setFrame(rect, display: true, animate: animate)
    }

    /// Called when the panel changes in the settings: the same width, the new
    /// proportions.
    func refit() { resizeScreen(toWidth: frame.width - PhoneChrome.side * 2) }
}

/// Lets the first click on an inactive window reach the guest as a touch,
/// instead of only bringing the window forward.
private final class FirstClickHostingView<Content: View>: NSHostingView<Content> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}

/// Whether the chrome is out, shared between the window and its content.
final class ChromeState: ObservableObject {
    /// Full screen: no frame and no corners, and the screen as large as fits.
    @Published var fullScreen = false
    /// `-forceChrome YES` keeps it out, for looking at it without a pointer —
    /// iPhone Mirroring has the same switch for the same reason.
    static let forced = UserDefaults.standard.bool(forKey: "forceChrome")
    @Published var shown = ChromeState.forced
}

// MARK: - The content

struct MacRootView: View {
    @ObservedObject var model: VMModel
    @ObservedObject var chrome: ChromeState
    @ObservedObject private var settings = Settings.shared
    @State private var pane: Pane = .screen
    @State private var fullScreen = false
    @State private var pickFile = false
    @State private var pickIPA = false
    @State private var pickDEB = false
    @State private var askPath = false
    @State private var hideChrome: DispatchWorkItem?
    @State private var resizeStart: (mouse: CGPoint, width: CGFloat)?

    /// How close to the screen's top the pointer has to come for the chrome.
    private static let revealDepth: CGFloat = 60

    var body: some View {
        GeometryReader { geo in
            let screen = screenRect(in: geo.size)
            let radius = screen.width * GuestBezel.radiusOverWidth
            let outer = radius + PhoneChrome.side
            let full = chrome.fullScreen
            ZStack(alignment: .topLeading) {
                // The chrome: a body the size of the window. Invisible at rest,
                // so the window is only the screen. In full screen there is no
                // window to shape: the body is plain black, always there, and
                // has no corners.
                chromeBody(full: full, bottom: outer)
                    .overlay(
                        chromeShape(bottom: outer, top: PhoneChrome.windowCornerRadius)
                            .strokeBorder(.white.opacity(full ? 0 : 0.16), lineWidth: 1)
                    )
                    .overlay(alignment: .bottomTrailing) { if !full { resizeGrip } }
                    .opacity(chrome.shown || full ? 1 : 0)
                    .allowsHitTesting(chrome.shown && !full)

                screenContent
                    .frame(width: screen.width, height: screen.height)
                    .background(Color.black)
                    .clipShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: radius, style: .continuous)
                            .strokeBorder(.white.opacity(chrome.shown ? 0.06 : 0.16), lineWidth: 1)
                    )
                    .offset(x: screen.minX, y: screen.minY)

                // The bar across the top, over the body in a window and over
                // black in full screen.
                bar
                    .frame(width: geo.size.width)
                    .opacity(chrome.shown ? 1 : 0)
                    .allowsHitTesting(chrome.shown)
            }
            .onContinuousHover { phase in
                switch phase {
                case .active(let point):
                    if point.y < screen.minY + Self.revealDepth || chrome.shown && geo.frame(in: .local).contains(point) {
                        reveal(true)
                    }
                case .ended:
                    reveal(false)
                }
            }
        }
        .ignoresSafeArea()
        .modifier(GuestActions(model: model, pane: pane, pickFile: $pickFile, pickIPA: $pickIPA,
                               pickDEB: $pickDEB, askPath: $askPath))
        .onChange(of: settings.panel) { _ in MacAppDelegate.window?.refit() }
        // The phone's "full screen" is the Mac's own full screen.
        .onChange(of: fullScreen) { wanted in
            guard wanted else { return }
            MacAppDelegate.window?.toggleFullScreen(nil)
            fullScreen = false
        }
        .onChange(of: chrome.shown) { _ in
            // The shadow is traced from what is drawn, and what is drawn just
            // changed shape.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { MacAppDelegate.window?.invalidateShadow() }
        }
    }

    /// The chrome's outline: a window's corners at the top, the device's below.
    private func chromeShape(bottom: CGFloat, top: CGFloat) -> UnevenRoundedRectangle {
        UnevenRoundedRectangle(topLeadingRadius: top, bottomLeadingRadius: bottom,
                               bottomTrailingRadius: bottom, topTrailingRadius: top, style: .continuous)
    }

    /// The chrome's body. In a window it is whatever lies behind the window,
    /// blurred, as iPhone Mirroring's frame is: a material in a transparent
    /// window blends with the desktop, not with the window. Under a dark veil,
    /// and dark whatever the system's appearance, so that the white glyphs and
    /// the traffic lights read on a light desktop too.
    @ViewBuilder
    private func chromeBody(full: Bool, bottom: CGFloat) -> some View {
        if full {
            Color.black
        } else {
            let shape = chromeShape(bottom: bottom, top: PhoneChrome.windowCornerRadius)
            shape.fill(.ultraThinMaterial)
                .overlay(shape.fill(Color.black.opacity(0.35)))
                .environment(\.colorScheme, .dark)
        }
    }

    /// Where the device's screen goes: inside the chrome's insets, at the
    /// panel's proportions, centred — which in a window the delegate keeps in
    /// proportion is simply the inset rectangle, and in full screen is the
    /// largest screen that fits.
    private func screenRect(in size: CGSize) -> CGRect {
        let panel = PhoneWindow.screenSize(for: settings.panel)
        let area = CGRect(x: PhoneChrome.side, y: PhoneChrome.top,
                          width: max(size.width - PhoneChrome.side * 2, 1),
                          height: max(size.height - PhoneChrome.top - PhoneChrome.side, 1))
        let scale = min(area.width / panel.width, area.height / panel.height)
        let fitted = CGSize(width: panel.width * scale, height: panel.height * scale)
        return CGRect(x: area.midX - fitted.width / 2, y: area.midY - fitted.height / 2,
                      width: fitted.width, height: fitted.height)
    }

    @ViewBuilder
    private var screenContent: some View {
        if !model.missing.isEmpty {
            SetupView(model: model)
        } else {
            switch pane {
            case .screen:
                ScreenView(model: model, picture: model.picture, serial: model.serial, fullScreen: $fullScreen)
            case .terminal:
                TerminalView(model: model)
            }
        }
    }

    /// The bar above the screen: the traffic lights, then the menu and the
    /// settings on the right. Dragged, it moves the window, as a title bar does.
    private var bar: some View {
        HStack(spacing: 8) {
            TrafficLights()
                .fixedSize(width: 54, height: 16)
            Spacer()
            ControlMenu(model: model, pane: $pane, fullScreen: $fullScreen,
                        pickFile: $pickFile, pickIPA: $pickIPA, pickDEB: $pickDEB,
                        askPath: $askPath, discSize: 24)
            Button {
                MacSettings.open()
            } label: {
                BarGlyph(systemName: "gearshape")
            }
            .buttonStyle(.plain)
            .help(L("Параметры…"))
        }
        .padding(.leading, PhoneChrome.side + 8)
        .padding(.trailing, PhoneChrome.side + 6)
        .frame(height: PhoneChrome.top)
        .contentShape(Rectangle())
        .gesture(WindowDragGesture())
    }

    /// A corner to pull, as a titled window's edge would be. The window keeps
    /// its proportions, so the width decides.
    private var resizeGrip: some View {
        Color.clear
            .frame(width: 22, height: 22)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside { NSCursor.frameResize(position: .bottomRight, directions: .all).push() } else { NSCursor.pop() }
            }
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { _ in
                        guard let window = MacAppDelegate.window else { return }
                        let mouse = NSEvent.mouseLocation
                        if resizeStart == nil {
                            resizeStart = (mouse, window.frame.width - PhoneChrome.side * 2)
                        }
                        guard let start = resizeStart else { return }
                        window.resizeScreen(toWidth: start.width + (mouse.x - start.mouse.x))
                    }
                    .onEnded { _ in resizeStart = nil }
            )
    }

    /// Out at once; away a moment after the pointer leaves, so that crossing
    /// the top of the window on the way somewhere else does not flash it.
    private func reveal(_ show: Bool) {
        guard !ChromeState.forced else { return }
        hideChrome?.cancel()
        if show {
            guard !chrome.shown else { return }
            withAnimation(.easeOut(duration: 0.18)) { chrome.shown = true }
        } else {
            let work = DispatchWorkItem { withAnimation(.easeIn(duration: 0.25)) { chrome.shown = false } }
            hideChrome = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0, execute: work)
        }
    }
}

/// A bar button as iPhone Mirroring draws them: a plain glyph, with a soft
/// rounded highlight while the pointer is on it.
struct BarGlyph: View {
    let systemName: String
    @State private var hovered = false

    var body: some View {
        Image(systemName: systemName)
            .font(.system(size: 15, weight: .regular))
            .foregroundStyle(.white.opacity(hovered ? 0.95 : 0.7))
            .frame(width: 30, height: 26)
            .background(.white.opacity(hovered ? 0.12 : 0), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .contentShape(Rectangle())
            .onHover { hovered = $0 }
    }
}

/// The three standard window buttons, the real ones, placed in the bar the way
/// iPhone Mirroring places them — so they look, highlight and behave exactly as
/// on any other window. Green is greyed out, as it is there; full screen is
/// still in the menu.
private struct TrafficLights: NSViewRepresentable {
    func makeNSView(context: Context) -> ButtonStack { ButtonStack() }
    func updateNSView(_ view: ButtonStack, context: Context) {}

    final class ButtonStack: NSStackView {
        private var buttons: [NSButton] = []

        init() {
            super.init(frame: .zero)
            orientation = .horizontal
            spacing = 6
            for kind in [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton] {
                guard let button = NSWindow.standardWindowButton(kind, for: [.titled, .closable, .miniaturizable, .resizable])
                else { continue }
                buttons.append(button)
                addArrangedSubview(button)
            }
        }

        required init?(coder: NSCoder) { fatalError("not used") }

        /// Aimed at the window once there is one: a borderless window does not
        /// wire its buttons up by itself. Red and yellow land in PhoneWindow's
        /// own performClose: and performMiniaturize:, since NSWindow's do
        /// nothing without a title bar. Green gets no action and is disabled.
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window, buttons.count == 3 else { return }
            buttons[0].target = window
            buttons[0].action = #selector(NSWindow.performClose(_:))
            buttons[1].target = window
            buttons[1].action = #selector(NSWindow.performMiniaturize(_:))
            buttons[2].action = nil
            buttons[2].isEnabled = false
        }

        /// The glyphs appear on all three together while the pointer is over
        /// any of them, as in a title bar. Each button asks the container
        /// through _mouseInGroup:, but only once it is told the pointer came or
        /// went — mouseEnteredOrExited, which a title bar sends. Marked for
        /// redrawing alone, a button draws what it drew before, and only green,
        /// which follows the pointer by itself, ever lit up.
        private var inside = false
        override func updateTrackingAreas() {
            super.updateTrackingAreas()
            trackingAreas.forEach(removeTrackingArea)
            addTrackingArea(NSTrackingArea(rect: bounds, options: [.mouseEnteredAndExited, .activeAlways],
                                           owner: self, userInfo: nil))
        }
        override func mouseEntered(with event: NSEvent) { pointer(inside: true) }
        override func mouseExited(with event: NSEvent) { pointer(inside: false) }
        @objc func _mouseInGroup(_ button: NSButton) -> Bool { inside }

        private func pointer(inside: Bool) {
            self.inside = inside
            let notify = NSSelectorFromString("mouseEnteredOrExited")
            for button in buttons {
                if button.responds(to: notify) { _ = button.perform(notify) }
                button.needsDisplay = true
            }
        }
    }
}

private extension View {
    func fixedSize(width: CGFloat, height: CGFloat) -> some View {
        frame(width: width, height: height)
    }
}
#endif
