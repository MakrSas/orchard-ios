#if os(iOS)
import UIKit
import ObjectiveC

/// Hands the screen edges to the guest.
///
/// The guest is an iPhone too, and its gestures start at the same edges as the
/// host's: a swipe up from the bottom reaches the host's home screen long
/// before the guest ever sees it. `preferredScreenEdgesDeferringSystemGestures`
/// makes the system ask for a second swipe, which leaves the first one for the
/// guest.
///
/// It does nothing while the home indicator is hidden: iOS ignores the deferral
/// then, and the swipe goes straight home. So whoever turns it on keeps the
/// indicator visible.
///
/// SwiftUI has no modifier for it: the value is read from the window's root
/// view controller, and `WindowGroup` owns that controller. So the getter is
/// installed on its class at runtime. It reads a flag rather than returning a
/// constant, so the deferral can be switched off again without undoing the
/// installation.
enum SystemGestures {
    /// Read by the installed getter; set through `apply(deferEdges:)`.
    private(set) static var deferEdges = false
    private static var installed = false

    static func apply(deferEdges: Bool) {
        Self.deferEdges = deferEdges
        guard let root = rootViewController() else { return }
        install(on: root)
        root.setNeedsUpdateOfScreenEdgesDeferringSystemGestures()
    }

    private static func install(on controller: UIViewController) {
        guard !installed, let cls: AnyClass = object_getClass(controller) else { return }
        let getter: @convention(block) (AnyObject) -> UInt = { _ in
            SystemGestures.deferEdges ? UIRectEdge.all.rawValue : 0
        }
        // UIRectEdge is an NSUInteger option set, hence "Q" for the return type.
        class_replaceMethod(cls,
                            #selector(getter: UIViewController.preferredScreenEdgesDeferringSystemGestures),
                            imp_implementationWithBlock(getter),
                            "Q@:")
        installed = true
    }

    private static func rootViewController() -> UIViewController? {
        let windows = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .flatMap(\.windows)
        return (windows.first(where: \.isKeyWindow) ?? windows.first)?.rootViewController
    }
}
#else
/// A Mac window has no system gestures at its edges to defer.
enum SystemGestures {
    static func apply(deferEdges: Bool) {}
}
#endif
