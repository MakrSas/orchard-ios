#if canImport(UIKit)
import SwiftUI
import UIKit

/// The screen as a trackpad: the guest's pointer moves by how far a finger
/// travels, not to where it lands, so it is never under the finger.
///
/// - one finger: move the pointer; tap to click
/// - one finger, held still for a moment and then moved: drag
/// - two fingers: tap for a right click; move to scroll
///
/// The pointer's position is kept here, in guest pixels, and sent as an
/// absolute position — the machine's pointing device is a tablet.
final class Trackpad {
    private typealias PointerFn = @convention(c) (Int32, Int32, UInt32, Int32) -> Void
    private var pointerFn: PointerFn? {
        QemuBridge.shared.symbol("orchard_input_pointer").map { unsafeBitCast($0, to: PointerFn.self) }
    }

    /// Guest pixels per point of finger travel, before acceleration.
    var speed: CGFloat = 1.2
    private(set) var cursor = CGPoint(x: -1, y: -1)
    private var buttons: UInt32 = 0
    private var scrollCarry: CGFloat = 0

    private func clamp(_ size: CGSize) {
        if cursor.x < 0 { cursor = CGPoint(x: size.width / 2, y: size.height / 2) }
        cursor.x = min(max(cursor.x, 0), size.width - 1)
        cursor.y = min(max(cursor.y, 0), size.height - 1)
    }

    private func send(_ size: CGSize, wheel: Int32 = 0) {
        clamp(size)
        pointerFn?(Int32(cursor.x), Int32(cursor.y), buttons, wheel)
    }

    /// A finger moved by `delta` points at `velocity` points a second; `scale`
    /// is guest pixels per point of the drawn picture.
    func move(by delta: CGPoint, velocity: CGPoint, scale: CGFloat, size: CGSize) {
        let v = hypot(velocity.x, velocity.y)
        // Slow and precise when the finger is, faster when it flicks.
        let gain = speed * scale * min(1 + v / 1200, 3)
        clamp(size)
        cursor.x += delta.x * gain
        cursor.y += delta.y * gain
        send(size)
    }

    func click(right: Bool, size: CGSize) {
        let bit: UInt32 = right ? 2 : 1
        buttons |= bit
        send(size)
        buttons &= ~bit
        send(size)
    }

    func press(_ down: Bool, size: CGSize) {
        if down { buttons |= 1 } else { buttons &= ~1 }
        send(size)
    }

    /// Two fingers moved by `dy` points; every 12 points is one wheel step.
    func scroll(by dy: CGFloat, size: CGSize) {
        scrollCarry += dy
        let steps = Int32(scrollCarry / 12)
        guard steps != 0 else { return }
        scrollCarry -= CGFloat(steps) * 12
        // Fingers moving up read as the wheel turning down, as on a Mac's own
        // trackpad before the guest applies its scroll direction.
        send(size, wheel: -steps)
    }
}

/// Catches the gestures `Trackpad` understands over the guest's picture.
struct TrackpadSurface: UIViewRepresentable {
    let trackpad: Trackpad
    /// Guest framebuffer size, and guest pixels per point of the picture.
    let size: CGSize
    let scale: CGFloat

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        let c = context.coordinator

        let move = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.move(_:)))
        move.minimumNumberOfTouches = 1
        move.maximumNumberOfTouches = 1

        let drag = UILongPressGestureRecognizer(target: c, action: #selector(Coordinator.drag(_:)))
        drag.minimumPressDuration = 0.35
        drag.allowableMovement = 8

        let scroll = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.scroll(_:)))
        scroll.minimumNumberOfTouches = 2
        scroll.maximumNumberOfTouches = 2

        let tap = UITapGestureRecognizer(target: c, action: #selector(Coordinator.tap(_:)))
        let rightTap = UITapGestureRecognizer(target: c, action: #selector(Coordinator.rightTap(_:)))
        rightTap.numberOfTouchesRequired = 2

        // A held finger is a drag, not a move, once it has waited long enough.
        move.require(toFail: drag)
        for g in [move, drag, scroll, tap, rightTap] as [UIGestureRecognizer] {
            g.delegate = c
            view.addGestureRecognizer(g)
        }
        return view
    }

    func updateUIView(_ view: UIView, context: Context) {
        context.coordinator.parent = self
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var parent: TrackpadSurface
        private var last = CGPoint.zero

        init(_ parent: TrackpadSurface) { self.parent = parent }

        private var pad: Trackpad { parent.trackpad }

        @objc func move(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            pad.move(by: t, velocity: g.velocity(in: g.view), scale: parent.scale, size: parent.size)
            g.setTranslation(.zero, in: g.view)
        }

        @objc func drag(_ g: UILongPressGestureRecognizer) {
            let p = g.location(in: g.view)
            switch g.state {
            case .began:
                last = p
                pad.press(true, size: parent.size)
            case .changed:
                pad.move(by: CGPoint(x: p.x - last.x, y: p.y - last.y), velocity: .zero,
                         scale: parent.scale, size: parent.size)
                last = p
            default:
                pad.press(false, size: parent.size)
            }
        }

        @objc func scroll(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            pad.scroll(by: t.y, size: parent.size)
            g.setTranslation(.zero, in: g.view)
        }

        @objc func tap(_ g: UITapGestureRecognizer) {
            pad.click(right: false, size: parent.size)
        }

        @objc func rightTap(_ g: UITapGestureRecognizer) {
            pad.click(right: true, size: parent.size)
        }

        // Taps and pans of different finger counts may be recognised together;
        // the finger counts keep them apart.
        func gestureRecognizer(_ g: UIGestureRecognizer,
                               shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
            false
        }
    }
}
#endif
