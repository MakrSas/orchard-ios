import Foundation
import AVFoundation

/// Plays the guest's sound.
///
/// The machine has a virtio-sound card whose output QEMU's `orchard` audio
/// backend keeps in a ring (qemu/audio/orchardaudio.c); this pulls it out of
/// the ring from the render callback of an `AVAudioSourceNode`, so the
/// timing is the phone's own audio clock. What the ring does not have when
/// the callback asks is played as silence rather than waited for.
///
/// The session mixes with others: a guest's sounds should not stop the
/// music the phone is already playing, nor take over its route.
final class GuestSound {
    static let shared = GuestSound()

    private typealias ReadFn = @convention(c) (UnsafeMutableRawPointer, Int) -> Int
    private typealias FormatFn = @convention(c) (UnsafeMutablePointer<UInt32>, UnsafeMutablePointer<UInt32>) -> Void

    private var engine: AVAudioEngine?
    /// Interleaved 16-bit samples from the ring; sized once, off the audio thread.
    private var scratch = [Int16](repeating: 0, count: 8192 * 2)

    func start() {
        guard engine == nil,
              let readSym = QemuBridge.shared.symbol("orchard_audio_read"),
              let formatSym = QemuBridge.shared.symbol("orchard_audio_format") else {
            LogCapture.shared.note(L("Звук: в этой сборке библиотеки нет звукового вывода"))
            return
        }
        let read = unsafeBitCast(readSym, to: ReadFn.self)
        var rate: UInt32 = 0
        var channels: UInt32 = 0
        unsafeBitCast(formatSym, to: FormatFn.self)(&rate, &channels)
        guard rate > 0, channels == 2,
              let format = AVAudioFormat(standardFormatWithSampleRate: Double(rate), channels: 2) else { return }

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, options: [.mixWithOthers])
        try? session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let scratchCount = scratch.count
        let source = AVAudioSourceNode(format: format) { [unowned self] _, _, frameCount, bufferList in
            let buffers = UnsafeMutableAudioBufferListPointer(bufferList)
            let frames = min(Int(frameCount), scratchCount / 2)
            let got = self.scratch.withUnsafeMutableBytes { raw in
                read(raw.baseAddress!, frames * 4) / 4
            }
            guard buffers.count >= 2,
                  let left = buffers[0].mData?.assumingMemoryBound(to: Float.self),
                  let right = buffers[1].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            self.scratch.withUnsafeBufferPointer { s in
                for i in 0..<Int(frameCount) {
                    if i < got {
                        left[i] = Float(s[2 * i]) / 32768
                        right[i] = Float(s[2 * i + 1]) / 32768
                    } else {
                        left[i] = 0
                        right[i] = 0
                    }
                }
            }
            return noErr
        }
        engine.attach(source)
        engine.connect(source, to: engine.mainMixerNode, format: format)
        do {
            try engine.start()
            self.engine = engine
            LogCapture.shared.note(L("Звук: вывод гостя включён (%d Гц)", Int(rate)))
        } catch {
            LogCapture.shared.note(L("Звук: не удалось запустить вывод — %@", error.localizedDescription))
        }
    }

    func stop() {
        engine?.stop()
        engine = nil
    }
}
