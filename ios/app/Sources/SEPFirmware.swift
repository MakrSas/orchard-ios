import Foundation
import CommonCrypto

/// The SEP's firmware, made ready for the emulated Secure Enclave.
///
/// Apple ships it encrypted. The key is not a secret any more — it is published
/// per build on The Apple Wiki, and the person restoring looks it up for their
/// own firmware and hands it over. With the payload in the clear it is packed
/// again as an `IMG4`, this time vouched for by the forged SEP ticket, which is
/// what the emulator's patched SEP ROM accepts.
///
/// Verified against `img4lib`, the tool ChefKiss's guide uses: the decrypted
/// payload and the finished container come out byte for byte the same.
enum SEPFirmware {
    enum Failure: LocalizedError {
        case badKey
        case notBlockSized(Int)
        case decryption(Int32)

        var errorDescription: String? {
            switch self {
            case .badKey:
                return L("Ключ SEP должен быть 96 шестнадцатеричных цифр: 32 на IV и 64 на ключ")
            case .notBlockSized(let count):
                return L("Прошивка SEP не кратна 16 байтам (%d) — это не шифрованный payload", count)
            case .decryption(let status):
                return L("Расшифровка SEP не удалась (%d)", Int(status))
            }
        }
    }

    /// The IV and key, as the wiki writes them: one run of hex, the IV first.
    struct Key {
        var iv: [UInt8]
        var key: [UInt8]

        /// Takes what the person pasted. Spaces and a leading `0x` are ignored,
        /// since that is how these get copied around.
        init(hex text: String) throws {
            let cleaned = text.lowercased()
                .replacingOccurrences(of: "0x", with: "")
                .filter { $0.isHexDigit }
            guard cleaned.count == 96 else { throw Failure.badKey }
            var bytes: [UInt8] = []
            var index = cleaned.startIndex
            while index < cleaned.endIndex {
                let next = cleaned.index(index, offsetBy: 2)
                guard let byte = UInt8(cleaned[index..<next], radix: 16) else { throw Failure.badKey }
                bytes.append(byte)
                index = next
            }
            iv = Array(bytes[0..<16])
            key = Array(bytes[16..<48])
        }
    }

    /// Decrypts an `IM4P` payload with AES-256-CBC and no padding: the firmware
    /// is a whole number of blocks, and every byte of it is the firmware.
    static func decrypt(_ payload: [UInt8], with key: Key) throws -> [UInt8] {
        guard payload.count % kCCBlockSizeAES128 == 0 else {
            throw Failure.notBlockSized(payload.count)
        }
        var out = [UInt8](repeating: 0, count: payload.count)
        var moved = 0
        let status = CCCrypt(CCOperation(kCCDecrypt), CCAlgorithm(kCCAlgorithmAES), CCOptions(0),
                             key.key, key.key.count, key.iv,
                             payload, payload.count,
                             &out, out.count, &moved)
        guard status == kCCSuccess, moved == payload.count else {
            throw Failure.decryption(status)
        }
        return out
    }

    /// The whole step, from what the IPSW carries to what the machine loads.
    ///
    /// The rebuilt payload is `rsep` — the restore SEP — and carries no version
    /// and none of the properties the encrypted original had: they described the
    /// encryption, and there is none left.
    static func rebuild(im4p encrypted: [UInt8], key: Key, ticket: [UInt8],
                        type: String = "rsep") throws -> [UInt8] {
        let payload = try IMG4.readIM4P(encrypted)
        let plain = try decrypt(payload.data, with: key)
        return try IMG4.make(payload: IMG4.Payload(type: type, version: "none", data: plain, extras: []),
                             ticket: ticket)
    }
}
