import Foundation
import CommonCrypto

/// The host half of a restore: everything `idevicerestore` does once the guest's
/// `restored` has answered.
///
/// Two conversations run here. `restored` itself speaks property lists behind a
/// four-byte length and asks for the firmware one piece at a time; ASR is a
/// second connection with no framing at all, over which the filesystem image is
/// streamed straight out of the IPSW — it is stored there uncompressed, so
/// nothing has to be unpacked first, which is what makes this fit on a phone.
///
/// Every component is personalised the same way: the `IM4P` out of the IPSW is
/// wrapped together with the forged ticket. Nothing contacts Apple.
final class RestoreClient {
    enum Failure: LocalizedError {
        case brokenProtocol(String)
        case unsupported(String)
        case guestFailed(UInt64)

        var errorDescription: String? {
            switch self {
            case .brokenProtocol(let what): return L("Рестор: гость сказал не то — %@", what)
            case .unsupported(let what):    return L("Рестор: этого я не умею — %@", what)
            case .guestFailed(let code):    return L("Рестор: гость прервал установку, код %@",
                                                    RestoreClient.statusName(code))
            }
        }
    }

    private let usb: GuestUSB
    private let ipsw: IPSW
    private let identity: [String: Any]
    private let ticket: [UInt8]
    /// A real device's own Cryptex1 IM4M, for `Cryptex1.forge` to build this
    /// build's ticket out of -- only iOS 16+ ever asks for one.
    private let cryptexTemplate: [UInt8]?
    private let note: (String) -> Void
    private let reportProgress: (String, Double) -> Void
    /// The last time progress was passed on, so that a transfer running at ten
    /// megabytes a second does not redraw the interface eighty times a second.
    private var lastProgress = Date.distantPast

    init(usb: GuestUSB, ipsw: IPSW, identity: [String: Any], ticket: [UInt8],
         cryptexTemplate: [UInt8]? = nil,
         note: @escaping (String) -> Void,
         progress: @escaping (String, Double) -> Void = { _, _ in }) {
        self.usb = usb
        self.ipsw = ipsw
        self.identity = identity
        self.ticket = ticket
        self.cryptexTemplate = cryptexTemplate
        self.note = note
        self.reportProgress = progress
    }

    private func progress(_ what: String, _ fraction: Double) {
        let now = Date()
        guard now.timeIntervalSince(lastProgress) >= 0.25 || fraction >= 1 else { return }
        lastProgress = now
        reportProgress(what, fraction)
    }

    // MARK: - The run

    /// Drives the restore from `StartRestore` to the guest's final status.
    ///
    /// Blocks until the guest is done, which is minutes of work; call it off the
    /// main thread.
    func run(on restored: GuestUSB.Channel, protocolVersion: Int) throws {
        try RestoreClient.send(restored, startRestore(protocolVersion: protocolVersion))
        note(L("Рестор: начат, протокол %d", protocolVersion))

        while true {
            let message = try RestoreClient.receive(restored, timeout: 900)
            guard let type = message["MsgType"] as? String else {
                note(L("Рестор: сообщение без типа"))
                continue
            }
            switch type {
            // The device asks for one piece of the firmware at a time. Async
            // requests are answered on the spot too: they carry their own port,
            // so nothing is lost by taking them in turn.
            case "DataRequestMsg", "AsyncDataRequestMsg":
                try answer(request: message, over: restored)

            case "ProgressMsg":
                let percent = (message["Progress"] as? NSNumber)?.doubleValue ?? 0
                let operation = (message["Operation"] as? NSNumber)?.uint64Value ?? 0
                if percent <= 100 {
                    progress(RestoreClient.operationName(operation), percent / 100)
                }

            case "StatusMsg":
                if let log = message["Log"] as? String, !log.isEmpty {
                    note(L("Рестор: журнал гостя — %@", log))
                }
                let status = (message["Status"] as? NSNumber)?.uint64Value ?? 0
                guard status == 0 else { throw Failure.guestFailed(status) }
                try RestoreClient.send(restored, ["MsgType": "ReceivedFinalStatusMsg"])
                note(L("Рестор: гость доложил, что закончил"))
                return

            case "CheckpointMsg":
                let name = message["CHECKPOINT_NAME"] as? String ?? "?"
                let result = (message["CHECKPOINT_RESULT"] as? NSNumber)?.int64Value ?? 0
                if result != 0 { note(L("Рестор: этап %@ вернул %d", name, Int(result))) }

            case "PreviousRestoreLogMsg":
                if let log = message["PreviousRestoreLog"] as? String {
                    note(L("Рестор: журнал прошлой попытки — %@", log))
                }

            case "CrashLog", "RestoredCrash":
                note(L("Рестор: гость уронил restored"))

            default:
                note(L("Рестор: сообщение %@ пропущено", type))
            }
        }
    }

    // MARK: - StartRestore

    /// The options `idevicerestore` sends for a device (as opposed to a Mac).
    ///
    /// `SupportedDataTypes` and `SupportedMessageTypes` are copied from Apple
    /// Configurator as they are: the booleans in them do not mean "the host can
    /// do this" — `RootTicket` and `NORData` sit there as false and are asked
    /// for and answered all the same — so they are not to be "fixed".
    private func startRestore(protocolVersion: Int) -> [String: Any] {
        var options: [String: Any] = [
            "AutoBootDelay": 0,
            "BootImageType": "User",
            "DFUFileType": "RELEASE",
            "DataImage": false,
            "FirmwareDirectory": ".",
            "FlashNOR": true,
            "KernelCacheType": "Release",
            "NORImageType": "production",
            "RestoreBundlePath": "/tmp/Per2.tmp",
            "SystemImageType": "User",
            "UpdateBaseband": true,
            "InstallDiags": false,
            "PersonalizedDuringPreflight": true,
            "RootToInstall": false,
            "CreateFilesystemPartitions": true,
            "SystemImage": true,
            "UUID": UUID().uuidString,
            "SupportedDataTypes": RestoreClient.supportedDataTypes,
            "SupportedMessageTypes": RestoreClient.supportedMessageTypes,
        ]
        let info = identity["Info"] as? [String: Any]
        options["SystemPartitionPadding"] = info?["SystemPartitionPadding"] as? [String: Any]
            ?? ["8": 80, "16": 160, "32": 320, "64": 640,
                "128": 1280, "256": 1280, "512": 1280, "768": 1280, "1024": 1280]
        return [
            "Request": "StartRestore",
            "Label": "Inferno",
            "RestoreOptions": options,
            "RestoreProtocolVersion": protocolVersion,
        ]
    }

    // MARK: - Answering a data request

    private func answer(request: [String: Any], over restored: GuestUSB.Channel) throws {
        guard let type = request["DataType"] as? String else {
            note(L("Рестор: запрос без типа данных"))
            return
        }
        let arguments = request["Arguments"] as? [String: Any] ?? [:]
        note(L("Рестор: гость просит %@", type))

        switch type {
        case "SystemImageData":
            try sendFilesystem(request)

        case "RootTicket":
            try reply(["RootTicketData": Data(ticket)], to: request, over: restored)

        case "KernelCache", "DeviceTree":
            try reply(["\(type)File": Data(try personalized(type))], to: request, over: restored)

        case "SystemImageRootHash":
            try reply(["SystemImageRootHashFile": Data(try personalized("SystemVolume"))],
                      to: request, over: restored)

        case "SystemImageCanonicalMetadata":
            try reply(["SystemImageCanonicalMetadataFile":
                        Data(try personalized("Ap,SystemVolumeCanonicalMetadata"))],
                      to: request, over: restored)

        case "NORData":
            try reply(try norData(arguments), to: request, over: restored)

        case "BuildIdentityDict":
            try reply(["BuildIdentityDict": identity,
                       "Variant": arguments["Variant"] as? String ?? "Erase"],
                      to: request, over: restored)

        case "ReceiptManifest":
            try reply(["ReceiptManifest": identity["Manifest"] as? [String: Any] ?? [:]],
                      to: request, over: restored)

        // What iTunes sends is an empty dictionary, and that is enough to make
        // the device carry on with FDR. Both preflights (`idevicerestore`
        // answers them with the exact same function) take the same nothing.
        case "FDRTrustData", "FirmwareUpdaterPreflight", "DeviceRestoreInfoPreflight":
            try reply([:], to: request, over: restored)

        // A family of requests that all mean the same thing: look through the
        // manifest for components of a kind, and send either their names or
        // their personalised selves.
        case "PersonalizedData":
            try reply(try imageData(arguments, list: "ImageList", type: nil, data: "ImageData"),
                      to: request, over: restored)

        case "FUDData":
            try reply(try imageData(arguments, list: "FUDImageList", type: "IsFUDFirmware",
                                    data: "FUDImageData"),
                      to: request, over: restored)

        case "EANData":
            try reply(try imageData(arguments, list: "EANImageList", type: "IsEarlyAccessFirmware",
                                    data: "EANData"),
                      to: request, over: restored)

        case "PersonalizedBootObjectV3":
            try sendBootObject(request, over: restored, personalize: true)

        case "SourceBootObjectV4", "SourceBootObjectV5":
            try sendBootObject(request, over: restored, personalize: false)

        case "FirmwareUpdaterData":
            let updater = arguments["MessageArgUpdaterName"] as? String ?? "?"
            // iOS 16's own OS and app cryptexes -- everything else here is a
            // firmware for a coprocessor the emulated phone does not have,
            // and Apple is the only one who can sign those.
            guard updater == "Cryptex1" || updater == "Cryptex1LocalPolicy" else {
                throw Failure.unsupported(L("прошивка сопроцессора %@ — её подписывает Apple", updater))
            }
            guard let cryptexTemplate else {
                throw Failure.unsupported(L("нет шаблона Cryptex1 — см. RESTORE.md"))
            }
            let responseKey = ((arguments["DeviceGeneratedTags"] as? [String: Any])?["ResponseTags"] as? [Any])?
                .first as? String ?? "Cryptex1,Ticket"
            let im4m = try Cryptex1.forge(identity: identity, template: cryptexTemplate)
            note(L("Рестор: подписан тикет Cryptex1, %d байт", im4m.count))
            try reply(["FirmwareResponseData": [responseKey: Data(im4m)]], to: request, over: restored)

        default:
            throw Failure.unsupported(type)
        }
    }

    /// Where an answer goes. From iOS 17 the guest names a port of its own for
    /// each request; before that everything rides the `restored` channel.
    private func reply(_ body: [String: Any], to request: [String: Any],
                       over restored: GuestUSB.Channel) throws {
        if let port = dataPort(of: request) {
            let channel = try usb.connect(to: port, timeout: 60)
            defer { channel.close() }
            try RestoreClient.send(channel, body)
        } else {
            try RestoreClient.send(restored, body)
        }
    }

    private func dataPort(of request: [String: Any]) -> UInt16? {
        guard let port = (request["DataPort"] as? NSNumber)?.uint16Value, port != 0 else { return nil }
        return port
    }

    // MARK: - Components

    /// One component of the firmware, wrapped with our ticket.
    private func personalized(_ component: String) throws -> [UInt8] {
        let raw = try raw(component)
        return try IMG4.make(payload: IMG4.readIM4P(raw), ticket: ticket)
    }

    private func raw(_ component: String) throws -> [UInt8] {
        guard let path = IPSW.path(of: component, in: identity) else {
            throw Failure.unsupported(L("в манифесте нет компонента %@", component))
        }
        return try ipsw.read(path)
    }

    /// Everything that goes into NOR: the boot chain and the SEP.
    private func norData(_ arguments: [String: Any]) throws -> [String: Any] {
        var body: [String: Any] = ["LlbImageData": Data(try personalized("LLB"))]

        // `idevicerestore` prefers a `manifest` file next to LLB and falls back
        // to the build identity. No IPSW since iOS 14 carries that file, so the
        // fallback is the real path: every firmware payload, plus the secondary
        // ones iBoot loads itself.
        var names = firmwareComponents()
        names.remove("LLB")             // already sent, in its own key
        names.remove("RestoreSEP")      // sent below, in its own key

        // iBoot has to come first; the rest keep a stable order so that two runs
        // of the same restore send the same bytes.
        var images = [(name: String, blob: Data)]()
        for name in names.sorted() {
            let blob = Data(try personalized(name))
            if name.hasPrefix("iBoot") { images.insert((name, blob), at: 0) } else { images.append((name, blob)) }
        }
        if arguments["FlashVersion1"] != nil {
            body["NorImageData"] = Dictionary(uniqueKeysWithValues: images.map { ($0.name, $0.blob) })
        } else {
            body["NorImageData"] = images.map(\.blob)
        }

        for (component, key) in [("RestoreSEP", "RestoreSEPImageData"),
                                 ("SEP", "SEPImageData"),
                                 ("SepStage1", "SEPPatchImageData")] {
            guard IPSW.path(of: component, in: identity) != nil else { continue }
            body[key] = Data(try personalized(component))
        }
        return body
    }

    /// The answer to `PersonalizedData` and its relatives.
    ///
    /// The guest either wants the list of components of some kind — the kind is
    /// a flag on each entry of the manifest, like `IsFUDFirmware` — or the
    /// components themselves, one named or all of them at once.
    private func imageData(_ arguments: [String: Any], list: String, type: String?,
                           data: String) throws -> [String: Any] {
        let wantList = (arguments[list] as? NSNumber)?.boolValue ?? false
        let imageName = arguments["ImageName"] as? String
        guard let kind = type ?? arguments["ImageType"] as? String else {
            throw Failure.brokenProtocol("ImageType")
        }
        guard let manifest = identity["Manifest"] as? [String: Any] else {
            throw Failure.unsupported(L("в манифесте нет компонента %@", kind))
        }

        var names = [String]()
        var blobs = [String: Any]()
        for (component, entry) in manifest.sorted(by: { $0.key < $1.key }) {
            guard let info = (entry as? [String: Any])?["Info"] as? [String: Any],
                  (info[kind] as? Bool) == true
            else { continue }
            if wantList {
                names.append(component)
            } else if imageName == nil || imageName == component {
                blobs[component] = Data(try personalized(component))
            }
        }

        if wantList {
            note(L("Рестор: %@ — %d штук", kind, names.count))
            return [list: names]
        }
        if let imageName {
            var body: [String: Any] = ["ImageName": imageName]
            if let blob = blobs[imageName] { body[data] = blob }
            return body
        }
        return [data: blobs]
    }

    private func firmwareComponents() -> Set<String> {
        guard let manifest = identity["Manifest"] as? [String: Any] else { return [] }
        var names = Set<String>()
        for (name, entry) in manifest {
            guard let info = (entry as? [String: Any])?["Info"] as? [String: Any],
                  info["Path"] is String
            else { continue }
            let firmware = info["IsFirmwarePayload"] as? Bool ?? false
            let secondary = info["IsSecondaryFirmwarePayload"] as? Bool ?? false
            let byBoot = info["IsLoadedByiBoot"] as? Bool ?? false
            if firmware || (secondary && byBoot) { names.insert(name) }
        }
        return names
    }

    // MARK: - Boot objects, in chunks

    /// Large objects go as a stream of `FileData` messages, 8 KB at a time,
    /// closed by `FileDataDone`.
    private func sendBootObject(_ request: [String: Any], over restored: GuestUSB.Channel,
                                personalize: Bool) throws {
        guard let arguments = request["Arguments"] as? [String: Any],
              let name = arguments["ImageName"] as? String
        else { throw Failure.brokenProtocol("ImageName") }

        let data: [UInt8]
        switch name {
        case "__RestoreVersion__": data = try ipsw.read("RestoreVersion.plist")
        case "__SystemVersion__":  data = try ipsw.read("SystemVersion.plist")
        case "__GlobalManifest__":
            // Global signing needs a manifest Apple signs for the whole build.
            // An offline restore has one ticket, and it is not that.
            throw Failure.unsupported(L("глобальный манифест — его подписывает Apple"))
        default:
            data = personalize ? try personalized(name) : try raw(name)
        }

        let channel: GuestUSB.Channel
        var ownChannel: GuestUSB.Channel?
        if let port = dataPort(of: request) {
            let opened = try usb.connect(to: port, timeout: 60)
            ownChannel = opened
            channel = opened
        } else {
            channel = restored
        }
        defer { ownChannel?.close() }

        var sent = 0
        while sent < data.count {
            let end = min(sent + 8192, data.count)
            try RestoreClient.send(channel, ["FileData": Data(data[sent..<end])])
            sent = end
            if data.count > 0x1000000 {
                progress(name, Double(sent) / Double(data.count))
            }
        }
        try RestoreClient.send(channel, ["FileDataDone": true])
        note(L("Рестор: отдан %@ (%d КБ)", name, data.count / 1024))
    }

    // MARK: - The filesystem, over ASR

    /// Streams the system image out of the IPSW.
    ///
    /// The image is stored in the archive uncompressed, so ASR's requests for
    /// arbitrary offsets are answered by reading the firmware file itself —
    /// nothing is unpacked, and nothing but a chunk is ever held in memory.
    private func sendFilesystem(_ request: [String: Any]) throws {
        guard let path = IPSW.path(of: "OS", in: identity) else {
            throw Failure.unsupported(L("в манифесте нет образа системы"))
        }
        let range = try ipsw.rangeOfStored(path)
        let file = try FileHandle(forReadingFrom: ipsw.url)
        defer { try? file.close() }

        let port = dataPort(of: request) ?? GuestUSB.Port.asr.rawValue
        let channel = try openASR(on: port)
        defer { channel.close() }
        let asr = ASR(channel: channel)

        note(L("Рестор: ASR на порту %d, образ %d МБ", Int(port), range.length / 1024 / 1024))

        // The first word is always an Initiate, and it says whether every chunk
        // has to carry its own checksum.
        let hello = try asr.receive()
        guard (hello["Command"] as? String) == "Initiate" else {
            throw Failure.brokenProtocol("ASR: \(hello["Command"] as? String ?? "?")")
        }
        asr.checksumChunks = (hello["Checksum Chunks"] as? NSNumber)?.boolValue ?? false
        try asr.sendPacketInfo(size: range.length)

        // Validation: the device reads pieces of the image at offsets it picks
        // itself, and only then asks for the whole thing.
        while true {
            // A pool here too: the guest picks offsets to check and can ask for
            // a great many of them.
            let payload = try autoreleasepool { () -> Bool in
                let packet = try asr.receive()
                switch packet["Command"] as? String {
                case "Initiate":
                    asr.checksumChunks = (packet["Checksum Chunks"] as? NSNumber)?.boolValue ?? asr.checksumChunks
                    try asr.sendPacketInfo(size: range.length)
                case "OOBData":
                    guard let offset = (packet["OOB Offset"] as? NSNumber)?.intValue,
                          let length = (packet["OOB Length"] as? NSNumber)?.intValue
                    else { throw Failure.brokenProtocol("ASR: OOB") }
                    try file.seek(toOffset: UInt64(range.offset + offset))
                    let chunk = file.readData(ofLength: length)
                    guard chunk.count == length else { throw Failure.brokenProtocol("ASR: OOB") }
                    try channel.write([UInt8](chunk))
                case "Payload":
                    return true
                case let other:
                    throw Failure.brokenProtocol("ASR: \(other ?? "?")")
                }
                return false
            }
            if payload { break }
        }

        note(L("Рестор: образ проверен, отдаю"))
        try file.seek(toOffset: UInt64(range.offset))
        var sent = 0
        // A pool per chunk. Five gigabytes in 128 KB pieces is forty thousand
        // turns of this loop, all inside one dispatched block, and what the
        // file hands back is autoreleased: without a pool of its own none of it
        // is given back until the whole restore ends. Measured on the phone
        // before this: the process grew about 70 MB for every percent of the
        // image, passed three gigabytes around a fifth of the way in, and was
        // killed with no crash report to say why.
        while sent < range.length {
            try autoreleasepool {
                let size = min(131072, range.length - sent)
                var chunk = file.readData(ofLength: size)
                guard chunk.count == size else { throw Failure.brokenProtocol("IPSW") }
                if asr.checksumChunks { chunk.append(RestoreClient.sha1(chunk)) }
                try channel.write([UInt8](chunk))
                sent += size
                progress(L("Образ системы"), Double(sent) / Double(range.length))
            }
        }
        note(L("Рестор: образ системы отдан целиком"))
    }

    /// Knocks on the ASR port until it answers.
    ///
    /// The guest does not start listening the moment `restored` asks for the
    /// image — it takes a few seconds, and a connection tried in between is
    /// refused outright rather than left waiting. `idevicerestore` knocks
    /// thirty times, two seconds apart, and so does this.
    private func openASR(on port: UInt16) throws -> GuestUSB.Channel {
        var lastFailure: Error?
        for attempt in 1...30 {
            do {
                return try usb.connect(to: port, timeout: 10)
            } catch {
                lastFailure = error
                if attempt == 1 || attempt % 5 == 0 {
                    note(L("Рестор: ASR на %d ещё не слушает, попытка %d", Int(port), attempt))
                }
                Thread.sleep(forTimeInterval: 2)
            }
        }
        throw lastFailure ?? Failure.brokenProtocol("ASR")
    }

    /// ASR speaks property lists with no length in front of them: a message ends
    /// where its XML ends, and more than one can arrive in a single read.
    private final class ASR {
        let channel: GuestUSB.Channel
        var checksumChunks = false
        private var buffer = [UInt8]()

        init(channel: GuestUSB.Channel) { self.channel = channel }

        func receive(timeout: TimeInterval = 300) throws -> [String: Any] {
            while true {
                if let end = ASR.endOfPlist(buffer) {
                    let raw = Data(buffer[0..<end])
                    buffer.removeFirst(end)
                    let plist = try PropertyListSerialization.propertyList(from: raw, options: [], format: nil)
                    return plist as? [String: Any] ?? [:]
                }
                buffer += try channel.readAvailable(timeout: timeout)
            }
        }

        func send(_ body: [String: Any]) throws {
            let payload = try PropertyListSerialization.data(fromPropertyList: body, format: .xml, options: 0)
            try channel.write([UInt8](payload))
        }

        /// What the device expects after every Initiate, and nothing else will
        /// move it along.
        func sendPacketInfo(size: Int) throws {
            var info: [String: Any] = [
                "FEC Slice Stride": 40,
                "Packet Payload Size": 1450,
                "Packets Per FEC": 25,
                "Payload": ["Port": 1, "Size": size],
                "Stream ID": 1,
                "Version": 1,
            ]
            if checksumChunks { info["Checksum Chunk Size"] = 131072 }
            try send(info)
        }

        private static let terminator = Array("</plist>".utf8)

        private static func endOfPlist(_ bytes: [UInt8]) -> Int? {
            guard bytes.count >= terminator.count else { return nil }
            for start in 0...(bytes.count - terminator.count) where Array(bytes[start..<start + terminator.count]) == terminator {
                var end = start + terminator.count
                while end < bytes.count, bytes[end] == 0x0A || bytes[end] == 0x0D { end += 1 }
                return end
            }
            return nil
        }
    }

    // MARK: - Framing and plumbing

    /// `restored` speaks property lists behind a four-byte length, big-endian —
    /// the same framing lockdownd uses.
    static func send(_ channel: GuestUSB.Channel, _ body: [String: Any]) throws {
        let payload = try PropertyListSerialization.data(fromPropertyList: body, format: .xml, options: 0)
        var framed = [UInt8]()
        let length = UInt32(payload.count)
        framed += (0..<4).reversed().map { UInt8((length >> (8 * $0)) & 0xFF) }
        framed += [UInt8](payload)
        try channel.write(framed)
    }

    static func receive(_ channel: GuestUSB.Channel, timeout: TimeInterval = 60) throws -> [String: Any] {
        let head = try channel.read(4, timeout: timeout)
        let size = (0..<4).reduce(UInt32(0)) { ($0 << 8) | UInt32(head[$1]) }
        guard size > 0, size < 64 * 1024 * 1024 else {
            throw Failure.brokenProtocol(L("пакет длиной %d", Int(size)))
        }
        let raw = try channel.read(Int(size), timeout: timeout)
        let plist = try PropertyListSerialization.propertyList(from: Data(raw), options: [], format: nil)
        return plist as? [String: Any] ?? [:]
    }

    static func request(_ channel: GuestUSB.Channel, _ body: [String: Any]) throws -> [String: Any] {
        try send(channel, body)
        return try receive(channel)
    }

    private static func sha1(_ data: Data) -> Data {
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA1_DIGEST_LENGTH))
        data.withUnsafeBytes { _ = CC_SHA1($0.baseAddress, CC_LONG(data.count), &digest) }
        return Data(digest)
    }

    private static func statusName(_ code: UInt64) -> String {
        switch code {
        case UInt64.max: return L("ошибка проверки")
        case 6:          return L("диск не отвечает")
        case 14:         return L("сбой")
        case 27:         return L("файловые системы не смонтировались")
        case 50, 51:     return L("прошивка SEP не загрузилась")
        case 53:         return L("данные FDR не восстановились")
        default:         return "\(code)"
        }
    }

    private static func operationName(_ code: UInt64) -> String {
        switch code {
        case 3:  return L("Очистка")
        case 8:  return L("Раздел с данными")
        case 12: return L("Прошивка NOR")
        case 13: return L("Раздел с системой")
        case 14: return L("Система")
        case 19: return L("Обновление SEP")
        case 21: return L("Проверка SEP")
        case 27: return L("Дерево устройства")
        default: return L("Установка")
        }
    }

    /// Copied from Apple Configurator, as `idevicerestore` has it.
    private static let supportedDataTypes: [String: Any] = [
        "AuthInstallCACert": true,
        "BasebandBootData": false,
        "BasebandData": false,
        "BasebandStackData": false,
        "BasebandUpdaterOutputData": false,
        "BootabilityBundle": false,
        "BootabilityBundleV2": false,
        "BuildIdentityDict": false,
        "BuildIdentityDictV2": false,
        "Cryptex1LocalPolicy": true,
        "DataType": false,
        "DeviceRestoreInfoPreflight": false,
        "DiagData": false,
        "EANData": false,
        "FDRMemoryCommit": false,
        "FDRTrustData": false,
        "FUDData": false,
        "FileData": false,
        "FileDataDone": false,
        "FirmwareUpdaterData": false,
        "FirmwareUpdaterDataV2": false,
        "FirmwareUpdaterDataV3": true,
        "FirmwareUpdaterPreflight": true,
        "GrapeFWData": false,
        "HPMFWData": false,
        "HostSystemTime": true,
        "KernelCache": false,
        "MessageUseStreamedImageFile": true,
        "NORData": false,
        "NitrogenFWData": true,
        "OpalFWData": false,
        "OverlayRootDataCount": false,
        "OverlayRootDataForKey": true,
        "OverlayRootDataForKeyIndex": true,
        "PeppyFWData": true,
        "PersonalizedBootObjectV3": false,
        "PersonalizedData": true,
        "ProvisioningData": false,
        "RamdiskFWData": true,
        "ReceiptManifest": true,
        "RecoveryOSASRImage": true,
        "RecoveryOSAppleLogo": true,
        "RecoveryOSDeviceTree": true,
        "RecoveryOSFileAssetImage": true,
        "RecoveryOSIBEC": true,
        "RecoveryOSIBootFWFilesImages": true,
        "RecoveryOSImage": true,
        "RecoveryOSKernelCache": true,
        "RecoveryOSLocalPolicy": true,
        "RecoveryOSOverlayRootDataCount": false,
        "RecoveryOSRootTicketData": true,
        "RecoveryOSStaticTrustCache": true,
        "RecoveryOSVersionData": true,
        "RestoreLocalPolicy": true,
        "RootData": false,
        "RootTicket": false,
        "S3EOverride": false,
        "SourceBootObjectV3": false,
        "SourceBootObjectV4": false,
        "SourceBootObjectV5": false,
        "SsoServiceTicket": false,
        "StockholmPostflight": false,
        "SystemImageCanonicalMetadata": false,
        "SystemImageData": false,
        "SystemImageRootHash": false,
        "URLAsset": true,
        "USBCFWData": false,
        "USBCOverride": false,
        "UpdateVolumeOverlayRootDataCount": true,
    ]

    private static let supportedMessageTypes: [String: Any] = [
        "AsyncDataRequestMsg": true,
        "AsyncWait": true,
        "BBUpdateStatusMsg": false,
        "CheckpointMsg": true,
        "CrashLog": true,
        "DataRequestMsg": false,
        "FDRSubmit": true,
        "MsgType": false,
        "PreviousRestoreLogMsg": false,
        "ProgressMsg": false,
        "ProvisioningAck": false,
        "ProvisioningInfo": false,
        "ProvisioningStatusMsg": false,
        "ReceivedFinalStatusMsg": false,
        "RestoreAttestation": true,
        "RestoreProtocol": true,
        "RestoredCrash": true,
        "StatusMsg": false,
    ]
}
