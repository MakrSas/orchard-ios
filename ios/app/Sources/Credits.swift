import SwiftUI

/// Who this app stands on, and what each of them did.
///
/// The avatars come from GitHub (`github.com/<login>.png`) rather than from X.
/// X was tried first and does not work: it has no open avatar endpoint, and the
/// usual third-party resolver answers with a 569-byte placeholder SVG instead of
/// a photo. GitHub serves the real picture, first-party and without a key, so
/// that is the source; X stays as a link on the card, and only for people whose
/// handle is actually known — an invented handle would be worse than none.
///
/// Opening this screen fetches the avatars from github.com. Nothing else leaves
/// the phone.
struct Credit: Identifiable, Hashable {
    /// The GitHub login, which is also where the avatar comes from. Empty for
    /// entries that are not a person — QEMU is thousands of people.
    let github: String
    let name: String
    /// What they did, in one or two sentences. Written to be true rather than
    /// flattering.
    let did: String
    /// Only when the handle is genuinely known.
    let x: String?
    let site: String?

    var id: String { github.isEmpty ? name : github }

    /// Asked for large: on the card the picture is the background, not a
    /// thumbnail, and it is drawn the full width of the sheet. GitHub resizes
    /// on request and sends the original when that is smaller.
    var avatarURL: URL? {
        guard !github.isEmpty else { return nil }
        return URL(string: "https://github.com/\(github).png?size=800")
    }

    var githubURL: URL? { github.isEmpty ? nil : URL(string: "https://github.com/\(github)") }
    var xURL: URL? { x.flatMap { URL(string: "https://x.com/\($0)") } }
    var siteURL: URL? { site.flatMap { URL(string: $0) } }

    /// The list, in the order the work stacks up: the emulator first, because
    /// without it there is nothing to show, and the two who wrote this app last.
    static let all: [Credit] = [
        Credit(github: "yaelliethy",
               name: "Youssef Elliethy",
               did: L("Автор Orchard — форка QEMU, в котором macOS грузится штатной цепочкой Apple на машине apple-vm. Гость работает на этом эмуляторе, а это приложение — перенос Orchard на iPhone."),
               x: nil, site: nil),
        Credit(github: "steelbrain",
               name: "Anees Iqbal",
               did: L("Автор reims-vgpu — паравиртуального GPU, который переводит графику гостя в Metal. Вся картинка на экране идёт через него."),
               x: "aneesbhatti", site: "https://aneesiqbal.ai"),
        Credit(github: "agraf",
               name: "Alexander Graf",
               did: L("Автор машины vmapple в QEMU — той, что повторяет виртуальную машину Apple из Virtualization.framework. На ней построена apple-vm, где работает гость."),
               x: nil, site: nil),
        Credit(github: "VisualEhrmanntraut",
               name: "Visual Ehrmanntraut",
               did: L("Автор эмулятора Apple Silicon на QEMU, с которого всё началось. На его работе стоит большая часть того, что тут под капотом, — в том числе поддержка PAC, без которой ядро macOS не грузится."),
               x: "HeWhomCodes", site: "https://chefkiss.dev"),
        Credit(github: "ChefKissInc",
               name: "ChefKiss",
               did: L("Команда этого эмулятора. Их форк QEMU — основа большой части того, что здесь работает."),
               x: nil, site: "https://chefkiss.dev"),
        Credit(github: "",
               name: "QEMU",
               did: L("Orchard — форк QEMU, а QEMU написан очень многими людьми. Отдельного человека тут назвать нельзя, но без их работы не было бы ни эмулятора, ни приложения."),
               x: nil, site: "https://www.qemu.org"),
        // The avatar and the link go to the org, because that is where there is
        // a picture to fetch; Claude itself has no GitHub account.
        Credit(github: "anthropics",
               name: "Claude",
               did: L("Сделал всю работу."),
               x: nil, site: "https://claude.ai"),
        Credit(github: "MakrSas",
               name: "Makr",
               did: L("Управлял Клодом."),
               x: "makrotasuss", site: nil),
    ]
}

/// Keeps fetched avatars for the life of the screen.
///
/// `URLSession`'s own cache does the real work; this only avoids decoding the
/// same picture again every time a row scrolls back into view.
@MainActor
final class AvatarStore: ObservableObject {
    @Published private(set) var images: [String: PlatformImage] = [:]
    private var inFlight: Set<String> = []

    func load(_ credit: Credit) {
        let key = credit.id
        guard images[key] == nil, !inFlight.contains(key), let url = credit.avatarURL else { return }
        inFlight.insert(key)
        Task { [weak self] in
            var request = URLRequest(url: url)
            request.cachePolicy = .returnCacheDataElseLoad
            request.timeoutInterval = 20
            let image = (try? await URLSession.shared.data(for: request))
                .flatMap { PlatformImage(data: $0.0) }
            await MainActor.run {
                guard let self else { return }
                self.inFlight.remove(key)
                if let image { self.images[key] = image }
            }
        }
    }
}

struct CreditsView: View {
    @StateObject private var avatars = AvatarStore()
    @State private var chosen: Credit?

    var body: some View {
        List {
            Section {
                ForEach(Credit.all) { credit in
                    Button { chosen = credit } label: { row(credit) }
                        .buttonStyle(.plain)
                }
            } footer: {
                Text(L("Приложение неофициальное и никак не связано ни с Apple, ни с ChefKiss. Аватарки берутся с GitHub — у X открытого способа их получить нет."))
            }
        }
        .navigationTitle(L("Благодарности"))
        .inlineNavigationTitle()
        .sheet(item: $chosen) { CreditCard(credit: $0, avatars: avatars).phoneSheetSize() }
    }

    private func row(_ credit: Credit) -> some View {
        HStack(spacing: 12) {
            avatar(credit, size: 44)
            VStack(alignment: .leading, spacing: 2) {
                Text(credit.name).font(.body)
                Text(credit.did)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
            Image(systemName: "chevron.right")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 4)
        .onAppear { avatars.load(credit) }
    }

    @ViewBuilder
    private func avatar(_ credit: Credit, size: CGFloat) -> some View {
        if let image = avatars.images[credit.id] {
            Image(platformImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: size, height: size)
                .clipShape(Circle())
        } else {
            // A person who has no picture yet, and QEMU, which is not a person.
            ZStack {
                Circle().fill(.quaternary)
                Image(systemName: credit.github.isEmpty ? "shippingbox" : "person.fill")
                    .font(.system(size: size * 0.4))
                    .foregroundStyle(.secondary)
            }
            .frame(width: size, height: size)
        }
    }
}

/// One contributor, laid out the way the Contacts card is: the picture is the
/// whole background, sharp at the top and dissolving into blur further down,
/// with the name and the glass on top of it.
private struct CreditCard: View {
    let credit: Credit
    @ObservedObject var avatars: AvatarStore
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL

    var body: some View {
        GeometryReader { geometry in
            let size = geometry.size
            ZStack(alignment: .top) {
                backdrop(size)
                ScrollView {
                    VStack(spacing: 18) {
                        // The picture owns this much of the sheet; the writing
                        // starts where the blur has taken hold of it.
                        Color.clear.frame(height: size.width * 0.78)

                        Text(credit.name)
                            .font(.system(size: 34, weight: .bold))
                            .foregroundStyle(.white)
                            .shadow(color: .black.opacity(0.45), radius: 14, y: 2)
                            .multilineTextAlignment(.center)
                            // The longest name here fills the width at this
                            // size; a narrower phone shrinks it rather than
                            // breaking it in two.
                            .lineLimit(2)
                            .minimumScaleFactor(0.65)

                        links
                        about

                        if !credit.github.isEmpty {
                            Text("@" + credit.github)
                                .font(.footnote)
                                .foregroundStyle(.white.opacity(0.45))
                        }
                    }
                    .padding(.horizontal, 22)
                    .padding(.bottom, 40)
                    .frame(maxWidth: .infinity)
                }
                .scrollIndicators(.hidden)
            }
            .frame(width: size.width, height: size.height)
            .overlay(alignment: .topTrailing) { done }
        }
        .background(Color.black)
        .ignoresSafeArea()
        // The card is a photograph with writing on it, whatever the phone is
        // set to; the glass and the materials have to be told that.
        .environment(\.colorScheme, .dark)
        .onAppear { avatars.load(credit) }
    }

    // MARK: - The picture behind everything

    @ViewBuilder
    private func backdrop(_ size: CGSize) -> some View {
        ZStack(alignment: .top) {
            // What shows wherever the picture does not, and the floor the glass
            // has to sit on.
            Color.black

            if let image = avatars.images[credit.id] {
                wash(image, size: size)
                // SwiftUI has no blur that varies down the view, so the ramp is
                // made of copies of the same picture: the blurriest lies at the
                // bottom of the stack, and the sharper ones are laid over it and
                // faded out before they reach the writing.
                sheet(image, side: size.width, blur: 16, from: 0.62, to: 1)
                sheet(image, side: size.width, blur: 0, from: 0.46, to: 0.86)
            } else {
                emblem(size)
            }

            // The writing sits low on the picture, and this is what it reads
            // against — a light picture would swallow white text otherwise.
            LinearGradient(stops: [
                .init(color: .clear, location: 0.28),
                .init(color: .black.opacity(0.40), location: 0.58),
                .init(color: .black.opacity(0.90), location: 1),
            ], startPoint: .top, endPoint: .bottom)
        }
        .frame(width: size.width, height: size.height)
        .clipped()
    }

    /// The whole sheet filled with the picture, blurred past recognition. It is
    /// the colour of the card rather than anything to look at.
    private func wash(_ image: PlatformImage, size: CGSize) -> some View {
        Image(platformImage: image)
            .resizable()
            .scaledToFill()
            .frame(width: size.width, height: size.height)
            .clipped()
            .blur(radius: 70, opaque: true)
            .saturation(1.25)
            .overlay(Color.black.opacity(0.18))
    }

    /// One square copy of the picture at the top of the sheet, blurred by
    /// `blur` and faded out between the two heights, given as fractions of the
    /// square.
    private func sheet(_ image: PlatformImage, side: CGFloat, blur: CGFloat,
                       from: CGFloat, to: CGFloat) -> some View {
        Image(platformImage: image)
            .resizable()
            .scaledToFill()
            .frame(width: side, height: side)
            .clipped()
            .blur(radius: blur, opaque: true)
            .mask(alignment: .top) {
                LinearGradient(stops: [
                    .init(color: .black, location: from),
                    .init(color: .clear, location: to),
                ], startPoint: .top, endPoint: .bottom)
            }
    }

    /// For QEMU, which is not a person, and for a picture that has not arrived.
    private func emblem(_ size: CGSize) -> some View {
        ZStack {
            LinearGradient(colors: [Color(red: 0.17, green: 0.19, blue: 0.26),
                                    Color(red: 0.05, green: 0.05, blue: 0.07)],
                           startPoint: .topLeading, endPoint: .bottomTrailing)
            Image(systemName: credit.github.isEmpty ? "shippingbox.fill" : "person.fill")
                .font(.system(size: size.width * 0.28))
                .foregroundStyle(.white.opacity(0.16))
                .offset(y: -size.height * 0.16)
        }
    }

    // MARK: - What sits on it

    private var done: some View {
        Button { dismiss() } label: {
            Text(L("Готово"))
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 18)
                .padding(.vertical, 10)
        }
        .buttonStyle(GlassFace(shape: Capsule()))
        .padding(.top, 14)
        .padding(.trailing, 18)
    }

    private var links: some View {
        HStack(spacing: 16) {
            if let url = credit.githubURL {
                link(L("GitHub")) { openURL(url) } icon: {
                    Brandmark.github
                        .fill(.white)
                        .frame(width: 24, height: 24)
                }
            }
            if let url = credit.xURL {
                link("X") { openURL(url) } icon: {
                    Brandmark.x
                        .fill(.white)
                        .frame(width: 21, height: 21)
                }
            }
            if let url = credit.siteURL {
                link(L("Сайт")) { openURL(url) } icon: {
                    Image(systemName: "globe")
                        .font(.system(size: 21, weight: .semibold))
                        .foregroundStyle(.white)
                }
            }
        }
    }

    /// The caption is deliberately outside the button: the disc is what is
    /// pressed, so the place that looks tappable is exactly the place that is.
    private func link<Icon: View>(_ title: String,
                                  tap: @escaping () -> Void,
                                  @ViewBuilder icon: () -> Icon) -> some View {
        VStack(spacing: 7) {
            Button(action: tap) {
                icon().frame(width: 58, height: 58)
            }
            .buttonStyle(GlassFace(shape: Circle()))
            Text(title)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.white.opacity(0.7))
        }
    }

    /// Straight on the picture, with no panel under it: the gradient below the
    /// blur is what it reads against.
    private var about: some View {
        Text(credit.did)
            .font(.callout)
            .foregroundStyle(.white.opacity(0.92))
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .shadow(color: .black.opacity(0.5), radius: 10, y: 1)
            .padding(.horizontal, 6)
    }
}

/// A button whose face is glass.
///
/// The glass is put on through a button style rather than laid on the label as
/// `glassEffect(.interactive)`, and that is not a matter of taste: interactive
/// glass answers touches itself, so a `Button` wearing it loses the first tap
/// to the glass and only works on the second. A style leaves the touch handling
/// to the button alone, and still has somewhere to show the press.
///
/// The hit area is stated outright (`contentShape`), so what is pressed is the
/// shape that is drawn — not whatever frame the effect happened to leave.
private struct GlassFace<S: Shape>: ButtonStyle {
    let shape: S

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .glassy(shape)
            .contentShape(shape)
            .opacity(configuration.isPressed ? 0.72 : 1)
            .scaleEffect(configuration.isPressed ? 0.96 : 1)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

private extension View {
    /// Liquid glass where the system has it, and a material that reads much the
    /// same on anything older.
    ///
    /// `glassEffect` is declared only in the iOS 26 SDK, so the compiler is
    /// gated as well as the run time: the same source has to build with an
    /// older Xcode, as CI's still is.
    @ViewBuilder
    func glassy<S: Shape>(_ shape: S) -> some View {
        #if compiler(>=6.2)
        if #available(iOS 26.0, macOS 26.0, *) {
            glassEffect(.regular, in: shape).clipShape(shape)
        } else {
            frosted(shape)
        }
        #else
        frosted(shape)
        #endif
    }

    func frosted<S: Shape>(_ shape: S) -> some View {
        background(.ultraThinMaterial, in: shape)
            .overlay(shape.stroke(.white.opacity(0.18), lineWidth: 0.5))
            .clipShape(shape)
    }
}
