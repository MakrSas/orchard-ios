//  The whole UI: run the probe at launch, put the report on screen, and make
//  it copyable. A device probe whose answer you cannot paste anywhere is an
//  answer you will retype by hand from a photograph.

import SwiftUI

@main
struct MetallibProbeApp: App {
    var body: some Scene {
        WindowGroup { ProbeView() }
    }
}

struct ProbeView: View {
    @State private var report: String = "Running…"

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(report)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .navigationTitle("metallib probe")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                Button("Copy") { UIPasteboard.general.string = report }
            }
        }
        .task {
            let text = MetallibProbe.report()
            print(text)          // also to the console, for a wired run
            report = text
        }
    }
}
