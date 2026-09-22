import Foundation

/// Puts the guest's package manager back on its feet.
///
/// Cydia and apt fail on every operation with `Sub-process
/// /usr/libexec/cydia/cydo returned an error code (2)`, and the package being
/// installed has nothing to do with it — a plain refresh fails the same way.
/// Three things are wrong at once on a restored image, and each is quiet on its
/// own:
///
/// - the bootstrap keeps its database in `/Library/dpkg`, while dpkg and apt
///   look for `/var/lib/dpkg`; `/var` is a separate data volume the bootstrap
///   never unpacked into, so that path does not exist at all;
/// - the root volume is mounted read-only, and packages write to `/usr` and
///   `/Library`, so a symlink alone changes nothing;
/// - the bootstrap's own packages were unpacked from an archive and never
///   configured — a jailbreak does that on its first boot, and this image had
///   no first boot.
///
/// The remount is the one part that does not survive a guest reboot, which is
/// why this is written to be run again as often as needed: every step is
/// idempotent.
///
/// And dpkg's state was only half of it. Cydia is a user app: it asks for root
/// through `/usr/libexec/cydia/cydo`, a setuid helper. Nothing in this guest is
/// setuid — every volume is mounted `nosuid`, the root included once it is
/// remounted writable — and `/usr/lib/libjailbreak.dylib`, which is how a real
/// jailbreak hands out root instead, is not in this image. So cydo ran dpkg as
/// `mobile`, dpkg said `requested operation requires superuser privilege` and
/// exited 2 — the very code Cydia showed. Measured in the guest: the same
/// command is 0 as root and 2 as mobile.
///
/// What stands in for the missing library is a small root-side script that
/// launchd starts when a request appears in a queue directory, and a
/// replacement for cydo that puts the arguments there and plays back what came
/// out. The original is kept as `cydo.real`.
///
/// Everything here blocks; run it off the main thread.
enum GuestPackages {
    enum Failure: LocalizedError {
        case noShell
        case step(String, Int64)
        case silent(String)
        case stillReadOnly
        case crashed(String)

        var errorDescription: String? {
            switch self {
            case .noShell:
                return L("Гость не отвечает на консоли — дождитесь загрузки и повторите.")
            case .step(let what, let code):
                return L("Шаг «%@» в госте вернул %d.", what, Int(code))
            case .silent(let what):
                return L("Шаг «%@» не ответил вовремя.", what)
            case .stillReadOnly:
                return L("Корень гостя остался только для чтения — пакеты писать некуда.")
            case .crashed(let line):
                return L("Гость упал и перезагружается, команда не доведена до конца: %@", line)
            }
        }
    }

    /// Where the guest keeps what the long steps printed, so a failure can be
    /// read here instead of being guessed at from an exit status. Cydia shows
    /// only `cydo returned an error code (2)`, and that 2 is dpkg's own — cydo
    /// answers a refusal with 77 — so dpkg's words are the thing worth having.
    private static let log = "/var/mobile/.inferno/packages.log"

    private struct Step {
        let title: String
        let command: String
        let timeout: TimeInterval
        /// Whether a non-zero status stops everything. The two long steps at the
        /// end report rather than stop: on an image that is already half fixed
        /// they can grumble and still leave dpkg usable, and the check at the
        /// end is what actually decides.
        let fatal: Bool
    }

    private static let queue = "/var/tmp/inferno-cydo"

    /// Runs dpkg as root for whoever asks.
    ///
    /// A plain background process rather than a launchd job: launchd on this
    /// image takes its daemons from a cache (`launchd_unsecure_cache=1`), so a
    /// plist dropped into /Library/LaunchDaemons lasts only until the guest
    /// reboots — and a helper that is quietly gone is worse than none, because
    /// Cydia then waits on it until the watchdog kills Cydia.
    private static let rootScript = [
        "#!/bin/bash",
        "# Runs dpkg as root on behalf of Cydia. Started by Inferno, because",
        "# nothing in this guest is setuid and cydo cannot elevate itself.",
        "set -u",
        "queue=\(queue)",
        "mkdir -p \"$queue\"",
        "chmod 777 \"$queue\"",
        // Whoever was here before steps aside. Two helpers on one queue is not
        // fatal — they race for requests and both answer — but it is a slow
        // guest, and every extra one costs it.
        "old=$(cat \"$queue/pid\" 2>/dev/null || echo 0)",
        "[ \"$old\" != 0 ] && [ \"$old\" != $$ ] && kill -9 \"$old\" 2>/dev/null",
        "echo $$ > \"$queue/pid\"",
        "trap \"rm -f $queue/pid\" EXIT",
        // Woken rather than polling. A bash loop that wakes three times a
        // second costs nothing on a phone and a great deal in a guest this
        // slow — every wake is instructions somebody has to translate, and it
        // was taking bandwidth away from the guest's own work.
        "[ -p \"$queue/wake\" ] || { rm -f \"$queue/wake\"; mkfifo -m 666 \"$queue/wake\"; }",
        "while true; do",
        "    for req in \"$queue\"/*.req; do",
        "        [ -e \"$req\" ] || continue",
        "        id=\"${req%.req}\"",
        "        mapfile -t args < \"$req\"",
        "        rm -f \"$req\"",
        "        printf %s\\\\n \"--- $id\" \"${args[@]}\" >> \"$queue/args.log\"",
        // apt hands dpkg an open file descriptor to report progress on, and it
        // cannot cross into this process. dpkg is pointed at a file instead,
        // which the client pours into apt's own descriptor — otherwise apt
        // counts nothing as done and says so: `planned for dpkg to do more
        // than it reported back (0 vs 5)`.
        "        keep=()",
        "        skip=0",
        "        for a in \"${args[@]}\"; do",
        "            if [ $skip = 1 ]; then skip=0; continue; fi",
        "            case \"$a\" in",
        "                --status-fd) skip=1; keep+=(--status-fd 9); continue;;",
        "                --status-fd=*) keep+=(--status-fd 9); continue;;",
        "                --log-fd) skip=1; continue;;",
        "                --log-fd=*) continue;;",
        "            esac",
        "            keep+=(\"$a\")",
        "        done",
        // cydo is not only dpkg. Cydia hands it a program to run as root —
        // /bin/rm, /bin/ln, /bin/cp, setnsfpn, firmware.sh — and passes bare
        // options only when it means dpkg itself. Running dpkg either way is
        // how `/bin/rm -f …` turned into `dpkg -f …` and answered `need an
        // action option`.
        "        case \"${keep[0]-}\" in",
        // firmware.sh takes tens of seconds here and Cydia blocks its main
        // thread on this call, which on a guest this slow is long enough for
        // iOS to kill Cydia as unresponsive. The repair button runs it for
        // real; from here it is let go of and answered at once.
        "            */firmware.sh)",
        "                (nohup \"${keep[@]}\" >/dev/null 2>&1 &)",
        "                : > \"$id.out\"",
        "                ;;",
        "            /*) \"${keep[@]}\" > \"$id.out\" 2>&1;;",
        "            *)  /usr/bin/dpkg \"${keep[@]}\" > \"$id.out\" 2>&1 9> \"$id.status\";;",
        "        esac",
        "        echo $? > \"$id.rc\"",
        "    done",
        // The timeout is a safety net, not a schedule: if a wake-up is ever
        // missed the queue is still looked at once a minute.
        "    read -r -t 60 _ < \"$queue/wake\" || true",
        "done",
    ]

    /// Takes cydo's place: same arguments, same output, same exit status.
    ///
    /// Refuses at once when the helper is not running, and says why. Cydia
    /// blocks on this call, so waiting in silence ends with the watchdog
    /// killing Cydia — which is how the first version of this went wrong.
    private static let clientScript = [
        "#!/bin/bash",
        "# Stands in for Cydia's setuid helper. The real one is cydo.real.",
        "set -u",
        "queue=\(queue)",
        "pid=$(cat \"$queue/pid\" 2>/dev/null || echo 0)",
        // `ps`, not `kill -0`: the client runs as mobile and the helper is
        // root's, so a signal to it comes back as "not permitted" rather than
        // as "alive", and the client would refuse every time.
        "if ! ps -p \"$pid\" >/dev/null 2>&1; then",
        "    echo \"cydo: Inferno helper is not running - press Repair the package manager in the app\" >&2",
        "    exit 2",
        "fi",
        "id=\"$queue/$$-$RANDOM\"",
        // apt reads dpkg's progress off a descriptor it opened itself. The work
        // happens in another process, so the descriptor is found here and the
        // helper's report is poured into it.
        "fd=\"\"",
        "prev=\"\"",
        "for a in \"$@\"; do",
        "    case \"$prev\" in --status-fd) fd=\"$a\";; esac",
        "    case \"$a\" in --status-fd=*) fd=\"${a#--status-fd=}\";; esac",
        "    prev=\"$a\"",
        "done",
        "printf %s\\\\n \"$@\" > \"$id.tmp\"",
        "touch \"$id.out\" \"$id.status\"",
        "mv \"$id.tmp\" \"$id.req\"",
        // Wakes the helper. In the background because opening the pipe waits
        // for the other end, and the helper may still be busy with the last
        // request — which is fine: it will find this one when it comes back.
        "(echo x > \"$queue/wake\" &) 2>/dev/null",
        "tail -n +1 -f \"$id.out\" 2>/dev/null &",
        "tail_pid=$!",
        "status_pid=0",
        // Only when it is really open: apt's descriptor is, but anyone calling
        // cydo by hand leaves it closed, and tail then says so on Cydia's
        // screen for no reason.
        "if [ -n \"$fd\" ] && { : >&\"$fd\"; } 2>/dev/null; then",
        "    tail -n +1 -f \"$id.status\" >&\"$fd\" 2>/dev/null &",
        "    status_pid=$!",
        "fi",
        "for _ in $(seq 1 3600); do",
        "    [ -e \"$id.rc\" ] && break",
        "    sleep 1",
        "done",
        "sleep 1",
        "kill \"$tail_pid\" 2>/dev/null",
        "[ \"$status_pid\" != 0 ] && kill \"$status_pid\" 2>/dev/null",
        "rc=$(cat \"$id.rc\" 2>/dev/null || echo 2)",
        "rm -f \"$id.out\" \"$id.rc\" \"$id.status\"",
        "exit \"$rc\"",
    ]

    /// Installs a `.deb` in the guest, without Cydia in the way.
    ///
    /// Cydia blocks its own main thread while dpkg runs, and on a guest this
    /// slow that is long enough for iOS to kill it as unresponsive — which
    /// also kills the helper's client and leaves dpkg holding the database.
    /// Here nothing is waiting on a screen, so a package can take its minutes.
    ///
    /// `--force-overwrite` because the bootstrap and the packages it came from
    /// disagree about who owns a handful of files in /etc/apt; every install in
    /// this image trips over that and nothing else.
    static func installDeb(_ local: URL, serial: SerialConsole, files: GuestFiles,
                           progress: @escaping (Int64, Int64) -> Void,
                           note: @escaping (String) -> Void) throws -> String {
        let remote = "/var/mobile/.inferno/install.deb"

        try awaitShell(serial)
        try ensureWritable(serial)

        // The carrying is one conversation and has to be: it is a sequence of
        // commands the guest answers in order. Everything after it is a single
        // long command, and those are left to run on their own.
        let package: String = try serial.exclusive {
            let shell = GuestShell(serial: serial)
            shell.line("mkdir -p /var/mobile/.inferno", timeout: 60)

            note(L("Переношу пакет в гостя…"))
            try files.carry(local, to: remote, plain: remote, shell: shell, progress: progress, note: note)

            // The name is read before the file goes away: it is what the apps
            // below are looked up by.
            return shell.text("dpkg-deb -f \(remote) Package", timeout: 300) ?? ""
        }

        note(L("Ставлю пакет…"))
        guard let code = try runDetached("dpkg -i --force-overwrite \(remote) >> \(log) 2>&1",
                                         serial: serial, timeout: 1800)
        else { throw Failure.silent(L("Ставлю пакет")) }

        note(L("Настраиваю пакеты…"))
        try runDetached("dpkg --configure -a >> \(log) 2>&1", serial: serial, timeout: 1800)

        // A package that brings an app leaves it on disk and nothing else:
        // SpringBoard learns about it from uicache. Only about this app, though
        // — `uicache --all` walks every app on the system, which on this guest
        // takes minutes with SpringBoard wedged for all of them.
        if !package.isEmpty { try refreshIcons(of: package, serial: serial, note: note) }

        let said = serial.exclusive { () -> String in
            let shell = GuestShell(serial: serial)
            shell.line("rm -f \(remote)", timeout: 60)
            return shell.text("grep -v '^$' \(log) | tail -3 | tr '\\n' ' ' | cut -c1-240", timeout: 120) ?? ""
        }
        if code != 0 { throw Failure.step(L("Ставлю пакет") + (said.isEmpty ? "" : ": " + said), code) }
        return said
    }

    /// What the guest has installed, as package name to version.
    ///
    /// Read through a file rather than off the console: the answer is three
    /// hundred lines long, and the console loses bytes inside long ones.
    static func installed(serial: SerialConsole, files: GuestFiles) throws -> [String: String] {
        let listing = "/var/mobile/.inferno/installed.txt"

        try awaitShell(serial)
        try serial.exclusive {
            let shell = GuestShell(serial: serial)
            shell.line("mkdir -p /var/mobile/.inferno", timeout: 60)
            guard shell.line("dpkg-query -W -f='${Package}\t${Version}\n' > \(listing) 2>/dev/null",
                             timeout: 600) != nil
            else { throw Failure.silent(L("Читаю установленное")) }
        }

        let file = try files.receive(listing, progress: { _, _ in })
        let text = String(decoding: (try? Data(contentsOf: file)) ?? Data(), as: UTF8.self)
        var out: [String: String] = [:]
        for line in text.split(separator: "\n") {
            let parts = line.split(separator: "\t", maxSplits: 1)
            guard parts.count == 2 else { continue }
            out[String(parts[0])] = String(parts[1])
        }
        return out
    }

    /// Removes a package, with its configuration files left where they are —
    /// the same thing Cydia's own remove does.
    static func remove(_ package: String, serial: SerialConsole) throws -> String {
        try ensureWritable(serial)
        try awaitShell(serial)
        serial.exclusive { GuestShell(serial: serial).line(": > \(log)", timeout: 60) }
        // Read before the removal: afterwards dpkg no longer knows what the
        // package owned.
        let apps = appPaths(of: package, serial: serial)

        guard let code = try runDetached("dpkg -r \(package) >> \(log) 2>&1", serial: serial, timeout: 1800)
        else { throw Failure.silent(L("Удаляю пакет")) }
        for app in apps { try runDetached("uicache -p \(app) >> \(log) 2>&1", serial: serial, timeout: 1800) }

        let said = serial.exclusive {
            GuestShell(serial: serial).text("grep -v '^$' \(log) | tail -3 | tr '\\n' ' ' | cut -c1-240",
                                            timeout: 120) ?? ""
        }
        if code != 0 { throw Failure.step(L("Удаляю пакет") + (said.isEmpty ? "" : ": " + said), code) }
        return said
    }

    /// Waits for a shell on the console instead of demanding one at once.
    ///
    /// The guest answers `echo 1` in a moment when it is idle and in a minute
    /// when it is not, and it is never idle right after a boot or in the middle
    /// of a package. One look and a refusal cost a download that had already
    /// been carried in.
    private static func awaitShell(_ serial: SerialConsole) throws {
        for attempt in 0..<10 {
            // The console is taken for the asking, not for the waiting: holding
            // it through the sleeps kept the shell pane from ever opening while
            // this ran.
            let answered = serial.exclusive { GuestShell(serial: serial).number("echo 1", timeout: 30) == 1 }
            if answered { return }
            if attempt < 9 { Thread.sleep(forTimeInterval: 5) }
        }
        throw Failure.noShell
    }

    /// Remounts the root writable, and makes sure it stayed that way.
    ///
    /// `mount -uw /` answers 0 and changes nothing when the guest is still
    /// booting — iOS puts the root back read-only on its way up — so the answer
    /// is not to be trusted and the mount table is read instead. Asked again
    /// every ten seconds: whatever the guest is doing to it, it stops doing it
    /// within a minute.
    private static func ensureWritable(_ serial: SerialConsole) throws {
        for attempt in 0..<6 {
            let writable = serial.exclusive { () -> Bool in
                let shell = GuestShell(serial: serial)
                shell.line("mount -uw /", timeout: 120)
                return shell.number("mount | grep -c ' on / .*read-only'") == 0
            }
            if writable { return }
            if attempt < 5 { Thread.sleep(forTimeInterval: 10) }
        }
        throw Failure.stillReadOnly
    }

    /// Runs a long command in the guest without sitting on the console.
    ///
    /// dpkg can take minutes here, and holding the conversation lock for all of
    /// them meant nothing else could say a word — the shell pane waited for its
    /// turn until it looked broken. The command is started detached, its status
    /// left in a file, and the console is only taken for the asking: a short
    /// check every few seconds.
    @discardableResult
    private static func runDetached(_ command: String, serial: SerialConsole,
                                    timeout: TimeInterval) throws -> Int64? {
        let done = "/var/mobile/.inferno/step.rc"
        // A guest that panics answers nothing ever again, and the wait below is
        // half an hour long. The console says when that has happened, so the
        // count is taken now and watched for the rest of the wait.
        let alive = serial.guestDeaths
        let started = serial.exclusive { () -> Bool in
            let shell = GuestShell(serial: serial)
            shell.line("rm -f \(done)", timeout: 60)
            return shell.line("(nohup /bin/bash -c '{ \(command) ;}; echo $? > \(done)' >/dev/null 2>&1 &)",
                              timeout: 120) == 0
        }
        guard started else { return nil }

        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            Thread.sleep(forTimeInterval: 3)
            if serial.guestDeaths != alive { throw Failure.crashed(serial.guestDeathReason) }
            let code = serial.exclusive { () -> Int64? in
                let shell = GuestShell(serial: serial)
                guard shell.number("test -e \(done); echo $?", timeout: 60) == 0 else { return nil }
                return shell.number("cat \(done)", timeout: 60)
            }
            if let code { return code }
        }
        return nil
    }

    /// The `/Applications` entries a package owns, if any.
    private static func appPaths(of package: String, serial: SerialConsole) -> [String] {
        let listed = serial.exclusive {
            GuestShell(serial: serial).text(
                "dpkg -L \(package) 2>/dev/null | grep -E '^/Applications/[^/]+\\.app$' | tr '\\n' ' '",
                timeout: 300) ?? ""
        }
        return listed.split(separator: " ").map(String.init)
    }

    /// Shows SpringBoard what a package brought, and nothing else.
    private static func refreshIcons(of package: String, serial: SerialConsole,
                                     note: @escaping (String) -> Void) throws {
        let apps = appPaths(of: package, serial: serial)
        guard !apps.isEmpty else { return }
        note(L("Показываю приложение SpringBoard…"))
        for app in apps { try runDetached("uicache -p \(app) >> \(log) 2>&1", serial: serial, timeout: 1800) }
    }

    /// Restarts SpringBoard.
    ///
    /// Tweaks are loaded into it when it starts, so one that was just installed
    /// does nothing until this happens. Cydia calls it a respring and asks
    /// first; here it is a menu entry for the same reason.
    static func respring(serial: SerialConsole) throws {
        try awaitShell(serial)
        serial.exclusive {
            let shell = GuestShell(serial: serial)
            shell.line("killall -9 SpringBoard", timeout: 120)
        }
    }

    /// Writes a file in the guest a line at a time. One line per command on
    /// purpose: the console drops bytes inside long lines, and a here-document
    /// would leave our own text sitting in the terminal while the guest is
    /// busy — which is exactly how transfers used to break.
    private static func write(_ lines: [String], to path: String, shell: GuestShell) throws {
        guard shell.line("rm -f \(path)", timeout: 60) == 0 else { throw Failure.step(path, -1) }
        for line in lines {
            let quoted = line.replacingOccurrences(of: "'", with: "'\\''")
            guard shell.line("printf '%s\\n' '\(quoted)' >> \(path)", timeout: 60) == 0 else {
                throw Failure.step(path, -1)
            }
        }
    }

    /// What a guest loses on every reboot, and what therefore has to be done
    /// again each time the machine starts. Seconds, not minutes.
    private static var fastSteps: [Step] {
        [
            Step(title: L("Готовлю папки apt"),
                 command: "mkdir -p /var/lib /var/cache/apt/archives/partial /var/lib/apt/lists/partial",
                 timeout: 60, fatal: true),
            Step(title: L("Ставлю ссылку на базу dpkg"), command: "ln -sfn /Library/dpkg /var/lib/dpkg",
                 timeout: 60, fatal: true),
        ]
    }

    /// Needed once per image and slow enough to be worth a button: on a phone
    /// these two are minutes, and they hold the console while they run.
    private static var slowSteps: [Step] {
        [
            Step(title: L("Регистрирую прошивку"),
                 command: "/usr/libexec/cydia/firmware.sh >> \(log) 2>&1",
                 timeout: 900, fatal: false),
            Step(title: L("Настраиваю пакеты"), command: "dpkg --configure -a >> \(log) 2>&1",
                 timeout: 1200, fatal: false),
            Step(title: L("Проверяю базу"), command: "dpkg --audit >> \(log) 2>&1",
                 timeout: 600, fatal: false),
        ]
    }

    /// Runs the repair. `note` is called with the step that is about to run, and
    /// returns the lines worth showing when it is over.
    /// Everything, including the slow steps. This is the button.
    static func repair(serial: SerialConsole, note: @escaping (String) -> Void) throws -> [String] {
        note(L("Перемонтирую корень на запись…"))
        try ensureWritable(serial)
        return try run(serial: serial, steps: fastSteps + slowSteps, note: note)
    }

    /// Only what the reboot undid: the remount, the folders, the helper. Run at
    /// every start, quietly, because without it Cydia is broken again and the
    /// error it gives says nothing about why.
    static func prepare(serial: SerialConsole) throws -> [String] {
        try ensureWritable(serial)
        return try run(serial: serial, steps: fastSteps, note: { _ in })
    }

    private static func run(serial: SerialConsole, steps: [Step],
                            note: @escaping (String) -> Void) throws -> [String] {
        var complaints: [String] = []

        // Nothing is typed into a console that has no shell on it yet: the
        // commands would land in the boot log and look like they ran. Waited
        // for outside the lock, so the shell pane can still open meanwhile.
        try awaitShell(serial)

        serial.exclusive {
            GuestShell(serial: serial).line("mkdir -p /var/mobile/.inferno; : > \(log)", timeout: 60)
        }

        // Each step on its own, and the long ones detached: firmware.sh and
        // dpkg take minutes here, and holding the console for them is what made
        // the shell pane look broken while a repair was running.
        for step in steps {
            note(step.title + "…")
            guard let code = try runDetached(step.command, serial: serial, timeout: step.timeout) else {
                if step.fatal { throw Failure.silent(step.title) }
                complaints.append(L("«%@» не ответил вовремя.", step.title))
                continue
            }
            guard code == 0 else {
                if step.fatal { throw Failure.step(step.title, code) }
                complaints.append(L("«%@» вернул %d.", step.title, Int(code)))
                continue
            }
        }

        try serial.exclusive {
            let shell = GuestShell(serial: serial)
            // The stand-in for the setuid helper. Written every time: it is
            // cheap, and an image where Cydia was reinstalled has the original
            // back in place.
            note(L("Ставлю помощника для Cydia…"))
            // Writing the two scripts costs a command per line, and the console
            // is held for all of them — long enough at every start for the shell
            // pane to give up waiting for its turn. So they are written only
            // when they are not already there in this exact shape. The marker
            // alone does not say so: installing or upgrading Cydia puts the
            // setuid original back over cydo and leaves the marker as it was,
            // and Cydia was broken again with every check here passing. So cydo
            // itself is looked at too.
            var stamp = PosixChecksum()
            let text = (rootScript + clientScript).joined(separator: "\n")
            text.utf8CString.withUnsafeBytes { stamp.update($0) }
            let version = String(format: "%08x", stamp.value)
            let marker = "/var/mobile/.inferno/cydo.version"
            let ours = "grep -qs '^# Stands in for Cydia' /usr/libexec/cydia/cydo"
            let same = shell.number("test -x /usr/libexec/cydia/cydo.real && \(ours) && "
                                    + "grep -qs '^\(version)$' \(marker) && echo 1 || echo 0") == 1

            if !same {
                try write(rootScript, to: "/usr/libexec/cydia/cydo-root.sh", shell: shell)
                try write(clientScript, to: "/tmp/cydo.new", shell: shell)
            }
            let install = [
                "chmod 755 /usr/libexec/cydia/cydo-root.sh",
                // Any cydo but the stand-in is the original: the bootstrap's the
                // first time, a newer Cydia's after an upgrade. Nothing runs
                // cydo.real; it is kept to put back.
                "\(ours) || mv -f /usr/libexec/cydia/cydo /usr/libexec/cydia/cydo.real",
                same ? "true" : "cp /tmp/cydo.new /usr/libexec/cydia/cydo",
                "chmod 755 /usr/libexec/cydia/cydo",
                "rm -f /tmp/cydo.new",
                "echo \(version) > \(marker)",
                // An earlier version of this shipped a launchd job. It does not
                // survive a reboot on this image, so it is taken back out.
                "launchctl unload /Library/LaunchDaemons/com.inferno.cydo.plist >/dev/null 2>&1",
                "rm -f /Library/LaunchDaemons/com.inferno.cydo.plist",
                // Requests left behind while no helper was running: nobody is
                // waiting on them any more.
                "mkdir -p \(queue)",
                "chmod 777 \(queue)",
                "rm -f \(queue)/*.req \(queue)/*.out \(queue)/*.rc \(queue)/*.tmp \(queue)/*.status \(queue)/wake",
                // By the process list, not by name: the guest's pkill does not
                // take `-f`, so helpers from earlier runs survived it — two of
                // them had burned a minute of the guest's own CPU apiece,
                // polling, which is exactly the slowdown this was meant to end.
                "ps ax | grep '[c]ydo-root.sh' | sed 's/^ *//;s/ .*//' | xargs kill -9 2>/dev/null",
                // In a subshell, and nothing after it on the line: a bare `&`
                // with the marker appended behind it is a syntax error, and
                // bash then runs none of this at all.
                "(nohup /bin/bash /usr/libexec/cydia/cydo-root.sh >/dev/null 2>&1 &)",
            ].joined(separator: "; ")
            if shell.line(install, timeout: 180) != 0 { complaints.append(L("Помощника не удалось разложить.")) }
            // It writes its pid as its first act, so a short wait tells a
            // running helper from one that fell over on startup.
            if shell.number("sleep 2; test -e \(queue)/pid; echo $?") != 0 {
                complaints.append(L("Помощник не запустился — Cydia останется без прав."))
            }

            note(L("Проверяю…"))
            // Still worth asking at the end: the guest can put the root back
            // while the slow steps are running.
            if shell.number("mount | grep -c ' on / .*read-only'") != 0 { throw Failure.stillReadOnly }
            if shell.number("test -e /var/lib/dpkg/status; echo $?") != 0 {
                complaints.append(L("База dpkg на месте не найдена — Cydia может всё ещё ругаться."))
            }
            // Whatever dpkg had to say, in one line: the console is no place for
            // paragraphs, and the last of it is the part that matters.
            if let said = shell.text("grep -v '^$' \(log) | tail -3 | tr '\\n' ' ' | cut -c1-240", timeout: 120),
               !said.isEmpty {
                complaints.append(L("dpkg сказал: %@", said))
            }
        }

        return complaints
    }
}
