# vmapple-guest-config.sh

Configures the provisioned vmapple macOS guest for headless agent-driven testing:

- disables sleep / display-sleep / disk-sleep / powernap / standby / autopoweroff (`pmset`),
- disables the screensaver, App Nap, and the screen lock,
- keeps **Remote Login (SSH)** on.

Applied from the host over SSH. **Run it against an `--interactive` (pristine) boot** so the
settings persist into the golden snapshot; a `--testing` boot is a throwaway clone.

Auto-login is deliberately not scripted. Enable it manually in System Settings if the snapshot
should reach the desktop without a password.

## Run

```sh
scripts/vmapple-guest-config/vmapple-guest-config.sh
```

Env: `GUEST_USER` (default `macosvm`), `GUEST_PW` (default = the username), `SSH_PORT` (2222),
`SSH_KEY` (`~/.ssh/vmapple_guest`).

## Guest access

The guest is reachable at `localhost:2222` (QEMU hostfwd → guest :22). A key was installed for
keyless access; the host SSH alias `vmapple-guest` wraps it:

```sh
ssh vmapple-guest            # keyless, via ~/.ssh/vmapple_guest
```

Password auth also works (user `macosvm`, password = the username) for the console / sudo.

## Verify

After a reboot, confirm SSH reaches the guest and the console user is the expected account:

```sh
ssh vmapple-guest 'stat -f%Su /dev/console'
```
