#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// The program that makes a restored system usable, run inside the guest.
//
// ChefKiss's guide ends with a computer: mount what the restore wrote and run
// the filesystem patcher over the shared cache. This is that step, done in the
// same ramdisk boot the restore itself runs in, so no computer is needed at
// all. The app puts this program and the patcher into the ramdisk (see
// RestoreRamdisk.swift) and launchd starts it at boot.
//
// Timing is everything here, and so is keeping still. Mounting the volumes
// while ASR still had them broke one restore; asking NVRAM every few seconds
// what the outcome was broke another, right as the restore was writing NOR.
//
// So this only reads a directory. The data volume is the last thing a restore
// makes -- it showed up nine minutes in, after everything else -- and by then
// the writing is over. The machine waits for us afterwards because the emulator
// is told to hold the reset the guest asks for.

#define MOUNTPOINT "/mnt9"
#define CACHE_DIR  MOUNTPOINT "/System/Library/Caches/com.apple.dyld"
#define SYSTEM_VOLUME "/dev/disk0s1s1"

static int run_to(const char* out_path, const char* path, char* const argv[])
{
    pid_t child = fork();
    if (child == 0) {
        if (out_path != NULL) {
            int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) { dup2(fd, 1); }
        }
        execv(path, argv);
        _exit(127);
    }
    int status = 0;
    waitpid(child, &status, 0);
    // A signalled child's WEXITSTATUS is 0, which reads exactly like success:
    // an unsigned binary is killed by AMFI the moment it execs, and that is how
    // a patcher that never ran once looked like one that finished cleanly.
    if (WIFSIGNALED(status)) { return -WTERMSIG(status); }
    return WEXITSTATUS(status);
}

static int run(const char* path, char* const argv[]) { return run_to(NULL, path, argv); }

// The data volume, the last thing the restore creates.
#define DATA_VOLUME "/dev/disk0s1s2"

static int restore_finished(void)
{
    struct stat st;
    return stat(DATA_VOLUME, &st) == 0 && stat(SYSTEM_VOLUME, &st) == 0;
}

static int cache_path(char* out, int size)
{
    DIR* dir = opendir(CACHE_DIR);
    if (dir == NULL) { return 0; }
    int found = 0;
    for (struct dirent* e = readdir(dir); e != NULL && !found; e = readdir(dir)) {
        if (strncmp(e->d_name, "dyld_shared_cache", 17) != 0 || strstr(e->d_name, ".map") != NULL) { continue; }
        snprintf(out, size, "%s/%s", CACHE_DIR, e->d_name);
        found = 1;
    }
    closedir(dir);
    return found;
}

/*
 * The other half of the filesystem patches: five services have to be off, or
 * the system never reaches its setup screen. ChefKiss's guide adds `Disabled`
 * to each of them inside launchd's own service cache, which is a binary plist
 * and awkward to edit in place. launchd reads an override file as well, so this
 * writes that instead -- same effect, one small file.
 *
 * It lives on the data volume, which the running system mounts at /private/var.
 */
#define DATA_MOUNTPOINT "/mnt8"
#define OVERRIDES_DIR   DATA_MOUNTPOINT "/db/com.apple.xpc.launchd"

static void disable_services(void)
{
    static const char* const names[] = {
        "com.apple.voicemail.vmd",       "com.apple.CommCenter", "com.apple.CommCenterMobileHelper",
        "com.apple.CommCenterRootHelper", "com.apple.locationd",
    };

    mkdir(DATA_MOUNTPOINT, 0755);
    char* mount_args[] = {"/sbin/mount_apfs", "-o", "rw", DATA_VOLUME, DATA_MOUNTPOINT, NULL};
    int   mounted = 1;
    for (int tries = 0; tries < 100 && mounted != 0; tries++) {
        mounted = run(mount_args[0], mount_args);
        if (mounted != 0) { usleep(200000); }
    }
    printf("*** PATCHER: mount_apfs %s says %d\n", DATA_VOLUME, mounted);
    fflush(stdout);
    if (mounted != 0) { return; }

    mkdir(DATA_MOUNTPOINT "/db", 0755);
    mkdir(OVERRIDES_DIR, 0755);

    int fd = open(OVERRIDES_DIR "/disabled.plist", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
        dprintf(fd, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
                    "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
                    "<plist version=\"1.0\">\n<dict>\n");
        for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
            dprintf(fd, "\t<key>%s</key>\n\t<true/>\n", names[i]);
        }
        dprintf(fd, "</dict>\n</plist>\n");
        close(fd);
        printf("*** PATCHER: five services switched off\n");
    }
    else { printf("*** PATCHER: could not write the override file\n"); }
    fflush(stdout);

    char* umount[] = {"/sbin/umount", DATA_MOUNTPOINT, NULL};
    run(umount[0], umount);
}

int main(void)
{
    int console = open("/dev/console", O_WRONLY);
    if (console >= 0) { dup2(console, 1); dup2(console, 2); }
    printf("*** PATCHER: waiting for the restore to finish\n");
    fflush(stdout);

    while (!restore_finished()) { sleep(2); }
    printf("*** PATCHER: the data volume is here -- the restore is done writing\n");
    fflush(stdout);

    // The restore keeps the system volume to itself until the very last of its
    // work -- the snapshot it takes after the data volume exists -- so the
    // answer to "resource busy" is to ask again. There is no hurry: the machine
    // is held for us.
    mkdir(MOUNTPOINT, 0755);
    char* mount_args[] = {"/sbin/mount_apfs", "-o", "rw", SYSTEM_VOLUME, MOUNTPOINT, NULL};
    // The volume frees up for a moment at the very end, between the restore
    // letting go of it and the guest asking to reset -- and once it has asked,
    // the kernel gives the reset about half a minute before it panics with
    // "Halt/Restart Timed Out". So this grabs at it rather than waiting
    // politely.
    int mounted = 1;
    for (int tries = 0; tries < 4000 && mounted != 0; tries++) {
        mounted = run(mount_args[0], mount_args);
        if (mounted != 0) { usleep(200000); }
    }
    printf("*** PATCHER: mount_apfs %s says %d\n", SYSTEM_VOLUME, mounted);
    fflush(stdout);
    if (mounted != 0) { return 1; }

    char cache[512];
    if (!cache_path(cache, sizeof(cache))) {
        printf("*** PATCHER: no shared cache on the volume\n");
        fflush(stdout);
        char* umount[] = {"/sbin/umount", MOUNTPOINT, NULL};
        run(umount[0], umount);
        return 1;
    }

    struct stat st;
    stat(cache, &st);
    printf("*** PATCHER: patching %s, %lld MB\n", cache, (long long)st.st_size >> 20);
    fflush(stdout);

    // Kept rather than left to the console: the patcher prints every address it
    // writes, and the console dropped all of it last time -- which is exactly
    // what was needed to see that the pair it wrote landed an instruction out.
    char* patch_args[] = {"/usr/standalone/firmware/nfrestore/firmware/jcop-prod/JCOP-11.04-012.2-P.bin", cache, NULL};
    int   code = run_to(MOUNTPOINT "/orchard_patch.log", patch_args[0], patch_args);
    printf("*** PATCHER: the patcher exited with %d\n", code);
    fflush(stdout);

    int log = open(MOUNTPOINT "/orchard_patch.log", O_RDONLY);
    if (log >= 0) {
        static char text[65536];
        ssize_t     got = read(log, text, sizeof(text) - 1);
        close(log);
        if (got > 0) {
            text[got] = '\0';
            printf("*** PATCHER: what it wrote ---\n%s\n*** PATCHER: --- end\n", text);
            fflush(stdout);
        }
    }

    char* umount[] = {"/sbin/umount", MOUNTPOINT, NULL};
    run(umount[0], umount);

    disable_services();

    printf("*** PATCHER: done, the machine can be reset now\n");
    fflush(stdout);
    return 0;
}
