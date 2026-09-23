/*
 * drag.c — synthesise a real mouse drag, and report what was actually posted.
 *
 * Goal 6 is "window-dragging performance of Safari should be a stable 120 fps".
 * Measuring it needs the drag to be a *drag*: a window moved by setting its
 * accessibility position does not take the same path through the window server
 * as a pointer held down across a title bar, so it does not produce the work
 * being complained about.
 *
 * This is C rather than the obvious Python because the guest's /usr/bin/python3
 * (3.9.6, Command Line Tools) has no PyObjC — `import Quartz` fails — while
 * clang and the ApplicationServices headers are present. Compiled on the guest
 * by the harness; nothing is shipped as a binary.
 *
 * Prints one line of JSON. The posted rate is reported rather than assumed
 * because a drag posted at 40 Hz cannot show a 120 Hz device, and reporting the
 * device as slow for that would be measuring this program.
 *
 * Build:
 *   clang -O2 -o drag drag.c -framework ApplicationServices -lm
 */
#include <ApplicationServices/ApplicationServices.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void post_at(CGEventType type, double x, double y)
{
    CGEventRef e = CGEventCreateMouseEvent(NULL, type, CGPointMake(x, y),
                                           kCGMouseButtonLeft);
    if (!e) {
        return;
    }
    CGEventPost(kCGHIDEventTap, e);
    CFRelease(e);
}

/* Monotonic: the pacing loop must not be perturbed by a clock adjustment, and
 * the guest's wall clock is settled by the host. */
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "usage: drag X Y SECONDS HZ AMPLITUDE_X AMPLITUDE_Y\n");
        return 2;
    }
    const double x0 = atof(argv[1]), y0 = atof(argv[2]);
    const double secs = atof(argv[3]), hz = atof(argv[4]);
    const double ax = atof(argv[5]), ay = atof(argv[6]);
    if (hz <= 0.0 || secs <= 0.0) {
        fprintf(stderr, "drag: seconds and hz must be positive\n");
        return 2;
    }
    const double period = 1.0 / hz;

    post_at(kCGEventLeftMouseDown, x0, y0);
    /* Let the window server latch the drag before moving, or the first samples
     * measure a click rather than a drag. */
    usleep(50000);

    const double start = now_s();
    double next = start, worst_late = 0.0;
    long posted = 0, late = 0;

    for (;;) {
        const double t = now_s() - start;
        if (t >= secs) {
            break;
        }
        /* Two incommensurate frequencies, so the path is a Lissajous curve
         * rather than a straight line: a straight uniform slide is the one
         * motion a compositor may coalesce or predict. Closed over the run, so
         * the window ends where it started and repeated runs do not walk it off
         * the screen. */
        const double x = x0 + ax * sin(2.0 * M_PI * 0.5 * t);
        const double y = y0 + ay * sin(2.0 * M_PI * 0.31 * t);
        post_at(kCGEventLeftMouseDragged, x, y);
        posted++;

        next += period;
        const double slack = next - now_s();
        if (slack > 0.0) {
            usleep((useconds_t)(slack * 1e6));
        } else {
            /* Could not keep up with the requested rate. Counted, not hidden. */
            late++;
            if (-slack > worst_late) {
                worst_late = -slack;
            }
            next = now_s();
        }
    }

    const double elapsed = now_s() - start;
    post_at(kCGEventLeftMouseUp, x0, y0);

    printf("{\"posted\":%ld,\"elapsed\":%.3f,\"posted_hz\":%.1f,"
           "\"late\":%ld,\"worst_late_s\":%.4f}\n",
           posted, elapsed, elapsed > 0.0 ? (double)posted / elapsed : 0.0,
           late, worst_late);
    return 0;
}
