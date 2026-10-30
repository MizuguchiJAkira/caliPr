/* caliPr.app's executable: run the app's own Python on the app's entry script.
 *
 * A compiled executable rather than a shell script, because macOS takes the
 * app's name, Dock icon and menu bar from the bundle whose Contents/MacOS holds
 * the running program. The Python this starts lives in the same folder, so the
 * window that opens is caliPr's, not Python's.
 */
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    char exe[PATH_MAX], real[PATH_MAX], python[PATH_MAX], script[PATH_MAX];
    uint32_t size = sizeof exe;
    if (_NSGetExecutablePath(exe, &size) != 0 || realpath(exe, real) == NULL) {
        fprintf(stderr, "caliPr: cannot find where the app is\n");
        return 1;
    }
    const char *macos = dirname(real);
    snprintf(python, sizeof python, "%s/python3", macos);
    snprintf(script, sizeof script, "%s/../Resources/app/packaging/macos/calipr_app.py", macos);
    /* -E -s: nothing from the user's own Python setup leaks in.
       -B: never write bytecode into the signed bundle (it ships compiled). */
    char *args[] = {python, "-E", "-s", "-B", script, NULL};
    execv(python, args);
    perror("caliPr: could not start Python");
    return 1;
}
