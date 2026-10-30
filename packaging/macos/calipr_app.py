"""caliPr as a Mac app: the labeler in a window of its own.

The window shows the same page the labeler serves to a browser, from the same
server, started here on a port of the app's own. What the app adds is where things
are kept, since its code sits inside the application and is not to be written to
(``fish_morpho.paths``):

- studies and exports in ``~/Documents/caliPr``, where they can be found;
- the trained models and Auto-label's environment in
  ``~/Library/Application Support/caliPr``, along with this app's log.

Auto-label is not built in: it needs about 2.3 GB, and hand labelling does not.
The page offers to install it the first time Auto-label is clicked.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

# Either can be set beforehand to keep a test, or a second copy, apart.
HOME = Path(os.environ.setdefault(
    "CALIPR_HOME", str(Path.home() / "Documents" / "caliPr")))
SUPPORT = Path(os.environ.setdefault(
    "CALIPR_MODELS", str(Path.home() / "Library" / "Application Support" / "caliPr")))
os.environ["CALIPR_APP"] = "1"
# The helper scripts the server starts are this same Python; none of them may
# write bytecode into the signed bundle either.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

#: The page keeps unsent drafts in the window's storage, which belongs to one
#: address. A fixed port keeps them across launches; the next ones are for when
#: something else holds it.
PORTS = range(18765, 18775)


def free_port() -> int:
    for port in PORTS:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit("caliPr: no free port between "
                     f"{PORTS.start} and {PORTS.stop - 1}")


def serve(port: int) -> None:
    import label_server
    label_server.main(["--port", str(port)])


def wait_for(url: str, seconds: float = 30) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + "api/authstate", timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def let_pages_choose_folders() -> None:
    """Make ``<input webkitdirectory>`` open a folder chooser, as it does in Safari.

    pywebview answers every file input with a file chooser, so the labeler's "Add
    folder" could only pick files. WebKit says which one the page asked for; this
    passes that on, and WebKit itself then lists the folder's files for the page.
    """
    import AppKit
    from webview.platforms import cocoa

    base = cocoa.BrowserView.BrowserDelegate

    class Delegate(base):
        def webView_runOpenPanelWithParameters_initiatedByFrame_completionHandler_(
                self, webview, params, frame, handler):
            panel = AppKit.NSOpenPanel.openPanel()
            folders = bool(params.allowsDirectories())
            panel.setCanChooseDirectories_(folders)
            panel.setCanChooseFiles_(not folders)
            panel.setAllowsMultipleSelection_(bool(params.allowsMultipleSelection()))
            if panel.runModal() == AppKit.NSModalResponseOK:
                handler(panel.URLs())
            else:
                handler(None)

    cocoa.BrowserView.BrowserDelegate = Delegate


def main() -> int:
    HOME.mkdir(parents=True, exist_ok=True)
    SUPPORT.mkdir(parents=True, exist_ok=True)
    log = open(SUPPORT / "caliPr.log", "a", buffering=1)
    sys.stdout = sys.stderr = log
    print(f"\n--- caliPr starting {time.strftime('%Y-%m-%d %H:%M:%S')} from {ROOT}")

    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    threading.Thread(target=serve, args=(port,), daemon=True).start()

    import webview
    webview.settings["ALLOW_DOWNLOADS"] = True
    if not wait_for(url):
        webview.create_window("caliPr", html=(
            "<p style='font:15px -apple-system;padding:2em'>caliPr's labeler did not "
            f"start. What went wrong is in<br><code>{SUPPORT / 'caliPr.log'}</code></p>"))
        webview.start()
        return 1

    window = webview.create_window("caliPr", url, width=1440, height=900,
                                   min_size=(960, 600), text_select=True)

    flushed = threading.Event()

    def flush_then_close():
        done = threading.Event()
        try:
            window.evaluate_js("Promise.resolve(typeof flushSave==='function' && flushSave())",
                               lambda _result: done.set())
            done.wait(8)
        except Exception as exc:
            print(f"flush on close: {exc}")
        flushed.set()
        window.destroy()

    def closing():
        # Labels save themselves a moment after each change; a save still waiting
        # is sent before the window goes. Closing is held for that, on another
        # thread: this one runs the window, and the page cannot answer while it waits.
        if flushed.is_set():
            return True
        threading.Thread(target=flush_then_close, daemon=True).start()
        return False

    window.events.closing += closing
    let_pages_choose_folders()
    webview.start(private_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
