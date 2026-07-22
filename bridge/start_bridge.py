"""
Launcher for the Atlas bridge.

Exists so the bridge can be started from the **MAXScript** listener, which is
the one that opens by default, without fighting MaxScript string quoting:

    python.ExecuteFile @"<path-to-repo>\\bridge\\start_bridge.py"

From the Python listener you can equally just import atlas_max_bridge directly.

Either way this must run on the main thread — the bridge creates its QTimer
here, and a QTimer created off the main thread would never fire, leaving every
command to time out.
"""

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

try:
    import importlib

    # Stop any previously-loaded bridge FIRST. Reloading the module builds a new
    # _Bridge instance, which loses the reference to the old one — and the old
    # one is still holding the listening socket, so the fresh instance fails to
    # bind with a confusing "address in use". Release it before reloading.
    _previous = sys.modules.get("atlas_max_bridge")
    if _previous is not None:
        try:
            _previous.stop()
        except Exception:
            pass

    import atlas_max_bridge

    # Reload so re-running this file picks up edits without restarting Max.
    atlas_max_bridge = importlib.reload(atlas_max_bridge)

    atlas_max_bridge.start()

    print("")
    print("=" * 58)
    print(" Atlas bridge ready")
    print(f"   port    : {atlas_max_bridge.PORT}")
    print(f"   maxscript: {'ENABLED' if atlas_max_bridge.ALLOW_MAXSCRIPT else 'disabled'}")
    print("   stop with: atlas_max_bridge.stop()")
    print("=" * 58)

except Exception:
    print("")
    print("!" * 58)
    print(" Atlas bridge FAILED to start")
    print("!" * 58)
    traceback.print_exc()
    raise
