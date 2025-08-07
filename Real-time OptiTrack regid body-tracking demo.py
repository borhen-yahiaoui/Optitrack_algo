"""
Real-time OptiTrack hand-tracking demo
=====================================

•  Prints a summary block only when the qualitative hand position changes
•  Speaks “Hand up / Hand down” every time the vertical state flips
•  Logs every frame to `mocap_data.csv`
•  Hides NatNet’s verbose debug output
"""

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import builtins
import csv
import queue
import sys
import threading
import time
from pathlib import Path

import pyttsx3
from NatNetClient import NatNetClient

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Constants & Configuration
# ──────────────────────────────────────────────────────────────────────────────
RIGHT_HAND_ID: int = 77

# Movement thresholds (metres)
UP_T: float = 0.05
SIDE_T: float = 0.05
FWD_T: float = 0.05

CSV_PATH = Path("mocap_data.csv")

# Strings NatNet prints that we want to hide
_NATNET_PREFIXES = (
    "MoCap Frame", "MarkerData", "Model Name", "Marker Count", "Unlabeled",
    "Rigid Body", "Skeleton", "Labeled Marker", "Force Plate", "Device Count",
    "ID ", "Position", "Orientation", "Marker Error", "Tracking Valid",
    "No Asset Data", "Timestamp", "-----------------", "MoCap Frame Begin",
    "MoCap Frame End",
)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Utility: Silence NatNet debug output
# ──────────────────────────────────────────────────────────────────────────────
_original_print = builtins.print


def _quiet_print(*args, **kwargs):
    if args and any(str(args[0]).lstrip().startswith(p) for p in _NATNET_PREFIXES):
        return
    _original_print(*args, **kwargs)


builtins.print = _quiet_print  # monkey-patch built-in print

# ──────────────────────────────────────────────────────────────────────────────
# 4.  Text-to-speech infrastructure (single worker thread + queue)
# ──────────────────────────────────────────────────────────────────────────────
_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", 175)
_tts_queue: queue.Queue[str | None] = queue.Queue()


def _tts_worker() -> None:
    while True:
        msg = _tts_queue.get()
        if msg is None:  # sentinel for clean shutdown
            break
        _tts_engine.stop()             # interrupt any ongoing utterance
        _tts_engine.say(msg)
        _tts_engine.runAndWait()
        _tts_queue.task_done()


threading.Thread(target=_tts_worker, daemon=True).start()


def speak(msg: str) -> None:
    """Enqueue a phrase to be spoken asynchronously."""
    _tts_queue.put(msg)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Global state updated by callbacks
# ──────────────────────────────────────────────────────────────────────────────
cur_frame: int | None = None
cur_ts: float | None = None

ref_pos: tuple[float, float, float] | None = None      # (x0, y0, z0)
last_label: tuple[str, str, str] | None = None         # (vert, depth, side)
last_vert_state: str | None = None                     # "UP" / "DOWN" / "LEVEL"

# ──────────────────────────────────────────────────────────────────────────────
# 6.  CSV logger
# ──────────────────────────────────────────────────────────────────────────────
csv_file = CSV_PATH.open("w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ["frameNumber", "timestamp", "rigidBodyID",
     "x", "y", "z", "qx", "qy", "qz", "qw"]
)

# ──────────────────────────────────────────────────────────────────────────────
# 7.  NatNet callback functions
# ──────────────────────────────────────────────────────────────────────────────
def on_new_frame(data_dict: dict) -> None:
    """Stores the current frame number & timestamp."""
    global cur_frame, cur_ts
    cur_frame = data_dict.get("frameNumber")
    cur_ts = data_dict.get("timestamp")


def on_rigid_body(rb_id: int, pos: tuple[float, float, float] | None,
                  rot: tuple[float, float, float, float]) -> None:
    """Processes right-hand rigid-body data each frame."""
    global ref_pos, last_label, last_vert_state

    if rb_id != RIGHT_HAND_ID or pos is None:
        return

    x, y, z = pos

    # Initial reference centre
    if ref_pos is None:
        ref_pos = pos
        print(f"\nReference set at: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        return

    # Relative offsets
    dx, dy, dz = x - ref_pos[0], y - ref_pos[1], z - ref_pos[2]

    # Qualitative labels
    vert = "UP" if dy > UP_T else "DOWN" if dy < -UP_T else "LEVEL"
    depth = "FORWARD" if dz > FWD_T else "BACK" if dz < -FWD_T else "CENTER"
    side = "RIGHT" if dx > SIDE_T else "LEFT" if dx < -SIDE_T else "CENTER"
    label = (vert, depth, side)

    # Speak on every vertical flip
    if vert in ("UP", "DOWN") and vert != last_vert_state:
        speak(f"Hand {vert.lower()}")
        last_vert_state = vert

    # Print block only when any label changes
    if label != last_label:
        print(f"\nFrame {cur_frame}")
        print(f"  x={x:7.3f}  y={y:7.3f}  z={z:7.3f}")
        print(f"  Vertical: {vert}   Depth: {depth}   Side: {side}")
        last_label = label

    # Log raw data every frame
    csv_writer.writerow([cur_frame, cur_ts, rb_id,
                         x, y, z, *rot])
    csv_file.flush()

# ──────────────────────────────────────────────────────────────────────────────
# 8.  Main routine
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    client = NatNetClient()
    client.set_client_address("127.0.0.1")
    client.set_server_address("169.254.162.216")
    client.set_use_multicast(True)

    client.new_frame_listener = on_new_frame
    client.rigid_body_listener = on_rigid_body

    print("Starting NatNet client …")
    if not client.run():
        sys.exit("❌ Could not start NatNet client")

    time.sleep(0.5)  # small handshake delay
    client.send_request(
        client.command_socket,
        client.NAT_REQUEST_MODELDEF,
        "",
        (client.server_ip_address, client.command_port)
    )

    print("Running — voice every UP/DOWN change; block printed on any change.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.01)  # keep main thread alive
    except KeyboardInterrupt:
        pass
    finally:
        client.shutdown()
        _tts_queue.put(None)   # stop TTS worker
        csv_file.close()
        print("\nSaved to", CSV_PATH.resolve())


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
