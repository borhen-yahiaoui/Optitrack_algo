"""
hand_tracker.py  –  OptiTrack real-time demo
• Prints a summary block whenever qualitative position changes
• Speaks “Hand up / Hand down” every time the vertical state flips
• Logs every frame to mocap_data.csv
"""

# ──────────────────────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────────────────────
from __future__ import annotations

import builtins
import csv
import sys
import threading
import time
from pathlib import Path

import pyttsx3
from NatNetClient import NatNetClient

# ──────────────────────────────────────────────────────────────
# 2. Configuration constants
# ──────────────────────────────────────────────────────────────
RIGHT_HAND_ID = 77
UP_T, SIDE_T, FWD_T = 0.05, 0.05, 0.05       # thresholds in metres
CSV_PATH = Path("mocap_data.csv")

# NatNet debug prefixes to suppress
_HIDDEN_PREFIXES = (
    "MoCap Frame", "MarkerData", "Model Name", "Marker Count", "Unlabeled",
    "Rigid Body", "Skeleton", "Labeled Marker", "Force Plate", "Device Count",
    "ID ", "Position", "Orientation", "Marker Error", "Tracking Valid",
    "No Asset Data", "Timestamp", "-----------------", "MoCap Frame Begin",
    "MoCap Frame End",
)

# ──────────────────────────────────────────────────────────────
# 3. Silence NatNet’s verbose prints
# ──────────────────────────────────────────────────────────────
_builtin_print = builtins.print
def quiet_print(*args, **kwargs):
    if args and any(str(args[0]).lstrip().startswith(p) for p in _HIDDEN_PREFIXES):
        return
    _builtin_print(*args, **kwargs)
builtins.print = quiet_print

# ──────────────────────────────────────────────────────────────
# 4. Thread-safe text-to-speech (single engine + lock)
# ──────────────────────────────────────────────────────────────
_tts = pyttsx3.init()
_tts.setProperty("rate", 175)
_tts_lock = threading.Lock()

def say_async(text: str) -> None:
    def _worker() -> None:
        with _tts_lock:
            _tts.say(text)
            _tts.runAndWait()
    threading.Thread(target=_worker, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# 5. Runtime state holders
# ──────────────────────────────────────────────────────────────
cur_frame: int | None = None
cur_ts: float | None = None
ref_pos: tuple[float, float, float] | None = None
last_label: tuple[str, str, str] | None = None
last_vert_state: str | None = None          # "UP", "DOWN", or None

# ──────────────────────────────────────────────────────────────
# 6. CSV logger (open once)
# ──────────────────────────────────────────────────────────────
csv_file = CSV_PATH.open("w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frameNumber", "timestamp", "rigidBodyID",
                     "x", "y", "z", "qx", "qy", "qz", "qw"])

# ──────────────────────────────────────────────────────────────
# 7. NatNet callback functions
# ──────────────────────────────────────────────────────────────
def on_new_frame(data: dict) -> None:
    """Stores frame number & timestamp each frame."""
    global cur_frame, cur_ts
    cur_frame, cur_ts = data.get("frameNumber"), data.get("timestamp")

def on_rigid_body(rb_id: int,
                  pos: tuple[float, float, float] | None,
                  rot: tuple[float, float, float, float]) -> None:
    """Processes right-hand data, prints & speaks on state change."""
    global ref_pos, last_label, last_vert_state

    if rb_id != RIGHT_HAND_ID or pos is None:
        return

    x, y, z = pos

    # First valid sample sets reference centre
    if ref_pos is None:
        ref_pos = pos
        print(f"\nReference set at: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        return

    dx, dy, dz = x - ref_pos[0], y - ref_pos[1], z - ref_pos[2]

    vert  = "UP"      if dy >  UP_T  else "DOWN" if dy < -UP_T  else "LEVEL"
    depth = "FORWARD" if dz >  FWD_T else "BACK" if dz < -FWD_T else "CENTER"
    side  = "RIGHT"   if dx >  SIDE_T else "LEFT" if dx < -SIDE_T else "CENTER"
    label = (vert, depth, side)

    # ---- VOICE: speak on every new UP/DOWN, reset after LEVEL ----
    if vert in ("UP", "DOWN") and vert != last_vert_state:
        say_async(f"Hand {vert.lower()}")
        last_vert_state = vert
    elif vert == "LEVEL":
        last_vert_state = None        # ensures next UP/DOWN will speak

    # ---- PRINT: only when qualitative label changes ----
    if label != last_label:
        print(f"\nFrame {cur_frame}")
        print(f"  x={x:7.3f}  y={y:7.3f}  z={z:7.3f}")
        print(f"  Vertical: {vert}   Depth: {depth}   Side: {side}")
        last_label = label

    # ---- LOG raw data every frame ----
    csv_writer.writerow([cur_frame, cur_ts, rb_id,
                         x, y, z, *rot])
    csv_file.flush()

# ──────────────────────────────────────────────────────────────
# 8. Main routine
# ──────────────────────────────────────────────────────────────
def main() -> None:
    client = NatNetClient()
    client.set_client_address("127.0.0.1")
    client.set_server_address("169.254.162.216")
    client.set_use_multicast(True)

    client.new_frame_listener   = on_new_frame
    client.rigid_body_listener  = on_rigid_body

    print("Starting NatNet client …")
    if not client.run():
        sys.exit("❌ Could not start NatNet client")

    time.sleep(0.5)  # let sockets settle
    client.send_request(client.command_socket, client.NAT_REQUEST_MODELDEF, "",
                        (client.server_ip_address, client.command_port))

    print("Running — speaks every UP/DOWN; prints block on any change. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.01)  # keep main thread alive
    except KeyboardInterrupt:
        pass
    finally:
        client.shutdown()
        csv_file.close()
        print("\nSaved to", CSV_PATH.resolve())

# ──────────────────────────────────────────────────────────────
# 9. Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
