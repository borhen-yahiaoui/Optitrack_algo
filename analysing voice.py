# test_lstm_only.py — Improved version with debugging, better LSTM checks,
# faster feedback for dancing, and safer parsing.

import os, sys, glob, re, time, math, threading
from pathlib import Path
import numpy as np
import pandas as pd

# ======== DEBUG SWITCH (turn ON while fixing) ========
DEBUG = True
# =====================================================

# ============ SETTINGS ============
RESAMPLE_HZ           = 20
REFERENCE_SECONDS     = 2.0
ROLLING_MEDIAN_MS     = 300

DWELL_SECONDS         = 0.6    # MORE SENSITIVE FOR DANCING
COMMENT_MIN_GAP_S     = 0.8
USE_VOICE             = True
PROGRESS_PER_SECOND   = True
PROGRESS_DELAY_S      = 0.75
COMMENT_DELAY_S       = 0.30

# Positive feedback (faster)
SAY_OK_ON_CLEAR       = True
SAY_GOOD_AFTER_S      = 2.0
SAY_EXCELLENT_AFTER_S = 5.0
# ==================================

HERE  = Path(__file__).parent
CLEAN = HERE / "clean_out"
CLEAN.mkdir(exist_ok=True)

# ---------- voice ----------
_tts = None
_tts_lock = threading.Lock()

def voice_init():
    global _tts
    if not USE_VOICE: return
    try:
        import pyttsx3
        _tts = pyttsx3.init()
        _tts.setProperty("rate", 175)
    except Exception:
        _tts = None

def say(text: str):
    if not USE_VOICE or _tts is None:
        return
    with _tts_lock:
        try:
            _tts.say(text)
            _tts.runAndWait()
        except Exception:
            pass

# ---------- CSV parsing (Motive ; + multi-row header) ----------
def parse_motive_semicolon_csv(path: Path):
    raw = pd.read_csv(path, sep=';', header=None, dtype=str, engine='python')
    if len(raw) < 7:
        raise SystemExit(f"CSV looks too short: {path}")

    names_row  = list(raw.iloc[3])
    header_row = list(raw.iloc[6])
    data = pd.read_csv(path, sep=';', header=None, skiprows=7, engine='python')

    bases = {}
    for j in range(2, len(header_row), 3):
        if j + 2 >= len(header_row): break
        base = names_row[j]
        if base is None or (isinstance(base, float) and pd.isna(base)):
            base = f"col_{j}"
        base = str(base).lower().strip()

        # normalize several naming variants
        base = re.sub(r"\bright\s*(elbow|hand)\b", "right hand", base)
        base = re.sub(r"\bleft\s*(elbow|hand)\b",  "left hand",  base)
        base = re.sub(r"\bforehead\b",             "head",       base)
        base = re.sub(r"\bhead.*",                 "head",       base)

        bases.setdefault(base, []).append((j, j+1, j+2))

    time_vals = pd.to_numeric(data.iloc[:, 1], errors='coerce').to_numpy()

    base_xyz = {}
    for base, triplets in bases.items():
        xs, ys, zs = [], [], []
        for (xc, yc, zc) in triplets:
            xs.append(pd.to_numeric(data.iloc[:, xc], errors='coerce').to_numpy())
            ys.append(pd.to_numeric(data.iloc[:, yc], errors='coerce').to_numpy())
            zs.append(pd.to_numeric(data.iloc[:, zc], errors='coerce').to_numpy())
        X = np.nanmean(np.stack(xs, axis=1), axis=1) if xs else np.array([])
        Y = np.nanmean(np.stack(ys, axis=1), axis=1) if ys else np.array([])
        Z = np.nanmean(np.stack(zs, axis=1), axis=1) if zs else np.array([])
        if X.size and Y.size and Z.size:
            base_xyz[base] = np.stack([X, Y, Z], axis=1)
    return time_vals, base_xyz

def group_to_three(keys):
    groups = {"head": [], "left hand": [], "right hand": []}
    for name in keys:
        n = name.lower()
        if "left hand"  in n: groups["left hand"].append(name)
        elif "right hand" in n: groups["right hand"].append(name)
        elif "head" in n: groups["head"].append(name)
    return groups

# ---------- signal utils ----------
def resample_uniform(df_pos: pd.DataFrame, hz: int):
    t0, t1 = float(df_pos["time"].iloc[0]), float(df_pos["time"].iloc[-1])
    dt = 1.0 / hz
    tunif = np.arange(t0, t1 + 1e-9, dt)
    dfu = pd.DataFrame({"time": tunif})
    out = pd.merge_asof(dfu, df_pos.sort_values("time"), on="time", direction="nearest")
    return out.interpolate().ffill().bfill()

def heal_series(s: pd.Series, hz: float, max_hold_s=0.5, max_interp_s=2.0):
    s = s.copy()
    if not s.isna().any(): return s
    s = s.ffill(limit=int(max_hold_s*hz))
    s = s.interpolate(limit=int(max_interp_s*hz), limit_direction="both")
    return s.bfill().ffill()

def rolling_median_cols(df: pd.DataFrame, cols, win):
    sm = df[cols].rolling(win, center=True, min_periods=max(1, win//3)).median()
    return sm.bfill().ffill()

def build_pos_df(time_vals: np.ndarray, xyz: np.ndarray):
    if xyz.size == 0:
        raise SystemExit("No marker data found.")
    mm2m = 0.001 if np.nanmedian(np.abs(xyz)) > 5.0 else 1.0
    df = pd.DataFrame({
        "time": pd.to_numeric(time_vals, errors="coerce"),
        "x_m": xyz[:,0] * mm2m,
        "y_m": xyz[:,1] * mm2m,
        "z_m": xyz[:,2] * mm2m
    }).sort_values("time")
    df[["x_m","y_m","z_m"]] = df[["x_m","y_m","z_m"]].interpolate().ffill().bfill()
    return df

# ---------- clean newest CSV → head/left/right ----------
def clean_three_from_csv(path: Path):
    time_vals, base_xyz = parse_motive_semicolon_csv(path)
    groups = group_to_three(base_xyz.keys())

    slug = path.stem.replace(" ", "_").replace(".", "_")
    outs = {}

    for part in ["head","left hand","right hand"]:
        names = groups.get(part, [])
        if not names:
            raise SystemExit(f"Missing '{part}' markers in CSV.")

        arrs = [base_xyz[n] for n in names if n in base_xyz]
        xyz  = np.nanmean(np.stack(arrs, axis=2), axis=2)

        df = build_pos_df(time_vals, xyz)
        out = resample_uniform(df, RESAMPLE_HZ)
        hz = RESAMPLE_HZ

        t0 = out["time"].iloc[0]
        ref = out[out["time"] <= (t0 + REFERENCE_SECONDS)][["x_m","y_m","z_m"]].mean().to_numpy()

        out[["dx_m","dy_m","dz_m"]] = out[["x_m","y_m","z_m"]].to_numpy() - ref
        for c in ["dx_m","dy_m","dz_m"]:
            out[c] = heal_series(out[c], hz)

        win = max(1, int((ROLLING_MEDIAN_MS/1000.0) * hz))
        out[["dx_m","dy_m","dz_m"]] = rolling_median_cols(out, ["dx_m","dy_m","dz_m"], win)

        out["vx_mps"] = out["dx_m"].diff().fillna(0.0) * hz
        out["vy_mps"] = out["dy_m"].diff().fillna(0.0) * hz
        out["vz_mps"] = out["dz_m"].diff().fillna(0.0) * hz

        canon = part.replace(" ", "_")
        csvp = CLEAN / f"clean_{canon}__{slug}_for_lstm.csv"
        out.to_csv(csvp, index=False)
        outs[canon] = csvp

        print(f"  ✓ Cleaned {canon:10s} → {csvp.name} rows={len(out)}")

    return slug, outs

# ---------- models (IMPROVED) ----------
def load_models_or_fail():
    try:
        import torch, torch.nn as nn
    except Exception as e:
        raise SystemExit("PyTorch is required. Install torch.") from e

    models = {}
    for part in ["head","right_hand","left_hand"]:
        pt = CLEAN / f"{part}_lstm.pt"
        if not pt.exists():
            raise SystemExit(f"Missing model: {pt}")

        blob = torch.load(str(pt), map_location="cpu")

        if "state_dict" not in blob:
            raise SystemExit(f"{pt} has no 'state_dict'. Wrong save format.")

        state   = blob["state_dict"]
        in_cols = blob.get("input_cols", ["dx_m","dy_m","dz_m","vx_mps","vy_mps","vz_mps"])
        win_len = int(blob.get("win_len", 20))

        if DEBUG:
            print(f"[DEBUG] Loaded model {part}:")
            print(f"        expects columns: {in_cols}")
            print(f"        window length  : {win_len}")

        import torch.nn as nn
        class Tiny(nn.Module):
            def __init__(self, in_dim=len(in_cols), hid=64, out_dim=3):
                super().__init__()
                self.lstm = nn.LSTM(in_dim, hid, num_layers=1, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hid, hid//2),
                    nn.ReLU(),
                    nn.Linear(hid//2, out_dim)
                )
       
        
            def forward(self, x):
                o,_ = self.lstm(x)
                return self.head(o[:, -1, :])

        m = Tiny()
        m.load_state_dict(state)
        m.eval()

        models[part] = (m, in_cols, win_len)

    return models

def predict_seq(model_tuple, df, part_name="unknown", debug=False):
    import torch
    m, cols, need = model_tuple

    # check columns exist
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"DF for {part_name} missing columns needed by model: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    X = df[cols].to_numpy().astype(np.float32)
    states = np.full(len(df), 1, dtype=np.int64)

    if len(df) < need:
        if debug: print(f"[DEBUG] {part_name}: sequence too short.")
        return states

    for i in range(len(df) - need + 1):
        seg = torch.from_numpy(X[i:i+need][None, :, :])
        with torch.no_grad():
            y = m(seg).argmax(dim=1).item()
        states[i+need-1] = y

    states[:need-1] = states[need-1]

    if debug:
        c = np.bincount(states, minlength=3)
        print(f"[DEBUG] {part_name} state counts: DOWN={c[0]} LEVEL={c[1]} UP={c[2]}")

    return states

# ---------- newest CSV ----------
def find_newest_source_csv() -> Path:
    candidates = []
    for f in glob.glob(str(HERE / "*.csv")):
        name = os.path.basename(f).lower()
        if name.startswith("clean_"):
            continue
        if name.startswith("coach_comments__"):
            continue
        if name == "pro_reference.json":
            continue
        candidates.append((Path(f), Path(f).stat().st_mtime))
    if not candidates:
        raise SystemExit("No raw CSV found.")
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

# ---------- main logic ----------
def main():
    voice_init()

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else find_newest_source_csv()
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    print("Using source CSV:", src)

    print("\nStep A) Cleaning CSV → 3 parts …")
    slug, paths = clean_three_from_csv(src)

    def load_df(p):
        df = pd.read_csv(p)
        req = {"time","dx_m","dy_m","dz_m","vx_mps","vy_mps","vz_mps"}
        if not req.issubset(df.columns):
            raise SystemExit(f"Unexpected columns in {p.name}")
        return df

    dfH = load_df(paths["head"])
    dfL = load_df(paths["left_hand"])
    dfR = load_df(paths["right_hand"])

    n = min(len(dfH), len(dfL), len(dfR))
    dfH, dfL, dfR = dfH.iloc[:n], dfL.iloc[:n], dfR.iloc[:n]

    if DEBUG:
        print("\n[DEBUG] Head cleaned preview:\n", dfH.head())
        print("\n[DEBUG] Left hand cleaned preview:\n", dfL.head())
        print("\n[DEBUG] Right hand cleaned preview:\n", dfR.head())

    print("\nStep B) Loading LSTM models …")
    models = load_models_or_fail()

    print("\nStep C) Predicting states …")
    sH = predict_seq(models["head"],       dfH, part_name="head",       debug=DEBUG)
    sL = predict_seq(models["left_hand"],  dfL, part_name="left_hand",  debug=DEBUG)
    sR = predict_seq(models["right_hand"], dfR, part_name="right_hand", debug=DEBUG)

    # Coach logic
    dwell_frames_needed = max(1, int(DWELL_SECONDS * RESAMPLE_HZ))
    min_gap_frames      = int(COMMENT_MIN_GAP_S * RESAMPLE_HZ)

    last_spoken_frame = -10**9
    last_issue_set = ()
    dwell_count = 0
    clean_streak = 0
    last_second_print = -1

    say("Starting LSTM-only check.")

    state_name = {0:"DOWN", 1:"LEVEL", 2:"UP"}

    for i in range(n):
        t = float(dfH["time"].iloc[i])

        # progress
        if PROGRESS_PER_SECOND:
            cur_sec = int(round(i / RESAMPLE_HZ))
            if cur_sec != last_second_print:
                last_second_print = cur_sec
                print(f"{t:.2f}s  states: H={state_name[sH[i]]}, L={state_name[sL[i]]}, R={state_name[sR[i]]}")
                time.sleep(PROGRESS_DELAY_S)

        fm = []

        # RIGHT hand
        if sR[i] == 0: fm.append("Raise right hand.")
        if sR[i] == 2: fm.append("Lower right hand.")

        # LEFT hand
        if sL[i] == 0: fm.append("Raise left hand.")
        if sL[i] == 2: fm.append("Lower left hand.")

        # HEAD
        if sH[i] == 0: fm.append("Lift head slightly.")
        if sH[i] == 2: fm.append("Relax head slightly down.")

        fm = tuple(sorted(set(fm)))

        if fm == last_issue_set and fm:
            dwell_count += 1
        else:
            last_issue_set = fm
            dwell_count = 1

        if fm:
            clean_streak = 0
            if dwell_count >= dwell_frames_needed and (i - last_spoken_frame) >= min_gap_frames:
                print(f"{t:.2f}s:")
                for m in fm:
                    print("  -", m)
                    say(m)
                    time.sleep(COMMENT_DELAY_S)
                last_spoken_frame = i
        else:
            # clean
            dwell_count = 0
            clean_streak += 1
            secs_clean = clean_streak / RESAMPLE_HZ

            if SAY_OK_ON_CLEAR and last_issue_set and (i - last_spoken_frame) >= min_gap_frames:
                print(f"{t:.2f}s: now it's ok")
                say("Now it’s okay.")
                last_spoken_frame = i
                time.sleep(COMMENT_DELAY_S)

            if abs(secs_clean - SAY_GOOD_AFTER_S) < (1.0/RESAMPLE_HZ) and (i - last_spoken_frame) >= min_gap_frames:
                print(f"{t:.2f}s: good, keep it")
                say("Good, keep it.")
                last_spoken_frame = i
                time.sleep(COMMENT_DELAY_S)

            if abs(secs_clean - SAY_EXCELLENT_AFTER_S) < (1.0/RESAMPLE_HZ) and (i - last_spoken_frame) >= min_gap_frames:
                print(f"{t:.2f}s: excellent posture")
                say("Excellent posture.")
                last_spoken_frame = i
                time.sleep(COMMENT_DELAY_S)

    print("\nDone.")

if __name__ == "__main__":
    main()
