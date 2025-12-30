# 01vis.py — Ballroom LSTM visual dashboard
# ----------------------------------------
# - Finds newest CSV in this folder
# - Cleans head / left hand / right hand
# - Runs trained LSTM models from clean_out\{head,right_hand,left_hand}_lstm.pt
# - Computes scores, progress, mistake segments, comparison to expert limits
# - Outputs:
#     * HTML dashboard with interactive Plotly charts
#     * PNG charts (if kaleido installed)
#     * TXT summary with scores + mistakes
#     * PDF summary report (if fpdf installed)
#
# Run (in venv):
#   python .\ballroom_lstm\01vis.py
#
# Optional: pass a specific CSV:
#   python .\ballroom_lstm\01vis.py ".\dancing today.csv"

import os, sys, glob, re, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
# ==========================================================
# NEW VISUALIZATION HELPERS (ADD THIS BELOW ALL IMPORTS)
# ==========================================================

import plotly.graph_objects as go
import numpy as np

# ----------------------------------------------------------
# Radar Chart (Technique Profile)
# ----------------------------------------------------------
def make_radar_chart(scores):
    labels = ["Head", "Left Arm", "Right Arm", "Vertical Smoothness", "Symmetry"]
    values = [
        scores.get("head", 0),
        scores.get("left_hand", 0),
        scores.get("right_hand", 0),
        max(0, 100 - scores.get("dy_var", 40)),   # derived smoothness
        scores.get("symmetry", 60),               # default symmetry
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        fillcolor='rgba(56,189,248,0.4)',
        line_color='rgba(56,189,248,1)',
        marker_color='white'
    ))

    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )

    return fig

# ----------------------------------------------------------
# Heatmap (Consistency Map)
# ----------------------------------------------------------
def make_heatmap(mistakes, total_segments=20, total_duration=None):
    """
    Builds a heatmap of technique consistency.
    If 'segment' is not in mistake dict, compute position from start time.
    """
    z = np.zeros((1, total_segments))

    # if total duration unknown, estimate from mistakes
    if total_duration is None:
        if len(mistakes) > 0:
            max_time = max(m["end_s"] for m in mistakes)
            total_duration = max_time
        else:
            total_duration = 1  # avoid division by zero

    for m in mistakes:
        start_t = m.get("start_s", 0)

        # Convert time to segment index
        seg = int((start_t / total_duration) * (total_segments - 1))
        seg = max(0, min(total_segments - 1, seg))

        # Color:
        # 1.0 → red (bad)
        # 0.5 → yellow (warning)
        # 0   → blue (good)
        if m["state"] == "DOWN":
            z[0, seg] = 1.0
        else:
            z[0, seg] = 0.5

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=list(range(total_segments)),
        colorscale=[
            [0, "#0ea5e9"],     # blue good
            [0.5, "#facc15"],   # yellow medium
            [1, "#f87171"],     # red bad
        ],
        showscale=False
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        height=120
    )

    return fig


# ----------------------------------------------------------
# Coaching Tips Generator
# ----------------------------------------------------------
def generate_coaching_tips(scores, mistakes):
    tips = []

    if scores["head"] < 50:
        tips.append("Keep your head lifted and stable — avoid sudden tilting.")

    if scores["left_hand"] < 50:
        tips.append("Extend your left arm more to maintain a strong frame.")

    if scores["right_hand"] < 50:
        tips.append("Right arm frame needs more control — avoid dropping the elbow.")

    if len(mistakes) > 0:
        tips.append("Review the red segments in your mistake map and repeat slowly.")

    if scores["overall"] < 40:
        tips.append("Focus on basics: balance, rhythm, and posture alignment.")

    if len(tips) == 0:
        tips.append("Excellent work — your movement is very clean!")

    return tips

HERE  = Path(__file__).parent
CLEAN = HERE / "clean_out"
CLEAN.mkdir(exist_ok=True)

# ============ SETTINGS ============
RESAMPLE_HZ           = 20
REFERENCE_SECONDS     = 2.0
ROLLING_MEDIAN_MS     = 300

MIN_MISTAKE_SEC       = 0.7    # min duration to count a mistake segment
SEGMENTS_FOR_PROGRESS = 3      # split whole dance into N segments
# ==================================


# ---------- CSV parsing (Motive ; + multi-row header) ----------
def parse_motive_semicolon_csv(path: Path):
    """
    Parse a Motive CSV exported with ';' and multi-row header.
    Returns:
      time_vals : np.ndarray [N]
      base_xyz  : dict base_name -> np.ndarray [N,3]
    """
    raw = pd.read_csv(path, sep=';', header=None, dtype=str, engine='python')
    if len(raw) < 7:
        raise SystemExit(f"CSV too short or wrong format: {path}")

    names_row = list(raw.iloc[3])
    data = pd.read_csv(path, sep=';', header=None, skiprows=7, engine='python')

    bases = {}
    for j in range(2, len(names_row), 3):
        base = names_row[j]
        if base is None or (isinstance(base, float) and pd.isna(base)):
            continue
        base = str(base).lower().strip()

        # normalize several naming variants
        base = re.sub(r"\bright\s*(elbow|hand).*", "right hand", base)
        base = re.sub(r"\bleft\s*(elbow|hand).*",  "left hand",  base)
        base = re.sub(r"\bforehead\b",             "head",       base)
        base = re.sub(r"\bhead.*",                 "head",       base)

        bases.setdefault(base, []).append((j, j+1, j+2))

    time_vals = pd.to_numeric(data.iloc[:, 1], errors='coerce').to_numpy()

    base_xyz = {}
    for base, triplets in bases.items():
        xs, ys, zs = [], [], []
        for xc, yc, zc in triplets:
            xs.append(pd.to_numeric(data.iloc[:, xc], errors='coerce').to_numpy())
            ys.append(pd.to_numeric(data.iloc[:, yc], errors='coerce').to_numpy())
            zs.append(pd.to_numeric(data.iloc[:, zc], errors='coerce').to_numpy())
        if not xs:
            continue
        X = np.nanmean(np.stack(xs, axis=1), axis=1)
        Y = np.nanmean(np.stack(ys, axis=1), axis=1)
        Z = np.nanmean(np.stack(zs, axis=1), axis=1)
        base_xyz[base] = np.stack([X, Y, Z], axis=1)

    return time_vals, base_xyz


def group_to_three(keys):
    groups = {"head": [], "left hand": [], "right hand": []}
    for name in keys:
        n = name.lower()
        if "left hand" in n:
            groups["left hand"].append(name)
        elif "right hand" in n:
            groups["right hand"].append(name)
        elif "head" in n:
            groups["head"].append(name)
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
    if not s.isna().any():
        return s
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
    dfs = {}

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
        dfs[canon] = out.reset_index(drop=True)
        print(f"  ✓ Cleaned {canon:10s} rows={len(out)}")

    return slug, dfs


# ---------- models ----------
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


def predict_seq(model_tuple, df):
    import torch
    m, cols, need = model_tuple
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"DF missing columns needed by model: {missing}")

    X = df[cols].to_numpy().astype(np.float32)
    states = np.full(len(df), 1, dtype=np.int64)  # default LEVEL
    if len(df) < need:
        return states

    for i in range(len(df) - need + 1):
        seg = torch.from_numpy(X[i:i+need][None, :, :])
        with torch.no_grad():
            y = m(seg).argmax(dim=1).item()
        states[i+need-1] = y

    states[:need-1] = states[need-1]
    return states


# ---------- find newest CSV ----------
def find_newest_source_csv() -> Path:
    candidates = []
    for f in glob.glob(str(HERE / "*.csv")):
        name = os.path.basename(f).lower()
        if name.startswith("clean_"):
            continue
        if name.startswith("coach_comments__"):
            continue
        candidates.append((Path(f), Path(f).stat().st_mtime))
    if not candidates:
        raise SystemExit("No raw CSV found in this folder.")
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# ---------- expert limits ----------
def load_expert_limits():
    """
    Safely load pro_reference.json ensuring it's always a dictionary.
    Older versions or broken files sometimes store only numbers.
    """
    p = CLEAN / "pro_reference.json"
    if not p.exists():
        print("[WARN] pro_reference.json not found → using defaults.")
        return {
            "dy_head_m": 0.02,
            "dy_left_m": 0.03,
            "dy_right_m": 0.03
        }

    try:
        with open(p, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print("[WARN] pro_reference.json was not a dict → resetting to defaults.")
            return {
                "dy_head_m": 0.02,
                "dy_left_m": 0.03,
                "dy_right_m": 0.03
            }

        out = {}
        out["dy_head_m"]  = float(data.get("dy_head_m", 0.02))
        out["dy_left_m"]  = float(data.get("dy_left_m", 0.03))
        out["dy_right_m"] = float(data.get("dy_right_m", 0.03))
        return out

    except Exception as e:
        print("[WARN] Could not load pro_reference.json → using defaults.", e)
        return {
            "dy_head_m": 0.02,
            "dy_left_m": 0.03,
            "dy_right_m": 0.03
        }


# ---------- scoring & mistakes ----------
def extract_mistake_segments(states, time_array, part_name, min_dur_sec=0.7):
    """
    states: np[int] 0=DOWN,1=LEVEL,2=UP
    time_array: np[float]
    Returns list of dicts: start, end, duration, state, suggestion
    """
    n = len(states)
    segments = []
    i = 0
    while i < n:
        st = states[i]
        if st == 1:
            i += 1
            continue
        j = i + 1
        while j < n and states[j] == st:
            j += 1
        start_t = float(time_array[i])
        end_t   = float(time_array[j-1])
        dur = end_t - start_t
        if dur >= min_dur_sec:
            state_str = "DOWN" if st == 0 else "UP"
            if part_name == "head":
                msg = "Lift head slightly." if st == 0 else "Relax head slightly down."
            else:
                direction = "Raise" if st == 0 else "Lower"
                msg = f"{direction} {part_name.replace('_',' ')}."
            segments.append({
                "part": part_name,
                "start_s": round(start_t,2),
                "end_s": round(end_t,2),
                "duration_s": round(dur,2),
                "state": state_str,
                "suggestion": msg
            })
        i = j
    return segments


def compute_scores(states_dict):
    """
    states_dict: {"head": np[int], "left_hand": np[int], "right_hand": np[int]}
    returns dict with per-part score and overall
    """
    scores = {}
    for k, s in states_dict.items():
        correct = np.mean(s == 1) if len(s) > 0 else 0.0
        scores[k] = round(float(correct * 100.0), 1)
    overall = round(float(np.mean(list(scores.values()))), 1)
    scores["overall"] = overall
    return scores


def split_into_segments(n, k):
    indices = []
    base = n // k
    extra = n % k
    start = 0
    for i in range(k):
        length = base + (1 if i < extra else 0)
        end = start + length
        indices.append((start, end))
        start = end
    return indices


def progress_scores(states_dict, time_array, segments_k=3):
    n = len(time_array)
    segs = split_into_segments(n, segments_k)
    out = []
    for idx, (a,b) in enumerate(segs):
        if a >= b:
            continue
        seg_states = {k: v[a:b] for k,v in states_dict.items()}
        sc = compute_scores(seg_states)
        out.append({
            "segment": idx+1,
            "start_s": round(float(time_array[a]),2),
            "end_s":   round(float(time_array[b-1]),2),
            "score_head": sc["head"],
            "score_left_hand": sc["left_hand"],
            "score_right_hand": sc["right_hand"],
            "score_overall": sc["overall"],
        })
    return out


# ---------- chart builders ----------
def make_fig_states(t, states):
    state_name = {0:"DOWN", 1:"LEVEL", 2:"UP"}
    fig = go.Figure()
    for part, s in states.items():
        fig.add_trace(go.Scatter(
            x=t, y=s,
            mode="lines",
            name=part.replace("_"," "),
            line=dict(shape="hv")
        ))
    fig.update_layout(
        title="LSTM predicted states (0=DOWN, 1=LEVEL, 2=UP)",
        xaxis_title="Time (s)",
        yaxis=dict(
            tickmode="array",
            tickvals=[0,1,2],
            ticktext=["DOWN","LEVEL","UP"]
        ),
        legend=dict(orientation="h")
    )
    return fig


def make_fig_dy(t, dfs, limits):
    fig = go.Figure()
    dy_head = dfs["head"]["dy_m"].to_numpy()
    dy_l    = dfs["left_hand"]["dy_m"].to_numpy()
    dy_r    = dfs["right_hand"]["dy_m"].to_numpy()

    fig.add_trace(go.Scatter(x=t, y=dy_head, name="head dy (m)", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=dy_l,    name="left hand dy (m)", mode="lines"))
    fig.add_trace(go.Scatter(x=t, y=dy_r,    name="right hand dy (m)", mode="lines"))

    fig.add_hrect(y0=-limits["dy_head_m"],  y1=limits["dy_head_m"],
                  fillcolor="lightblue", opacity=0.2, line_width=0,
                  annotation_text="head comfort zone", annotation_position="top left")
    fig.add_hrect(y0=-limits["dy_left_m"],  y1=limits["dy_left_m"],
                  fillcolor="lightgreen", opacity=0.15, line_width=0,
                  annotation_text="left hand zone", annotation_position="bottom left")
    fig.add_hrect(y0=-limits["dy_right_m"], y1=limits["dy_right_m"],
                  fillcolor="lightpink", opacity=0.15, line_width=0,
                  annotation_text="right hand zone", annotation_position="bottom right")

    fig.update_layout(
        title="Vertical offset dy (m) vs expert comfort zones",
        xaxis_title="Time (s)",
        yaxis_title="dy (meters)"
    )
    return fig


def make_fig_progress(segments):
    labels = [f"S{d['segment']} ({d['start_s']}–{d['end_s']}s)" for d in segments]
    vals   = [d["score_overall"] for d in segments]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=vals,
        text=[f"{v}%" for v in vals],
        textposition="auto",
        name="overall score"
    ))
    fig.update_layout(
        title="Progress across time segments",
        xaxis_title="Segment",
        yaxis_title="Score (%)"
    )
    return fig


def make_fig_mistakes(mistakes):
    fig = go.Figure()
    if mistakes:
        colors = {"DOWN":"blue", "UP":"red"}
        xs = [m["start_s"] for m in mistakes]
        ys = [m["part"] for m in mistakes]
        texts = [f"{m['part']} {m['state']}: {m['suggestion']}" for m in mistakes]
        cs = [colors.get(m["state"], "black") for m in mistakes]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=10, color=cs),
            text=texts, hoverinfo="text"
        ))
    fig.update_layout(
        title="Mistakes detected (start time vs body part)",
        xaxis_title="Time (s)",
        yaxis_title="Body part"
    )
    return fig


# ---------- dashboard + exports ----------
def build_html(slug, scores, segments, mistakes, figs):
    # --- small helpers for summary cards ---
    overall = scores.get("overall", 0.0)
    head_sc = scores.get("head", 0.0)
    left_sc = scores.get("left_hand", 0.0)
    right_sc = scores.get("right_hand", 0.0)

    total_mistakes = len(mistakes)
    head_m = sum(1 for m in mistakes if m["part"] == "head")
    left_m = sum(1 for m in mistakes if m["part"] == "left_hand")
    right_m = sum(1 for m in mistakes if m["part"] == "right_hand")

    best_seg = max(segments, key=lambda d: d["score_overall"]) if segments else None
    worst_seg = min(segments, key=lambda d: d["score_overall"]) if segments else None
    avg_seg = round(
        sum(d["score_overall"] for d in segments) / len(segments), 1
    ) if segments else overall

    if overall >= 70:
        label = "Strong overall technique"
        tag_class = ""
        tag_text = "Great foundation"
    elif overall >= 40:
        label = "Mixed – needs refinement"
        tag_class = "warn"
        tag_text = "Work on consistency"
    else:
        label = "Needs focused coaching"
        tag_class = "bad"
        tag_text = "Plan a technique block"

    # --- HTML rows for segments table ---
    seg_rows = ""
    for d in segments:
        seg_rows += (
            f"<tr>"
            f"<td>S{d['segment']}</td>"
            f"<td>{d['start_s']}–{d['end_s']}s</td>"
            f"<td>{d['score_head']}%</td>"
            f"<td>{d['score_left_hand']}%</td>"
            f"<td>{d['score_right_hand']}%</td>"
            f"<td>{d['score_overall']}%</td>"
            f"</tr>"
        )

    # --- HTML rows for mistakes table ---
    mistakes_rows = ""
    for m in mistakes[:40]:  # cap list to keep UI readable
        state_cls = "down" if m["state"] == "DOWN" else ""
        mistakes_rows += (
            f"<tr>"
            f"<td>{m['start_s']}</td>"
            f"<td>{m['end_s']}</td>"
            f"<td>{m['part'].replace('_',' ')}</td>"
            f"<td><span class='chip-state {state_cls}'>{m['state']}</span></td>"
            f"<td>{m['suggestion']}</td>"
            f"</tr>"
        )
    if not mistakes_rows:
        mistakes_rows = "<tr><td colspan='5'>No mistakes above threshold were detected in this recording.</td></tr>"

    # --- Plotly charts as HTML fragments ---
    fig_states_html   = pio.to_html(figs["states"],   full_html=False, include_plotlyjs="cdn")
    fig_progress_html = pio.to_html(figs["progress"], full_html=False, include_plotlyjs=False)
    fig_dy_html       = pio.to_html(figs["dy"],       full_html=False, include_plotlyjs=False)
    fig_mist_html     = pio.to_html(figs["mistakes"], full_html=False, include_plotlyjs=False)

    # --- neon dark CSS (sidebar + cards + animations) ---
    CSS = """
    <style>
    :root {
      --bg: #020617;
      --bg-elevated: #020617;
      --card-bg: #020617;
      --card-soft: #020617;
      --accent: #22c55e;
      --accent-soft: rgba(34,197,94,0.18);
      --accent-strong: #22c55e;
      --accent-glow: 0 0 16px rgba(34,197,94,0.6);
      --text-main: #e5e7eb;
      --text-muted: #9ca3af;
      --danger: #f97373;
      --warn: #fbbf24;
      --border-subtle: rgba(148,163,184,0.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #020617 0, #020617 40%, #000 100%);
      color: var(--text-main);
      -webkit-font-smoothing: antialiased;
    }
    .app-shell {
      display: grid;
      grid-template-columns: 250px minmax(0, 1fr);
      min-height: 100vh;
      background: radial-gradient(circle at top left, rgba(34,197,94,0.18), transparent 52%),
                  radial-gradient(circle at bottom right, rgba(16,185,129,0.12), transparent 55%),
                  #020617;
    }
    @media (max-width: 960px) {
      .app-shell {
        grid-template-columns: minmax(0, 1fr);
      }
      .sidebar {
        display: none;
      }
    }
    .sidebar {
      padding: 22px 18px;
      border-right: 1px solid rgba(15,23,42,0.9);
      background: radial-gradient(circle at top, #020617 0, #020617 55%, #000 100%);
    }
    .sidebar-logo {
      font-size: 18px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-main);
      margin-bottom: 22px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .logo-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: var(--accent-glow);
    }
    .sidebar-section-title {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-muted);
      margin: 16px 0 8px;
    }
    .sidebar-nav {
      list-style: none;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .sidebar-nav li {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 9px;
      border-radius: 10px;
      cursor: default;
      color: var(--text-muted);
      font-size: 13px;
    }
    .sidebar-nav li.active {
      background: rgba(15,23,42,0.95);
      color: var(--accent-strong);
      box-shadow: 0 0 0 1px rgba(34,197,94,0.6), var(--accent-glow);
    }
    .sidebar-pill {
      margin-top: 16px;
      padding: 10px 11px;
      border-radius: 12px;
      border: 1px dashed rgba(148,163,184,0.5);
      font-size: 12px;
      color: var(--text-muted);
      background: rgba(15,23,42,0.9);
    }
    .main {
      padding: 22px 22px 26px;
    }
    .main-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }
    .main-title-block h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 600;
    }
    .main-title-block p {
      margin: 4px 0 0;
      font-size: 13px;
      color: var(--text-muted);
    }
    .tag-live {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      padding: 4px 9px;
      border-radius: 999px;
      border: 1px solid rgba(148,163,184,0.4);
      background: rgba(15,23,42,0.9);
      color: var(--text-muted);
    }
    .tag-live-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #22c55e;
      box-shadow: 0 0 0 4px rgba(34,197,94,0.38);
    }
    .kpi-row {
      display: grid;
      grid-template-columns: 2.1fr 1.4fr;
      gap: 18px;
      margin-bottom: 16px;
    }
    @media (max-width: 1100px) {
      .kpi-row { grid-template-columns: minmax(0, 1fr); }
    }
    .card {
      background: rgba(15,23,42,0.95);
      border-radius: 18px;
      padding: 16px 18px 16px;
      border: 1px solid rgba(15,23,42,0.95);
      box-shadow:
        0 18px 45px rgba(15,23,42,0.9),
        0 0 0 1px rgba(15,23,42,0.7);
      position: relative;
      overflow: hidden;
    }
    .card::before {
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top left, rgba(34,197,94,0.16), transparent 60%);
      opacity: 0.9;
      pointer-events: none;
    }
    .card-inner { position: relative; z-index: 1; }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }
    .card-title {
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    .card-subtitle {
      font-size: 11px;
      color: var(--text-muted);
    }

    .overall-layout {
      display: flex;
      gap: 18px;
      align-items: center;
    }
    @media (max-width: 640px) {
      .overall-layout { flex-direction: column; align-items: flex-start; }
    }

    /* overall score ring – keep your old style but switch to green */
    .score-ring-wrapper {
      display: flex;
      align-items: center;
      gap: 18px;
    }
    .score-ring {
      position: relative;
      width: 150px;
      height: 150px;
      border-radius: 999px;
      background:
        radial-gradient(circle at center, #020617 0, #020617 56%, transparent 57%),
        conic-gradient(var(--accent-strong) 0deg, var(--accent-strong) calc(var(--p)*3.6deg), rgba(15,23,42,0.6) 0deg);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow:
        0 0 0 6px rgba(15,23,42,0.98),
        0 26px 60px rgba(15,23,42,0.95),
        var(--accent-glow);
      transition: background 1s ease-out;
    }
    .score-ring-inner {
      position: relative;
      width: 96px;
      height: 96px;
      border-radius: 999px;
      background: radial-gradient(circle at top, #020617 0, #020617 55%, #000 100%);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }
    .score-main {
      font-size: 32px;
      font-weight: 700;
    }
    .score-unit {
      font-size: 14px;
      color: var(--text-muted);
      margin-top: -4px;
    }
    .score-label {
      margin-top: 5px;
      font-size: 10px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    .score-tag {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(34,197,94,0.12);
      color: #bbf7d0;
      border: 1px solid rgba(22,163,74,0.5);
      font-size: 10px;
      margin-top: 6px;
    }
    .score-tag.warn {
      background: rgba(251,191,36,0.10);
      color: #facc15;
      border-color: rgba(234,179,8,0.45);
    }
    .score-tag.bad {
      background: rgba(248,113,113,0.10);
      color: #fecaca;
      border-color: rgba(248,113,113,0.6);
    }

    .score-detail {
      flex: 1;
      font-size: 13px;
      color: var(--text-muted);
      line-height: 1.5;
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
      font-size: 11px;
    }
    .pill-row span {
      padding: 3px 8px;
      border-radius: 999px;
      background: rgba(15,23,42,0.95);
      border: 1px solid rgba(148,163,184,0.4);
      color: var(--text-muted);
    }

    .score-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 14px;
    }
    .score-mini {
      background: rgba(15,23,42,0.9);
      border-radius: 12px;
      padding: 9px 11px 10px;
      border: 1px solid rgba(30,64,175,0.7);
      position: relative;
      overflow: hidden;
    }
    .score-mini::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(34,197,94,0.12), transparent 55%);
      opacity: 0.9;
      pointer-events: none;
    }
    .score-mini-label {
      font-size: 11px;
      color: var(--text-muted);
    }
    .score-mini-value {
      font-size: 18px;
      font-weight: 600;
    }
    .score-mini-bar {
      margin-top: 3px;
      width: 100%;
      height: 6px;
      border-radius: 999px;
      background: rgba(15,23,42,1);
      overflow: hidden;
    }
    .score-mini-bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), #4ade80);
      width: 0;
      transition: width 0.9s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* small KPI cards on the right */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0,1fr));
      gap: 10px;
      margin-top: 4px;
    }
    .kpi {
      background: rgba(15,23,42,0.9);
      border-radius: 12px;
      padding: 9px 10px;
      border: 1px solid rgba(30,64,175,0.75);
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .kpi-label {
      color: var(--text-muted);
      font-size: 11px;
    }
    .kpi-number {
      font-size: 18px;
      font-weight: 600;
    }
    .kpi-note {
      font-size: 11px;
      color: var(--text-muted);
    }

    /* lower layout rows */
    .charts-row,
    .bottom-row {
      display: grid;
      grid-template-columns: 1.4fr 1.4fr;
      gap: 18px;
      margin-bottom: 16px;
    }
    @media (max-width: 1100px) {
      .charts-row,
      .bottom-row {
        grid-template-columns: minmax(0, 1fr);
      }
    }

    .panel-title {
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin: 0 0 8px;
    }
    .panel {
      margin-top: 6px;
      padding: 8px 0 0;
      border-radius: 12px;
    }

    .panel table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 2px;
    }
    .panel th,
    .panel td {
      padding: 6px 6px;
      text-align: left;
      border-bottom: 1px solid rgba(30,64,175,0.6);
    }
    .panel th {
      font-size: 11px;
      color: var(--text-muted);
    }
    .panel tr:last-child td {
      border-bottom: none;
    }

    .chip-state {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 500;
      background: rgba(34,197,94,0.14);
      color: var(--accent-strong);
      border: 1px solid rgba(34,197,94,0.5);
    }
    .chip-state.down {
      background: rgba(248,113,113,0.12);
      color: #fecaca;
      border-color: rgba(248,113,113,0.65);
    }
    .footer-note {
      margin-top: 10px;
      font-size: 11px;
      color: var(--text-muted);
      opacity: 0.85;
    }

    /* chart containers */
    .chart-wrapper {
      width: 100%;
      min-height: 240px;
    }
    .chart-wrapper > div.plotly-graph-div {
      min-height: 240px !important;
    }

    /* animate counters */
    .count-up {
      opacity: 0.96;
    }
    </style>
    """

    # --- JS for bar animation + count-up numbers ---
    JS = """
    <script>
    window.addEventListener('load', function() {
      // animate mini bars
      document.querySelectorAll('.score-mini-bar-fill').forEach(function(el) {
        var target = el.getAttribute('data-target-width');
        el.style.width = '0';
        setTimeout(function() {
          el.style.width = target;
        }, 180);
      });

      // animate numeric counters
      document.querySelectorAll('[data-count]').forEach(function(el) {
        var target = parseFloat(el.getAttribute('data-count') || '0');
        var decimals = el.hasAttribute('data-decimals') ? 1 : 0;
        var current = 0;
        var frames = 70;
        var step = target / frames;
        function tick() {
          current += step;
          if (current >= target) { current = target; }
          el.textContent = decimals ? current.toFixed(1) : Math.round(current);
          if (current < target) {
            requestAnimationFrame(tick);
          }
        }
        requestAnimationFrame(tick);
      });
    });
    </script>
    """

    # --- full HTML ---
    full_html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Ballroom LSTM Dashboard – {slug}</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        {CSS}
      </head>
      <body>
        <div class="app-shell">
          <aside class="sidebar">
            <div class="sidebar-logo">
              <span class="logo-dot"></span>
              <span>Dance Coach</span>
            </div>
            <div class="sidebar-section-title">Session</div>
            <ul class="sidebar-nav">
              <li class="active">Overview</li>
              <li>States &amp; timing</li>
              <li>Vertical motion</li>
              <li>Coaching notes</li>
            </ul>
            <div class="sidebar-section-title">Summary</div>
            <ul class="sidebar-nav">
              <li>Best segment: {best_seg['score_overall']}% S{best_seg['segment'] if best_seg else '-'} </li>
              <li>Average: {avg_seg}%</li>
              <li>Total mistakes: {total_mistakes}</li>
            </ul>
            <div class="sidebar-pill">
              This dashboard compares your motion to a professional reference.
              Use it after each practice to track your technical progress.
            </div>
          </aside>

          <main class="main">
            <div class="main-header">
              <div class="main-title-block">
                <h1>Overview</h1>
                <p>Session <strong>{slug}</strong> · LSTM motion analysis</p>
              </div>
              <div class="tag-live">
                <span class="tag-live-dot"></span>
                Last run just now
              </div>
            </div>

            <!-- TOP ROW: Overall technique + KPI mini cards -->
            <section class="kpi-row">
              <!-- Overall card (your original style, adapted) -->
              <div class="card">
                <div class="card-inner">
                  <div class="card-header">
                    <div>
                      <div class="card-title">Overall technique</div>
                      <div class="card-subtitle">{label}</div>
                    </div>
                  </div>

                  <div class="overall-layout">
                    <div class="score-ring-wrapper">
                      <div class="score-ring" style="--p:{overall}">
                        <div class="score-ring-inner">
                          <div class="score-main count-up" data-count="{overall:.1f}" data-decimals="1">0</div>
                          <div class="score-unit>/ 100</div>
                          <div class="score-label">overall score</div>
                          <div class="score-tag {tag_class}">{tag_text}</div>
                        </div>
                      </div>
                    </div>
                    <div class="score-detail">
                      <p>
                        This score summarises how stable and aligned your
                        <strong>head</strong>, <strong>left hand</strong> and
                        <strong>right hand</strong> posture were compared to the
                        reference professional pattern.
                      </p>
                      <p>
                        Use this as a quick “at a glance” indicator: aim for scores
                        above <strong>70</strong> for exam or competition-ready technique.
                      </p>
                      <div class="pill-row">
                        <span>Head: {head_sc:.1f}%</span>
                        <span>Left hand: {left_sc:.1f}%</span>
                        <span>Right hand: {right_sc:.1f}%</span>
                      </div>
                    </div>
                  </div>

                  <div class="score-grid">
                    <div class="score-mini">
                      <div class="score-mini-label">Head line</div>
                      <div class="score-mini-value count-up" data-count="{head_sc:.1f}" data-decimals="1">0</div>
                      <div class="score-mini-bar">
                        <div class="score-mini-bar-fill" data-target-width="{head_sc}%"></div>
                      </div>
                    </div>
                    <div class="score-mini">
                      <div class="score-mini-label">Left arm &amp; frame</div>
                      <div class="score-mini-value count-up" data-count="{left_sc:.1f}" data-decimals="1">0</div>
                      <div class="score-mini-bar">
                        <div class="score-mini-bar-fill" data-target-width="{left_sc}%"></div>
                      </div>
                    </div>
                    <div class="score-mini">
                      <div class="score-mini-label">Right arm &amp; frame</div>
                      <div class="score-mini-value count-up" data-count="{right_sc:.1f}" data-decimals="1">0</div>
                      <div class="score-mini-bar">
                        <div class="score-mini-bar-fill" data-target-width="{right_sc}%"></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- KPI side card like in second screenshot -->
              <div class="card">
                <div class="card-inner">
                  <div class="card-header">
                    <div>
                      <div class="card-title">Session snapshot</div>
                      <div class="card-subtitle">Quick stats based on this recording</div>
                    </div>
                  </div>
                  <div class="kpi-grid">
                    <div class="kpi">
                      <div class="kpi-label">Average segment score</div>
                      <div class="kpi-number count-up" data-count="{avg_seg}" data-decimals="1">0</div>
                      <div class="kpi-note">Across all time segments.</div>
                    </div>
                    <div class="kpi">
                      <div class="kpi-label">Best segment</div>
                      <div class="kpi-number">{best_seg['score_overall'] if best_seg else '-'}%</div>
                      <div class="kpi-note">Segment S{best_seg['segment'] if best_seg else '-'} performed best.</div>
                    </div>
                    <div class="kpi">
                      <div class="kpi-label">Total mistakes</div>
                      <div class="kpi-number count-up" data-count="{total_mistakes}">0</div>
                      <div class="kpi-note">Head: {head_m}, Left: {left_m}, Right: {right_m}</div>
                    </div>
                    <div class="kpi">
                      <div class="kpi-label">Focus recommendation</div>
                      <div class="kpi-note">
                        Use the table and charts below to pick one body part and time window to improve on your next run.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <!-- MIDDLE ROW: Progress + States charts -->
            <section class="charts-row">
              <div class="card">
                <div class="card-inner">
                  <h3 class="panel-title">Progress across time segments</h3>
                  <div class="panel">
                    <div class="chart-wrapper">
                      {fig_progress_html}
                    </div>
                    <table>
                      <thead>
                        <tr>
                          <th>Seg</th><th>Time</th>
                          <th>Head</th><th>Left</th><th>Right</th><th>Overall</th>
                        </tr>
                      </thead>
                      <tbody>
                        {seg_rows}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <div class="card">
                <div class="card-inner">
                  <h3 class="panel-title">LSTM states over time</h3>
                  <div class="panel">
                    <div class="chart-wrapper">
                      {fig_states_html}
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <!-- LOWER ROW: Vertical motion + mistakes -->
            <section class="bottom-row">
              <div class="card">
                <div class="card-inner">
                  <h3 class="panel-title">Vertical motion vs expert comfort zones</h3>
                  <div class="panel">
                    <div class="chart-wrapper">
                      {fig_dy_html}
                    </div>
                  </div>
                </div>
              </div>

              <div class="card">
                <div class="card-inner">
                  <h3 class="panel-title">Mistake map & coaching</h3>
                  <div class="panel">
                    <div class="chart-wrapper" style="min-height: 200px;">
                      {fig_mist_html}
                    </div>
                  </div>
                  <div class="panel">
                    <table>
                      <thead>
                        <tr>
                          <th>From (s)</th>
                          <th>To (s)</th>
                          <th>Body part</th>
                          <th>State</th>
                          <th>Coaching tip</th>
                        </tr>
                      </thead>
                      <tbody>
                        {mistakes_rows}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </section>

            <p class="footer-note">
              This dashboard is designed to feel like a performance analytics panel.
              Use it after each practice session to choose one <strong>body part</strong> and one
              <strong>time segment</strong> to refine on your next attempt.
            </p>
          </main>
        </div>

        {JS}
      </body>
    </html>
    """

    out_html = CLEAN / f"lstm_dashboard_{slug}.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(full_html)
    return out_html



def export_pngs(slug, figs):
    paths = {}
    try:
        for key, fig in figs.items():
            p = CLEAN / f"{slug}_{key}.png"
            fig.write_image(str(p))
            paths[key] = p
        return paths
    except Exception as e:
        print("[WARN] Could not export PNGs (install kaleido).", e)
        return {}


def export_txt(slug, scores, segments, mistakes):
    p = CLEAN / f"summary_{slug}.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"Ballroom LSTM Summary – {slug}\n")
        f.write("="*40 + "\n\n")
        f.write("Scores:\n")
        f.write(f"  Head       : {scores['head']}%\n")
        f.write(f"  Left hand  : {scores['left_hand']}%\n")
        f.write(f"  Right hand : {scores['right_hand']}%\n")
        f.write(f"  OVERALL    : {scores['overall']}%\n\n")

        f.write("Progress (segments):\n")
        for d in segments:
            f.write(
                f"  Segment {d['segment']} ({d['start_s']}–{d['end_s']}s): "
                f"head={d['score_head']}%, left={d['score_left_hand']}%, right={d['score_right_hand']}%, "
                f"overall={d['score_overall']}%\n"
            )
        f.write("\nMistakes:\n")
        if not mistakes:
            f.write("  None above threshold.\n")
        else:
            for m in mistakes:
                f.write(
                    f"  {m['start_s']}–{m['end_s']}s  {m['part']} {m['state']}: {m['suggestion']} "
                    f"(duration={m['duration_s']}s)\n"
                )
    return p


def export_pdf(slug, scores, segments, mistakes, png_paths):
    try:
        from fpdf import FPDF
    except Exception as e:
        print("[WARN] fpdf not installed → skipping PDF.", e)
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Ballroom LSTM Report – {slug}", ln=1)

    pdf.set_font("Arial", size=12)
    pdf.ln(4)
    pdf.cell(0, 8, "Scores:", ln=1)
    pdf.cell(0, 7, f"  Head: {scores['head']}%", ln=1)
    pdf.cell(0, 7, f"  Left hand: {scores['left_hand']}%", ln=1)
    pdf.cell(0, 7, f"  Right hand: {scores['right_hand']}%", ln=1)
    pdf.cell(0, 7, f"  OVERALL: {scores['overall']}%", ln=1)

    pdf.ln(5)
    pdf.cell(0, 8, "Progress segments:", ln=1)
    for d in segments:
        pdf.cell(
            0, 7,
            f"  S{d['segment']} ({d['start_s']}–{d['end_s']}s): "
            f"head={d['score_head']}%, left={d['score_left_hand']}%, right={d['score_right_hand']}%, "
            f"overall={d['score_overall']}%",
            ln=1
        )

    pdf.ln(5)
    pdf.cell(0, 8, "Mistakes (summary):", ln=1)
    if not mistakes:
        pdf.cell(0, 7, "  None above threshold.", ln=1)
    else:
        max_list = min(10, len(mistakes))
        for m in mistakes[:max_list]:
            pdf.multi_cell(
                0, 7,
                f"  {m['start_s']}–{m['end_s']}s  {m['part']} {m['state']}: {m['suggestion']} "
                f"(dur={m['duration_s']}s)"
            )
    pdf.ln(5)

    # Add one or two images if we have them
    for key in ["states", "dy", "progress"]:
        if key in png_paths:
            try:
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, f"Figure: {key}", ln=1)
                pdf.image(str(png_paths[key]), w=180)
            except Exception as e:
                print(f"[WARN] Could not embed PNG '{key}' in PDF:", e)

    out_pdf = CLEAN / f"report_{slug}.pdf"
    pdf.output(str(out_pdf))
    return out_pdf


# ---------- main ----------
def main():
    # 1) pick CSV
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        src = find_newest_source_csv()
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    print("Using source CSV:", src)

    # 2) cleaning
    print("\nStep A) Cleaning CSV → head / left hand / right hand …")
    slug, dfs = clean_three_from_csv(src)

    # 3) load models
    print("\nStep B) Loading LSTM models …")
    models = load_models_or_fail()

    # 4) predict states
    print("\nStep C) Predicting states …")
    dfH = dfs["head"]
    dfL = dfs["left_hand"]
    dfR = dfs["right_hand"]

    n = min(len(dfH), len(dfL), len(dfR))
    dfH, dfL, dfR = dfH.iloc[:n], dfL.iloc[:n], dfR.iloc[:n]
    t = dfH["time"].to_numpy()

    sH = predict_seq(models["head"],       dfH)
    sL = predict_seq(models["left_hand"],  dfL)
    sR = predict_seq(models["right_hand"], dfR)

    states = {
        "head": sH,
        "left_hand": sL,
        "right_hand": sR
    }

    # 5) scoring & mistakes
    print("\nStep D) Computing scores & mistakes …")
    scores = compute_scores(states)
    print("Scores:", scores)

    mistakes = []
    for part, s in states.items():
        mistakes.extend(extract_mistake_segments(
            s, t, part, min_dur_sec=MIN_MISTAKE_SEC
        ))

    segments = progress_scores(states, t, segments_k=SEGMENTS_FOR_PROGRESS)
    expert_limits = load_expert_limits()

    # 6) figures
    print("\nStep E) Building charts and dashboard …")
    figs = {}
    figs["states"]   = make_fig_states(t, states)
    figs["dy"]       = make_fig_dy(t, dfs, expert_limits)
    figs["progress"] = make_fig_progress(segments)
    figs["mistakes"] = make_fig_mistakes(mistakes)

    # 👉 Build HTML output
    html_path = build_html(slug, scores, segments, mistakes, figs)

    # 👉 Export extras
    png_paths = export_pngs(slug, figs)
    txt_path  = export_txt(slug, scores, segments, mistakes)
    pdf_path  = export_pdf(slug, scores, segments, mistakes, png_paths)

    print("\nOutputs:")
    print("  HTML dashboard:", html_path.resolve())
    print("  TXT summary   :", txt_path.resolve())
    if png_paths:
        print("  PNG charts   :", ", ".join(str(p) for p in png_paths.values()))
    if pdf_path:
        print("  PDF report    :", pdf_path.resolve())
    print("\nOpen the HTML in your browser for interactive visualization.")

    # 👉 VERY IMPORTANT: return HTML path for auto-open
    return html_path


# ==========================================================
# AUTO OPEN DASHBOARD
# ==========================================================
if __name__ == "__main__":
    out = main()      # now returns the correct file path
    import webbrowser
    # Force string path and open in default browser
    webbrowser.open(str(out.resolve()))
