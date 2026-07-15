from tensorflow.keras.models import load_model

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import sys
import csv
import json
import urllib.request
import datetime
from collections import deque

# ── Optional: TTS ─────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty('rate', 160)
    _tts_engine.setProperty('volume', 0.9)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  §1  MODEL AUTO-DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

def ensure_models():
    for fname, url in MODEL_URLS.items():
        if not os.path.exists(fname):
            print(f"[DOWNLOAD] Fetching {fname} …")
            try:
                urllib.request.urlretrieve(url, fname)
                print(f"[OK] {fname} saved.")
            except Exception as e:
                print(f"[ERROR] Could not download {fname}: {e}")
                sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  §2  SYSTEM CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# --- Capture ---
CAP_W, CAP_H = 640, 480
CAP_FPS      = 30
POSE_SKIP    = 3          # run heavy pose model every N frames

# --- Smoothing ---
EMA_ALPHA    = 0.40       # exponential moving average weight (lower = smoother)
SMOOTH_WIN   = 12         # secondary rolling window size

# --- Risk Feature Weights (sum = 100) ---
# Derived from paper's paresis prevalence statistics:
#   Face paresis: 54.6% → split over 4 facial features
#   Arm paresis:  75.5% → allocated 35%
W = {
    "mouth": 12,
    "eye": 6,
    "tilt": 6,
    "brow": 4,
    "cheek": 4,
    "rigid": 3,
    "arm": 30,
}
# --- Feature Thresholds ---
TH = {
    "mouth":  0.030,
    "eye":    0.012,
    "tilt":   10.0,
    "brow":   0.022,
    "cheek":  0.015,
    "rigid":  0.008,
    "arm":    0.12,
}

# --- Risk Levels ---
RISK_LOW  = 30
RISK_MED  = 55
RISK_HIGH = 70

# --- Sustained alert: risk must be >= HIGH for this many seconds ---
ALERT_SUSTAIN_SEC = 1.5

# --- Session logging ---
LOG_DIR = "stroke_logs"

# ══════════════════════════════════════════════════════════════════════════════
#  §3  MEDIAPIPE LANDMARK INDICES
# ══════════════════════════════════════════════════════════════════════════════

# Face
MOUTH_L, MOUTH_R   = 61,  291
MOUTH_T, MOUTH_B   = 13,  14
EYE_L_T, EYE_L_B   = 159, 145
EYE_R_T, EYE_R_B   = 386, 374
BROW_L,  BROW_R    = 107, 336
NOSE_TIP, CHIN     = 1,   152
L_TEMPLE, R_TEMPLE = 234, 454
CHEEK_L,  CHEEK_R  = 50,  280    # nasolabial fold reference landmarks
NASAL_L,  NASAL_R  = 64,  294    # nose wing — cheek fold line endpoints

# Pose (shoulders=11/12, elbows=13/14, wrists=15/16)
SH_L, SH_R   = 11, 12
EL_L, EL_R   = 13, 14
WR_L, WR_R   = 15, 16

# ══════════════════════════════════════════════════════════════════════════════
#  §4  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_KEYS = list(W.keys())
class DualSmooth:
    """
    Two-pass smoother: EMA first, then rolling mean.
    Chosen over Kalman for lower CPU cost; suitable for 30fps medical overlay.
    """
    def __init__(self, keys, win=SMOOTH_WIN, alpha=EMA_ALPHA):
        self._ema  = {k: 0.0 for k in keys}
        self._roll = {k: deque([0.0]*win, maxlen=win) for k in keys}
        self._alpha = alpha

    def update(self, key, val):
        # Pass 1: EMA
        self._ema[key] = self._alpha * val + (1 - self._alpha) * self._ema[key]
        # Pass 2: Rolling mean of EMA
        self._roll[key].append(self._ema[key])

    def get(self, key):
        return sum(self._roll[key]) / len(self._roll[key])

    def get_all(self):
        return {k: self.get(k) for k in self._ema}


class SessionLogger:
    """
    Writes per-frame scores to CSV for post-session analysis.
    The CSV can be opened in Excel / pandas for academic evaluation.
    """
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(LOG_DIR, f"session_{ts}.csv")
        self._file = open(self.path, "w", newline="")
        cols = ["timestamp", "frame", "risk"] + FEATURE_KEYS + \
               ["alert_active", "face_detected", "pose_detected", "fps"]
        self._writer = csv.DictWriter(self._file, fieldnames=cols)
        self._writer.writeheader()
        self._file.flush()
        print(f"[LOG] Session log: {self.path}")

    def write(self, row: dict):
        self._writer.writerow(row)

    def flush(self):
        self._file.flush()

    def close(self):
        self._file.close()
        print(f"[LOG] Session saved → {self.path}")


class AlertTracker:
    """
    Requires risk to stay >= HIGH for ALERT_SUSTAIN_SEC before triggering alert.
    Prevents single-frame spikes from creating false alarms.
    Counts total alerts in session.
    """
    def __init__(self):
        self._high_since = None
        self.alert_active = False
        self.alert_count  = 0
        self._cooldown_until = 0.0
        self.COOLDOWN_SEC = 15.0

    def update(self, risk: int) -> bool:
        now = time.time()
        if risk >= RISK_HIGH:
            if self._high_since is None:
                self._high_since = now
            elapsed = now - self._high_since
            if elapsed >= ALERT_SUSTAIN_SEC and now > self._cooldown_until:
                if not self.alert_active:
                    self.alert_active  = True
                    self.alert_count  += 1
                    self._cooldown_until = now + self.COOLDOWN_SEC
                    return True   # NEW alert fired
        else:
            self._high_since  = None
            self.alert_active = False
        return False

    def sustain_progress(self) -> float:
        """0.0–1.0 progress toward alert trigger."""
        if self._high_since is None:
            return 0.0
        return min((time.time() - self._high_since) / ALERT_SUSTAIN_SEC, 1.0)


class TrendBuffer:
    """Stores last N risk values for live sparkline graph."""
    def __init__(self, n=300):
        self._buf = deque([0]*n, maxlen=n)
        self.n = n

    def push(self, v):
        self._buf.append(int(v))

    def as_array(self):
        return list(self._buf)


class Calibrator:
    """
    Per-user baseline calibration.
    User holds neutral face for ~2s; mean of each feature stored.
    All subsequent raw values have the baseline subtracted (floored at 0).
    Eliminates natural inter-person asymmetry (not addressed in original paper).
    """
    CAP_FRAMES = 150

    def __init__(self):
        self.baseline  = None
        self._samples  = {k: [] for k in FEATURE_KEYS}
        self._active   = False
        self._count    = 0

    def start(self):
        self._samples = {k: [] for k in FEATURE_KEYS}
        self._active  = True
        self._count   = 0
        self.baseline = None
        print("[CALIBRATE] Hold neutral expression …")

    def feed(self, raw: dict):
        if not self._active:
            return
        for k in FEATURE_KEYS:
            self._samples[k].append(raw.get(k, 0.0))
        self._count += 1
        if self._count >= self.CAP_FRAMES:
            self.baseline = {k: float(np.mean(v)) for k, v in self._samples.items()}
            self._active  = False
            print(f"[CALIBRATE] Baseline set: { {k:f'{v:.4f}' for k,v in self.baseline.items()} }")

    def apply(self, raw: dict) -> dict:
        if self.baseline is None:
            return dict(raw)
        return {k: max(float(raw.get(k,0)) - self.baseline.get(k,0), 0.0)
                for k in FEATURE_KEYS}

    @property
    def progress(self):
        return self._count / self.CAP_FRAMES if self._active else 0.0

    @property
    def active(self):
        return self._active

# ══════════════════════════════════════════════════════════════════════════════
#  §5  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def lm_xy(lm) -> np.ndarray:
    return np.array([lm.x, lm.y], dtype=np.float32)

def dist(a, b) -> float:
    d = lm_xy(a) - lm_xy(b)
    return float(np.hypot(d[0], d[1]))

def face_h(lm) -> float:
    return max(dist(lm[NOSE_TIP], lm[CHIN]) * 2.0, 1e-5)

def face_w(lm) -> float:
    return max(dist(lm[L_TEMPLE], lm[R_TEMPLE]), 1e-5)

def px(lm, W, H):
    """Normalised → pixel coords."""
    return int(lm.x * W), int(lm.y * H)

# ══════════════════════════════════════════════════════════════════════════════
#  §6  FEATURE EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def feat_mouth(lm) -> float:
    fh = face_h(lm)
    val = abs(lm[MOUTH_L].y - lm[MOUTH_R].y) / fh

    if val < 0.015:
        return 0.0

    return val

def feat_eye(lm) -> float:
    fh = face_h(lm)
    l = dist(lm[EYE_L_T], lm[EYE_L_B]) / fh
    r = dist(lm[EYE_R_T], lm[EYE_R_B]) / fh

    val = abs(l - r)

    if val < 0.012:
        return 0.0

    return val

def feat_tilt(lm) -> float:
    """
    Nose-to-chin axis deviation from vertical (degrees).
    Paper ref: 'face get tilted towards the either left or the right side' — p.934.
    """
    n = lm_xy(lm[NOSE_TIP])
    c = lm_xy(lm[CHIN])
    d = c - n
    return float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-6)))

def feat_brow(lm) -> float:
    fh = face_h(lm)
    val = abs(lm[BROW_L].y - lm[BROW_R].y) / fh

    if val < 0.012:
        return 0.0

    return val

def feat_cheek(lm) -> float:
    fh = face_h(lm)

    l_drop = abs(lm[CHEEK_L].y - lm[MOUTH_L].y) / fh
    r_drop = abs(lm[CHEEK_R].y - lm[MOUTH_R].y) / fh

    val = abs(l_drop - r_drop)

    if val < 0.010:
        return 0.0

    return val

def feat_rigid(lm, prev_lm) -> float:
    """
    Expression rigidity: measures how much key landmarks moved since last frame.
    Paper ref: 'inability of the patient to change its facial expressions' — p.934.
    A rigid face (very low movement) combined with other flags = elevated risk.
    NOTE: Inverted — HIGH rigidity (LOW movement) contributes to risk.
    """
    if prev_lm is None:
        return 0.0
    fh = face_h(lm)
    pts = [MOUTH_L, MOUTH_R, BROW_L, BROW_R, EYE_L_T, EYE_R_T]
    total_mv = sum(
        np.hypot(lm[i].x - prev_lm[i].x, lm[i].y - prev_lm[i].y)
        for i in pts
    ) / (len(pts) * fh)
    # Inverted: near-zero movement = high rigidity score
    rigidity = max(0.03 - total_mv, 0.0)
    return rigidity

def feat_arm(pose_lm) -> float:
    """
    Bilateral arm weakness score combining:
      (a) Arm elevation asymmetry (unilateral weakness causes arm to drift down)
      (b) Angular asymmetry of shoulder-wrist vectors
    Paper ref: arm paresis 75.5% of confirmed stroke patients — Rathore et al.
    cited in paper as [7], p.933.
    """
    if pose_lm is None:
        return 0.0

    Ls = lm_xy(pose_lm[SH_L]);  Rs = lm_xy(pose_lm[SH_R])
    Lw = lm_xy(pose_lm[WR_L]);  Rw = lm_xy(pose_lm[WR_R])

    span = max(float(np.linalg.norm(Rs - Ls)), 1e-5)

    # Wrist elevation relative to shoulder (positive = wrist above shoulder)
    L_lift = float((Ls[1] - Lw[1]) / span)
    R_lift = float((Rs[1] - Rw[1]) / span)

    # Arm angle (shoulder → wrist, from horizontal)
    def ang(sh, wr):
        d = wr - sh
        return float(np.degrees(np.arctan2(-d[1], d[0] + 1e-6)))

    ang_asym = abs(ang(Ls, Lw) - ang(Rs, Rw)) / 180.0

    combined_lift = (L_lift + R_lift) / 2.0
    drift_penalty = max(0.0, TH["arm"] - combined_lift)

    return ang_asym + drift_penalty

# ══════════════════════════════════════════════════════════════════════════════
#  §7  RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════

def score_feature(val, threshold, saturation=None) -> float:
    """
    Piecewise-linear mapping: 0 at/below threshold → 1.0 at saturation.
    Saturation defaults to 2.5× threshold.
    """
    if saturation is None:
        saturation = threshold * 2.5
    if val <= threshold:
        return 0.0
    if val >= saturation:
        return 1.0
    return (val - threshold) / (saturation - threshold)

def compute_risk(smoothed: dict) -> tuple:
    s = {k: score_feature(smoothed[k], TH[k]) for k in FEATURE_KEYS}
    risk = sum(s[k] * W[k] for k in FEATURE_KEYS)
    return min(int(risk), 100), s

# ══════════════════════════════════════════════════════════════════════════════
#  §8  RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

def risk_color(r):
    if r < RISK_LOW:  return (70, 210, 90)
    if r < RISK_MED:  return (40, 185, 255)
    if r < RISK_HIGH: return (30, 120, 255)
    return (50, 50, 230)

def risk_label(r):
    if r < RISK_LOW:  return "NORMAL",   (70, 210, 90)
    if r < RISK_MED:  return "MONITOR",  (40, 185, 255)
    if r < RISK_HIGH: return "ELEVATED", (30, 120, 255)
    return "HIGH RISK", (50, 50, 230)

def draw_horizontal_bar(img, x, y, w, h, fill, bg_col, fill_col, border=(100,100,100)):
    cv2.rectangle(img, (x,y), (x+w, y+h), bg_col, -1)
    fw = int(w * max(0, min(fill, 1.0)))
    if fw > 0:
        cv2.rectangle(img, (x,y), (x+fw, y+h), fill_col, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), border, 1)

def draw_gauge(img, cx, cy, r, risk):
    """Circular arc risk gauge."""
    rc = risk_color(risk)
    cv2.circle(img, (cx, cy), r+5, (35,35,35), -1)
    cv2.circle(img, (cx, cy), r+5, (80,80,80), 1)
    for i in range(360):
        a = np.radians(i - 90)
        col = risk_color(min(int(i/3.6), 100)) if i <= int(3.6*risk) else (45,45,45)
        x1 = int(cx + (r-9)*np.cos(a));  y1 = int(cy + (r-9)*np.sin(a))
        x2 = int(cx + r    *np.cos(a));  y2 = int(cy + r    *np.sin(a))
        cv2.line(img, (x1,y1), (x2,y2), col, 2)
    cv2.circle(img, (cx,cy), r-11, (22,22,22), -1)
    val_s = str(risk)
    tw = cv2.getTextSize(val_s, FONT, 0.95, 2)[0][0]
    cv2.putText(img, val_s, (cx - tw//2, cy+7), FONT, 0.95, rc, 2, cv2.LINE_AA)
    cv2.putText(img, "RISK %", (cx-24, cy+22), FONT, 0.38, (160,160,160), 1, cv2.LINE_AA)

def draw_sparkline(img, data, x, y, w, h, max_val=100):
    """Live trend graph of last N risk values."""
    n = len(data)
    if n < 2:
        return
    # Background
    cv2.rectangle(img, (x,y), (x+w, y+h), (25,25,25), -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (70,70,70), 1)
    # Reference lines at 30, 55, 75
    for level, col in [(RISK_LOW,(50,80,50)),(RISK_MED,(50,70,100)),(RISK_HIGH,(80,50,50))]:
        ry = y + h - int(h * level / max_val)
        cv2.line(img, (x, ry), (x+w, ry), col, 1)
    # Sparkline
    pts = []
    for i, v in enumerate(data):
        px_x = x + int(i * w / n)
        px_y = y + h - int(h * min(v, max_val) / max_val)
        pts.append((px_x, px_y))
    for i in range(1, len(pts)):
        col = risk_color(data[i])
        cv2.line(img, pts[i-1], pts[i], col, 1, cv2.LINE_AA)
    # Label
    cv2.putText(img, "RISK TREND (last 10s)", (x+2, y+h+12),
                FONT, 0.38, (130,130,130), 1, cv2.LINE_AA)

def draw_asymmetry_lines(frame, lm, W_, H_):
    """
    Draw connecting lines between symmetric landmark pairs.
    Line colour encodes asymmetry magnitude — green=OK, red=asymmetric.
    Provides visual explanation of what the algorithm is measuring.
    """
    fh = face_h(lm)

    def pair_line(i, j, asym_val, threshold):
        p1 = px(lm[i], W_, H_)
        p2 = px(lm[j], W_, H_)
        ratio = min(asym_val / max(threshold, 1e-6), 1.0)
        # Interpolate green → red
        g = int(200 * (1 - ratio))
        r = int(200 * ratio)
        col = (30, g, r)
        cv2.line(frame, p1, p2, col, 1, cv2.LINE_AA)
        cv2.circle(frame, p1, 3, col, -1)
        cv2.circle(frame, p2, 3, col, -1)

    pair_line(MOUTH_L,  MOUTH_R,  abs(lm[MOUTH_L].y - lm[MOUTH_R].y)/fh, TH["mouth"])
    pair_line(EYE_L_T,  EYE_R_T,  abs(lm[EYE_L_T].y - lm[EYE_R_T].y)/fh,  TH["eye"])
    pair_line(BROW_L,   BROW_R,   abs(lm[BROW_L].y  - lm[BROW_R].y)/fh,   TH["brow"])
    pair_line(CHEEK_L,  CHEEK_R,  abs(lm[CHEEK_L].y - lm[CHEEK_R].y)/fh,  TH["cheek"])

    # Face tilt axis
    n = px(lm[NOSE_TIP], W_, H_)
    c = px(lm[CHIN],     W_, H_)
    tilt = feat_tilt(lm)
    t_ratio = min(tilt / TH["tilt"], 1.0)
    tg = int(200 * (1 - t_ratio));  tr_ = int(200 * t_ratio)
    cv2.line(frame, n, c, (30, tg, tr_), 2, cv2.LINE_AA)

def draw_pose_overlay(frame, pose_lm, W_, H_):
    """Draws shoulder-wrist vectors and colours by arm weakness."""
    if pose_lm is None:
        return
    score = feat_arm(pose_lm)
    ratio = min(score / (TH["arm"] * 2), 1.0)
    g = int(200*(1-ratio));  r_ = int(200*ratio)
    col = (30, g, r_)

    for sh, wr in [(SH_L, WR_L), (SH_R, WR_R)]:
        p1 = px(pose_lm[sh], W_, H_)
        p2 = px(pose_lm[wr], W_, H_)
        cv2.line(frame, p1, p2, col, 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 5, col, -1)
        cv2.circle(frame, p2, 5, col, -1)

# ══════════════════════════════════════════════════════════════════════════════
#  §9  FULL HUD
# ══════════════════════════════════════════════════════════════════════════════

def draw_hud(frame, risk, feat_scores, raw, smoothed,
             fps, alert_tracker, trend_buf, sess_start,
             face_ok, pose_ok, cal, frame_idx):

    H_, W_ = frame.shape[:2]
    rc = risk_color(risk)

    # ── Left panel background ────────────────────────────────────────────────
    panel_w = 252
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (panel_w, H_), (16,16,16), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    y = 26
    cv2.putText(frame, "STROKE DETECTION v2.0", (10, y),
                FONT, 0.5, (170,220,255), 1, cv2.LINE_AA)
    y += 14
    cv2.putText(frame, "Stroke Risk Detection Dashboard", (10, y),
            FONT_MONO, 0.85, (90,120,130), 1, cv2.LINE_AA)
    y += 8
    cv2.line(frame, (10,y), (panel_w-10,y), (55,75,85), 1)
    y += 14

    # ── Feature bars ─────────────────────────────────────────────────────────
    bar_info = [
        ("Mouth Drop",    "mouth",  (80,210,110)),
        ("Eye Fissure",   "eye",    (120,180,230)),
        ("Face Tilt",     "tilt",   (200,160,70)),
        ("Brow Raise",    "brow",   (180,130,220)),
        ("Cheek Wrinkle", "cheek",  (220,200,80)),
        ("Expression Rig","rigid",  (150,200,200)),
        ("Arm Weakness",  "arm",    (230,90,90)),
    ]
    cv2.putText(frame, "FEATURE SCORES", (10, y),
                FONT, 0.42, (160,220,255), 1, cv2.LINE_AA)
    y += 14

    for label, key, col in bar_info:
        fs   = feat_scores.get(key, 0.0)
        raw_v = raw.get(key, 0.0)
        draw_horizontal_bar(frame, 10, y, 150, 11, fs, (45,45,45), col)
        # Raw value label
        if key == "tilt":
            val_s = f"{raw_v:.1f}°"
        else:
            val_s = f"{raw_v:.4f}"
        cv2.putText(frame, label, (10, y-2), FONT, 0.35, (190,190,190), 1, cv2.LINE_AA)
        cv2.putText(frame, val_s, (164, y+9), FONT, 0.35, (160,160,160), 1, cv2.LINE_AA)
        y += 22

    cv2.line(frame, (10,y), (panel_w-10,y), (55,75,85), 1)
    y += 12

    # ── FAST indicators (clinical checklist) ──────────────────────────────────
    cv2.putText(frame, "FAST PROTOCOL", (10, y),
                FONT, 0.42, (160,220,255), 1, cv2.LINE_AA)
    y += 14

    fast_flags = [
        ("F — Face droop",   feat_scores.get("mouth",0)>0.4 or feat_scores.get("brow",0)>0.4),
        ("A — Arm weakness", feat_scores.get("arm",0)>0.4),
        ("S — Speech*",      False),      # cannot assess speech via video
        ("T — Time →911",    risk >= RISK_HIGH),
    ]
    for txt, flag in fast_flags:
        col = (50,50,200) if flag else (80,180,80)
        sym = "■" if flag else "□"
        cv2.putText(frame, f"{sym} {txt}", (12, y), FONT, 0.38, col, 1, cv2.LINE_AA)
        y += 14

    cv2.putText(frame, "  *Speech: manual check needed", (10, y),
                FONT_MONO, 0.75, (80,80,80), 1, cv2.LINE_AA)
    y += 12

    cv2.line(frame, (10,y), (panel_w-10,y), (55,75,85), 1)
    y += 12

    # ── Session stats ─────────────────────────────────────────────────────────
    uptime  = int(time.time() - sess_start)
    um, us  = divmod(uptime, 60)
    cv2.putText(frame, "SESSION STATS", (10, y),
                FONT, 0.42, (160,220,255), 1, cv2.LINE_AA)
    y += 14
    stat_lines = [
        f"Uptime:    {um:02d}:{us:02d}",
        f"Alerts:    {alert_tracker.alert_count}",
        f"Frames:    {frame_idx}",
        f"FPS:       {fps:.1f}",
        f"Calib:     {'YES' if cal.baseline else 'NO'}",
        f"Pose:      {'ON' if pose_ok else 'OFF'}",
    ]
    for s in stat_lines:
        cv2.putText(frame, s, (12, y), FONT_MONO, 0.85, (150,160,160), 1, cv2.LINE_AA)
        y += 14

    # ── Gauge (right side) ────────────────────────────────────────────────────
    cx, cy, gr = W_ - 82, 82, 62
    draw_gauge(frame, cx, cy, gr, risk)

    lbl, lc = risk_label(risk)
    lw = cv2.getTextSize(lbl, FONT, 0.52, 2)[0][0]
    cv2.putText(frame, lbl, (cx - lw//2, cy + gr + 18),
                FONT, 0.52, lc, 2, cv2.LINE_AA)

    # ── Sustained-risk progress bar ───────────────────────────────────────────
    sp = alert_tracker.sustain_progress()
    if sp > 0:
        bx = W_ - 155;  by = cy + gr + 28
        cv2.putText(frame, "Alert threshold:", (bx, by),
                    FONT, 0.36, (160,160,160), 1, cv2.LINE_AA)
        draw_horizontal_bar(frame, bx, by+4, 130, 8, sp,
                            (45,45,45), (50,50,220))

    # ── Risk sparkline (top-right under gauge) ────────────────────────────────
    spark_x = W_ - 170;  spark_y = cy + gr + 55
    draw_sparkline(frame, trend_buf.as_array(),
                   spark_x, spark_y, 155, 50)

    # ── Calibration overlay ───────────────────────────────────────────────────
    if cal.active:
        p = cal.progress
        cv2.putText(frame, f"CALIBRATING  {int(p*100)}%",
                    (W_//2 - 110, H_//2 - 15),
                    FONT, 0.8, (50,200,255), 2, cv2.LINE_AA)
        draw_horizontal_bar(frame, W_//2-110, H_//2, 220, 18, p,
                            (50,50,50), (50,200,255))

    # ── High-risk flashing alert banner ───────────────────────────────────────
    if risk >= RISK_HIGH and int(time.time() * 2) % 2 == 0:
        bw, bh = 290, 42
        bx = (W_ - bw) // 2;  by_ = H_ - 60
        cv2.rectangle(frame, (bx, by_), (bx+bw, by_+bh), (40,40,180), -1)
        cv2.rectangle(frame, (bx, by_), (bx+bw, by_+bh), (0,0,255), 2)
        cv2.putText(frame, "⚠  HIGH RISK — SEEK HELP NOW",
                    (bx+8, by_+27), FONT, 0.56, (255,255,255), 1, cv2.LINE_AA)

    # ── No-face warning ───────────────────────────────────────────────────────
    if not face_ok:
        cv2.putText(frame, "No face detected — adjust camera",
                    (panel_w + 20, H_//2), FONT, 0.65, (80,80,200), 2, cv2.LINE_AA)

    # ── Controls footer ───────────────────────────────────────────────────────
    cv2.putText(frame, "C=Calibrate  S=Snapshot  R=Reset  ESC=Quit",
                (panel_w+6, H_-8), FONT_MONO, 0.78, (75,95,105), 1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════════════════════════════
#  §10  SNAPSHOT / REPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_snapshot(frame, risk, raw, smoothed, feat_scores, alert_tracker, sess_start):
    os.makedirs(LOG_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path  = os.path.join(LOG_DIR, f"snapshot_{ts}.png")
    json_path = os.path.join(LOG_DIR, f"report_{ts}.json")

    cv2.imwrite(img_path, frame)

    report = {
        "timestamp":     ts,
        "session_uptime_sec": int(time.time() - sess_start),
        "risk_score":    risk,
        "risk_level":    risk_label(risk)[0],
        "alert_count":   alert_tracker.alert_count,
        "raw_features":  {k: round(v, 6) for k,v in raw.items()},
        "smoothed_features": {k: round(v, 6) for k,v in smoothed.items()},
        "feature_scores": {k: round(v, 4) for k,v in feat_scores.items()},
        "feature_weights": W,
        "thresholds":    TH,
        "paper_reference": "Ahmad et al., IJ-AI Vol.13 No.1, 2024, DOI:10.11591/ijai.v13.i1.pp933-940",
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[SNAPSHOT] {img_path}")
    print(f"[REPORT]   {json_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  §11  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_models()
    cnn_model = load_model("model.h5")
    print("[INFO] CNN model loaded successfully.")

    print("=" * 60)
    print("  Real-Time Stroke Detection System  v2.0")
    print("=" * 60)
    print("  C   — Calibrate (hold neutral face, ~2 seconds)")
    print("  S   — Save snapshot + JSON report")
    print("  R   — Reset session stats")
    print("  ESC — Quit (auto-saves session CSV)")
    if TTS_AVAILABLE:
        print("  TTS — Speech alerts ENABLED")
    else:
        print("  TTS — Speech alerts DISABLED (install pyttsx3 to enable)")
    print("=" * 60)

    # ── MediaPipe setup ───────────────────────────────────────────────────────
    face_opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="pose_landmarker_lite.task"),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_det = vision.FaceLandmarker.create_from_options(face_opts)
    pose_det = vision.PoseLandmarker.create_from_options(pose_opts)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS,          CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    # ── State objects ─────────────────────────────────────────────────────────
    smooth      = DualSmooth(FEATURE_KEYS)
    logger      = SessionLogger()
    alert_tr    = AlertTracker()
    trend       = TrendBuffer(300)
    cal         = Calibrator()
    sess_start  = time.time()

    frame_idx   = 0
    previous_risk = 0
    prev_time   = time.time()
    last_pose   = None
    prev_lm     = None          # for rigidity feature
    fps         = 0.0

    # ── Peak risk tracker for session stats ───────────────────────────────────
    peak_risk = 0
    cnn_label = "Waiting..."
    cnn_prob = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            H_, W_ = frame.shape[:2]
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # ── Face detection (every frame) ──────────────────────────────────
            face_res = face_det.detect(mp_img)
            face_ok  = bool(face_res.face_landmarks)

            raw = {k: 0.0 for k in FEATURE_KEYS}

            if face_ok:
                lm = face_res.face_landmarks[0]
                # ================= CNN FACE CROP =================

                x_coords = [p.x for p in lm]
                y_coords = [p.y for p in lm]

                x_min = int(min(x_coords) * W_)
                x_max = int(max(x_coords) * W_)

                y_min = int(min(y_coords) * H_)
                y_max = int(max(y_coords) * H_)

                # Add padding
                padding = 25

                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)

                x_max = min(W_, x_max + padding)
                y_max = min(H_, y_max + padding)
                cv2.rectangle(
    frame,
    (x_min, y_min),
    (x_max, y_max),
    (0, 255, 255),
    2
)

                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size != 0:
                    cv2.imshow("CNN Face Input", face_crop)
                cnn_label = "N/A"
                cnn_prob = 0.0

                if face_crop.size != 0:

                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                    gray = cv2.resize(gray, (48, 48))

                    gray = gray.astype("float32") / 255.0

                    gray = gray.reshape(1, 48, 48, 1)

                    pred = cnn_model.predict(gray, verbose=0)

                    cnn_prob = float(pred[0][0])

                    if cnn_prob >= 0.5:
                        cnn_label = "Stroke"
                    else:
                        cnn_label = "No Stroke"

# ============================================

                head_angle = feat_tilt(lm)

                if head_angle > 30:
                    cv2.putText(frame, "Keep head straight for analysis",
                        (280, 40),
                        FONT,
                        0.6,
                        (0, 0, 255),
                        2)
                    cv2.imshow("Stroke Detection System v2.0", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

                    frame_idx += 1
                    continue

                raw["mouth"] = feat_mouth(lm)
                raw["eye"]   = feat_eye(lm)
                raw["tilt"]  = head_angle
                raw["brow"]  = feat_brow(lm)
                raw["cheek"] = feat_cheek(lm)
                raw["rigid"] = feat_rigid(lm, prev_lm)

                draw_asymmetry_lines(frame, lm, W_, H_)
                prev_lm = lm
            else:
                prev_lm = None
            # ── Pose (every POSE_SKIP frames) ─────────────────────────────────
            if frame_idx % POSE_SKIP == 0:
                pose_res = pose_det.detect(mp_img)
                last_pose = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None

            raw["arm"] = feat_arm(last_pose)
            draw_pose_overlay(frame, last_pose, W_, H_)

            # ── Calibration ───────────────────────────────────────────────────
            cal.feed(raw)
            raw_adj = cal.apply(raw)

            # ── Smooth ────────────────────────────────────────────────────────
            for k, v in raw_adj.items():
                smooth.update(k, v)
            smoothed = smooth.get_all()

# -------------------------------------------------
# Rule-Based Risk
# -------------------------------------------------

            rule_risk, feat_scores = compute_risk(smoothed)

# -------------------------------------------------
# CNN Risk (0-100)
# -------------------------------------------------

            cnn_risk = cnn_prob * 100

# -------------------------------------------------
# Hybrid Fusion
# -------------------------------------------------

            risk = int((0.6 * rule_risk) + (0.4 * cnn_risk))

# -------------------------------------------------
# Decision Refinement
# -------------------------------------------------

# Both systems strongly agree
            if rule_risk >= 70 and cnn_prob >= 0.80:
                risk = min(risk + 10, 100)

# CNN strongly suspects stroke but rules don't
            elif rule_risk < 20 and cnn_prob >= 0.90:
                risk = max(risk, 60)

# Rules strongly suspect stroke but CNN doesn't
            elif rule_risk >= 80 and cnn_prob < 0.20:
                risk = max(risk, 70)

# -------------------------------------------------
# Smooth sudden changes
# -------------------------------------------------

            if frame_idx == 0:
                previous_risk = risk

# Limit sudden increase
            if risk > previous_risk + 20:
                risk = previous_risk + 5

# Limit sudden decrease
            elif risk < previous_risk - 20:
                risk = previous_risk - 5

            previous_risk = risk
            peak_risk = max(peak_risk, risk)
            trend.push(risk)

            # ── Alert system ──────────────────────────────────────────────────
            new_alert = alert_tr.update(risk)
            if new_alert:
                print(f"[ALERT] High stroke risk sustained! Risk={risk}%  "
                      f"Time={datetime.datetime.now().strftime('%H:%M:%S')}")
                if TTS_AVAILABLE:
                    try:
                        _tts_engine.say("Warning. High stroke risk detected. "
                                        "Please seek medical attention immediately.")
                        _tts_engine.runAndWait()
                    except Exception:
                        pass
                # Auto-snapshot on alert
                save_snapshot(frame.copy(), risk, raw_adj, smoothed,
                              feat_scores, alert_tr, sess_start)

            # ── FPS ───────────────────────────────────────────────────────────
            now      = time.time()
            fps      = 0.9*fps + 0.1*(1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # ── CSV logging (every 5th frame to reduce I/O) ───────────────────
            if frame_idx % 5 == 0:
                log_row = {
                    "timestamp":    datetime.datetime.now().isoformat(),
                    "frame":        frame_idx,
                    "risk":         risk,
                    "alert_active": int(alert_tr.alert_active),
                    "face_detected":int(face_ok),
                    "pose_detected":int(last_pose is not None),
                    "fps":          round(fps, 1),
                }
                log_row.update({k: round(smoothed[k], 6) for k in FEATURE_KEYS})
                logger.write(log_row)
                if frame_idx % 150 == 0:
                    logger.flush()

            # ── HUD ───────────────────────────────────────────────────────────
            draw_hud(frame, risk, feat_scores, raw_adj, smoothed,
                     fps, alert_tr, trend,
                     sess_start, face_ok, last_pose is not None,
                     cal, frame_idx)
            cv2.putText(
                frame,
                f"Rule Risk: {rule_risk}%",
                (280, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2)

            cv2.putText(
                frame,
                f"CNN: {cnn_label} ({cnn_prob*100:.1f}%)",
                (280, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)

            cv2.putText(
                frame,
                f"Final Risk: {risk}%",
                (280, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

            cv2.imshow("Stroke Detection System v2.0", frame)

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                         # ESC
                break
            elif key in (ord('c'), ord('C')):
                cal.start()
            elif key in (ord('s'), ord('S')):
                save_snapshot(frame.copy(), risk, raw_adj, smoothed,
                              feat_scores, alert_tr, sess_start)
            elif key in (ord('r'), ord('R')):
                alert_tr.alert_count = 0
                peak_risk = 0
                sess_start = time.time()
                print("[RESET] Session statistics reset.")

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_det.close()
        pose_det.close()
        logger.close()

        # ── End-of-session summary ────────────────────────────────────────────
        uptime = int(time.time() - sess_start)
        print("\n" + "="*50)
        print("  SESSION SUMMARY")
        print(f"  Duration   : {uptime//60:02d}:{uptime%60:02d}")
        print(f"  Peak Risk  : {peak_risk}%")
        print(f"  Alerts     : {alert_tr.alert_count}")
        print(f"  Total Frames: {frame_idx}")
        print(f"  Log saved  : {logger.path}")
        print("="*50)

if __name__ == "__main__":
    main()
