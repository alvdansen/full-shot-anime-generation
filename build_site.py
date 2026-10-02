"""The record of how this repository was built.

This is the script that generated this repository's index.html, case-study.md, media/ and workflows/. It is kept here so
every number on the page can be traced to the code that computed it. Its inputs (the generated clips, our grade
sheets, the run records, the prompt files, the measurement module and the graph builders it imports) stay in our
working directory, so the script does not run from here.

What it does: it reads the grades and the run records; measures every clip (twos score, drawings per second, first-key
gap and flicker, plus the seed spread between a route's two attempts); asserts each qualitative claim in the prose
against those data; passes every statement drawn from the grading through said() and said_in(), which assert that the
comment or note it rests on carries the evidence named; builds the comparison grids and the Part B figures with ffmpeg;
rebuilds the four workflows from the same graph builders that made the study's clips, against a stub that cannot reach
Floyo; and writes the page, the Markdown and the repository files, checking every published text file against a
private deny list and checking the article's voice (no em dash, no surviving comment verbatim).
"""
from __future__ import annotations

import html
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import types
from collections import Counter
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import cv2
import markdown
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
R2 = ROOT / "outputs" / "round2"
S2 = R2 / "stage2"
PKG = R2 / "package" / "emmy_lofi_i2v_package"
REVIEW = ROOT / "review"
OUT = ROOT / "outputs" / "case_study_site"
MEDIA = OUT / "media"
WF = OUT / "workflows"
MD_OUT = OUT / "case-study.md"                         # the article as Markdown, at the repo root so its links resolve
MD_POINTER = ROOT / "posts" / "CASE_STUDY_full.md"     # the old location, now a pointer
NUMBERS_OUT = R2 / "case_study_numbers.json"           # the numbers record stays in the working directory, not the repo
OLD_NUMBERS = OUT / "data" / "numbers.json"            # where it used to ship; removed
DENY_FILE = SCRIPTS / "publish_denylist.json"          # private: the patterns no published file may contain
ORG = "Alvdansen Labs"                                   # the byline the companion papers carry (Timothy to confirm)
AUTHORS = ["Minta Carlson", "Timothy Bielec"]
AUTHORS_BIB = "Carlson, Minta and Bielec, Timothy"
REPO_BYLINE = f"{ORG} · {', '.join(AUTHORS)}"
REPO_URL = "https://alvdansen.github.io/full-shot-anime-generation/"
FLOYO_WF = "[Floyo workflow link: to come]"
AOT_URL = "https://alvdansen.github.io/animating-on-twos/"          # the companion paper (Carlson & Bielec, August 2026)
HF_URL = "https://huggingface.co/alvdansen/h3-keyframe-animation"    # the adapters; gated, access is requested on the page
# the adapters' contracts as their model card states them (HF_URL, "The conditioning contract" and "Inference settings")
MODEL_CARD_SEQ = {"refs": 2, "frames": 22}
MODEL_CARD_HERO = {"refs": 1, "frames": 22}

FORCE = "--force-media" in sys.argv
NO_MEDIA = "--no-media" in sys.argv

FONT_DIR = Path(os.environ.get("WINDIR", "")) / "Fonts"          # system fonts, used only when media is built
FONT_PIL = (FONT_DIR / "arialbd.ttf").as_posix()
FONT_PIL_REG = (FONT_DIR / "arial.ttf").as_posix()
FONT_FF = FONT_PIL.replace(":", "\\:")                           # ffmpeg drawtext escapes the drive colon

sys.path.insert(0, str(SCRIPTS))
import score_clips  # noqa: E402  (cv2 + numpy only)

GREY_MAX = int(np.iinfo(np.uint8).max)

# ----------------------------------------------------------------------------------------------- sources
def jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


BRIEF = (REVIEW / "MINTA_CASESTUDY_BRIEF_2026-10-01.txt").read_text(encoding="utf-8")
CHECK = (REVIEW / "MINTA_STAGE2_CHECK_2026-10-01.txt").read_text(encoding="utf-8")
FINAL_PASS = (REVIEW / "MINTA_FINAL_PASS_2026-10-02.txt").read_text(encoding="utf-8")
GR = jload(R2 / "minta_grades_r2s1.json")
RES = jload(R2 / "results.json")
SHOTS_DOC = jload(PKG / "shots.json")
S2RES = jload(S2 / "results.json")
SPANS = jload(S2 / "spans.json")
PROBE = jload(R2 / "stage2_probe.json")
PR = jload(REVIEW / "prompts_r2.json")["routes"]
ADP = jload(REVIEW / "prompts_r2_adapters.json")
PR2 = jload(REVIEW / "prompts_r2s2.json")
TWOS_MIN = float(re.search(r"^TWOS_MIN\s*=\s*([\d.]+)", (SCRIPTS / "r2s2_analyze.py").read_text(encoding="utf-8"), re.M).group(1))
for _t in (json.dumps(PR, ensure_ascii=False), json.dumps(ADP, ensure_ascii=False), json.dumps(PR2, ensure_ascii=False)):
    assert "\u00e2\u20ac" not in _t, "mojibake in a prompt file"

# ----------------------------------------------------------------------------------------------- routes and shots
ROUTES = ["h3api", "seedance", "flux3", "wan30", "h3ow", "seq124", "hero124"]       # the seven graded routes
FINAL_WF = ["wan30", "h3ow", "h3api", "seq124"]                                     # Minta's smoothest four
NAME = {"h3api": "MiniMax H3 API", "seedance": "Seedance 2.5", "flux3": "FLUX 3", "wan30": "Wan 3.0",
        "h3ow": "H3 open weights"}
# reader-facing anchors for the route cards (the working codes stay out of the page)
ANCHOR = {"h3api": "route-h3-api", "seedance": "route-seedance", "flux3": "route-flux-3", "wan30": "route-wan-3",
          "h3ow": "route-h3-open-weights", "seq124": "route-seq-adapter", "hero124": "route-hero-adapter"}
TILE = {"h3api": "H3 API", "seedance": "Seedance 2.5", "flux3": "FLUX 3", "wan30": "Wan 3.0", "h3ow": "H3 open weights"}
SHOT_NAME = {"01_closeup_train": "Train window", "02_wide_chill_bus_shelter": "Bus shelter in the rain",
             "03b_action_bike_downpour": "Bike in the downpour", "03e_action_platform": "Station platform",
             "03f_action_kicksled": "Kicksled", "04_zoom_out_roofs": "Roofs, pulling back",
             "05b_fisheye_peephole": "The peephole"}

ALL_SHOTS = SHOTS_DOC["shots"]
# a shot is dropped where Minta's comment says so (03f: "we are dropping this whole category")
DROPPED = {s["id"] for s in ALL_SHOTS for c in GR["clips"].values()
           if s["id"].startswith(c.get("shot_key", "") + "_") and "dropping this whole category" in (c.get("Comment") or "")}
assert DROPPED == {"03f_action_kicksled"}, DROPPED
SHOTS = [s for s in ALL_SHOTS if s["id"] not in DROPPED]
SID = [s["id"] for s in SHOTS]
SHOT = {s["id"]: s for s in SHOTS}
CLIP_S = SHOTS_DOC["clip_length_seconds"]


def short(sid: str) -> str:
    return f"{sid.split('_')[0]} {SHOT_NAME[sid]}"


def row(shot, arm, seed):
    rs = [r for r in RES if r["shot"] == shot and r["arm"] == arm and r["seed"] == seed]
    assert len(rs) == 1, (shot, arm, seed, len(rs))
    return rs[0]


SEEDS = {a: sorted({r["seed"] for r in RES if r["arm"] == a and r["shot"] in SID}) for a in ROUTES}
for a in ROUTES:
    assert len(SEEDS[a]) == 2, (a, SEEDS[a])
    for s in SID:
        assert len([r for r in RES if r["arm"] == a and r["shot"] == s]) == 2, (a, s)
FRAMES = {a: sorted({r["frames"] for r in RES if r["arm"] == a and r["shot"] in SID}) for a in ROUTES}
FPS = {a: sorted({r["fps"] for r in RES if r["arm"] == a and r["shot"] in SID}) for a in ROUTES}
for a in ROUTES:
    assert len(FRAMES[a]) == 1 and len(FPS[a]) == 1, (a, FRAMES[a], FPS[a])
NAME["seq124"] = f"seq adapter, {FRAMES['seq124'][0]}f"
NAME["hero124"] = f"hero adapter, {FRAMES['hero124'][0]}f"
TILE["seq124"] = f"seq {FRAMES['seq124'][0]}f"
TILE["hero124"] = f"hero {FRAMES['hero124'][0]}f"
ATT = [0, 1]                      # attempt index: 0 = seed 42 / FLUX roll 1, 1 = seed 1042 / FLUX roll 2
SEED_A = {a: SEEDS[a] for a in ROUTES}


def attempt_label(i: int, long=True) -> str:
    s = SEEDS["h3api"][i]
    fr = SEEDS["flux3"][i]
    return f"Attempt {i + 1}: seed {s} (FLUX 3 roll {fr})" if long else f"Attempt {i + 1}"


# Minta's grades keyed (shot, arm, seed)
G = {}
for k, c in GR["clips"].items():
    sid = next((s for s in SID if s.startswith(c["shot_key"] + "_")), None)
    if sid:
        G[(sid, c["arm"], c["seed"])] = c
for a in ROUTES:
    for s in SID:
        for sd in SEEDS[a]:
            assert G.get((s, a, sd), {}).get("Usable"), ("ungraded", s, a, sd)


def probs(c) -> list[str]:
    p = c.get("Problems") or []
    return p if isinstance(p, list) else [x.strip() for x in p.split(";") if x.strip()]


GRADE_MAX = max(int(c["Believable in-between"].split()[0]) for c in G.values() if c.get("Believable in-between"))
GRADE_MIN = min(int(c["Believable in-between"].split()[0]) for c in G.values() if c.get("Believable in-between"))

# ----------------------------------------------------------------------------------------------- statements from the grading
# The article is written in our words and quotes no grading comment. A statement that rests on a comment goes
# through said(): it returns the statement unchanged, after asserting that the comment on that take carries every
# piece of evidence named (case-insensitive substrings). said_in() does the same for the brief, the Stage 2 check and
# the final-pass notes. The count of checked statements goes to the numbers record.
BACKED = []


def said(text: str, shot, arm, seed, *evidence: str) -> str:
    """A statement of ours, asserted to be supported by the grading comment on one take."""
    assert evidence, "a statement drawn from the grading names its evidence"
    c = (G[(shot, arm, seed)].get("Comment") or "").lower()
    for e in evidence:
        assert e.lower() in c, f"not supported by the comment on {shot} {arm} {seed}: {e!r}"
    BACKED.append((shot, arm, seed, evidence))
    return text


def said_in(text: str, source: str, *evidence: str) -> str:
    """A statement of ours, asserted to be supported by one of the other texts (brief, Stage 2 check, final pass)."""
    assert evidence, "a statement drawn from a text names its evidence"
    for e in evidence:
        assert e.lower() in source.lower(), f"not supported by the source text: {e!r}"
    BACKED.append(("text", evidence))
    return text


MODEL_NOTE_H3 = GR["models"]["MiniMax H3 API"]["note"]
ASKS = GR["asks_raw"]

# ----------------------------------------------------------------------------------------------- formatting
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven", 12: "twelve"}


def w(n: int) -> str:
    return WORDS.get(n, str(n))


def W(n: int) -> str:
    s = w(n)
    return s[0].upper() + s[1:]


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def f1(x):
    return "–" if x is None else f"{x:.1f}"


def usd(x):
    return f"${x:.2f}"


def secs(x):
    return f"{x:.0f} s"


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def g_int(c, k):
    v = c.get(k)
    return int(v.split()[0]) if v else None


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def k2k(frames: int, fps: float) -> str:
    """Key-to-key time of a clip, first frame to last: (frames - 1) / fps, two places, half up. Part B uses this one
    convention throughout (the source span 25 f = 1.00 s, 22 f = 0.88 s, 39 f = 1.58 s), labelled "key to key"; it is
    also the mark the tween's alignment line gives Picture 2."""
    return str((Decimal(frames - 1) / Decimal(str(fps))).quantize(Decimal("0.01"), ROUND_HALF_UP))


def calibrate_twos() -> dict:
    """score_clips' twos score on synthetic clips held on ones, twos and threes (a square stepping across white),
    written losslessly, so the page can say what the scale means without typing it."""
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for k in (1, 2, 3):
            path = Path(td) / f"cadence{k}.mp4"
            w_, h_, n = 320, 180, 48
            pr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w_}x{h_}", "-r", "24",
                                   "-i", "-", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
            for i in range(n):
                fr = np.full((h_, w_), 245, np.uint8)
                x = 20 + 5 * (i // k)
                fr[70:110, x:x + 40] = 30
                pr.stdin.write(fr.tobytes())
            pr.stdin.close(); pr.wait()
            out[k] = score_clips.score(path)["twos_score"]
    return out


TWOS_CAL = calibrate_twos()
assert TWOS_CAL[2] > TWOS_CAL[3] > TWOS_CAL[1], TWOS_CAL


# ----------------------------------------------------------------------------------------------- measurements
def gray_frames(path: Path, w_px: int = score_clips.W):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h = max(1, round(f.shape[0] * w_px / f.shape[1]))
        out.append(cv2.cvtColor(cv2.resize(f, (w_px, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return out


def bgr_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


SPREAD_N = 25    # time points compared between a route's two attempts


def seed_spread(arm: str, shot: str) -> float:
    """Mean absolute grey-level difference between the two attempts at matched moments (SPREAD_N points over the
    shorter clip), grayscale at score_clips.W px. Low = the two attempts look alike; high = they differ."""
    a, b = (row(shot, arm, sd) for sd in SEEDS[arm])
    A, B = gray_frames(R2 / a["file"]), gray_frames(R2 / b["file"])
    fps = a["fps"]
    n = min(len(A), len(B))
    idx = sorted({min(n - 1, int(t * fps)) for t in np.linspace(0, (n - 1) / fps, SPREAD_N)})
    return float(np.mean([np.mean(np.abs(A[i] - B[i])) for i in idx]))


print("measuring seed spread ...", flush=True)
SPREAD = {a: {s: round(seed_spread(a, s), 2) for s in SID} for a in ROUTES}


def route_stats(a: str) -> dict:
    rs = [row(s, a, sd) for s in SID for sd in SEEDS[a]]
    gs = [G[(r["shot"], a, r["seed"])] for r in rs]
    us = Counter(c["Usable"] for c in gs)
    shots_asis = sum(1 for s in SID if any(G[(s, a, sd)]["Usable"] == "As is" for sd in SEEDS[a]))
    return {
        "clips": len(rs), "as_is": us.get("As is", 0), "with_fixes": us.get("With fixes", 0), "no": us.get("No", 0),
        "shots_as_is": shots_asis, "shots": len(SID),
        "twos_eye": Counter(c.get("Reads on twos", "–") for c in gs),
        "look": Counter(c.get("Looks like") for c in gs if c.get("Looks like")),
        "problems": Counter(p for c in gs for p in probs(c)),
        "fix": Counter(c.get("Fix") for c in gs if c.get("Fix")),
        "believable_mean": statistics.mean([g_int(c, "Believable in-between") for c in gs if c.get("Believable in-between")]),
        "mood_mean": statistics.mean([g_int(c, "Lofi-anime mood") for c in gs if c.get("Lofi-anime mood")]),
        "twos": med([r["twos_score"] for r in rs]), "dps": med([r["drawings_per_s"] for r in rs]),
        "first_mad": med([r["first_mad"] for r in rs]), "flotime": med([r["flotime_s"] for r in rs]),
        "cost": med([r["cost_usd"] or 0 for r in rs]), "spend": sum(r["cost_usd"] or 0 for r in rs),
        "spread": med(list(SPREAD[a].values())), "frames": FRAMES[a][0], "fps": FPS[a][0],
        "size": sorted({f"{r['width']}x{r['height']}" for r in rs}),
        "node": rs[0]["node"], "params": rs[0]["prov"].get("params") or {}, "steps": rs[0]["prov"].get("steps"),
        "lora": rs[0].get("lora") or "",
    }


ST = {a: route_stats(a) for a in ROUTES}
N_CLIPS = sum(ST[a]["clips"] for a in ROUTES)
SPEND = sum(ST[a]["spend"] for a in ROUTES)
PAID = [a for a in ROUTES if ST[a]["cost"] > 0]
OPEN = [a for a in ROUTES if ST[a]["cost"] == 0]

# all graded clips of the seven routes on the kept shots
ALLG = [G[(s, a, sd)] for a in ROUTES for s in SID for sd in SEEDS[a]]
PROB_ALL = Counter(p for c in ALLG for p in probs(c))
LOOK_ALL = Counter(c.get("Looks like") for c in ALLG if c.get("Looks like"))
EYE_ALL = Counter(c.get("Reads on twos") for c in ALLG if c.get("Reads on twos"))
EYE_TWOS = {k: [row(s, a, sd)["twos_score"] for a in ROUTES for s in SID for sd in SEEDS[a]
                if G[(s, a, sd)].get("Reads on twos") == k] for k in ("Yes", "Partly", "No")}
ROOFS = "04_zoom_out_roofs"
ROOFS_ZERO = [(a, sd) for a in ROUTES for sd in SEEDS[a] if row(ROOFS, a, sd)["twos_score"] == 0]
PLATFORM = "03e_action_platform"
PLATFORM_ASIS = [(a, sd) for a in ROUTES for sd in SEEDS[a] if G[(PLATFORM, a, sd)]["Usable"] == "As is"]
PLATFORM_FIX = [(a, sd) for a in ROUTES for sd in SEEDS[a] if G[(PLATFORM, a, sd)]["Usable"] == "With fixes"]


def leaders(key, hi=True):
    best = (max if hi else min)(ST[a][key] for a in ROUTES)
    return [a for a in ROUTES if ST[a][key] == best], best


def join_names(arms, short_names=False):
    n = [(TILE if short_names else NAME)[a] for a in arms]
    return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]


# ----------------------------------------------------------------------------------------------- Stage 2 (span B)
SPAN = SPANS["B"]
SPAN_SHOT = SPAN["shot"]
A0, B0 = SPAN["proposed"]
m = re.search(r"SPAN B:.*?Frames: Proposed: frames (\d+) to (\d+)", CHECK, re.S)
assert m and (int(m.group(1)), int(m.group(2))) == (A0, B0), "span B frames differ from Minta's check"
SEG = SPAN["segments"][f"{A0}-{B0}"]
SRC_ROW = next(r for r in RES if r["run_id"] == SPAN["run_id"])
assert SRC_ROW["shot"] == SPAN_SHOT and SRC_ROW["arm"] == SPAN["route"] and SRC_ROW["seed"] == SPAN["seed"]
SRC_ATT = SEEDS[SRC_ROW["arm"]].index(SRC_ROW["seed"])
B_ROWS = sorted([r for r in S2RES if r["span"] == "B"], key=lambda r: (["h3ow22", "seq22", "seq39"].index(r["arm"]), r["seed"]))
assert len(B_ROWS) == 6
B_ARMS = ["h3ow22", "seq22", "seq39"]
B_SEEDS = sorted({r["seed"] for r in B_ROWS})


def brow(arm, seed):
    return next(r for r in B_ROWS if r["arm"] == arm and r["seed"] == seed)


B_FR = {a: brow(a, B_SEEDS[0])["frames"] for a in B_ARMS}
B_NAME = {"h3ow22": f"H3 open weights, {B_FR['h3ow22']}f", "seq22": f"seq adapter, {B_FR['seq22']}f", "seq39": f"seq adapter, {B_FR['seq39']}f"}
B_TILE = {"h3ow22": f"H3 open weights {B_FR['h3ow22']}f", "seq22": f"seq {B_FR['seq22']}f", "seq39": f"seq {B_FR['seq39']}f"}
# reader-facing media names for Part B (the working codes stay out of the published file names)
B_SLUG = {"h3ow22": f"h3-open-weights-{B_FR['h3ow22']}f", "seq22": f"seq-adapter-{B_FR['seq22']}f", "seq39": f"seq-adapter-{B_FR['seq39']}f"}
SPAN_FRAMES = SEG["frames"]
FPS_B = brow("seq22", B_SEEDS[0])["fps"]

# duration floors (the probe record) and the open-weight length rule
PROBE_RE = {"h3api_first_last": ("h3api", r"duration enum '(\d+)'"), "seedance_25": ("seedance", r"duration '(\d+)'\.\.'(\d+)' accepted"),
            "flux3_first_last": ("flux3", r"duration INT min (\d+)"), "wan30": ("wan30", r"duration '(\d+)'\.\. accepted")}
FLOOR = {}
for k, (arm, rx) in PROBE_RE.items():
    FLOOR[arm] = int(re.search(rx, PROBE[k]).group(1))
OW_RULE = re.search(r"any (\d+n\+\d+) frame count at (\d+) fps", PROBE["h3_open_weights_and_adapters"])
RULE_TXT, RULE_FPS = OW_RULE.group(1), int(OW_RULE.group(2))
RULE_K, RULE_C = (int(x) for x in re.match(r"(\d+)n\+(\d+)", RULE_TXT).groups())
LEGAL = [RULE_K * n + RULE_C for n in range(1, 6)]
assert B_FR["seq22"] in LEGAL and B_FR["seq39"] in LEGAL and B_FR["h3ow22"] in LEGAL
BELOW = max(x for x in LEGAL if x <= SPAN_FRAMES)
ABOVE = min(x for x in LEGAL if x > SPAN_FRAMES)
assert BELOW == B_FR["seq22"] and ABOVE == B_FR["seq39"]
MARK22 = k2k(B_FR["seq22"], RULE_FPS)                       # where the tween's alignment line puts Picture 2
TWEEN_MARK_LINE = f"Picture 2 aligns with the {MARK22}-second mark"    # checked against the built tween graph
# two notes the page and the workflows README share (the grader's issue 11)
SEQ_ADAPTER_SETUP = "\n".join([          # a lead line, then numbered steps (joined, so no drive-like "x:" + backslash in the source)
    "Both seq workflows load our seq adapter in their LoRA node (LoraLoaderModelOnly). To set it up:",
    "",
    f"1. Request access at [alvdansen/h3-keyframe-animation]({HF_URL}) on Hugging Face.",
    "2. Take `h3_seq_step12000` from the repository's `adapters` folder. It is already converted for H3.",
    "3. Put it in your ComfyUI `models/loras` folder and select it in that node.",
    "4. The model card gives the license, the inference settings and the conditioning contract. The adapters run on the full H3 "
    "model, not Turbo, under MiniMax's H3 license.",
    ""])
assert B_FR["seq22"] == MODEL_CARD_SEQ["frames"] in LEGAL      # the tween workflow renders the card's own window length
SEQ_CONTRACT_NOTE = (f"the seq adapter was trained on {w(MODEL_CARD_SEQ['refs'])} references, a window's first drawing and its natural "
                     f"end, rendered as {MODEL_CARD_SEQ['frames']} frames. The long first-frame workflow gives it one reference and "
                     f"{FRAMES['seq124'][0]} frames, which asks more of it; Part A records how it behaves there. The tween workflow "
                     "matches its training.")
TWEEN_LENGTH_NOTE = (f"the tween's prompt opens with an alignment line that puts Picture 2 at the {MARK22}-second mark, the key-to-key "
                     f"time of {B_FR['seq22']} frames at {RULE_FPS} fps. If you change CLIP LENGTH, change that mark to "
                     f"(frames - 1) / {RULE_FPS} seconds for the frame count the length snaps to: {ABOVE} frames, for example, is the "
                     f"{k2k(ABOVE, RULE_FPS)}-second mark.")

# the span's first and last frames, given to the models
KEY_A = S2 / "frames" / f"spanB_{SPAN_SHOT}_f{A0:03d}.png"
KEY_B = S2 / "frames" / f"spanB_{SPAN_SHOT}_f{B0:03d}.png"
assert KEY_A.exists() and KEY_B.exists()


def source_span_metrics() -> dict:
    """score_clips.score() on the source take's frames A0..B0, through a lossless temporary copy."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "span.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(R2 / SRC_ROW["file"]), "-vf",
                        f"trim=start_frame={A0}:end_frame={B0 + 1},setpts=PTS-STARTPTS", "-c:v", "libx264", "-qp", "0",
                        "-pix_fmt", "yuv420p", "-r", f"{SRC_ROW['fps']:g}", str(tmp)], check=True)
        met = score_clips.score(tmp, fps_override=SRC_ROW["fps"])
    assert met["frames"] == SPAN_FRAMES, met["frames"]
    return met


print("measuring the source span ...", flush=True)
SRC_MET = source_span_metrics()


def _share(frames, pairs) -> float:
    """The share of all frame-to-frame change in a clip that happens inside its held pairs."""
    d = [float(np.mean(np.abs(frames[i + 1] - frames[i]))) for i in range(len(frames) - 1)]
    return sum(d[a] for a, _, _ in pairs) / sum(d)


def tc_pairs(order):
    out, i = [], 0
    while i < len(order) - 1:
        if order[i] == order[i + 1]:
            out.append((i, i + 1, order[i])); i += 2
        else:
            i += 1
    return out


TC = {}
for r in B_ROWS:
    if "twosclean" not in r:
        continue
    order = r["twosclean"]["order"]
    pairs = tc_pairs(order)
    assert len(pairs) == r["twosclean"]["pairs"], (r["id"], len(pairs))
    before = gray_frames(R2 / r["file"])
    after_p = R2 / r["twosclean"]["file"]
    after = gray_frames(after_p)
    assert len(before) == len(after) == len(order)
    # the after frames are the before frames in the recorded order (checks the file against the record)
    worst = max(float(np.mean(np.abs(after[n] - before[order[n]]))) for n in range(len(order)))
    assert worst < 2.0, (r["id"], worst)
    met_after = score_clips.score(after_p)
    TC[r["id"]] = {
        "arm": r["arm"], "seed": r["seed"], "pairs": len(pairs), "frames": len(order), "order": order, "pair_list": pairs,
        "in_pair_before": float(np.mean([np.mean(np.abs(before[a] - before[b])) for a, b, _ in pairs])),
        "in_pair_after": float(np.mean([np.mean(np.abs(after[a] - after[b])) for a, b, _ in pairs])),
        "share_before": _share(before, pairs), "share_after": _share(after, pairs),
        "between_drawings": float(np.median([np.mean(np.abs(before[i + 1] - before[i])) for i in range(len(before) - 1)
                                             if not any(i == a for a, _, _ in pairs)])),
        "twos_before": r["twos_score"], "twos_after": met_after["twos_score"],
        "dps_before": r["drawings_per_s"], "dps_after": met_after["drawings_per_s"], "file": r["twosclean"]["file"], "src": r["file"],
    }
assert TC, "no twos-clean clips for span B"
FEATURED = max(TC, key=lambda k: TC[k]["in_pair_before"])
NOT_TC = [r for r in B_ROWS if "twosclean" not in r]

# ----------------------------------------------------------------------------------------------- media
FPS_GRID = 60     # the grid's frame rate: shows 30 fps evenly and 24 fps the way any 60 Hz screen does
assert FPS_GRID % int(FPS["wan30"][0]) == 0
TW, TH = 640, 360


def ff(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def dt_label(text: str) -> str:
    assert not re.search(r"[:,'%\\\[\];=]", text), f"label needs plain text: {text!r}"
    return (f"drawtext=fontfile='{FONT_FF}':text='{text}':x=12:y=10:fontsize=22:fontcolor=white:"
            f"box=1:boxcolor=black@0.62:boxborderw=7")


def card_png(lines: list[tuple[str, int, bool]], dst: Path, size=(TW, TH)):
    """An info tile: lines of (text, px, bold) on the dark tile colour."""
    im = Image.new("RGB", size, (27, 35, 33))
    d = ImageDraw.Draw(im)
    y = 40
    for text, px, bold in lines:
        f = ImageFont.truetype(FONT_PIL if bold else FONT_PIL_REG, px)
        d.text((36, y), text, font=f, fill=(236, 240, 238) if bold else (180, 192, 188))
        y += int(px * 1.45)
    im.save(dst)


def grid(tiles: list[dict], dst: Path, T: float, fps_out: int, crf: int = 22):
    """3x3 grid of 640x360 tiles. tile: {kind: clip|image, path, fps, trim, label}. Clips play at their native rate
    (setpts from the clip's fps) and hold their last drawing to T."""
    assert len(tiles) == 9
    ins, filt = [], []
    for i, t in enumerate(tiles):
        fit = (f"scale={TW}:{TH}:force_original_aspect_ratio=decrease,pad={TW}:{TH}:(ow-iw)/2:(oh-ih)/2:color=black")
        if t["kind"] == "image":
            ins += ["-loop", "1", "-framerate", str(fps_out), "-t", f"{T:.3f}", "-i", str(t["path"])]
            chain = f"[{i}:v]{fit},fps={fps_out},trim=duration={T:.3f},setsar=1,format=yuv420p"
        else:
            ins += ["-i", str(t["path"])]
            pre = f"trim=start_frame={t['trim'][0]}:end_frame={t['trim'][1]}," if t.get("trim") else ""
            chain = (f"[{i}:v]{pre}setpts=N/({t['fps']:g}*TB),{fit},fps={fps_out},"
                     f"tpad=stop_mode=clone:stop_duration={T:.3f},trim=duration={T:.3f},setsar=1,format=yuv420p")
        if t.get("label"):
            chain += "," + dt_label(t["label"])
        filt.append(chain + f"[v{i}]")
    layout = "|".join(f"{(k % 3) * TW}_{(k // 3) * TH}" for k in range(9))
    filt.append("".join(f"[v{k}]" for k in range(9)) + f"xstack=inputs=9:layout={layout}[out]")
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff(*ins, "-filter_complex", ";".join(filt), "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "slow",
       "-crf", str(crf), "-tune", "animation", "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2",
       "-r", str(fps_out), "-movflags", "+faststart", str(dst))


def poster(src: Path, dst: Path, at: float, width: int = 1280):
    ff("-ss", f"{at:.3f}", "-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(dst))


def need(p: Path) -> bool:
    return FORCE or not p.exists()


MEDIA_FILES = {}     # key -> relative path


def rel(p: Path) -> str:
    return str(p.relative_to(OUT)).replace("\\", "/")


def build_stills():
    for s in SID:
        dst = MEDIA / "stills" / f"{s}.jpg"
        if need(dst) and not NO_MEDIA:
            dst.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(PKG / SHOT[s]["file"]).convert("RGB")
            im.thumbnail((1600, 1600), Image.LANCZOS)
            im.save(dst, quality=86, optimize=True)
        MEDIA_FILES[f"still:{s}"] = rel(dst)


def build_stage1_grids(tmpdir: Path):
    for s in SID:
        for i in ATT:
            dst = MEDIA / "grids" / f"{s}_attempt{i + 1}.mp4"
            pst = dst.with_suffix(".jpg")
            rows = [row(s, a, SEEDS[a][i]) for a in ROUTES]
            T = max(r["frames"] / r["fps"] for r in rows) + 0.25
            if need(dst) and not NO_MEDIA:
                card = tmpdir / f"card_{s}_{i}.png"
                card_png([(short(s), 30, True), (SHOT[s]["shot_type"], 24, False), ("", 12, False),
                          (attempt_label(i, long=False), 26, True), (f"seed {SEEDS['h3api'][i]}", 22, False),
                          (f"FLUX 3 roll {SEEDS['flux3'][i]} (no seed)", 22, False), ("", 12, False),
                          ("Each tile plays at its", 20, False), ("model's native frame rate", 20, False)], card)
                tiles = [{"kind": "image", "path": PKG / SHOT[s]["file"], "label": "The still (first frame)"}]
                tiles += [{"kind": "clip", "path": R2 / r["file"], "fps": r["fps"], "label": f"{TILE[r['arm']]} · {r['fps']:g} fps"} for r in rows]
                tiles += [{"kind": "image", "path": card}]
                grid(tiles, dst, T, FPS_GRID, crf=18)
                poster(dst, pst, T * 0.55)
            MEDIA_FILES[f"grid:{s}:{i}"] = rel(dst)
            MEDIA_FILES[f"poster:{s}:{i}"] = rel(pst)


def build_partb(tmpdir: Path):
    d = MEDIA / "partb"
    d.mkdir(parents=True, exist_ok=True)
    for key, png, n, which in (("keyA", KEY_A, A0, "first"), ("keyB", KEY_B, B0, "last")):
        dst = d / f"knock_{which}-key_frame{n:02d}.jpg"
        if need(dst) and not NO_MEDIA:
            im = Image.open(png).convert("RGB")
            im.thumbnail((1280, 1280), Image.LANCZOS)
            im.save(dst, quality=88, optimize=True)
        MEDIA_FILES[f"partb:{key}"] = rel(dst)
    # the grid: keys and the source span on top, one seed per row below
    dst = d / "knock_repair-grid.mp4"
    durs = [SPAN_FRAMES / SRC_ROW["fps"]] + [r["frames"] / r["fps"] for r in B_ROWS]
    T = max(durs) + 0.5
    if need(dst) and not NO_MEDIA:
        tiles = [{"kind": "image", "path": KEY_A, "label": f"First key · frame {A0}"},
                 {"kind": "clip", "path": R2 / SRC_ROW["file"], "fps": SRC_ROW["fps"], "trim": (A0, B0 + 1),
                  "label": f"Source take · frames {A0} to {B0}"},
                 {"kind": "image", "path": KEY_B, "label": f"Last key · frame {B0}"}]
        for sd in B_SEEDS:
            for a in B_ARMS:
                r = brow(a, sd)
                tiles.append({"kind": "clip", "path": R2 / r["file"], "fps": r["fps"], "label": f"{B_TILE[a]} · seed {sd}"})
        grid(tiles, dst, T, int(FPS_B), crf=20)
        poster(dst, dst.with_suffix(".jpg"), 0.45)
    MEDIA_FILES["partb:grid"] = rel(dst)
    MEDIA_FILES["partb:grid_poster"] = rel(dst.with_suffix(".jpg"))
    # twos-clean before/after, one per clip
    for cid, t in TC.items():
        dst = d / f"twos-clean_{B_SLUG[t['arm']]}_seed{t['seed']}.mp4"
        if need(dst) and not NO_MEDIA:
            before, after = bgr_frames(R2 / t["src"]), bgr_frames(R2 / t["file"])
            held = {n for a, b, _ in t["pair_list"] for n in (a, b)}
            fb, fr_ = ImageFont.truetype(FONT_PIL, 26), ImageFont.truetype(FONT_PIL_REG, 22)
            hw, hh = 960, 548
            p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{hw * 2}x{hh}",
                                  "-r", f"{FPS_B:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-tune", "animation",
                                  "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.2", "-movflags", "+faststart", str(dst)],
                                 stdin=subprocess.PIPE)
            for n in range(len(t["order"])):
                canvas = Image.new("RGB", (hw * 2, hh), (0, 0, 0))
                for k, (fr, title) in enumerate(((before[n], "As generated"), (after[n], "Twos-clean"))):
                    im = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (hw, hh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                    dr = ImageDraw.Draw(im)
                    dr.rectangle((10, 10, 24 + dr.textlength(title, font=fb), 50), fill=(0, 0, 0))
                    dr.text((17, 14), title, font=fb, fill=(255, 255, 255))
                    sub = f"frame {n:02d}" if k == 0 else (f"frame {n:02d} shows {t['order'][n]:02d}" + ("  held" if n in held else ""))
                    tw_ = dr.textlength(sub, font=fr_)
                    dr.rectangle((10, hh - 46, 24 + tw_, hh - 12), fill=(0, 0, 0))
                    dr.text((17, hh - 43), sub, font=fr_, fill=(255, 214, 140) if (k == 1 and n in held) else (230, 230, 230))
                    canvas.paste(im, (k * hw, 0))
                p.stdin.write(canvas.tobytes())
            p.stdin.close(); p.wait()
            assert p.returncode == 0
            poster(dst, dst.with_suffix(".jpg"), 0.30)
        MEDIA_FILES[f"tc:{cid}"] = rel(dst)
        MEDIA_FILES[f"tcposter:{cid}"] = rel(dst.with_suffix(".jpg"))
    # the pair figure for the featured clip: the held pairs that moved most inside the hold
    t = TC[FEATURED]
    dst = d / f"twos-clean_held-pairs_{B_SLUG[t['arm']]}_seed{t['seed']}.jpg"
    before = bgr_frames(R2 / t["src"])
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in before]
    gs = gray_frames(R2 / t["src"])
    ranked = sorted(t["pair_list"], key=lambda p: -float(np.mean(np.abs(gs[p[0]] - gs[p[1]]))))[:PAIR_ROWS]
    ranked.sort()
    t["figure_pairs"] = [(a, b, k, round(float(np.mean(np.abs(gs[a] - gs[b]))), 2)) for a, b, k in ranked]
    if need(dst) and not NO_MEDIA:
        cw, ch, gap, mg, lab = 600, 343, 16, 24, 40
        Wd = mg * 2 + cw * 3 + gap * 2
        Hd = mg * 2 + len(ranked) * (ch + lab + gap)
        im = Image.new("RGB", (Wd, Hd), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        fb, frg = ImageFont.truetype(FONT_PIL, 22), ImageFont.truetype(FONT_PIL_REG, 22)
        for r_i, (a, b, keep) in enumerate(ranked):
            y = mg + r_i * (ch + lab + gap)
            diff = np.clip(np.abs(g[a] - g[b]) * DIFF_GAIN, 0, GREY_MAX)
            cells = [(before[a], f"frame {a:02d}", keep == a), (before[b], f"frame {b:02d}", keep == b),
                     (cv2.cvtColor((GREY_MAX - diff).astype(np.uint8), cv2.COLOR_GRAY2BGR), f"what changes inside the hold, x{DIFF_GAIN}", False)]
            for c_i, (fr, text, kept) in enumerate(cells):
                x = mg + c_i * (cw + gap)
                tile = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (cw, ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
                im.paste(tile, (x, y + lab))
                dr.text((x, y + 8), text + ("   kept, shown twice" if kept else ""), font=fb if kept else frg,
                        fill=(45, 106, 153) if kept else (60, 66, 64))
                if kept:
                    dr.rectangle((x - 4, y + lab - 4, x + cw + 3, y + lab + ch + 3), outline=(45, 106, 153), width=6)
                else:
                    dr.rectangle((x, y + lab, x + cw - 1, y + lab + ch - 1), outline=(205, 208, 204), width=1)
        im.save(dst, quality=86, optimize=True)
    MEDIA_FILES["tc:pairs"] = rel(dst)


PAIR_ROWS = 4      # pairs shown in the figure
DIFF_GAIN = 8      # the difference image is amplified by this factor

# ----------------------------------------------------------------------------------------------- workflows
FIRST, LAST = "#inputs/FIRST_FRAME.png", "#inputs/LAST_FRAME.png"
EXAMPLE_SHOT = "02_wide_chill_bus_shelter"     # the first-frame examples use the bus-shelter prompts


def _refuse(*a, **k):
    raise RuntimeError("the case-study builder never talks to Floyo")


def builders():
    stub = types.ModuleType("sprint_lib")
    stub.R1, stub.ROOT = R2, ROOT
    for n in ("launch", "settle", "download", "upload", "runs", "round_report"):
        setattr(stub, n, _refuse)
    stub.fc = types.SimpleNamespace()
    sys.modules["sprint_lib"] = stub
    # the graph builders assert the round and stage they belong to; set exactly the variables they assert
    for src in ("r2_runs.py", "r2s2_runs.py"):
        for var, val in re.findall(r'os\.environ\.get\("(\w+)"\) == "(\d+)"', (SCRIPTS / src).read_text(encoding="utf-8")):
            os.environ[var] = val
    import r2_runs, r2s2_runs  # noqa: E401
    return r2_runs, r2s2_runs


def frames_from_graph(g: dict, secs_node="132", math_node="131") -> int:
    expr = g[math_node]["inputs"]["expression"]
    a = g[secs_node]["inputs"]["value"]
    return int(eval(expr, {"__builtins__": {}}, {"a": a, "max": max, "round": round}))


def titled(g: dict, titles: dict) -> dict:
    for k, t in titles.items():
        g[k].setdefault("_meta", {})["title"] = t
    return g


def build_workflows() -> list[dict]:
    r2_runs, r2s2_runs = builders()
    WF.mkdir(parents=True, exist_ok=True)
    out = []
    p_wan = PR["wan30"][EXAMPLE_SHOT]
    p_h3 = PR["h3api"][EXAMPLE_SHOT]
    p_seq = r2_runs.adapter_caption(EXAMPLE_SHOT)
    tw_frames, tw_secs, _ = r2s2_runs.ROUTES["seq22"]
    p_tw = r2s2_runs.prompt("B", "seq22")

    partner_titles = lambda model, how: {  # noqa: E731
        "1": "FIRST FRAME: your key drawing (replace FIRST_FRAME.png)", "10": f"{model}: {how}",
        "90": "Frames + the model's native frame rate (output 2)", "91": "Create video at the model's native frame rate",
        "92": "Save video"}
    g = r2_runs.g_wan(FIRST, p_wan, EXAMPLE_SHOT, SEEDS["wan30"][0])
    g["92"]["inputs"]["filename_prefix"] = "video/anime_i2v_wan30_first_frame"
    titled(g, partner_titles("Wan 3.0", "first frame to 2D animation"))
    out.append({"arm": "wan30", "file": "anime_i2v_wan30_first_frame.json", "graph": g, "prompt": p_wan,
                "what": "Wan 3.0, first frame", "input": "first frame",
                "length": f"{g['10']['inputs']['duration']} s at the model's native rate ({FPS['wan30'][0]:g} fps)"})
    g = r2_runs.g_h3api(FIRST, p_h3, EXAMPLE_SHOT, SEEDS["h3api"][0])
    g["92"]["inputs"]["filename_prefix"] = "video/anime_i2v_minimax_h3_api_first_frame"
    titled(g, partner_titles("MiniMax H3 API", "first frame to 2D animation (last frame left empty)"))
    out.append({"arm": "h3api", "file": "anime_i2v_minimax_h3_api_first_frame.json", "graph": g, "prompt": p_h3,
                "what": "MiniMax H3 API, first frame", "input": "first frame",
                "length": f"{g['10']['inputs']['duration']} s at the model's native rate ({FPS['h3api'][0]:g} fps)"})
    adapter_titles = {
        "137": "FIRST FRAME: your key drawing -> <Picture 1> (replace FIRST_FRAME.png)",
        "138": "Prompt: alignment line, then Subject, Action, Camera, Preserve",
        "146": "seq adapter LoRA: 1.0 for seq; set 0 (or bypass) for the H3 open weights on this reference-to-video path",
        "132": f"CLIP LENGTH (seconds) -> frames, snapped to {RULE_TXT} at {RULE_FPS} fps",
        "130": f"Create video at {RULE_FPS} fps (H3's native rate)",
        "127": "MiniMax H3 open weights, reference-to-video (the base the seq adapter was trained on)"}
    g = r2_runs.g_adapter(FIRST, p_seq, EXAMPLE_SHOT, SEEDS["seq124"][0], "seq124")
    g["92"]["inputs"]["filename_prefix"] = "video/anime_seq_adapter_first_frame"
    titled(g, adapter_titles)
    n = frames_from_graph(g)
    assert n == FRAMES["seq124"][0], n
    out.append({"arm": "seq124", "file": "anime_seq_adapter_first_frame_124f.json", "graph": g, "prompt": p_seq,
                "what": "seq adapter, long first-frame generation", "input": "first frame (one reference)",
                "length": f"{n} frames at {RULE_FPS} fps"})
    g = r2s2_runs.g_seq(FIRST, LAST, p_tw, tw_frames, tw_secs, B_SEEDS[0], "anime_seq_adapter_tween")
    g["92"]["inputs"]["filename_prefix"] = "video/anime_seq_adapter_tween"
    titled(g, {**adapter_titles, "139": "LAST FRAME: the next key -> <Picture 2> (replace LAST_FRAME.png)",
               "138": f"Prompt: alignment line, then Subject, Action, Camera, Preserve. If you change CLIP LENGTH, set the "
                      f"Picture 2 mark to (frames - 1) / {RULE_FPS} seconds for the frame count it snaps to"})
    n = frames_from_graph(g)
    assert n == tw_frames == B_FR["seq22"], n
    assert TWEEN_MARK_LINE in g["138"]["inputs"]["value"], "the tween prompt's Picture 2 mark is not (frames - 1) / fps"
    out.append({"arm": "seq22", "file": "anime_seq_adapter_tween_first_last_22f.json", "graph": g, "prompt": p_tw,
                "what": "seq adapter, tween between two keys", "input": "first and last frame (two references)",
                "length": f"{n} frames at {RULE_FPS} fps"})
    for o in out:
        g = o["graph"]
        imgs = [n_["inputs"]["image"] for n_ in g.values() if n_.get("class_type") == "LoadImage"]
        assert all(i in (FIRST, LAST) for i in imgs), imgs
        cv = [n_ for n_ in g.values() if n_.get("class_type") == "CreateVideo"]
        assert len(cv) == 1
        fps_in = cv[0]["inputs"]["fps"]
        if o["arm"] in ("wan30", "h3api"):
            assert fps_in == ["90", 2], fps_in
        else:
            assert float(fps_in) == RULE_FPS == FPS["seq124"][0], fps_in
        txt = json.dumps(g)
        assert "C:" not in txt and "FLOYO" not in txt.upper().replace("_FLOYO", "")
        (WF / o["file"]).write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


# ----------------------------------------------------------------------------------------------- document model
DOC = []          # ("md", text) | ("fig", html, md)
ABSTRACT = {}     # the opening's title, dek and first paragraphs, reused by the repo README


def md(text: str):
    DOC.append(("md", text.strip("\n") + "\n"))


def fig(h: str, m: str):
    DOC.append(("fig", h, m))


VID_N = [0]


def video(src_key: str, poster_key: str | None, caption: str, loop=False, alt="") -> None:
    VID_N[0] += 1
    vid = f"v{VID_N[0]}"
    src = MEDIA_FILES[src_key]
    pst = f' poster="{MEDIA_FILES[poster_key]}"' if poster_key and poster_key in MEDIA_FILES else ""
    pressed = {r: (' aria-pressed="true"' if r == 1 else ' aria-pressed="false"') for r, _ in SPEEDS}
    rates = "".join(f'<button type="button" data-for="{vid}" data-rate="{r}"{pressed[r]}>{lab}</button>' for r, lab in SPEEDS)
    h = (f'<figure class="vid"><video id="{vid}" controls muted playsinline preload="metadata"{" loop" if loop else ""}{pst}'
         f' aria-label="{esc(alt or caption)}"><source src="{src}" type="video/mp4"></video>'
         f'<div class="speed" role="group" aria-label="Playback speed"><span>Speed</span>{rates}</div>'
         f'<figcaption>{caption}</figcaption></figure>')
    fig(h, f"[Video: {html.unescape(re.sub('<[^>]+>', '', caption))}]({src})\n")


SPEEDS = [(1, "1×"), (0.5, "½×"), (0.25, "¼×")]


def image(key: str, alt: str, caption: str = "", cls="img") -> None:
    src = MEDIA_FILES[key]
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    fig(f'<figure class="{cls}"><img src="{src}" alt="{esc(alt)}" loading="lazy" decoding="async">{cap}</figure>',
        f"![{alt}]({src})\n" + (f"\n*{html.unescape(re.sub('<[^>]+>', '', caption))}*\n" if caption else ""))


def table(head: list[str], rows: list[list[str]], align: str = "") -> str:
    al = align or "l" * len(head)
    sep = ["---:" if a == "r" else "---" for a in al]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def stats_block(a: str):
    s = ST[a]
    items = [
        ("Usable as is", f"{s['as_is']} of {s['clips']} clips"),
        ("Shots with an as-is take", f"{s['shots_as_is']} of {s['shots']}"),
        ("With fixes / not usable", f"{s['with_fixes']} / {s['no']}"),
        ("Reads on twos, by eye", " · ".join(f"{k.lower()} {s['twos_eye'].get(k, 0)}" for k in ("Yes", "Partly", "No"))),
        ("Twos score, median", f2(s["twos"])),
        ("Drawings per second, median", f1(s["dps"])),
        ("First-key gap, median", f1(s["first_mad"])),
        ("Seed spread, median", f1(s["spread"])),
        ("Partner fee per clip", usd(s["cost"]) if s["cost"] else "none (Floyo GPU time)"),
        ("Floyo run time, median", secs(s["flotime"])),
        ("Output", f"{s['frames']} frames at {s['fps']:g} fps, {', '.join(s['size'])}"),
    ]
    h = '<dl class="stats">' + "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in items) + "</dl>"
    m = "\n".join(f"- **{k}:** {v}" for k, v in items) + "\n"
    fig(h, m)


# ----------------------------------------------------------------------------------------------- content
AOT = f"[Animating on Twos]({AOT_URL})"


def content(wfs: list[dict]):
    n_sh, n_rt = len(SID), len(ROUTES)
    n_att = len(ATT)
    s42, s1042 = SEEDS["h3api"]
    fr1, fr2 = SEEDS["flux3"]
    top_clips, top_clips_n = leaders("as_is")
    top_shots, top_shots_n = leaders("shots_as_is")
    low_spread, low_spread_v = leaders("spread", hi=False)
    hi_twos, hi_twos_v = leaders("twos")
    cheap, cheap_v = min(((a, ST[a]["cost"]) for a in PAID), key=lambda x: x[1])
    dear, dear_v = max(((a, ST[a]["cost"]) for a in PAID), key=lambda x: x[1])
    fast = min(ROUTES, key=lambda a: ST[a]["flotime"])
    slow = max(ROUTES, key=lambda a: ST[a]["flotime"])
    adapters = ["seq124", "hero124"]
    i2v = [a for a in ROUTES if a not in adapters]
    gap_ad = med([r["first_mad"] for a in adapters for s in SID for r in (row(s, a, sd) for sd in SEEDS[a])])
    gap_i2v = med([r["first_mad"] for a in i2v for s in SID for r in (row(s, a, sd) for sd in SEEDS[a])])
    assert gap_ad > gap_i2v
    assert low_spread == ["wan30"], low_spread          # the prose reads Wan 3.0 as the narrowest
    assert set(hi_twos) == {"seq124"}, hi_twos           # and the seq adapter as the steadiest on twos
    assert ST["flux3"]["as_is"] == 0 and not PLATFORM_ASIS
    # every qualitative claim in the prose below, checked against the data it rests on
    assert top_shots == ["seedance"] and dear == "seedance" and top_clips == ["h3ow"], (top_shots, dear, top_clips)
    assert ST["seq124"]["twos_eye"]["Yes"] == max(ST[a]["twos_eye"]["Yes"] for a in ROUTES)
    assert ST["wan30"]["fix"].most_common(1)[0][0] == "Needs gen AI" and not ST["wan30"]["problems"] and ST["wan30"]["no"] == 1
    assert ST["wan30"]["mood_mean"] == max(ST[a]["mood_mean"] for a in ROUTES)
    assert ST["seq124"]["believable_mean"] == max(ST[a]["believable_mean"] for a in ROUTES)
    assert leaders("first_mad", hi=False)[0] == ["h3ow"]
    assert slow == "seq124" and fast == "flux3" and ST["h3ow"]["flotime"] == min(ST[a]["flotime"] for a in OPEN)
    assert set(ST["seq124"]["look"]) == {"Classic 2D"} and set(ST["seedance"]["look"]) == {"Classic 2D"}
    px = lambda a: max(int(x.split("x")[0]) * int(x.split("x")[1]) for x in ST[a]["size"])  # noqa: E731
    assert px("h3api") == max(px(a) for a in ROUTES)
    asis = lambda s, a: [G[(s, a, sd)]["Usable"] == "As is" for sd in SEEDS[a]]  # noqa: E731
    assert all(asis("01_closeup_train", "h3ow")) and asis("01_closeup_train", "h3api")[0]
    assert all(asis("03b_action_bike_downpour", "h3api")) and all(asis("02_wide_chill_bus_shelter", "seedance"))
    assert all(asis(ROOFS, "hero124")) and all(asis("02_wide_chill_bus_shelter", "wan30"))
    assert {a for a in ROUTES if any(asis("02_wide_chill_bus_shelter", a))} == set(ROUTES) - {"flux3"}
    sp_ad = med([v for a in adapters for v in SPREAD[a].values()])
    assert set(sorted(ROUTES, key=lambda a: ST[a]["spread"])[-2:]) == set(adapters)
    assert sp_ad / ST["wan30"]["spread"] > 2                       # "more than twice as much"
    # the invented hair motion in the train is on Wan 3.0 and on both adapters, by Minta's comments
    assert all(any("hair" in (G[("01_closeup_train", a, sd)].get("Comment") or "") for sd in SEEDS[a]) for a in ("wan30", "seq124", "hero124"))
    global N_SEQ_WF
    N_SEQ_WF = sum(1 for o in wfs if o["arm"].startswith("seq"))
    dated = date.fromisoformat(SHOTS_DOC["date"])

    # the throughlines' evidence, computed
    n_g = len(ALLG)
    use_all = Counter(c["Usable"] for c in ALLG)
    fix_all = Counter(c.get("Fix") for c in ALLG if c.get("Fix"))
    bel_all = Counter(g_int(c, "Believable in-between") for c in ALLG if c.get("Believable in-between"))
    n_classed = sum(LOOK_ALL.values())
    classic = LOOK_ALL["Classic 2D"]
    asis_classic = sum(1 for c in ALLG if c["Usable"] == "As is" and c.get("Looks like") == "Classic 2D")
    assert asis_classic == use_all["As is"]                          # every take usable as is looked classic 2D
    assert classic / n_classed > 0.85 and asis_classic / classic < 0.5   # nearly all looked right; fewer than half of those usable as is
    rejected = [c for c in ALLG if c["Usable"] == "No"]
    rej_noflag = [c for c in rejected if not probs(c)]
    assert 0.4 < len(rej_noflag) / len(rejected) < 0.5               # "nearly half of the takes she rejected carry no flag"
    assert 0.30 < use_all["With fixes"] / n_g < 0.36                  # "about a third of the takes were usable with a fix"
    use_twos = {k: med([row(s, a, sd)["twos_score"] for a in ROUTES for s in SID for sd in SEEDS[a] if G[(s, a, sd)]["Usable"] == k])
                for k in ("As is", "With fixes", "No")}
    use_n = {k: use_all[k] for k in use_twos}
    eye = {k: med(v) for k, v in EYE_TWOS.items()}
    assert max(use_twos.values()) - min(use_twos.values()) < 0.1     # "nearly the same median twos score"
    assert abs(eye["Yes"] - eye["No"]) < 0.05                        # on twos and not on twos, by her eye: the same median
    assert eye["Partly"] < min(eye["Yes"], eye["No"]) - 0.15          # "partly" is the outlier, well below both
    by_twos = sorted(ROUTES, key=lambda a: -ST[a]["twos"])
    assert by_twos[:2] == ["seq124", "flux3"], by_twos              # FLUX 3: the second-highest twos score, no take as is
    n_roofs = sum(len(SEEDS[a]) for a in ROUTES)
    assert len(ROOFS_ZERO) / n_roofs > 0.5                            # "more than half of the roofs takes score zero"
    assert min(ROUTES, key=lambda a: SPREAD[a]["01_closeup_train"]) == "seedance"   # the train pair Minta saw range in measures closest
    WAN_REPEAT = ["01_closeup_train", "04_zoom_out_roofs", "05b_fisheye_peephole"]   # her second-attempt comments open on "same"
    assert all((G[(s, "wan30", SEEDS["wan30"][1])].get("Comment") or "").lower().startswith("same") for s in WAN_REPEAT)
    assert ST["h3ow"]["with_fixes"] < min(ST["h3ow"]["as_is"], ST["h3ow"]["no"])     # the open weights: fine or gone
    assert G[(ROOFS, "h3ow", SEEDS["h3ow"][0])]["Usable"] == "No"                     # the roofs witness for "total" is a rejected take
    assert G[(PLATFORM, "h3ow", SEEDS["h3ow"][0])]["Usable"] == "No"
    flux_smooth = [sd for s in SID for sd in SEEDS["flux3"] if "smooth" in (G[(s, "flux3", sd)].get("Comment") or "")]
    assert len(flux_smooth) == 1                                                      # "its smoothest take smears": one take called smooth
    assert "Nano Banana" in SHOTS_DOC["still_source"] and "Floyo" in SHOTS_DOC["still_source"]
    hero_bw = G[("03b_action_bike_downpour", "hero124", SEEDS["hero124"][0])].get("Comment") or ""
    assert "lineweight" in hero_bw and "black and white lineart" in hero_bw          # the hero note is about lineweight, from B/W training
    assert FRAMES["hero124"][0] != MODEL_CARD_HERO["frames"] and MODEL_CARD_HERO["refs"] == 1
    assert all("hallucinations" in (G[(ROOFS, "h3api", sd)].get("Comment") or "") for sd in SEEDS["h3api"])
    seq_no = [G[(s, "seq124", sd)] for s in SID for sd in SEEDS["seq124"] if G[(s, "seq124", sd)]["Usable"] == "No"]
    assert seq_no and all("Morphing" in probs(c) for c in seq_no)                   # its rejects fail in the drawing
    assert not any("Wrong timing" in probs(G[(s, "seq124", sd)]) for s in SID for sd in SEEDS["seq124"])
    sd_miss = [G[(s, "seedance", sd)] for s in SID for sd in SEEDS["seedance"] if G[(s, "seedance", sd)]["Usable"] != "As is"]
    sd_timing = [c for c in sd_miss if "Wrong timing" in probs(c) or re.search(r"timing|abrupt|slow mo", c.get("Comment") or "")]
    assert len(sd_timing) / len(sd_miss) > 0.5                       # "most of its misses were timing misses"
    roofs_action = [a for a in ROUTES if any(re.search(r"hallucinat|what shes doing|puppet|raises the sheet|lifts the sheet",
                                                       G[(ROOFS, a, sd)].get("Comment") or "") for sd in SEEDS[a])]
    assert len(roofs_action) > n_rt / 2, roofs_action                 # "most routes gave her an action that does not read"
    assert ST["flux3"]["look"].get("Vector", 0) > max(ST[a]["look"].get("Vector", 0) for a in ROUTES if a != "flux3")
    assert ST["flux3"]["problems"].get("AI smear / ghosting", 0) > max(ST[a]["problems"].get("AI smear / ghosting", 0) for a in ROUTES if a != "flux3")
    rej_kinds = {(s, a, sd) for a in ROUTES for s in SID for sd in SEEDS[a] if G[(s, a, sd)]["Usable"] == "No" and not probs(G[(s, a, sd)])}
    assert ("05b_fisheye_peephole", "h3api", SEEDS["h3api"][0]) in rej_kinds and (ROOFS, "seedance", SEEDS["seedance"][1]) in rej_kinds
    assert ("02_wide_chill_bus_shelter", "flux3", SEEDS["flux3"][0]) in rej_kinds and ("03b_action_bike_downpour", "seedance", SEEDS["seedance"][0]) in rej_kinds
    one_s = w(round(SEG["seconds"]))
    assert one_s == "one"

    # ---------------------------------------------------------------- header and opening
    S = lambda text, s, a, i, *e: said(text, s, a, SEEDS[a][i], *e)  # noqa: E731   (attempt index, not seed)
    commercial = list(PAID)
    assert commercial == ["h3api", "seedance", "flux3", "wan30"] and [a for a in ROUTES if a not in commercial] == ["h3ow"] + adapters
    # the first frames are generated from hand-drawn references (the final pass corrects an opening that called them hand-drawn)
    refs_note = said_in("working from hand-drawn references", FINAL_PASS, "used hand drawn references for nano",
                        "the first frames are generative")
    title = f"Full-shot anime generation: which video models move convincingly, and a {one_s}-second repair"
    dek = (f"Anime can now be generated a whole shot at a time, scene and motion together, from a single first frame. We compared "
           f"{w(n_rt)} video routes on {w(n_sh)} shots to find the ones that move convincingly with little work from an artist on the "
           f"in-betweens, then regenerated {one_s} second of a failed take with our seq adapter.")
    opening = [
        said_in("""This study is about generating anime one full shot at a time. Each shot begins with a single generated frame, not layers
or line art. A video model renders the whole scene from it: character, background and motion together. What we set out to find is
which models move convincingly in this style with little direct work from an artist on
the in-betweens, the drawings that carry a movement from one pose to the next. Anime usually holds each of those drawings for two
frames, a rhythm called twos, which gives twelve drawings to every second of screen. Keeping that cadence, and keeping the motion
believable, is what we judged every take on.""", FINAL_PASS, "full shot generation", "layered or line art first",
                "little direct artist intervention on the tweens"),
        said_in(f"""Some teams want methods built deliberately into a multi-stage animation pipeline, where an artist draws the keys and
adapters trained on hand-drawn animation fill in the drawings around them. Our paper {AOT} covers that approach. This case study is
written for AI-native teams. It stays with shots generated whole, on tools a studio can open today, and asks how well each route
keeps the cadence and the believability of the motion.""", FINAL_PASS, "multi stage animation pipeline", "AI native teams",
                "maintaining the cadence and believability"),
        f"""We gave {w(n_rt)} routes the same character and the same {w(n_sh)} first frames, which we generated with Google's Nano Banana
image models, {refs_note}. A route here is a model together with the way it is run. Every route ran as a ComfyUI workflow on Floyo:
ComfyUI is the node-graph tool in which most open video pipelines are built, and Floyo runs ComfyUI workflows in the cloud. The
{w(n_rt)} routes are:""",
        f"""- {W(len(commercial))} commercial models called through Floyo's partner nodes: {join_names(commercial)}
- MiniMax H3, run from its open weights
- {W(len(adapters))} adapters we trained on top of those weights, the seq adapter and the hero adapter""",
        f"""Each route animated each first frame {'twice' if n_att == 2 else w(n_att) + ' times'}, into a {CLIP_S}-second shot. Our Creative
Lead, Minta Carlson, graded every take by eye, and we measured every take frame by frame. Then we set out to repair one take that
went wrong. Its knock hits the peephole lens instead of the door, so we regenerated that second with our seq adapter as a tween to cut
back into the take.""",
    ]
    ABSTRACT.clear()
    ABSTRACT.update(title=title, dek=dek, opening=opening, date=dated)
    md(f"""
# {title} {{#top}}

*{dek}*

**{ORG}** · {', '.join(AUTHORS)} · {dated.strftime('%B %Y')}

""" + "\n\n".join(opening) + "\n")

    # ---------------------------------------------------------------- what we found
    rows_ = []
    for a in ROUTES:
        s = ST[a]
        rows_.append([f"**{NAME[a]}**" if a in FINAL_WF else NAME[a], f"{s['as_is']} of {s['clips']}", f"{s['with_fixes']} of {s['clips']}",
                      f"{s['shots_as_is']} of {s['shots']}", f2(s["twos"]), usd(s["cost"]) if s["cost"] else "none", secs(s["flotime"]),
                      "yes" if a in FINAL_WF else "–"])
    final_four = said_in(f"Our final four came from the grading: {NAME['wan30']}, the H3 open weights, the H3 API and our seq adapter were the "
                         "smoothest, though each had high and low points.", BRIEF,
                         "Wan3, H3 Open Weights, H3 API and our Seq Adapter were the smoothest", "all had high and low points")
    assert FINAL_WF == ["wan30", "h3ow", "h3api", "seq124"]
    md(f"""
## What we found {{#summary}}

Five findings came out of the study, the first four from the comparison and the fifth from the repair:

1. **[The drawing holds before the motion does.](#motion)** Nearly every take looked hand-drawn. What separated the usable ones was
   motion: timing, contact, and movement the shot gives no reason for.
2. **[Each route fails differently, and the kind of failure decides the fix.](#signatures)** The four routes we found smoothest do not
   line up from best to worst. Each tends toward a different kind of failure, and each kind has a different repair.
3. **[Two attempts that agree may show the model's bias, not its control.](#range)** {NAME['wan30']} varied least between attempts and
   tended to repeat its flaws. A second attempt buys little there, and a frame-level fix buys a lot.
4. **[The measurements describe rhythm; the eye judges the take.](#measures)** Our cadence score did not predict which takes we would
   use. It is a good check on rhythm and on a cleanup, and it cannot judge whether a take works.
5. **[Generating a shot and repairing a second of it are different jobs.](#repair)** The commercial routes are built for whole shots,
   and none of them will render a {one_s}-second span, including {NAME['seedance']}, which covered the most shots. The H3 open weights
   and our seq adapter can, and they repair in different ways.

{final_four} Those four, in bold below, are the routes we publish [workflows](#workflows) for. The seq adapter gets {w(N_SEQ_WF)} of
them: a long generation from a first frame, and a short tween, the in-betweens that carry one key to the next. The measures are
defined under [How we measured](#method).

{table(["Route", "Usable as is", "Usable with fixes", "Shots with an as-is take", "Twos score (median)", "Partner fee per clip", "Floyo run time (median)", "Workflow"], rows_, "lrrrrrrl")}""")

    # ---------------------------------------------------------------- Part A: setup
    look = SHOTS_DOC["look_to_hold"]
    look_items = [said_in("thin, grainy pencil line", look, "thin grainy pencil line"),
                  said_in("flat, pale colour laid in loosely", look, "flat pale colour laid in loosely"),
                  said_in("almost no shading, a soft round blush and lots of paper white", look, "almost no shading", "soft round blush",
                          "lots of paper white"),
                  said_in("motion that keeps the drawn look, with no added rendering, gloss or depth of field", look,
                          "Motion should keep this drawn look", "no added rendering, gloss or depth-of-field")]
    md(f"""
## Part A · {W(n_rt)} routes, {w(n_sh)} shots {{#part-a}}

### The setup {{#setup}}

The character is Emmy: {SHOTS_DOC['character'].split('(', 1)[1].rstrip(')')}. Every route was asked to hold one look, the hand-drawn
lofi anime of the first frames:

""" + "".join(f"- {x}\n" for x in look_items) + f"""
Each shot is a generated first frame, which we call the still, and a {CLIP_S}-second scene written as timed beats with a list of what
must hold. Every route was given the still as its first frame (image-to-video from a first frame) and made {w(n_att)} attempts at every
shot. An attempt is fixed by its seed, the number that sets a model's random starting point. The same seed with the same inputs gives
the same take, and a new seed gives a new one. We used seeds {s42} and {s1042}. FLUX 3 takes no seed, so its {w(n_att)} attempts are
fresh rolls.

The four commercial routes ran through Floyo's partner nodes, ComfyUI nodes that send the job to the vendor's service and charge a fee
per clip. The other three ran on Floyo's GPUs. The H3 open weights are the weights MiniMax published, which can be run on hardware you
own or rent and adapted under MiniMax's license. Our two adapters are LoRAs, small sets of trained weights loaded on top of a base
model to steer it. We trained them on hand-drawn animation:

- the seq adapter surfaces short sequences held on twos
- the hero adapter draws the next key pose from the current one

{AOT} describes how they were made.

{table(["Shot", "Type", "Scene"], [[f"**{short(s)}**", SHOT[s]["shot_type"], SHOT[s]["scene"]] for s in SID])}
We wrote each route's prompt from its vendor's guide, the same brief in each model's dialect ([How each route was driven](#driven) has
the detail). Each take was graded on one sheet:

- believable in-betweens, from {GRADE_MIN} to {GRADE_MAX}
- whether it reads on twos
- what it looks like: classic 2D, digital 2D or vector
- the problems in it
- how it could be fixed
- its lofi-anime mood, from {GRADE_MIN} to {GRADE_MAX}
- whether it is usable as is, with fixes, or not at all
""")

    # ---------------------------------------------------------------- 1. motion
    n_bel, n_fix, n_eye = sum(bel_all.values()), sum(fix_all.values()), sum(EYE_ALL.values())
    assert set(bel_all) <= {3, 2, 1, 0} and set(fix_all) <= {"None needed", "Needs gen AI", "By hand", "Not fixable"}
    assert sum(use_all.values()) == n_g and n_eye == n_g and n_bel < n_g and n_fix < n_g   # the rows that count fewer say so
    of = lambda n, what: "" if n == n_g else f" (of {n} {what})"  # noqa: E731
    tally = [
        ["Looks like: classic 2D / digital 2D / vector", f"{LOOK_ALL.get('Classic 2D', 0)} / {LOOK_ALL.get('Digital 2D', 0)} / {LOOK_ALL.get('Vector', 0)}{of(n_classed, 'classed')}"],
        ["Believable in-between: 3 / 2 / 1 / 0", " / ".join(str(bel_all.get(k, 0)) for k in (3, 2, 1, 0)) + of(n_bel, "graded")],
        ["Reads on twos: yes / partly / no", f"{EYE_ALL['Yes']} / {EYE_ALL['Partly']} / {EYE_ALL['No']}{of(n_eye, 'read')}"],
        ["Usable: as is / with fixes / no", f"{use_all['As is']} / {use_all['With fixes']} / {use_all['No']}"],
        ["Fix: none needed / regenerate with gen AI / by hand / not fixable",
         f"{fix_all['None needed']} / {fix_all['Needs gen AI']} / {fix_all['By hand']} / {fix_all['Not fixable']}{of(n_fix, 'marked')}"],
    ] + [[f"Flagged: {p.lower().replace('ai smear', 'AI smear')}", str(n)] for p, n in PROB_ALL.most_common()]
    assert all(G[(ROOFS, "hero124", sd)]["Usable"] == "As is" for sd in SEEDS["hero124"])
    md(f"""
### 1. The drawing holds before the motion does {{#motion}}

Almost every take looked like hand-drawn anime. Of the takes we classed by look, nearly all read as classic 2D, the pencil and flat
colour of the stills, and every take we would use as is was one of them. Yet fewer than half of the classic-2D takes were usable as is.
Looking right was necessary and far from sufficient: the takes that failed did so between the frames, in how the drawings moved.
{AOT} describes that gap for one model, and here it ran through every route we tried.

Read together, the grading comments add up to a checklist for reviewing generated animation:

1. **Believable in-betweens.** Whether the drawings between the poses are ones an animator would make. Good ones carry acting, as
   {NAME['wan30']} did on the bus shelter, {S("where the take has nuance and a natural rhythm", "02_wide_chill_bus_shelter", "wan30", 0, "nuance and natural beats")}.
   Weak ones move like a rig: {S("on the roofs, a FLUX 3 take made Emmy look like a puppet", ROOFS, "flux3", 1, "puppet")}.
2. **Holding on twos.** Anime is limited animation, with fewer drawings held for longer, and the holds are what make its motion feel
   intended. {S("Without them everything moves at one speed and the motion loses its sense of intent, as in a FLUX 3 take of the peephole", "05b_fisheye_peephole", "flux3", 1, "moves at the same speed", "unintentionality")}.
3. **Line and look.** Classic 2D, the hand-drawn look of the stills, against digital 2D and vector. Vector here means the eased or
   constant-speed interpolation of motion graphics, in which shapes slide instead of being redrawn.
   {S("One FLUX 3 take of the bus shelter came out as clunky vector", "02_wide_chill_bus_shelter", "flux3", 1, "clunky vector")}, and
   {S("one hero take was light in lineweight", "03b_action_bike_downpour", "hero124", 0, "light in lineweight")}.
4. **AI smears and ghosting.** Frames that cross-dissolve two drawings into one. A drawn smear frame is different, because an animator
   places it on purpose to sell speed. {S("A smooth FLUX 3 take on the train has smeared frames that anyone who knows what to look for will spot", "01_closeup_train", "flux3", 1, "smearing", "someone who knows what theyre looking at")}.
5. **Morphing and drift.** Shapes that swell, a character who stops being herself, line that boils from frame to frame, colour that
   flickers. {S("On the roofs, one Seedance take lets the blanket grow far bigger", ROOFS, "seedance", 1, "blanket gets way bigger")}.
6. **Motion the shot gives no reason for.** {NAME['wan30']} blew Emmy's hair about inside a closed [train](#shot-01) carriage,
   {S("the kind of invented motion an animator would not draw", "01_closeup_train", "wan30", 0, "hair blowing inside the train car", "hallucination that an animator wouldnt make")}.
   On the [roofs](#shot-04), {S("the H3 API had her go through the motions of folding without folding anything", ROOFS, "h3api", 0, "not really folding anything")}.
7. **Timing.** Slow motion, rushed beats, endings that stop short of a settle. {S("Seedance played the station platform in slow motion", PLATFORM, "seedance", 0, "slow mo")}.
8. **Contacts.** Where a hand, a foot or a fist lands. The peephole knock was meant for the door.
   {S("The H3 API's first attempt punched the lens", "05b_fisheye_peephole", "h3api", 0, "punches the peep hole")}, while
   {S("the open weights placed the fist better", "05b_fisheye_peephole", "h3ow", 0, "better placement for the fist")} (the
   [peephole](#shot-05b) grids show both).
9. **Range.** How different two attempts are, which decides whether a second attempt is worth paying for. The section on
   [range](#range) takes it up.

Most of these happen across frames, not within one, which is how a take whose every frame passes can still fail. The problems flagged
on the sheet show the same thing from the other side, because they cover only part of what went wrong. Nearly half of the rejected
takes carry no flag at all. They fail on things only the comments record:

- timing
- invented motion
- a misplaced fist
- a slide into vector
- a blanket that grows

{table(["What the sheet recorded, over the " + str(n_g) + " takes", "Takes"], tally, "lr")}
Restraint counts as much as invention. On the slow pull-back over the roofs, most routes gave Emmy an action that does not read. The
hero adapter had her do very little, and we graded both of its attempts usable as is,
{S("because an action that makes no sense distracts more than a quiet one", ROOFS, "hero124", 0, "doing too much of something senseless", "more distracting")}.

The consequence for a studio is where review time goes. A single frame from almost any of these takes would pass inspection, so a take
has to be judged in motion, at speed and slowed down, with the checklist above in mind. The speed buttons under every grid in this study
are there for that.
""")

    # ---------------------------------------------------------------- 2. signatures
    sig = {
        "wan30": ("come close, and the same way twice", "regenerate the frames that are wrong"),
        "h3ow": ("hold a shot or lose it", "a take it holds rarely needs work; a take it loses needs another attempt or another route"),
        "h3api": ("invent action and contact", "edit: retime frames onto twos, repair the invented beat"),
        "seq124": ("keep the motion and, when it fails, lose the drawing", "repair the drawing where the motion is worth keeping"),
    }
    sig_rows = [[f"**{NAME[a]}**", f"{ST[a]['as_is']} · {ST[a]['with_fixes']} · {ST[a]['no']}", sig[a][0], sig[a][1]] for a in FINAL_WF]
    md(f"""
### 2. Each route fails differently {{#signatures}}

The four routes we found smoothest do not line up from best to worst, and none of them won every shot. The station platform, where
Emmy jumps down the last stairs and runs for a closing train door, beat all {w(n_rt)} routes. None produced a take of it usable as is,
{S("because the shot has too much small detail moving at speed for most models", PLATFORM, "flux3", 0, "too much small detail at higher speed for most models")}
(the [platform grids](#shot-03e) show each route losing it in a different way). What distinguishes the four is the kind of failure each
tends toward. For a studio that matters more than a ranking, because the kind of failure decides the fix.

{table(["Route", "As is · fixes · no", "Tends to", "The fix it points to"], sig_rows, "lrll")}
{NAME['wan30']} is the steadiest of the four on mood and the least likely to lose a take outright: we rejected only
{w(ST['wan30']['no'])} of its takes and flagged no problem on any of them. Its weakness is that it comes close in the same way twice,
which the section on [agreement between attempts](#range) takes up. For most of its takes, the fix the grading named was to regenerate
frames.

The H3 open weights gave more takes usable as is than any other route and started closest to the still. Their failures are total, not
partial: {S("on the platform too many of their frames are bad", PLATFORM, "h3ow", 0, "too many frames are bad")}, and
{S("on the roofs one attempt goes wrong as she lifts the sheet", ROOFS, "h3ow", 0, "when she lifts the sheet")}.

The H3 API tends to invent. It gave Emmy something to do on the roofs that she is not doing, on both attempts, and its first peephole
attempt drove the fist into the lens. {S("On the platform its characters fell apart, with flickering and degradation", PLATFORM, "h3api", 0, "characters fall apart", "flickering and degradation")}.
Its cadence also slips off twos in places, {S("and there an edit that retimes the frames is a fine solution", "01_closeup_train", "h3api", 1, "editing is a fine solution")}.

FLUX 3 sits outside the four for a plain reason: it gave no take usable as is, and its takes slid into vector and smeared their
in-betweens more often than any other route's. {NAME['seedance']} sits outside for a subtler one, and it makes the clearest contrast
with the seq adapter. Seedance gave a take usable as is on more shots than any other route and still missed our four, because most of
its misses were timing misses. {S("That is a pattern we saw in the strongest commercial models: high fidelity, with timing that can be very strange", "03b_action_bike_downpour", "seedance", 0, "SOTA models", "high fidelity but the timing can be so strange")}.

The seq adapter is the opposite case. It held twos most steadily of all {w(n_rt)} routes and earned the highest in-between grades. It
also added life the brief did not ask for: on the [bike](#shot-03b)
{S("it was the first route to add trees to the scene, and they are a fine touch", "03b_action_bike_downpour", "seq124", 0, "first one to add really nice tree elements")}.
Where it failed, the drawing gave way while the motion held. Every rejected take of it is flagged for morphing, and
{S("on its second platform attempt the movement is good while everything else fails to hold up", PLATFORM, "seq124", 1, "movementis good", "not standing up otherwise")}.

Seedance keeps the drawing and loses the timing, and the seq adapter keeps the timing and sometimes loses the drawing. Which of the two a
production can repair more cheaply is the real choice between them.
""")

    # ---------------------------------------------------------------- 3. range
    w1, w2 = SEEDS["wan30"]
    sh2, bk, pp = "02_wide_chill_bus_shelter", "03b_action_bike_downpour", "05b_fisheye_peephole"
    wan_rows = [
        [short(sh2), S("Subtle acting and natural beats", sh2, "wan30", 0, "nuance and natural beats"),
         S("Also good, and very close to attempt 1", sh2, "wan30", 1, "also good", "very consistent")],
        [short("01_closeup_train"), S("Hair blowing inside the closed carriage", "01_closeup_train", "wan30", 0, "hair blowing inside the train car"),
         S("The same flaw, less pronounced", "01_closeup_train", "wan30", 1, "same comment as the last", "less agregious")],
        [short(bk), S("A good ending and a weaker beginning", bk, "wan30", 0, "the end of this one is good", "the beginning not as much"),
         S("Better, and very similar to attempt 1", bk, "wan30", 1, "this is better", "how similar it is to the last seed")],
        [short(ROOFS), S("Trouble with how she raises the sheet", ROOFS, "wan30", 0, "issues with how shes raises the sheet"),
         S("The same trouble, as expected", ROOFS, "wan30", 1, "same unsurprisingly")],
        [short(pp), S("Good animation, with the fist landing somewhere odd", pp, "wan30", 0, "good animation", "fist ends up somewhere weird"),
         S("The same fist problem in an otherwise good animation", pp, "wan30", 1, "same issue but otherwise a good animation")],
    ]
    spread_rows = [[NAME[a], f1(ST[a]["spread"])] for a in sorted(ROUTES, key=lambda a: ST[a]["spread"])]
    assert G[(PLATFORM, "seq124", SEEDS["seq124"][0])]["Usable"] != "No" and G[(PLATFORM, "seq124", SEEDS["seq124"][1])]["Usable"] == "No"
    md(f"""
### 3. Agreement between attempts can be the model's bias {{#range}}

It is tempting to read two attempts that agree as a model that understood the prompt. The grading read {NAME['wan30']}'s agreement the
other way. On the [bus shelter](#shot-02) its two attempts were very consistent,
{S("and we tend to read consistency like that as heavy-handed pretraining that keeps a model's output stylistically uniform", sh2, "wan30", 1, "heavy handed pretraining", "stylistically consistent")}.
{S("That is a strength while the shot sits where the model is biased, and a risk for a precise, complex shot it is not biased toward", sh2, "wan30", 1, "precise, complex shot", "biased towards")}.
The notes on the other shots bear that out. On the train, the roofs and the [peephole](#shot-05b), the second attempt carries the first
one's flaw. On the bike {S("the two attempts are so alike that we read a sizeable bias in the model", bk, "wan30", 1, "how similar it is to the last seed", "sizeable bias")}:

{table([f"{NAME['wan30']}, by shot", f"Attempt 1 (seed {w1})", f"Attempt 2 (seed {w2})"], wan_rows)}
The measurement agrees with the grading here. Seed spread is our measure of how different a route's two attempts look over the length
of the clip. It is lowest for {NAME['wan30']} and highest for the two adapters, whose attempts differ by more than twice as much.

{table(["Route", "Seed spread (median)"], spread_rows, "lr")}
A wide route rewards the second attempt instead. On the [platform](#shot-03e) the seq adapter gave both the best take of the shot and a
rejected one. {S("One seed told the most coherent story of any take there", PLATFORM, "seq124", 0, "most coherent narrative")}; the other
was not usable. Commercial models can show range too. On the train,
{S("Seedance's two attempts showed real range between generations with no adapter involved", "01_closeup_train", "seedance", 1, "range between gens on the frontier models", "wihtout adapters")},
although our measure scores that pair as the closest of the shot. We come back to that disagreement under [measurement](#measures).

For a studio the consequence is practical. A second attempt from a narrow model returns the same answer with the same mistake, so the fix
for {NAME['wan30']} is to regenerate the frames that are wrong and keep the rest of the take.
""")

    # ---------------------------------------------------------------- 4. measures
    meas_rows = [[f"Usable {k.lower()}" if k != "No" else "Not usable", str(use_n[k]), f2(use_twos[k])] for k in ("As is", "With fixes", "No")]
    meas_rows += [[f"Reads on twos: {k.lower()}", str(len(EYE_TWOS[k])), f2(eye[k])] for k in ("Yes", "Partly", "No")]
    md(f"""
### 4. The measurements describe rhythm; the eye judges the take {{#measures}}

We measured every take, and the measurements are good at what they measure:

- The **twos score** is the share of a clip's frame-to-frame steps that follow a strict hold, move, hold, move pattern. It separates a
  clip held on twos from one drawn on ones, with a new drawing on every frame.
- The **first-key gap** says how far a take starts from the still it was given.
- The **seed spread** says how different two attempts look.

None of them predicts which takes we would use. Takes graded usable as is, usable with fixes and not usable have nearly the same median
twos score. FLUX 3 has the second-highest median twos score of the {w(n_rt)} routes, yet no take usable as is. Even the read of twos
by eye agrees with the score only in part. The takes read as partly on twos score well below the rest, while the takes read as fully on
twos and as not on twos at all score alike.

{table(["Takes, by grade", "Takes", "Twos score (median)"], meas_rows, "lrr")}
Part of the gap comes from how the score is built. It reads the whole frame, so rain, passing scenery or a camera move animated on ones
reads as ones even when the character holds on twos. On the slow pull-back over the [roofs](#shot-04), more than half of the takes score
zero. The larger part is that the score measures rhythm and cannot see quality: a take can hold perfect twos and still put the fist
through the lens. Range has the same limit. In the Seedance train pair the eye saw real variety, and a whole-frame difference scores the
pair as the closest of the shot. The per-shot tables in the Reference set the twos score beside the grades for every take.

We drew the same line in {AOT}, and it holds here. The numbers still did useful work in this study:

- They set the gate for the twos-clean trick in Part B, which cleans only takes the score finds properly on twos, and they checked that
  the clean did what it should.
- The first-key gap showed that the adapters start further from the still than the image-to-video routes do.
- The seed spread agreed with the grading that {NAME['wan30']}'s two attempts sit close together.

Whether a take works is a judgement for a trained eye.
""")

    # ---------------------------------------------------------------- Part B
    src_name = NAME[SRC_ROW["arm"]]
    h3 = [brow("h3ow22", sd) for sd in B_SEEDS]
    sq = [brow("seq22", sd) for sd in B_SEEDS] + [brow("seq39", sd) for sd in B_SEEDS]
    mark22 = MARK22
    assert mark22 == k2k(B_FR["seq22"], RULE_FPS)
    # Part B states durations key to key only: the span, the tweens and the alignment mark use one convention
    assert f"{SEG['seconds']:.2f}" == k2k(SPAN_FRAMES, SRC_ROW["fps"]), (SEG["seconds"], SPAN_FRAMES)
    gap_h3 = med([r[k] for r in h3 for k in ("first_mad", "last_mad")])
    gap_sq = med([r[k] for r in sq for k in ("first_mad", "last_mad")])
    assert round(gap_h3) < round(gap_sq)
    assert SRC_MET['flicker'] > max(r['flicker'] for r in h3) > max(r['flicker'] for r in sq)          # the light pumps least on seq
    assert max(max(r['first_mad'], r['last_mad']) for r in h3) < min(min(r['first_mad'], r['last_mad']) for r in sq)  # pinned vs redrawn
    assert min(brow("seq22", sd)["twos_score"] for sd in B_SEEDS) > max(r["twos_score"] for r in h3)   # seq holds twos more steadily
    assert FLOOR["seedance"] == min(FLOOR[a] for a in ("h3api", "seedance", "flux3")) and min(FLOOR.values()) > SEG["seconds"]
    # the span's description, read from the span record: the keys' poses, and both knocks inside the span and on the lens
    assert f"Frame {A0}: fist cocked high beside the lens" in SPAN["why"] and f"Frame {B0}: the hand coming down to her chest" in SPAN["why"]
    knocks = [int(x) for x in re.search(r"Knocks land on the lens at frames (\d+) and (\d+)", SPAN["why"]).groups()]
    assert all(A0 < k < B0 for k in knocks)
    assert "$0 validation probes" in PROBE["MEASURED_AT"] and "nothing could execute" in PROBE["MEASURED_AT"]   # the floors: probes, no render
    assert NOT_TC and all(r["twos_score"] < TWOS_MIN for r in NOT_TC)                 # the takes left as generated
    md(f"""
## Part B · Redoing keyframes with the seq adapter {{#part-b}}

### 5. Generating a shot and repairing a second are different jobs {{#repair}}

About a third of the takes in Part A were usable with a fix, and the fixes the grading named are local:

- {S("rerun the weak opening frames of a " + NAME["wan30"] + " take on the bike", bk, "wan30", 0, "the beginning not as much", "rerunning those frames")}
- {S("redo the end of a " + NAME["wan30"] + " take on the platform", PLATFORM, "wan30", 1, "the end needs to be redone")}
- {S("drop one frame during Emmy's head turn from a hero take on the bus shelter", sh2, "hero124", 1, "one frame during her head turn")}
- {S("slow a few frames of an H3 API take on the train to put it back on twos", "01_closeup_train", "h3api", 1, "slow down certain frames to get it on 2s")}

Those are repairs, not generations, and a repair asks different things of a model. It has to:

- render a short span
- start and end on frames that already exist in the take (its keys)
- hold the cadence of the take around it, so that the cut does not show

#### The take {{#problem}}

On the peephole shot, Emmy is meant to knock on the door. In the {src_name}'s first attempt (seed {SRC_ROW['seed']}) her fist goes
straight into the lens and fills the view. {S("The take is charming, but the punch on the peephole is plainly unintended, the kind of contact that marks lazy AI", SPAN_SHOT, SRC_ROW['arm'], SRC_ATT, "cute", "punches the peep hole", "unintentional", "hallmark of lazy ai")}.
We chose this take for the test: keep it, regenerate the knock between two of its frames, and have the fist strike the door beside the
peephole.

#### The span {{#span}}

We cut the span from frame {A0}, where her fist is cocked high beside the lens, to frame {B0}, where her hand comes down to her chest.
That {one_s} second holds both knocks, and both land on the lens.
""")
    h = (f'<div class="pair"><figure class="img"><img src="{MEDIA_FILES["partb:keyA"]}" alt="First key, frame {A0}" loading="lazy" decoding="async">'
         f'<figcaption>First key: frame {A0} of the source take</figcaption></figure>'
         f'<figure class="img"><img src="{MEDIA_FILES["partb:keyB"]}" alt="Last key, frame {B0}" loading="lazy" decoding="async">'
         f'<figcaption>Last key: frame {B0}</figcaption></figure></div>')
    fig(h, f"![First key, frame {A0}]({MEDIA_FILES['partb:keyA']})\n\n![Last key, frame {B0}]({MEDIA_FILES['partb:keyB']})\n")
    floors = [[NAME[a], f"{FLOOR[a]} s"] for a in ("h3api", "seedance", "flux3", "wan30")]
    floors.append(["H3 open weights and the seq adapter", f"any {RULE_TXT} frames at {RULE_FPS} fps ({B_FR['seq22']} frames span {k2k(B_FR['seq22'], RULE_FPS)} s key to key)"])
    assert "Handling: (c) Only the routes that can make the span" in CHECK      # the routes chosen at the frame check (an option ticked)
    assert re.search(rf"Lengths: {B_FR['seq39']} f", CHECK)                     # and the longer seq example
    edits_note = said_in("This is worth knowing before a pipeline depends on a partner model: the commercial models are built to generate "
                         "whole shots, and their length floors rule them out of small frame-to-frame edits.", FINAL_PASS,
                         "arent the best for small frame to frame edits", "useful to know going in")
    md(f"""
#### Why only two routes could run it {{#floors}}

Here the two jobs part company. A {one_s}-second repair needs a route that will render {one_s} second, and the partner nodes will not.
Each has a floor on clip length. {NAME['seedance']}, the route that covered the
most shots in Part A, will not go below {w(FLOOR['seedance'])} seconds. Generating a longer clip and keeping part of it would change the
timing between the two keys. The H3 open weights, and the seq adapter on top of them, render any length of the form {RULE_TXT} frames.

{table(["Route", "Shortest clip it accepts (the node's duration setting)"], floors)}
{edits_note} So we ran only the two routes that can render the span, and added a longer seq take to see a slower knock.

#### What ran {{#ran}}

We made {w(len(B_ROWS))} takes, all given the span's first and last frames, with seeds {B_SEEDS[0]} and {B_SEEDS[1]} for each of these:

- the H3 open weights at {B_FR['h3ow22']} frames
- the seq adapter at {B_FR['seq22']} frames, the nearest length the rule allows below the span's {SPAN_FRAMES}, so these tweens play
  slightly faster than the original
- the seq adapter at {B_FR['seq39']} frames, the next length up, for a slower knock

{said_in("The prompt asked for two knocks on the door beside the peephole, with her fist swinging forward to strike the door just past the right edge of the lens and drawing back each time.", PR2['seq']['B'], "her fist swinging forward to strike the door just past the right edge of the lens and drawing back each time")}

The two routes take the keys differently:

- The H3 open weights ran on MiniMax's first-and-last-frame graph, which treats the two frames as fixed endpoints and fills the frames
  between them.
- The seq adapter ran on the reference-to-video graph, which treats them as references, drawings to work from instead of frames to
  reproduce. An alignment line in the prompt places them in time (“Picture 2 aligns with the {mark22}-second mark” at
  {B_FR['seq22']} frames, the tween's key-to-key time).

The published tween workflow uses the second graph.
""")
    video("partb:grid", "partb:grid_poster",
          f"Top: the first key, the source take over the span, the last key. Middle row: seed {B_SEEDS[0]}. Bottom row: seed {B_SEEDS[1]}. "
          f"Every tile at {FPS_B:g} fps; short clips hold their last drawing.", loop=True,
          alt=f"The knock repair: keys, source take and {w(len(B_ROWS))} regenerated takes")
    trs = [[f"Source take ({TILE[SRC_ROW['arm']]})", "–", f"{SPAN_FRAMES}", k2k(SPAN_FRAMES, SRC_ROW['fps']), "–", "–",
            f2(SRC_MET["twos_score"]), f1(SRC_MET["drawings_per_s"]), f2(SRC_MET["flicker"]), "–"]]
    for r in B_ROWS:
        trs.append([B_NAME[r["arm"]], f"{r['seed']}", f"{r['frames']}", k2k(r['frames'], r['fps']), f1(r["first_mad"]), f1(r["last_mad"]),
                    f2(r["twos_score"]), f1(r["drawings_per_s"]), f2(r["flicker"]), secs(r["flotime_s"])])
    md(f"""
#### What the two routes did {{#measurements}}

On our first read of the frames, the seq takes bring the fist to the right edge of the lens on both seeds, with her face in view. The
open-weights takes still drive the fist well into the lens, which is the fault the repair was meant to remove. The
{B_FR['seq39']}-frame seq take at seed {B_SEEDS[0]} also smears the arm mid-swing. We have not yet graded these repairs cut back into
the take. Until then, the grid's {SPEEDS[-1][1]} speed setting lets anyone judge the knock frame by frame.

{table(["Take", "Seed", "Frames", "Key to key (s)", "First-key gap", "Last-key gap", "Twos score", "Drawings/s", "Flicker", "Floyo run time"], trs, "lrrrrrrrrr")}
The measurements show how differently the two routes repair. The open weights pin the keys, starting and ending on the frames they were
given. The seq adapter redraws them, so its first and last drawings are close to the keys without being the keys (the two gap columns).
A seq tween cut back into the take should therefore keep the source take's key frames at the cut points. In return the seq adapter holds twos
more steadily than the open weights at the same length, and its light pumps less through the knock than either the source take or the
open-weights takes. Which matters more depends on the cut. A repair that must match the take frame for frame wants the pinned keys, and
a repair whose job is the rhythm wants the redraw.
""")

    # ---------------------------------------------------------------- twos-clean
    ft = TC[FEATURED]
    tc_rows = [[B_NAME[t["arm"]], f"{t['seed']}", f2(t["twos_before"]), f"{2 * t['pairs']} of {t['frames']}", f"{t['in_pair_before']:.2f} → {t['in_pair_after']:.2f}",
                f"{t['between_drawings']:.2f}", f"{100 * t['share_before']:.1f}% → {100 * t['share_after']:.1f}%"] for t in TC.values()]
    tc_rows += [[B_NAME[r["arm"]], f"{r['seed']}", f2(r["twos_score"]), "not cleaned", "–", "–", "–"] for r in NOT_TC]
    same = all(t["twos_before"] == t["twos_after"] and t["dps_before"] == t["dps_after"] for t in TC.values())
    assert same, "the prose says the twos score is unchanged by the clean"
    assert all(t["twos_before"] >= TWOS_MIN for t in TC.values())
    keeper_note = said_in("That rule is our stand-in for judging which of the two frames is less coherent.", CHECK,
                          "if properly on twos, please remove the less coherent of the two frames")
    md(f"""
### The twos-clean trick {{#twos-clean}}

A model that animates on twos rarely draws a true hold. The second frame of a held pair is almost the first one, with a line shifted here
and the colour moved a little there, so the holds shimmer when the take plays. The cure is simple: where a take is properly on twos, keep
the more coherent frame of each held pair and show it on both frames. It is the cheapest repair in this study, because it regenerates
nothing and works on the frames a take already has. It takes five steps:

1. **Find the holds.** Measure how much each frame changes into the next. A pair that changes far less than the frames around it (under
   {score_clips.HOLD_REL:.2f} of its larger neighbour), or barely at all, is a drawing the model meant to hold for two frames.
2. **Check that the take is really on twos.** We clean only takes with a twos score of {TWOS_MIN:.2f} or more; below that the holds are too
   irregular to trust.
3. **Pick the keeper.** In each held pair, keep the frame that agrees better with the drawing before the pair and the drawing after it
   (the smaller summed difference to both), and drop the other. {keeper_note}
4. **Hold it.** Show the keeper on both frames of the pair. Frames outside a hold stay as they are.
5. **Check by eye.** The pick rule is a proxy, so we scrub the result and swap a pick by hand wherever the other frame is the better
   drawing.

{table(["Take", "Seed", "Twos score", "Frames in held pairs", "Change inside a hold, before → after", "Change between drawings (median)", "Share of all change inside holds"], tc_rows, "lrrrrrr")}
“Change inside a hold” is the mean grey-level difference between the two frames of each held pair, the shimmer the trick removes.
“Change between drawings” is the same measure between frames that are meant to differ, for scale. The twos score and the drawings per
second are the same before and after on every take, because by that measure these frames already were holds. What changes is what
happens inside each hold: the part the eye catches as shimmer, which the twos score cannot see. The {w(len(NOT_TC))} takes below the
threshold were left as generated.

The video below sets the take whose holds moved most beside its cleaned version, as generated on the left and cleaned on the right, with
frame numbers in the corners. The figure after it shows the {w(len(ft['figure_pairs']))} pairs from that take that moved most inside the
hold, with the change amplified {DIFF_GAIN} times.
""")
    video(f"tc:{FEATURED}", f"tcposter:{FEATURED}",
          f"Before and after: {esc(B_NAME[ft['arm']])}, seed {ft['seed']}, best seen at {SPEEDS[-1][1]} speed.", loop=True,
          alt="Twos-clean before and after, side by side")
    image("tc:pairs", "Held pairs before cleaning, with the difference between the two frames",
          f"Held pairs from {esc(B_NAME[ft['arm']])}, seed {ft['seed']}. The outlined frame is the keeper; the grey image shows what changes "
          f"between the two frames of the hold, amplified {DIFF_GAIN} times.")
    others = [k for k in TC if k != FEATURED]
    if others:
        md("The other cleaned takes:")
        h = '<div class="trio">' + "".join(
            f'<figure class="vid small"><video controls muted playsinline preload="metadata" loop poster="{MEDIA_FILES["tcposter:" + k]}" '
            f'aria-label="Twos-clean before and after: {esc(B_NAME[TC[k]["arm"]])}, seed {TC[k]["seed"]}"><source src="{MEDIA_FILES["tc:" + k]}" type="video/mp4"></video>'
            f'<figcaption>{esc(B_NAME[TC[k]["arm"]])}, seed {TC[k]["seed"]}</figcaption></figure>' for k in others) + "</div>"
        fig(h, "".join(f"- [Video: {B_NAME[TC[k]['arm']]}, seed {TC[k]['seed']}, before and after]({MEDIA_FILES['tc:' + k]})\n" for k in others))
    md(f"""
### In practice {{#take-away}}

- **Short spans need open weights.** A repair shorter than the partner nodes' floor goes to the H3 open weights or the seq adapter, which
  render in steps of {RULE_K} frames ({', '.join(str(x) for x in LEGAL[:3])} frames and up).
- **Pinned or redrawn.** The open weights start and end almost exactly on the keys. The seq adapter's first and last drawings sit visibly
  off them, and in exchange it keeps a cleaner twos rhythm.
- **Holds are cleaned after generation.** Twos-clean works on frames a take already has, regenerates nothing, and turns near-holds into
  true holds, on takes that are already mostly on twos.
- **A repair is judged in the cut.** The tween replaces frames inside a take that was already graded, so it is played inside the shot
  before it is kept.
""")

    # ---------------------------------------------------------------- closing
    assert BRIEF.count("drop Span A") == 1                                   # the platform span was run and dropped
    assert any(r["span"] == "A" for r in S2RES) and SPANS["A"]["shot"] == PLATFORM
    # the closing takeaway (the final pass), every part checked against the data it rests on
    ow_better = said_in("The H3 open weights did much better than we expected going in.", FINAL_PASS,
                        "minimax open weights performed much better than we initially expected")
    assert top_clips == ["h3ow"]                                                            # the most takes usable as is
    assert "open weights model better" in (G[(bk, "h3ow", SEEDS["h3ow"][0])].get("Comment") or "")
    assert max(commercial, key=lambda a: ST[a]["as_is"]) == "seedance" == top_shots[0]    # the strongest commercial route on the count
    assert ST["wan30"]["mood_mean"] == max(ST[a]["mood_mean"] for a in ROUTES) and low_spread == ["wan30"]
    ad_asis = {a: ST[a]["as_is"] for a in adapters}
    assert min(ad_asis.values()) >= ST["h3api"]["as_is"] > ST["wan30"]["as_is"] and max(ad_asis.values()) < ST["seedance"]["as_is"]
    assert ad_asis["hero124"] > ad_asis["seq124"]                                          # the order the sentence gives them in
    assert not any("Wrong timing" in probs(G[(s, a, sd)]) for a in adapters for s in SID for sd in SEEDS[a])
    assert all(ST[a]["cost"] == 0 for a in adapters) and all(ST[a]["cost"] > 0 for a in commercial)
    takeaway = said_in("So we still see a finetuning path as a good place to start for a studio that wants:", FINAL_PASS,
                       "we still think a finetuning path is a good start", "house style", "reduce generation cost",
                       "keep flexibility high between gens")
    platform_drop = said_in("too hard for every model we tried", BRIEF, "drop Span A, its too hard for any of the models")
    asis_order = ["seedance", "hero124", "seq124", "wan30"]                                 # "they sit between the two"
    assert [ST[a]["as_is"] for a in asis_order] == sorted((ST[a]["as_is"] for a in asis_order), reverse=True)
    asis_rows = [[NAME[a], f"{ST[a]['as_is']} of {ST[a]['clips']}"] for a in asis_order]
    md(f"""
## What a studio can take from this {{#studio}}

{ow_better} They gave the most takes usable as is of any route, in exchange for more setup than a partner node asks.

The strongest commercial routes ran into the two problems this study kept meeting:

- **Timing.** {NAME['seedance']} gave a take usable as is on the most shots, and most of its misses were timing misses.
- **Bias.** {NAME['wan30']} was the steadiest on mood, and its two attempts sat so close together that a second attempt gave little
  range, which we read as a strong bias in the model.

Our adapters performed on par with them without those problems. Neither adapter drew a timing flag, and their attempts differed more
than any other route's. On takes usable as is, they sit between the two:

{table(["Route", "Usable as is"], asis_rows, "lr")}
{takeaway}

- fidelity and style matching to its house style
- lower generation cost, since a run on open weights pays for GPU time and no partner fee per clip
- flexibility, with real range between generations

Each of the final four points to a different follow-up:

- {NAME['wan30']} comes close the same way twice, so it pairs better with frame repair than with a second roll.
- The H3 API invents action and contact, so a studio using it should plan to edit.
- The H3 open weights hold a shot or lose it, so a lost take calls for another attempt or another route.
- The seq adapter keeps the rhythm and leaves the drawing as the thing to repair when it fails.

A repair shorter than the partner nodes' floor on clip length needs open weights. Even then the eye decides what is kept: we also ran a
span on the station platform, the shot that beat every route in Part A, and dropped it as {platform_drop}. The labor between the poses is
still where the cost of this work sits, and these routes help with that labor under an artist's eye.
""")

    # ---------------------------------------------------------------- workflows
    wrows = [[o["what"], f"`{o['file']}`", o["input"], o["length"], FLOYO_WF] for o in wfs]
    assert N_SEQ_WF == 2 and all(o["graph"]["146"]["inputs"]["lora_name"] for o in wfs if o["arm"].startswith("seq"))
    md(f"""
## Workflows {{#workflows}}

{W(len(wfs))} ComfyUI workflows in API format cover the final four routes. {NAME['wan30']} and the MiniMax H3 API get one each, and the
seq adapter gets {w(N_SEQ_WF)}; its LoRA can be switched off to run the H3 open weights. Each is built by the same code that made
the takes in this study, with the study's prompt left in as an editable example. `FIRST_FRAME.png` (and `LAST_FRAME.png` for the tween)
is where your key drawings go.

{table(["Workflow", "File", "Input", "Length", "On Floyo"], [[w_[0], f"[{w_[1]}](workflows/{w_[1].strip('`')})", w_[2], w_[3], w_[4]] for w_ in wrows])}
**Setting up the seq adapter.** {SEQ_ADAPTER_SETUP}
Notes on the workflows:

- **What the long first-frame workflow asks of it:** {SEQ_CONTRACT_NOTE}
- **Changing the tween's length:** {TWEEN_LENGTH_NOTE}
- **H3 open weights:** open either seq workflow and set the LoRA strength to 0, or bypass the LoRA node. That runs the open weights on
  the reference-to-video path. The H3 open-weights takes in this study came from MiniMax's image-to-video graph, so results with the
  LoRA off can differ from them.
- **Frame rate:** on the partner graphs the video is saved at the model's native rate, read from the model's output. On the open-weight
  graphs it is saved at {RULE_FPS} fps, H3's native rate, which the length formula ({RULE_TXT} frames) also uses. A fixed rate that does
  not match the model plays the clip too fast or too slow.
- **Prompts:** the first-frame examples are the bus-shelter prompts from Part A, and the tween example is the peephole knock from Part B.
  Each model's dialect is worth keeping in your prompts, since the structure carries as much as the words.
""")
    h = "".join(f'<details class="prompt"><summary>Example prompt: {esc(o["what"])}</summary><pre>{esc(o["prompt"])}</pre></details>' for o in wfs)
    fig(h, "".join(f"\n**Example prompt: {o['what']}**\n\n```text\n{o['prompt']}\n```\n" for o in wfs))

    # ---------------------------------------------------------------- reference: the routes
    assert all(ST[a]["clips"] == ST["h3api"]["clips"] for a in ROUTES)
    md(f"""
## Reference {{#reference}}

### The routes, one by one {{#models}}

Each route's card gives its tally over its {ST['h3api']['clips']} takes ({w(n_sh)} shots, {w(n_att)} attempts) and its medians over the
same takes. The study's {N_CLIPS} graded takes cost {usd(SPEND)} in partner fees in all. The open-weight routes pay no partner fee and run
on Floyo GPU time instead, and each card gives its route's fee and run time.
""")
    assert ST["wan30"]["fix"].most_common(1)[0][0] == "Needs gen AI"
    model_text = {
        "h3api": f"""
**Out of the box.** A single partner node and the still. MiniMax rewrites the prompt before the model sees it (its prompt expansion), so
a complete prompt, with every beat timed, matters more than a clever one. It returns the largest frames of the study
({', '.join(ST['h3api']['size'])}).

**High points**

- The bike, usable as is on both attempts, {S("with a strong ending on the second", bk, "h3api", 1, "the end is quite cool")}.
- The first attempt on the train.

**Low points**

- The platform, where it fell apart.
- The roofs, where {S("it invented an action on both attempts", ROOFS, "h3api", 1, "similar hallucinations")}.
- The first peephole attempt, which punched the lens. It is the take Part B repairs.
- Where its cadence slips off twos we would retime it instead of rejecting it,
  {S("slowing certain frames to put it back on twos", "01_closeup_train", "h3api", 1, "slow down certain frames to get it on 2s")}.
""",
        "seedance": f"""
**Out of the box.** A single partner node, the most shots with a take usable as is, and the highest partner fee per clip. Every take
classed by look read as classic 2D.

**High points**

- The quiet shots: both bus-shelter attempts were usable as is, {S("and we liked the first one", sh2, "seedance", 0, "i like this result")}.
- On the train, {S("its two seeds landed far apart to the eye", "01_closeup_train", "seedance", 1, "range between gens")}.

**Low points**

- Timing on the action shots: {S("the first bike attempt ends abruptly and oddly", bk, "seedance", 0, "the end feels weird and abrupt")}.
- {S("The platform plays in slow motion", PLATFORM, "seedance", 0, "slow mo")}.
- {S("Its first peephole knock is less well timed than MiniMax's", pp, "seedance", 0, "compared to minimax, not as well timed")}.
- {S("One roofs attempt grows the blanket", ROOFS, "seedance", 1, "blanket gets way bigger")}.
""",
        "flux3": f"""
**Out of the box.** A single partner node, the shortest run time of the study, and no seed, so a good attempt cannot be reproduced.
Nothing it made was usable as is.

**High points**

- A smooth take on the train.
- {S("Acceptable linework on the bike", bk, "flux3", 0, "linework is fine")}.

**Low points**

- Its smoothest take smears.
- {S("The bus shelter veers away from anime into a vector look", sh2, "flux3", 0, "veers from anime")}.
- {S("The second bike roll freezes the rain", bk, "flux3", 1, "the rain is frozen")}.
- {S("On the roofs the action does not read at all", ROOFS, "flux3", 0, "idk what shes doing")}.
""",
        "wan30": f"""
**Out of the box.** A single partner node, the lowest partner fee, and the only route that renders at {ST['wan30']['fps']:g} fps. A
workflow has to save at that rate, or the clip plays at the wrong speed. When a take needs work, the usual fix is to regenerate frames.

**High points**

- Steady quality: {w(ST['wan30']['no'])} take rejected, none flagged with a problem, and the highest average lofi-anime mood of the
  {w(n_rt)} routes.
- The bus shelter, usable as is on both attempts, is its best shot.

**Low points**

- Narrow range and a repeated flaw (see [Agreement between attempts](#range)).
- Wind inside a closed train carriage.
- {S("On the platform, a take that feels flat", PLATFORM, "wan30", 0, "feels super flat")}.
""",
        "h3ow": f"""
**Out of the box.** More setup than a partner node, since the model, text encoder and sampler sit in the graph and the prompt is written
by hand in MiniMax's documented structure. It pays no partner fee and runs longer on Floyo GPUs. It starts closer to the still than any
other route.

**High points**

- The most takes usable as is.
- On the train, {S("the second attempt improves on the first and brings out the light on her face", "01_closeup_train", "h3ow", 1, "even better than the last one", "light on her face")}.
- On the bike, {S("a take we liked better than most, with her jacket flapping in the wind", bk, "h3ow", 0, "open weights model better", "jacket flapping")}.
- On the peephole, {S("the fist landed well on both attempts", pp, "h3ow", 1, "good placement")}.

**Low points**

- The platform, {S("bad from beginning to end, in style and in story", PLATFORM, "h3ow", 1, "bad beginning to end, both in style and narrative")}.
- The roofs, {S("where one attempt goes wrong as she lifts the sheet", ROOFS, "h3ow", 0, "when she lifts the sheet")}.
- {S("Even a good take on the bus shelter felt a little immature in the quality of its animation", sh2, "h3ow", 1, "immature")}.
""",
        "seq124": f"""
**Out of the box.** Our seq adapter on the H3 open weights, run here as a long generation from a first frame: {FRAMES['seq124'][0]} frames
from one reference, against the model card's {MODEL_CARD_SEQ['frames']} frames from {w(MODEL_CARD_SEQ['refs'])}. It is the slowest
route on Floyo GPUs and pays no partner fee. It redraws its
reference instead of copying it, so its first frame sits further from the still than any image-to-video route's.

**High points**

- The steadiest twos of the study and the highest average in-between grade.
- Every take classed by look read as classic 2D.
- On the platform, the shot that beat every route, it gave the most coherent take.
- On the peephole, {S("a good lean in toward the lens", pp, "seq124", 0, "nice pull in from her face")}.

**Low points**

- The roofs, {S("where the second attempt falls apart at the end", ROOFS, "seq124", 1, "falls apart at the end")}.
- The train, where it shares the hair flicker we found on {NAME['wan30']} and the hero adapter.
- The second peephole attempt, {S("where she points at the peephole", pp, "seq124", 1, "pointing on the peep hole")}.
""",
        "hero124": f"""
**Out of the box.** Our hero adapter on the same base and settings, with the same caption dialect and {w(MODEL_CARD_HERO['refs'])}
reference. It runs here at {FRAMES['hero124'][0]} frames instead of the model card's {MODEL_CARD_HERO['frames']}, as a whole shot rather
than a single next key.

**High points**

- The roofs, where the base model struggled and the hero adapter was usable as is on both attempts.
  {S("The boats drifting behind her are a nice touch", ROOFS, "hero124", 1, "boats moving in the background")}, and
  {S("the take shows why adapters are worth having: it improved on the base model here, with more delicacy than we expected it to handle", ROOFS, "hero124", 1, "it did improve the base in this case", "did not expect it to be able to handle such delicacy")}.

**Low points**

- {S("The hero adapter was optimized for black-and-white line art, which may be why its lineweight can come out light on colour work", bk, "hero124", 0, "optimized for for black and white lineart", "light in lineweight")}.
- {S("It failed the platform, as the base weights did", PLATFORM, "hero124", 0, "foundational weights also failed")}.
- The train has the same hair flicker as the seq adapter.
""",
    }
    for a in ROUTES:
        md(f"#### {NAME[a]} {{#{ANCHOR[a]}}}")
        stats_block(a)
        md(model_text[a])

    # ---------------------------------------------------------------- reference: how each route was driven
    p = {a: ST[a]["params"] for a in ROUTES}
    practice = {
        "h3api": f"Partner node, first frame only, {p['h3api'].get('resolution')}, prompt expansion “{p['h3api'].get('prompt_expansion_mode')}”. "
                 "MiniMax rewrites the prompt in front of the model, so the prompt states every beat with its time and leaves it nothing to invent: "
                 "an alignment line, then Subject, Action, Camera, Preserve, and a sound line.",
        "seedance": f"Partner node, image only, {p['seedance'].get('resolution')}. ByteDance's whole-second timeline (“0s-1s: …”). The 2D style is named "
                    "outright, because the model drifts toward live action when it is not, and hair and cloth are asked to move very little.",
        "flux3": f"Partner image-to-video node, {p['flux3'].get('resolution')}, no seed. Timestamped beats; a short Subject line that only pins which way she "
                 "faces (Black Forest Labs: prompt the motion, not the scene); the drawn look last, as a continuity constraint.",
        "wan30": f"Partner node, {p['wan30'].get('resolution')}. One timed segment for the whole take, because segment boundaries can turn into cuts; "
                 "steady exposure asked through the last frame.",
        "h3ow": f"Open weights on Floyo GPUs ({ST['h3ow']['steps']} steps). No prompt expansion, so the prompt is written by hand in MiniMax's documented "
                "structure for the open weights. The first-frame path stretches its input, so it gets a centre crop of the still at the output aspect.",
        "seq124": f"Our seq adapter, a LoRA on the H3 open weights (reference-to-video, {ST['seq124']['steps']} steps), with the still as its one reference. "
                  "The caption dialect it was trained on: alignment line, Subject, Action, Camera, Preserve.",
        "hero124": f"Our hero adapter on the same base and settings ({ST['hero124']['steps']} steps), one reference, same caption dialect.",
    }
    na_token = "non_diegetic_music: N/A"           # MiniMax's documented field token for a shot with no score
    assert all(PR["h3ow"][s].rstrip().endswith(na_token) for s in SID), "the open-weights prompts no longer end on the field token"
    assert not any(na_token in PR[a][s] for a in PR if a != "h3ow" for s in SID)
    assert "no added rendering, gloss or depth-of-field" in look
    matte = said_in("matte paper grain, even focus across the whole frame", PR["h3api"][EXAMPLE_SHOT], "matte paper grain, even focus across the whole frame")
    md(f"""
### How each route was driven {{#driven}}

We wrote each prompt from its vendor's guide and checked it against the brief and the stills. What to avoid is written as what should
happen instead: the brief's request for no added rendering, gloss or depth of field became “{matte}” in the H3 API prompt.

{table(["Route", "Output", "How it was driven and prompted"], [[f"**{NAME[a]}**", f"{ST[a]['frames']} frames at {ST[a]['fps']:g} fps, {', '.join(ST[a]['size'])}", practice[a]] for a in ROUTES])}""")

    # ---------------------------------------------------------------- reference: shot by shot
    md(f"""
### Shot by shot {{#shots}}

Each grid puts the still and the {w(n_rt)} routes in one frame, one grid per attempt. Every tile plays at its model's native frame rate,
and a clip that ends early holds its last drawing. The grids run at {FPS_GRID} fps, which shows {FPS['wan30'][0]:g} fps evenly and
{FPS['h3api'][0]:g} fps the way any {FPS_GRID} Hz screen does; the speed buttons slow them down. The tables give the grades and the
twos score per take.
""")
    shot_text = {
        "01_closeup_train": f"""
The open weights were usable as is on both attempts. Wind inside a closed carriage is the recurring failure: {NAME['wan30']} blew her
hair about, and the seq and hero adapters show the same flicker. {S("One seq take could pass as the train bouncing", "01_closeup_train", "seq124", 1, "read as the train bouncing")},
but {S("on a hero take the window is closed, so the wind has no source", "01_closeup_train", "hero124", 1, "the window isnt open")}.
By eye, Seedance's two attempts show how far apart two seeds of a commercial model can land.
""",
        "02_wide_chill_bus_shelter": f"""
On the quiet shot most routes got it right: every route but FLUX 3 has a take usable as is. {NAME['wan30']} is the standout for acting,
and this is the shot where its lack of range first showed. {S("FLUX 3 turned the scene into odd, clunky animation", sh2, "flux3", 1, "weird animation")}.
""",
        "03b_action_bike_downpour": f"""
The bike asks for tracking and weather, and the H3 API, the open weights and the seq adapter all delivered.
{S("The seq adapter's first attempt stood out for the trees it added", bk, "seq124", 0, "very unique", "tree elements")}. Seedance's first
attempt has the abrupt ending behind our note on its timing, and {S("FLUX 3's second roll froze the rain, which left it unusable", bk, "flux3", 1, "the rain is frozen", "not useable")}.
""",
        "03e_action_platform": f"""
Nothing usable as is came out of the platform, from any route. Seedance played it in slow motion, {NAME['wan30']} came out flat, and the H3
routes came apart. {S("The seq adapter's first attempt holds the story together well enough that we would consider fixing it with generative tools, though the fix would take time", PLATFORM, "seq124", 0, "open to fixing it with genai", "time intensive")}.
""",
        "04_zoom_out_roofs": f"""
The slow pull-back over a town with a tiny figure punishes invented action. The H3 routes and FLUX 3 gave her actions that do not read,
{NAME['wan30']} had trouble with the sheet on both attempts, and one Seedance take grew the blanket. The hero adapter did less, and did it
well on both attempts, {S("with more delicacy than we expected from it", ROOFS, "hero124", 1, "handle such delicacy")}.
""",
        "05b_fisheye_peephole": f"""
The peephole is a contact shot. The first H3 API attempt punches the lens, and
{S("the second lands the fist in a more acceptable place, with some humour", pp, "h3api", 1, "more acceptable where the fist lands", "funny")}.
{S("The open weights placed the fist well on both attempts, and the takes are funny", pp, "h3ow", 1, "funny and good placement")}.
{NAME['wan30']} animated well and put the fist somewhere odd both times. The seq adapter's first attempt
{S("keeps the rain falling about her and off the peephole", pp, "seq124", 0, "around her instead of on the peep hole")}.
""",
    }
    for s in SID:
        sh = SHOT[s]
        md(f"#### {short(s)} · {sh['shot_type']} {{#shot-{s.split('_')[0]}}}")
        beats = "".join(f"<li>{esc(b)}</li>" for b in sh["beats"])
        h = (f'<div class="shot-head"><figure class="img"><img src="{MEDIA_FILES["still:" + s]}" alt="{esc(short(s))}: the still" loading="lazy" decoding="async">'
             f'<figcaption>The still, the first frame every route was given.</figcaption></figure>'
             f'<div class="brief"><p>{esc(sh["scene"])}</p><p class="label">Camera</p><p>{esc(sh["camera"])}</p>'
             f'<p class="label">Beats</p><ul>{beats}</ul><p class="label">Must hold</p><p>{esc(sh["must_hold"])}</p></div></div>')
        m_ = (f"![{short(s)}: the still]({MEDIA_FILES['still:' + s]})\n\n{sh['scene']}\n\n**Camera:** {sh['camera']}\n\n**Beats:**\n\n"
              + "".join(f"- {b}\n" for b in sh["beats"]) + f"\n**Must hold:** {sh['must_hold']}\n")
        fig(h, m_)
        for i in ATT:
            video(f"grid:{s}:{i}", f"poster:{s}:{i}", f"{esc(short(s))} · {esc(attempt_label(i))}",
                  alt=f"{short(s)}, {attempt_label(i)}: the still and seven routes side by side")
        trs = []
        for a in ROUTES:
            for i, sd in enumerate(SEEDS[a]):
                c, r = G[(s, a, sd)], row(s, a, sd)
                trs.append([TILE[a], f"{i + 1}", (c.get("Believable in-between") or "–").split()[0], c.get("Reads on twos", "–"),
                            f2(r["twos_score"]), c.get("Looks like", "–"), "; ".join(probs(c)) or "–", c["Usable"].lower()])
        md(table(["Route", "Attempt", "In-between", "On twos (by eye)", "Twos score", "Looks like", "Problems", "Usable"], trs, "lrrlrlll"))
        md(f"**In the grading.** {shot_text[s].strip()}")

    # ---------------------------------------------------------------- reference: method
    md(f"""
### How we measured {{#method}}

All measures run on a grayscale copy of each clip scaled to {score_clips.W} px wide, and differences are mean absolute grey-level
differences on a scale where black to white is {GREY_MAX}.

- **Twos score:** the share of frame-to-frame steps that follow a strict hold, move, hold, move pattern. A frame pair is a hold when it
  changes far less than its neighbours or not at all. On synthetic test clips, a move held cleanly on twos scores {f2(TWOS_CAL[2])}, on
  threes {f2(TWOS_CAL[3])} and on ones {f2(TWOS_CAL[1])}. It reads the whole frame, with the consequences described under
  [The measurements describe rhythm](#measures).
- **Drawings per second:** the frame rate times the share of frame pairs that are not holds.
- **First-key gap:** the difference between the clip's first frame and the still it was given, with the still centre-cropped to the
  clip's aspect. In Part B the **last-key gap** does the same for the last frame and the span's last key. Low means the clip starts (or
  ends) on the key.
- **Seed spread:** the difference between a route's two attempts at {SPREAD_N} matched moments across the clip, averaged. Low means the
  two attempts look alike.
- **Flicker:** the wobble of each frame's mean brightness around its running average. Zero is steady light.
- **Key to key:** Part B times each clip from its first frame to its last: (frames - 1) / fps. The source span, the tweens and the
  prompt's alignment mark all use this convention.
- **Cost and run time:** Floyo's record for each run, the partner fee and the run time on Floyo.

Every grade in this study comes from a single reviewer, which is the study's main limit of scope.
""")


# ----------------------------------------------------------------------------------------------- render
CSS = r"""
:root {
  --bg: #F7F6F2; --surface: #FFFFFF; --ink: #1D2321; --muted: #5A6561; --rule: #DAD9D1;
  --accent: #2D6A99; --accent-soft: #E7EFF6; --warm: #9A5B22; --video-bg: #0F1211; --code: #F0EFE9;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --sans: "Inter Tight", "Inter", "Segoe UI", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #121514; --surface: #1A1F1E; --ink: #E5EAE8; --muted: #9EAAA6; --rule: #2E3634;
  --accent: #8EC3EA; --accent-soft: #1C2A35; --warm: #E3A867; --video-bg: #050606; --code: #1F2523; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #121514; --surface: #1A1F1E; --ink: #E5EAE8; --muted: #9EAAA6; --rule: #2E3634;
  --accent: #8EC3EA; --accent-soft: #1C2A35; --warm: #E3A867; --video-bg: #050606; --code: #1F2523; color-scheme: dark; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 64px; -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 18px/1.62 var(--serif); overflow-x: hidden; }
a { color: var(--accent); text-underline-offset: 2px; }
.top { position: sticky; top: 0; z-index: 10; background: var(--bg); border-bottom: 1px solid var(--rule); }
.top .bar { max-width: 1200px; margin: 0 auto; display: flex; align-items: center; gap: 12px; padding: 0 16px; }
.top nav { flex: 1 1 auto; min-width: 0; overflow-x: auto; scrollbar-width: none;
  -webkit-mask-image: linear-gradient(90deg, #000 calc(100% - 28px), transparent); mask-image: linear-gradient(90deg, #000 calc(100% - 28px), transparent); }
.top nav::-webkit-scrollbar { display: none; }
.top ol { list-style: none; margin: 0; padding: 0; display: flex; gap: 2px; white-space: nowrap; }
.top a { display: block; padding: 14px 10px; font: 600 14px/1 var(--sans); color: var(--muted); text-decoration: none; letter-spacing: .01em; }
.top a:hover, .top a:focus-visible { color: var(--ink); }
.top a.part { color: var(--accent); }
.theme { flex: 0 0 auto; font: 600 13px/1 var(--sans); color: var(--muted); background: none; border: 1px solid var(--rule); border-radius: 999px; padding: 7px 12px; cursor: pointer; }
main { max-width: 1200px; margin: 0 auto; padding: 8px 16px 96px; }
.prose > * { max-width: 72ch; }
.prose > .tw { max-width: 100%; }
h1, h2, h3, h4 { font-family: var(--sans); line-height: 1.12; text-wrap: balance; letter-spacing: -.01em; }
h1 { font-size: clamp(34px, 5.6vw, 58px); font-weight: 750; margin: 36px 0 12px; }
h1 + p em { font-size: 1.12em; color: var(--muted); font-style: normal; }
h2 { font-size: clamp(26px, 3.4vw, 36px); font-weight: 750; margin: 72px 0 12px; padding-top: 16px; border-top: 3px solid var(--ink); }
h3 { font-size: 24px; font-weight: 700; margin: 48px 0 8px; }
h4 { font-size: 20px; font-weight: 700; margin: 40px 0 8px; padding-top: 12px; border-top: 1px solid var(--rule); }
p, li { margin: .55em 0; }
ol, ul { padding-left: 1.3em; }
strong { font-weight: 650; }
code { font-family: var(--mono); font-size: .82em; background: var(--code); border: 1px solid var(--rule); border-radius: 4px; padding: 1px 5px; overflow-wrap: anywhere; }
.tw { overflow-x: auto; margin: 16px 0; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font: 14px/1.45 var(--sans); font-variant-numeric: tabular-nums; }
th, td { padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; text-align: left; }
th { font-size: 12px; font-weight: 650; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); white-space: nowrap; }
td { min-width: 4.5em; }
td[style*="right"], th[style*="right"] { white-space: nowrap; }
.tw code { white-space: nowrap; overflow-wrap: normal; }
tr:last-child td { border-bottom: 0; }
figure { margin: 18px 0; }
figcaption { font: 14px/1.45 var(--sans); color: var(--muted); margin-top: 8px; }
.vid video { display: block; width: 100%; height: auto; aspect-ratio: 16 / 9; background: var(--video-bg); border-radius: 8px; }
.vid.small video { aspect-ratio: auto; }
.speed { display: none; gap: 6px; align-items: center; margin-top: 8px; font: 600 13px/1 var(--sans); color: var(--muted); }
.js .speed { display: flex; }
.speed button { font: 600 13px/1 var(--sans); color: var(--ink); background: var(--surface); border: 1px solid var(--rule); border-radius: 999px; padding: 6px 11px; cursor: pointer; }
.speed button[aria-pressed="true"] { background: var(--accent); color: var(--bg); border-color: var(--accent); }
.img img { display: block; width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--rule); background: var(--surface); }
.shot-head { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 6fr); gap: 24px; align-items: start; }
.shot-head .brief { font-size: 16px; }
.shot-head .brief p { margin: .3em 0 .7em; }
.shot-head .brief ul { margin: .2em 0 .7em; }
.label { font: 650 12px/1 var(--sans) !important; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 14px 0 4px !important; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 900px; }
.trio { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
dl.stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 0; margin: 12px 0 8px; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); overflow: hidden; }
dl.stats div { padding: 10px 14px; border-bottom: 1px solid var(--rule); border-right: 1px solid var(--rule); }
dl.stats dt { font: 650 11px/1.3 var(--sans); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }
dl.stats dd { margin: 4px 0 0; font: 500 15px/1.35 var(--mono); }
details.prompt { border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); margin: 10px 0; max-width: 72ch; }
details.prompt summary { cursor: pointer; padding: 12px 14px; font: 600 15px/1.3 var(--sans); }
details.prompt pre { margin: 0; padding: 0 14px 14px; white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.55 var(--mono); color: var(--ink); }
footer { max-width: 1200px; margin: 0 auto; padding: 24px 16px 64px; font: 14px/1.5 var(--sans); color: var(--muted); border-top: 1px solid var(--rule); }
@media (max-width: 760px) {
  body { font-size: 17px; }
  .shot-head, .pair { grid-template-columns: 1fr; }
  h2 { margin-top: 56px; }
  th, td { padding: 7px 9px; }
  .tw table { min-width: 40em; }
}
"""

JS = r"""
(function () {
  var root = document.documentElement;
  root.classList.add('js');
  try { var t = localStorage.getItem('cs-theme'); if (t === 'light' || t === 'dark') root.setAttribute('data-theme', t); } catch (e) {}
  function current() {
    var t = root.getAttribute('data-theme');
    if (t) return t;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  document.addEventListener('DOMContentLoaded', function () {
    var b = document.querySelector('.theme');
    function label() { b.textContent = current() === 'dark' ? 'Light' : 'Dark'; }
    if (b) { label(); b.addEventListener('click', function () {
      var n = current() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', n);
      try { localStorage.setItem('cs-theme', n); } catch (e) {}
      label();
    }); }
  });
  document.addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('[data-rate]') : null;
    if (!b) return;
    var v = document.getElementById(b.getAttribute('data-for'));
    if (!v) return;
    var r = parseFloat(b.getAttribute('data-rate'));
    v.defaultPlaybackRate = r; v.playbackRate = r;
    var sib = b.parentNode.querySelectorAll('button');
    for (var i = 0; i < sib.length; i++) sib[i].setAttribute('aria-pressed', sib[i] === b ? 'true' : 'false');
  });
  // one video at a time
  document.addEventListener('play', function (e) {
    var vs = document.querySelectorAll('video');
    for (var i = 0; i < vs.length; i++) if (vs[i] !== e.target) vs[i].pause();
  }, true);
})();
"""

NAV = [("summary", "Findings", False), ("part-a", "Part A", True), ("setup", "Setup", False), ("motion", "1 Motion", False),
       ("signatures", "2 Failures", False), ("range", "3 Range", False), ("measures", "4 Measures", False), ("part-b", "Part B", True),
       ("repair", "5 Repair", False), ("twos-clean", "Twos-clean", False), ("studio", "For studios", True),
       ("workflows", "Workflows", True), ("reference", "Reference", True),
       ("models", "Routes", False), ("shots", "Shots", False), ("method", "Method", False)]


def render_html() -> str:
    parts = []
    for seg in DOC:
        if seg[0] == "md":
            h = markdown.markdown(seg[1], extensions=["tables", "attr_list"])
            h = re.sub(r"<table>(.*?)</table>", lambda m_: f'<div class="tw"><table>{m_.group(1)}</table></div>', h, flags=re.S)
            parts.append(f'<div class="prose">{h}</div>')
        else:
            parts.append(seg[1])
    body = "\n".join(parts)
    ids = set(re.findall(r'id="([^"]+)"', body))
    for i, _, _ in NAV:
        assert i in ids, f"nav anchor missing: {i}"
    for href in re.findall(r'href="#([^"]+)"', body):
        assert href in ids, f"broken anchor #{href}"
    cls = ' class="part"'
    nav = "".join(f'<li><a href="#{i}"{cls if p else ""}>{esc(t)}</a></li>' for i, t, p in NAV)
    title = ABSTRACT["title"].split(":", 1)[0]
    desc =re.sub(r"<[^>]+>", "", re.search(r"<em>(.*?)</em>", body, re.S).group(1)).strip()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{esc(" ".join(desc.split()))}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter+Tight:wght@500;600;650;700;750&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>{CSS}</style>
<script>{JS}</script>
</head>
<body>
<header class="top"><div class="bar"><nav aria-label="Contents"><ol>{nav}</ol></nav><button class="theme" type="button" aria-label="Switch colour theme">Dark</button></div></header>
<main>
{body}
</main>
<footer>{esc(ORG)} · {esc(", ".join(AUTHORS))} · {ABSTRACT["date"].strftime("%B %Y")} · <a href="#workflows">Workflows</a> on Floyo: {FLOYO_WF}</footer>
</body>
</html>
"""


def gh_slug(text: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped (letters, digits, spaces, hyphens and underscores
    kept), each space a hyphen. 'Part A · Seven routes, six shots' -> 'part-a--seven-routes-six-shots'."""
    t = re.sub(r"[*_`]", "", text.strip()).lower()
    return re.sub(r"[^\w\- ]", "", t).replace(" ", "-")


def render_md() -> str:
    """The article as Markdown for the repo root. Media and workflow links are already relative to the root; the page's own
    {#id} anchors become the anchors GitHub generates from the heading text, so in-page links resolve there too."""
    out = [seg[1] if seg[0] == "md" else seg[2] for seg in DOC]
    txt = "\n".join(out).replace("\n\n\n", "\n\n")
    slug_of, seen, in_code = {}, Counter(), False
    for line in txt.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"(#{1,6}) (.+?)(?:\s*\{#([\w-]+)\})?\s*$", line)
        if m:
            base = gh_slug(m.group(2))
            slug = base if not seen[base] else f"{base}-{seen[base]}"
            seen[base] += 1
            if m.group(3):
                slug_of[m.group(3)] = slug
    for ref in set(re.findall(r"\]\(#([\w-]+)\)", txt)):
        assert ref in slug_of, f"case-study.md links to #{ref}, which no heading carries"
    txt = re.sub(r"\]\(#([\w-]+)\)", lambda m_: f"](#{slug_of[m_.group(1)]})", txt)
    return re.sub(r"[ \t]*\{#[\w-]+\}", "", txt)


def deny_patterns() -> list[str]:
    """The patterns no published file may contain: local paths, keys, run ids and internal names. Kept in a private file
    beside this script, so the script itself can ship in the repo without carrying the names it screens for."""
    pats = json.loads(DENY_FILE.read_text(encoding="utf-8"))["patterns"]
    assert len(pats) >= 10 and all(isinstance(p, str) and p for p in pats)
    return pats


# loader fields in a workflow graph name the model files the graph needs; these, and only these, may carry a model file name
MODEL_FIELDS = ("vae_name", "unet_name", "clip_name", "lora_name")


def check_text(name: str, txt: str, pats: list[str]):
    """One published text carries no local path, key, run id or internal name."""
    if name.endswith(".html"):
        visible = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", txt, flags=re.S)
        assert "!" not in re.sub(r"<!doctype html>|<!--.*?-->", "", visible, flags=re.S | re.I), f"{name}: exclamation mark in page text"
    if name.endswith(".json"):
        g = json.loads(txt)
        if isinstance(g, dict) and all(isinstance(n, dict) and "class_type" in n for n in g.values()):     # a workflow graph
            files = [n["inputs"][f] for n in g.values() for f in MODEL_FIELDS if f in n.get("inputs", {})]
            assert not any(re.search(r"hero", f, re.I) for f in files), f"{name}: an unpublished adapter is referenced"
            for n in g.values():
                for f in MODEL_FIELDS:
                    if f in n.get("inputs", {}):
                        n["inputs"][f] = "MODEL_FILE"
            txt = json.dumps(g, ensure_ascii=False)
    for bad in pats:
        hit = re.search(bad, txt, re.I)
        assert not hit, f"{name}: forbidden text {bad!r}: …{txt[max(0, hit.start() - 60):hit.end() + 60]}…"


# the article's voice: written by us collectively, in our words. A run of VERBATIM_N words from any grading comment may
# not survive into the published prose; these patterns may not appear in it at all.
VERBATIM_N = 6
BANNED = [chr(0x2014),                                        # the em dash
          r"\b(?:our|its|their|your|his|her|my)\s+own\b", r"\b\w+['’]s\s+own\b",   # "X's own", "our own"
          r"\band nothing else\b", r"\bby (?:her|his|their) hand alone\b",         # boundary-reminder qualifiers
          r"\bindependent pass\b", r"\bwithout paying for\b", r"\bnot a build pipeline\b", r"\bno workflow in the final set\b",
          r"\bname them\b", r"\bfrontier\b",                                       # notes to ourselves; "commercial" is the term
          r"\bis an animator\b", r"\bMinta['’]s\b", r"\bMinta (?:noted|saw|said|sums|summed|called|wrote|would)\b"]
VOICE_FILES = ("index.html", "case-study.md", "README.md", "workflows/README.md")


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower().replace("’", "").replace("'", ""))


def article_prose(name: str, txt: str) -> str:
    """The reader-facing prose of a published text: tags, scripts, styles, code blocks and the example prompts removed."""
    if name.endswith(".html"):
        t = re.sub(r"<(script|style|pre)[^>]*>.*?</\1>", " ", txt, flags=re.S)
        return html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"```.*?```", " ", txt, flags=re.S)


def check_voice(texts: dict[str, str]) -> int:
    """No em dash, no "X's own", no boundary-reminder phrasing, no grading comment quoted, and Minta named only as an
    author (byline, footer, citation) and in the one introduction of the team. Returns the number of comment runs screened."""
    grams = set()
    for c in GR["clips"].values():
        ws = _words(c.get("Comment") or "")
        grams |= {" ".join(ws[i:i + VERBATIM_N]) for i in range(len(ws) - VERBATIM_N + 1)}
    for name in VOICE_FILES:
        prose = article_prose(name, texts[name])
        for pat in BANNED:
            hit = re.search(pat, prose, re.I)
            assert not hit, f"{name}: voice: {pat!r} at …{prose[max(0, hit.start() - 60):hit.end() + 60]}…"
        ws = _words(prose)
        runs = sorted({g for g in (" ".join(ws[i:i + VERBATIM_N]) for i in range(len(ws) - VERBATIM_N + 1)) if g in grams})
        assert not runs, f"{name}: grading comments survive verbatim: {runs}"
        assert all(m.group(1) for m in re.finditer(r"\bMinta\b( Carlson)?", prose)), f"{name}: Minta named outside her full name"
        flat = " ".join(prose.split())
        if name != "workflows/README.md":
            assert flat.count("Creative Lead") == 1 and "Creative Lead, Minta Carlson" in flat, name
            assert flat.count("Minta Carlson") == (3 if name == "index.html" else 2), (name, flat.count("Minta Carlson"))
    return len(grams)


TEXT_SUFFIXES = {".html", ".md", ".json", ".py", ".txt", ".css", ".js", ""}


def check_tree() -> list[str]:
    """Every text file in the repo folder, checked; returns the files checked."""
    pats = deny_patterns()
    done = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            check_text(rel(p), p.read_text(encoding="utf-8"), pats)
            done.append(rel(p))
    return done


# ----------------------------------------------------------------------------------------------- repository files
GITIGNORE = """# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
env/

# OS cruft
.DS_Store
Thumbs.db
desktop.ini

# Editors
.vscode/
.idea/
*.swp
*~
"""

LICENSE_TEXT = f"""Copyright (c) 2026 {' and '.join(AUTHORS)}, {ORG}.

The article text, figures and video in this repository are licensed under the
Creative Commons Attribution 4.0 International License (CC BY 4.0).

You are free to:
  - Share — copy and redistribute the material in any medium or format.
  - Adapt — remix, transform, and build upon the material for any purpose,
    even commercially.

Under the following terms:
  - Attribution — You must give appropriate credit, provide a link to the
    license, and indicate if changes were made. You may do so in any
    reasonable manner, but not in any way that suggests the licensor
    endorses you or your use.

No additional restrictions — You may not apply legal terms or technological
measures that legally restrict others from doing anything the license
permits.

The full legal text of the CC BY 4.0 license is available at:
https://creativecommons.org/licenses/by/4.0/legalcode

A human-readable summary is available at:
https://creativecommons.org/licenses/by/4.0/

Code in this repository (build_site.py and the ComfyUI workflow files in
workflows/) is separately licensed under MIT; see LICENSE-CODE.
"""

LICENSE_CODE_TEXT = f"""MIT License

Copyright (c) 2026 {' and '.join(AUTHORS)}, {ORG}.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

This license applies to the source code in this repository (currently
build_site.py and the ComfyUI workflow files in workflows/). The article
text, figures and video are separately licensed under CC BY 4.0; see LICENSE.
"""

BREADCRUMB_DOC = '''"""The record of how this repository was built.

This is the script that generated this repository's index.html, case-study.md, media/ and workflows/. It is kept here so
every number on the page can be traced to the code that computed it. Its inputs (the generated clips, our grade
sheets, the run records, the prompt files, the measurement module and the graph builders it imports) stay in our
working directory, so the script does not run from here.

What it does: it reads the grades and the run records; measures every clip (twos score, drawings per second, first-key
gap and flicker, plus the seed spread between a route's two attempts); asserts each qualitative claim in the prose
against those data; passes every statement drawn from the grading through said() and said_in(), which assert that the
comment or note it rests on carries the evidence named; builds the comparison grids and the Part B figures with ffmpeg;
rebuilds the four workflows from the same graph builders that made the study's clips, against a stub that cannot reach
Floyo; and writes the page, the Markdown and the repository files, checking every published text file against a
private deny list and checking the article's voice (no em dash, no surviving comment verbatim).
"""'''


def breadcrumb_copy() -> str:
    """This script's own source with the breadcrumb docstring in place of its working docstring."""
    src = Path(__file__).read_text(encoding="utf-8")
    m = re.match(r'""".*?"""', src, re.S)
    assert m, "the builder's docstring is not at the top of the file"
    return BREADCRUMB_DOC + src[m.end():]


def readme(wfs: list[dict]) -> str:
    title = ABSTRACT["title"]
    head, sub = (x.strip() for x in title.split(":", 1))
    dated = ABSTRACT["date"]
    # one line per paragraph, and one line per item where a block is a list
    one = lambda s: "\n".join(" ".join(x.split()) for x in re.split(r"\n(?=- )", s))  # noqa: E731
    abstract = "\n\n".join([f"*{one(ABSTRACT['dek'])}*"] + [one(p_) for p_ in ABSTRACT["opening"]])
    media = [p for p in MEDIA.rglob("*") if p.is_file()]
    n_mp4 = sum(1 for p in media if p.suffix == ".mp4")
    n_img = sum(1 for p in media if p.suffix in (".jpg", ".png"))
    mb = sum(p.stat().st_size for p in media) / 1e6
    assert n_mp4 + n_img == len(media)
    rows = [["[`index.html`](index.html)", "The case study as a web page: what the live site serves."],
            ["[`case-study.md`](case-study.md)", "The same article as Markdown, readable on GitHub."],
            ["[`media/`](media/)", f"{n_mp4} videos and {n_img} images: the comparison grids and their posters, the stills, and the Part B "
                                   f"clips and figures ({mb:.0f} MB)."],
            ["[`workflows/`](workflows/)", f"{W(len(wfs))} ComfyUI workflows in API format, with a README."],
            ["[`build_site.py`](build_site.py)", "The script that generated the page, the Markdown, the media and the workflows."],
            ["[`LICENSE`](LICENSE), [`LICENSE-CODE`](LICENSE-CODE)", "The licenses (see below)."]]
    key = "full_shot_anime_generation_" + str(dated.year)
    return f"""# {head[0].upper() + head[1:]}

**{sub[0].upper() + sub[1:]}**

A case study from {ORG} by {' and '.join(AUTHORS)} · {dated.strftime('%B %Y')}.

**Live site:** {REPO_URL}

---

## Abstract

{abstract}

---

## Repository contents

{table(["File", "Description"], rows)}
The script records how the page was built. Its inputs (the generated clips, grade sheets, run records and graph builders) stay in our
working directory.

---

## Citation

```bibtex
@misc{{{key},
  title        = {{{title}}},
  author       = {{{AUTHORS_BIB}}},
  year         = {{{dated.year}}},
  month        = {{{dated.strftime('%B')}}},
  howpublished = {{Case study, {ORG}}},
  url          = {{{REPO_URL}}}
}}
```

---

## License

- **Article text, figures and video:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE). You may copy, redistribute,
  remix and build on the work in any medium or format, including commercially, provided you give appropriate credit and link to the
  license.
- **Code** (`build_site.py` and the workflow files in `workflows/`): [MIT](LICENSE-CODE).
"""


def workflows_readme(wfs: list[dict]) -> str:
    cap = lambda s: s[0].upper() + s[1:]  # noqa: E731
    return ("# Workflows\n\nComfyUI API-format graphs from the case study. Replace `#inputs/FIRST_FRAME.png` (and `#inputs/LAST_FRAME.png` "
            "for the tween) with your key drawings; each prompt is an editable example from the study.\n\n"
            + table(["File", "What it is", "Input", "Length"], [[f"`{o['file']}`", o["what"], o["input"], o["length"]] for o in wfs])
            + f"\n**Setting up the seq adapter.** {SEQ_ADAPTER_SETUP}\n"
            + f"**What the long first-frame workflow asks of it.** {cap(SEQ_CONTRACT_NOTE)}\n\n"
            + f"**Changing the tween's length.** {cap(TWEEN_LENGTH_NOTE)}\n\n"
            + "**H3 open weights.** Open either seq workflow and set the LoRA strength to 0, or bypass the LoRA node. That runs the open "
              "weights on the reference-to-video path. The H3 open-weights clips in the case study came from MiniMax's image-to-video "
              "graph, so results with the LoRA off can differ from them.\n\n"
              f"{FLOYO_WF}\n")


POINTER_TEXT = """# Case study: moved

The full case study's Markdown now lives at the root of the case-study repository folder, next to the page it mirrors:
`outputs/case_study_site/case-study.md`, where its media and workflow links resolve. This file is a pointer, written by
`scripts/build_case_study_site.py`.
"""


def check_links(texts: dict[str, str]):
    """Every relative link and media source in the published texts points at a file in the repo, and every media file is used."""
    used = set()
    pending = {(OUT / n).resolve() for n in texts}      # written in this same pass
    for name, txt in texts.items():
        if not name.endswith((".html", ".md")):
            continue
        refs = re.findall(r'(?:src|poster|href)="([^"#:]+)"', txt) if name.endswith(".html") else re.findall(r"\]\(([^)#:\s]+)\)", txt)
        base = (OUT / name).parent
        for r in refs:
            p = (base / r).resolve()
            assert p.exists() or p in pending, f"{name}: link to a missing file: {r}"
            used.add(p)
    orphans = [rel(p) for p in MEDIA.rglob("*") if p.is_file() and p.resolve() not in used]
    assert not orphans, f"media files no page or Markdown links to: {orphans}"


def main():
    for d in (MEDIA, WF):
        d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        print("stills ...", flush=True); build_stills()
        print("stage 1 grids ...", flush=True); build_stage1_grids(tmp)
        print("part B media ...", flush=True); build_partb(tmp)
    print("workflows ...", flush=True)
    wfs = build_workflows()
    content(wfs)
    page, mdtxt = render_html(), render_md()
    texts = {"index.html": page, rel(MD_OUT): mdtxt, "README.md": readme(wfs), "LICENSE": LICENSE_TEXT, "LICENSE-CODE": LICENSE_CODE_TEXT,
             ".gitignore": GITIGNORE, "build_site.py": breadcrumb_copy(), "workflows/README.md": workflows_readme(wfs)}
    pats = deny_patterns()
    for name, txt in texts.items():                     # nothing is written until every new text passes
        check_text(name, txt, pats)
    n_runs = check_voice(texts)
    check_links(texts)
    for name, txt in texts.items():
        (OUT / name).write_text(txt, encoding="utf-8", newline="\n")
    # the numbers record stays in the working directory; the copy that used to ship in the repo is removed
    numbers = {"routes": {a: {k: (dict(v) if isinstance(v, Counter) else v) for k, v in ST[a].items() if k != "params"} for a in ROUTES},
               "seed_spread": SPREAD, "eye_vs_twos_median": {k: med(v) for k, v in EYE_TWOS.items()},
               "source_span": {k: SRC_MET.get(k) for k in ("frames", "twos_score", "drawings_per_s", "flicker")},
               "span_b": B_ROWS and [{k: r.get(k) for k in ("arm", "seed", "frames", "first_mad", "last_mad", "twos_score", "drawings_per_s", "flicker", "flotime_s")} for r in B_ROWS],
               "twosclean": {k: {kk: vv for kk, vv in v.items() if kk not in ("order", "pair_list", "file", "src")} for k, v in TC.items()},
               "floors_s": FLOOR, "grading_statements_checked": len(BACKED)}
    NUMBERS_OUT.write_text(json.dumps(numbers, indent=1, default=str), encoding="utf-8")
    if OLD_NUMBERS.exists():
        OLD_NUMBERS.unlink()
    if OLD_NUMBERS.parent.exists() and not any(OLD_NUMBERS.parent.iterdir()):
        OLD_NUMBERS.parent.rmdir()
    MD_POINTER.write_text(POINTER_TEXT, encoding="utf-8")
    checked = check_tree()                              # the whole folder, workflows included, as it now stands on disk
    sizes = sorted(((p.stat().st_size, rel(p)) for p in MEDIA.rglob("*") if p.is_file()), reverse=True)
    total = sum(s for s, _ in sizes)
    print(f"index.html {len(page) / 1024:.0f} KB; md {len(mdtxt) / 1024:.0f} KB; {len(BACKED)} statements checked against the grading and notes; "
          f"voice check passed ({n_runs} comment runs screened)")
    print(f"text files checked: {len(checked)}: " + ", ".join(checked))
    print(f"numbers record: {NUMBERS_OUT.relative_to(ROOT).as_posix()}")
    print(f"media: {len(sizes)} files, {total / 1e6:.1f} MB; largest: " + ", ".join(f"{p} {s / 1e6:.1f} MB" for s, p in sizes[:4]))
    assert total < 300e6 and all(s < 50e6 for s, _ in sizes)


if __name__ == "__main__":
    main()
