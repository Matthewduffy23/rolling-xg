"""
Rolling xG — standalone Streamlit app.

Upload Wyscout game-by-game XLSX exports (player, goalkeeper or team stats),
build a rolling-average over/under-performance chart, export it as a PNG.

No file paths are hardcoded anywhere: every byte of data comes from
st.file_uploader at runtime.
"""

from __future__ import annotations

import io
import math
import os
import re
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
BG = "#0E1117"
PANEL = "#161B22"
TEXT = "#F2F4F7"
MUTED = "#9AA4B2"

OVER_DEFAULT = "#1F3FE0"
UNDER_DEFAULT = "#B80D0D"
LINE_DEFAULT = "#FF1A1A"

MINUS = "−"  # unicode minus

EXPORT_SIZES = {
    "1600 x 900 (Standard)": (1600, 900),
    "1920 x 1080": (1920, 1080),
    "1080 x 1080": (1080, 1080),
    "1080 x 1350": (1080, 1350),
    "1080 x 1920": (1080, 1920),
}

# Columns that are never coerced to numbers.
TEXT_COLS = {
    "Match",
    "Competition",
    "Team",
    "Position",
    "Scheme",
    "Date",
    "Season",
    "Opponent",
}


# ==========================================================================
# Parsing
# ==========================================================================
def _isna(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _split_header(name: str) -> list[str]:
    """Split a Wyscout compound header into base + sub-parts.

    "Shots / on target"                        -> ["Shots", "on target"]
    "Losses / Low / Medium / High"             -> ["Losses", "Low", "Medium", "High"]
    "Penalty area entries (runs / crosses)"    -> ["Penalty area entries", "runs", "crosses"]
    """
    m = re.match(r"^(.*?)\s*\((.*)\)\s*$", name)
    if m and "/" in m.group(2):
        return [m.group(1).strip()] + [p.strip() for p in m.group(2).split("/")]
    return [p.strip() for p in name.split("/")]


def expand_headers(row) -> list[str]:
    """Expand a Wyscout header row into one flat, unique name per column.

    A compound header occupies its own cell plus every following NaN cell.
    Any column beyond the named sub-parts is the percentage column.
    """
    row = list(row)
    n = len(row)
    names: list[str] = []
    i = 0
    while i < n:
        raw = row[i]
        j = i + 1
        while j < n and _isna(row[j]):
            j += 1
        span = j - i

        if _isna(raw):
            base_name = f"Column {i + 1}"
        else:
            base_name = str(raw).strip()

        if span == 1:
            block = [base_name]
        else:
            parts = _split_header(base_name)
            base = parts[0]
            subs = parts[1:]
            block = [base]
            for k in range(1, span):
                if k - 1 < len(subs):
                    block.append(f"{base} / {subs[k - 1]}")
                else:
                    block.append(f"{base} %")
        names.extend(block)
        i = j

    # de-dupe
    seen: dict[str, int] = {}
    out: list[str] = []
    for nm in names:
        if nm in seen:
            seen[nm] += 1
            out.append(f"{nm} ({seen[nm]})")
        else:
            seen[nm] = 1
            out.append(nm)
    return out


def parse_dates(s: pd.Series) -> pd.Series:
    """Best-effort date parsing; unparseable values become NaT."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    best = None
    attempts = ({"format": "ISO8601"}, {"format": "mixed", "dayfirst": True}, {"dayfirst": True})
    for kw in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = pd.to_datetime(s, errors="coerce", **kw)
        except Exception:
            continue
        if best is None or got.notna().sum() > best.notna().sum():
            best = got
        if best is not None and best.notna().sum() == len(s):
            break
    if best is None:
        return pd.Series(pd.NaT, index=s.index)
    return best


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every non-text column to numeric, keeping genuinely textual ones."""
    for col in df.columns:
        if col in TEXT_COLS:
            continue
        s = df[col]
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() == 0 and s.notna().sum() > 0:
            continue  # looks like free text (e.g. "4-2-3-1 (100.0%)")
        df[col] = conv
    return df


def read_workbook(file, sheet=0) -> pd.DataFrame:
    """Read one Wyscout XLSX export into a tidy frame."""
    raw = pd.read_excel(file, sheet_name=sheet, header=None)
    if raw.empty:
        return pd.DataFrame()

    names = expand_headers(list(raw.iloc[0]))
    df = raw.iloc[1:].copy()
    df.columns = names[: df.shape[1]]
    df = df.reset_index(drop=True)

    date_col = None
    for c in df.columns:
        if str(c).strip().lower() == "date":
            date_col = c
            break
    if date_col is None:
        raise ValueError("No 'Date' column found in this file.")
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    dates = parse_dates(df["Date"])
    df = df[dates.notna()].copy()
    df["Date"] = dates[dates.notna()].values

    df = coerce_numeric(df)
    for c in ("Match", "Competition", "Team", "Position", "Scheme"):
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), None)
    return df.reset_index(drop=True)


def season_label(ts, calendar_year: bool = False) -> str:
    if pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    if calendar_year:
        return str(ts.year)
    start = ts.year if ts.month >= 7 else ts.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def load_files(files, calendar_year: bool = False) -> pd.DataFrame:
    """Read + concat + de-dupe + sort every uploaded workbook."""
    frames = []
    for f in files:
        try:
            f.seek(0)
        except Exception:
            pass
        frames.append(read_workbook(f))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    subset = [c for c in ("Date", "Match", "Team") if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="first")
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    df["Season"] = [season_label(d, calendar_year) for d in df["Date"]]
    return df


# ==========================================================================
# Mode helpers
# ==========================================================================
def pick_col(cols, *candidates, contains=None):
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    if contains:
        for c in cols:
            if contains.lower() in str(c).lower():
                return c
    return None


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def build_team_frame(df: pd.DataFrame, team: str, metric: str, basis: str):
    """Join each chosen-team row to its opponent row on Date + Match."""
    own = df[df["Team"] == team].copy()
    others = df[df["Team"] != team]

    key = ["Date", "Match"]
    opp = (
        others.drop_duplicates(subset=key)[key + ["Team", metric]]
        .rename(columns={"Team": "Opponent", metric: "__opp"})
    )
    merged = own.merge(opp, on=key, how="left")
    missing = int(merged["__opp"].isna().sum())

    own_v = pd.to_numeric(merged[metric], errors="coerce")
    opp_v = pd.to_numeric(merged["__opp"], errors="coerce")

    if basis == "Team - Opponent":
        diff = own_v - opp_v
    elif basis == "Opponent - Team":
        diff = opp_v - own_v
    elif basis == "Team":
        diff = own_v
    else:  # "Opponent"
        diff = opp_v

    merged["__own_v"] = own_v
    merged["__opp_v"] = opp_v
    merged["__diff"] = diff
    return merged, missing


# ==========================================================================
# Formatting
# ==========================================================================
def fmt_num(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    v = float(v)
    r = round(v, 2)
    if abs(r - round(r)) < 1e-9:
        return f"{int(round(r)):,}"
    return f"{r:,.2f}"


def fmt_signed(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    v = float(v)
    r = round(v, 2)
    body = fmt_num(abs(r))
    return f"{MINUS}{body}" if r < 0 else f"+{body}"


def slugify(text: str, fallback: str = "rolling_chart") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_")
    return s.lower() or fallback


def game_option_label(g: int, date, match) -> str:
    d = ""
    if not pd.isna(date):
        d = pd.Timestamp(date).strftime("%d %b %y")
    m = str(match) if match is not None and not pd.isna(match) else ""
    bits = [f"{g:>3}"]
    if d:
        bits.append(d)
    if m:
        bits.append(m)
    return " · ".join(bits)


# ==========================================================================
# Chart
# ==========================================================================
def _line_height(fs: float, dpi: int, height_px: int, mult: float = 1.35) -> float:
    """Line height as a fraction of figure height, in real pixels."""
    return fs * mult * dpi / 72.0 / height_px


def build_chart(res: pd.DataFrame, cfg: dict) -> bytes:
    W = int(cfg["width"])
    H = int(cfg["height"])
    dpi = int(cfg["dpi"])

    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=BG)

    left_px, right_px = 120.0, 58.0
    left, right = left_px / W, 1.0 - right_px / W
    avail_px = W - left_px - right_px

    # ---- title block (pixel-aware line heights) ----
    tfs = float(cfg["title_fs"])
    y = 1.0 - 26.0 / H

    cpl = max(12, int(avail_px / (0.56 * tfs * dpi / 72.0)))
    title_lines: list[tuple[str, bool]] = []
    for idx, t in enumerate((cfg.get("title1"), cfg.get("title2"))):
        t = str(t or "").strip()
        if not t:
            continue
        for w in textwrap.wrap(t, cpl) or [t]:
            title_lines.append((w, idx == 0))

    for txt, bold in title_lines:
        fig.text(
            left, y, txt,
            ha="left", va="top", color=TEXT, fontsize=tfs,
            fontweight="bold" if bold else "normal",
        )
        y -= _line_height(tfs, dpi, H)

    sub = str(cfg.get("subtitle") or "").strip()
    if sub:
        sfs = max(9.0, tfs * 0.46)
        if title_lines:
            y -= 5.0 / H
        scpl = max(12, int(avail_px / (0.56 * sfs * dpi / 72.0)))
        for w in textwrap.wrap(sub, scpl) or [sub]:
            fig.text(left, y, w, ha="left", va="top", color=MUTED,
                     fontsize=sfs, style="italic")
            y -= _line_height(sfs, dpi, H)

    top = y - 20.0 / H
    top = float(np.clip(top, 0.30, 0.97))

    # ---- footer ----
    ffs = max(7.5, tfs * 0.30)
    footer = str(cfg.get("footer") or "").strip()
    if footer:
        fig.text(left, 16.0 / H, footer, ha="left", va="bottom",
                 color=MUTED, fontsize=ffs)

    xmode = cfg.get("xaxis_mode", "Season")
    bottom_px = (30.0 if footer else 12.0) + (30.0 if xmode != "None" else 6.0)
    bottom = float(np.clip(bottom_px / H, 0.04, 0.4))

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)

    x = res["game"].to_numpy(dtype=float)
    roll = res["rolling"].to_numpy(dtype=float)
    diff = res["diff"].to_numpy(dtype=float)
    n = len(x)

    # ---- y limits ----
    if cfg.get("ylim_mode") == "Fixed":
        lim = float(cfg.get("ylim_value") or 1.0) or 1.0
    else:
        finite = roll[np.isfinite(roll)]
        m = float(np.nanmax(np.abs(finite))) if finite.size else 0.0
        lim = m * 1.15 if m > 0 else 1.0
    ax.set_ylim(-lim, lim)
    ax.set_xlim(x[0] if n else 0, x[-1] if n else 1)
    ax.margins(x=0)

    # ---- series ----
    if cfg.get("chart_type", "Area (over / under)").startswith("Area"):
        safe = np.nan_to_num(roll, nan=0.0)
        ax.fill_between(x, safe, 0, where=(safe >= 0), interpolate=True,
                        color=cfg.get("over_color", OVER_DEFAULT), linewidth=0)
        ax.fill_between(x, safe, 0, where=(safe <= 0), interpolate=True,
                        color=cfg.get("under_color", UNDER_DEFAULT), linewidth=0)
        ax.axhline(0, color="#FFFFFF", linewidth=1.2, zorder=4)
    else:
        ax.axhline(0, color=MUTED, linewidth=0.9, linestyle="--", alpha=0.35, zorder=1)
        ax.plot(x, roll, color=cfg.get("line_color", LINE_DEFAULT),
                linewidth=float(cfg.get("line_width", 2.5)), solid_capstyle="round",
                zorder=4)

    # ---- cosmetics ----
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", length=0, colors=MUTED,
                   labelsize=max(7.5, tfs * 0.30))
    ax.set_yticks([])
    if cfg.get("gridlines"):
        ax.set_yticks(np.linspace(-lim, lim, 5))
        ax.set_yticklabels([])
        ax.grid(axis="y", color=MUTED, alpha=0.14, linewidth=0.8)
        ax.set_axisbelow(True)

    # ---- x axis ----
    if xmode == "None" or n == 0:
        ax.set_xticks([])
    elif xmode == "Season":
        ticks, labels = [], []
        prev = None
        for g, s in zip(res["game"], res["Season"]):
            if s != prev:
                ticks.append(g)
                labels.append(s)
                prev = s
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif xmode == "Game number":
        step = max(1, int(round(n / 10)))
        ticks = list(range(1, n + 1, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
    else:  # Date
        step = max(1, int(round(n / 10)))
        idx = list(range(0, n, step))
        ax.set_xticks([res["game"].iloc[i] for i in idx])
        ax.set_xticklabels(
            [pd.Timestamp(res["Date"].iloc[i]).strftime("%b %y") for i in idx]
        )

    # ---- rotated y labels ----
    lab_fs = max(9.0, tfs * 0.38)
    lab_x = 40.0 / W
    fig.text(lab_x, bottom + 0.75 * (top - bottom), cfg.get("over_label", "Over Performance"),
             rotation=90, ha="center", va="center", color=MUTED, fontsize=lab_fs)
    fig.text(lab_x, bottom + 0.25 * (top - bottom), cfg.get("under_label", "Under Performance"),
             rotation=90, ha="center", va="center", color=MUTED, fontsize=lab_fs)

    # ---- annotations ----
    ann_fs = max(7.5, tfs * 0.28)
    pill = dict(boxstyle="round,pad=0.32", facecolor=PANEL, edgecolor=MUTED,
                linewidth=0.7, alpha=0.92)

    if cfg.get("show_overall") and n:
        avg = float(np.nanmean(diff))
        ax.axhline(avg, color="#FFFFFF", linewidth=1.1, linestyle="--", alpha=0.85, zorder=5)
        ax.text(x[-1], avg, "  " + str(cfg.get("overall_label") or ""),
                ha="right", va="center", color=TEXT, fontsize=ann_fs,
                bbox=pill, zorder=6)

    for seg in cfg.get("segments", []):
        g0, g1 = int(seg["from"]), int(seg["to"])
        if g1 < g0:
            g0, g1 = g1, g0
        vals = diff[(x >= g0) & (x <= g1)]
        if not vals.size:
            continue
        v = float(np.nanmean(vals))
        color = seg.get("color", "#FFFFFF")
        if seg.get("full_width"):
            xs = [x[0], x[-1]]
        else:
            xs = [g0, g1]
        ax.plot(xs, [v, v], color=color, linewidth=1.1, linestyle="--", alpha=0.9, zorder=5)
        label = str(seg.get("label") or "")
        if label:
            ax.text(xs[1], v, "  " + label, ha="right", va="center", color=TEXT,
                    fontsize=ann_fs, bbox=pill, zorder=6)

    for h in cfg.get("custom_h", []):
        try:
            yv = float(h["y"])
        except (TypeError, ValueError):
            continue
        ax.axhline(yv, color=h.get("color", MUTED), linewidth=1.1,
                   linestyle="--", alpha=0.9, zorder=5)
        if str(h.get("label") or ""):
            ax.text(x[-1] if n else 1, yv, "  " + str(h["label"]), ha="right", va="center",
                    color=TEXT, fontsize=ann_fs, bbox=pill, zorder=6)

    # vertical labels are staggered by type so a season boundary and a
    # competition change on the same game do not print on top of each other
    def vline(gx, label, color, style=":", tier=0):
        ax.axvline(gx, color=color, linewidth=1.0, linestyle=style, alpha=0.8, zorder=3)
        if label:
            ax.text(gx, lim * (0.95 - 0.075 * tier), " " + str(label),
                    ha="left", va="top", color=color, fontsize=ann_fs, zorder=6)

    if cfg.get("show_season_lines") and n:
        prev = None
        for g, s in zip(res["game"], res["Season"]):
            if prev is not None and s != prev:
                vline(g, s, MUTED, tier=0)
            prev = s

    if cfg.get("show_change_lines") and n:
        for tier, col in enumerate(("Competition", "Team"), start=1):
            if col not in res.columns:
                continue
            prev = None
            for g, v in zip(res["game"], res[col]):
                if prev is not None and v != prev:
                    vline(g, v, MUTED, style="--", tier=tier)
                prev = v

    for vdef in cfg.get("custom_v", []):
        try:
            gx = float(vdef["game"])
        except (TypeError, ValueError):
            continue
        vline(gx, vdef.get("label", ""), vdef.get("color", TEXT), style="-.", tier=3)

    # ---- stat box ----
    if cfg.get("stat_text"):
        corner = cfg.get("stat_corner", "Top left")
        pos = {
            "Top left": (0.015, 0.97, "left", "top"),
            "Top right": (0.985, 0.97, "right", "top"),
            "Bottom left": (0.015, 0.03, "left", "bottom"),
            "Bottom right": (0.985, 0.03, "right", "bottom"),
        }[corner]
        ax.text(
            pos[0], pos[1], str(cfg["stat_text"]), transform=ax.transAxes,
            ha=pos[2], va=pos[3], color=TEXT, fontsize=max(8.0, tfs * 0.30),
            linespacing=1.45, zorder=7,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=PANEL,
                      edgecolor=MUTED, linewidth=0.8, alpha=0.95),
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


# ==========================================================================
# UI
# ==========================================================================
CSS = f"""
<style>
  .stApp, [data-testid="stAppViewContainer"] {{ background:{BG}; color:{TEXT}; }}
  [data-testid="stSidebar"] > div:first-child {{ background:{PANEL}; }}
  [data-testid="stHeader"] {{ background:transparent; }}
  h1,h2,h3,h4,h5,h6,p,label,span,div {{ color:{TEXT}; }}
  .stCaption, small {{ color:{MUTED} !important; }}
  [data-testid="stExpander"], [data-testid="stFileUploaderDropzone"] {{
      background:{PANEL}; border:1px solid #2A313B; border-radius:10px; }}
  .stTextInput input, .stNumberInput input, .stTextArea textarea {{
      background:{BG}; color:{TEXT}; border-color:#2A313B; }}
  div[data-baseweb="select"] > div {{ background:{BG}; border-color:#2A313B; }}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="Rolling xG", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Rolling xG")
    st.caption("Upload Wyscout game-by-game exports, build a rolling-average "
               "performance chart, export a PNG.")

    mode = st.sidebar.radio("Mode", ["Player", "Goalkeeper", "Team"], index=0)
    calendar_year = st.sidebar.checkbox(
        "Calendar-year season labels (e.g. \"2026\")", value=False,
        help="Off = split-year labels: a game in Aug 2025 is \"2025-26\".",
    )

    files = st.file_uploader(
        "Wyscout XLSX export(s)", type=["xlsx", "xlsm"], accept_multiple_files=True,
        help="Player / Goalkeeper / Team stats exports. Multiple seasons can be "
             "uploaded together — duplicates are dropped automatically.",
    )
    if not files:
        st.info("Upload one or more Wyscout XLSX exports to begin.")
        return

    try:
        df = load_files(files, calendar_year=calendar_year)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the upload(s): {exc}")
        return

    if df.empty:
        st.error("No rows with a parseable date were found in the upload(s).")
        return

    st.success(f"Loaded {len(df):,} rows from {len(files)} file(s) "
               f"({df['Date'].min():%d %b %Y} - {df['Date'].max():%d %b %Y}).")

    nums = numeric_columns(df)

    # ---------------- filters ----------------
    st.sidebar.markdown("### Filters")
    work = df.copy()
    if "Competition" in work.columns:
        comps = sorted([c for c in work["Competition"].dropna().unique()])
        chosen = st.sidebar.multiselect("Competition", comps, default=comps)
        if chosen:
            work = work[work["Competition"].isin(chosen)]

    min_col = pick_col(work.columns, "Minutes played", contains="minutes")
    if mode in ("Player", "Goalkeeper") and min_col:
        min_minutes = st.sidebar.number_input("Min minutes played", 0, 130, 0, step=5)
        if min_minutes > 0:
            work = work[pd.to_numeric(work[min_col], errors="coerce").fillna(0) >= min_minutes]

    if work.empty:
        st.warning("No games left after filtering.")
        return

    # ---------------- mode: build diff ----------------
    st.sidebar.markdown("### Metric")
    over_label = "Over Performance"
    under_label = "Under Performance"
    missing_opp = 0

    if mode == "Player":
        a_def = pick_col(nums, "Goals") or (nums[0] if nums else None)
        e_def = pick_col(nums, "xG") or (nums[0] if nums else None)
        actual_col = st.sidebar.selectbox("Actual", nums,
                                          index=nums.index(a_def) if a_def in nums else 0)
        expected_col = st.sidebar.selectbox("Expected", nums,
                                            index=nums.index(e_def) if e_def in nums else 0)
        base = work.copy()
        a = pd.to_numeric(base[actual_col], errors="coerce")
        e = pd.to_numeric(base[expected_col], errors="coerce")
        base["diff"] = a - e
        stat_default = (f"{actual_col}: {fmt_num(a.sum())} / "
                        f"{expected_col}: {fmt_num(e.sum())} / "
                        f"{fmt_signed(a.sum() - e.sum())}")

    elif mode == "Goalkeeper":
        a_def = pick_col(nums, "Conceded goals", "Conceded", contains="conceded") or (nums[0] if nums else None)
        e_def = pick_col(nums, "xCG", "xGA", contains="xcg") or (nums[0] if nums else None)
        conceded_col = st.sidebar.selectbox("Conceded", nums,
                                            index=nums.index(a_def) if a_def in nums else 0)
        expected_col = st.sidebar.selectbox("Expected (xCG)", nums,
                                            index=nums.index(e_def) if e_def in nums else 0)
        base = work.copy()
        c = pd.to_numeric(base[conceded_col], errors="coerce")
        e = pd.to_numeric(base[expected_col], errors="coerce")
        base["diff"] = e - c  # positive = over-performing
        actual_col, expected_col = conceded_col, expected_col
        stat_default = (f"{conceded_col}: {fmt_num(c.sum())} / "
                        f"{expected_col}: {fmt_num(e.sum())} / "
                        f"{fmt_signed(e.sum() - c.sum())}")

    else:  # Team
        if "Team" not in work.columns:
            st.error("This upload has no 'Team' column - it does not look like a "
                     "Wyscout Team Stats export.")
            return
        counts = work["Team"].dropna().value_counts()
        if counts.empty:
            st.error("No team rows found.")
            return
        team = st.sidebar.selectbox("Team", list(counts.index), index=0)
        metric_default = pick_col(nums, "xG") or (nums[0] if nums else None)
        metric = st.sidebar.selectbox("Metric", nums,
                                      index=nums.index(metric_default) if metric_default in nums else 0)
        basis = st.sidebar.selectbox(
            "Basis", ["Team - Opponent", "Team", "Opponent", "Opponent - Team"], index=0)

        base, missing_opp = build_team_frame(work, team, metric, basis)
        base = base.rename(columns={"__diff": "diff"})
        if basis in ("Team", "Opponent"):
            over_label = f"{metric} ↑"
            under_label = f"{metric} ↓"

        own_tot = base["__own_v"].sum(skipna=True)
        opp_tot = base["__opp_v"].sum(skipna=True)
        if basis == "Team":
            stat_default = f"{metric} ({team}): {fmt_num(own_tot)}"
        elif basis == "Opponent":
            stat_default = f"{metric} (opponents): {fmt_num(opp_tot)}"
        elif basis == "Team - Opponent":
            stat_default = (f"{metric}: {fmt_num(own_tot)} / "
                            f"Opp: {fmt_num(opp_tot)} / {fmt_signed(own_tot - opp_tot)}")
        else:
            stat_default = (f"Opp {metric}: {fmt_num(opp_tot)} / "
                            f"{team}: {fmt_num(own_tot)} / {fmt_signed(opp_tot - own_tot)}")
        actual_col, expected_col = metric, f"Opponent {metric}"

        st.info(f"{team}: {len(base):,} matches. "
                + (f"**{missing_opp} opponent row(s) missing.**" if missing_opp
                   else "0 opponent rows missing."))

    base = base[base["diff"].notna()].copy()
    if base.empty:
        st.warning("No games with a usable value for the chosen metric.")
        return
    base = base.sort_values("Date", kind="mergesort").reset_index(drop=True)
    base["game"] = np.arange(1, len(base) + 1)

    # ---------------- rolling ----------------
    st.sidebar.markdown("### Rolling")
    window = int(st.sidebar.number_input("Window (games)", 1, 100, 10, step=1))
    full_only = st.sidebar.checkbox("Start only once full window available", value=True)
    base["rolling"] = base["diff"].rolling(
        window=window, min_periods=window if full_only else 1).mean()

    res = base.copy()
    for c in ("Season", "Competition", "Match", "Team"):
        if c not in res.columns:
            res[c] = ""

    n = len(res)

    # ---------------- chart settings ----------------
    st.sidebar.markdown("### Chart")
    chart_type = st.sidebar.radio("Type", ["Area (over / under)", "Line"], index=0)
    if chart_type.startswith("Area"):
        c1, c2 = st.sidebar.columns(2)
        over_color = c1.color_picker("Over", OVER_DEFAULT)
        under_color = c2.color_picker("Under", UNDER_DEFAULT)
        line_color, line_width = LINE_DEFAULT, 2.5
    else:
        over_color, under_color = OVER_DEFAULT, UNDER_DEFAULT
        line_color = st.sidebar.color_picker("Line", LINE_DEFAULT)
        line_width = st.sidebar.slider("Line width", 0.5, 8.0, 2.5, 0.1)

    ylim_mode = st.sidebar.radio("Y limits", ["Auto (symmetric)", "Fixed"], index=0)
    ylim_value = 1.0
    if ylim_mode == "Fixed":
        ylim_value = st.sidebar.number_input("Fixed y limit (±)", 0.01, 1000.0, 1.0, step=0.1)
    xaxis_mode = st.sidebar.radio("X-axis labels", ["Season", "Game number", "Date", "None"], index=0)
    gridlines = st.sidebar.checkbox("Y gridlines", value=False)

    st.sidebar.markdown("### Titles")
    default_t1 = ""
    if mode == "Team" and "Team" in res.columns:
        default_t1 = str(res["Team"].iloc[0]) if len(res) else ""
    title1 = st.sidebar.text_input("Title line 1", default_t1)
    title2 = st.sidebar.text_input("Title line 2", "")
    subtitle = st.sidebar.text_input("Subtitle", f"{window}-Game Rolling Average")
    footer = st.sidebar.text_input("Footer", "Data: Wyscout")
    title_fs = st.sidebar.slider("Title font size", 10, 60, 26, 1)

    st.sidebar.markdown("### Stat box")
    show_stats = st.sidebar.checkbox("Show stat box", value=True)
    stat_corner = st.sidebar.selectbox(
        "Corner", ["Top left", "Top right", "Bottom left", "Bottom right"], index=0)
    stat_text = st.sidebar.text_area("Stat text", stat_default, height=90) if show_stats else ""

    st.sidebar.markdown("### Export")
    size_key = st.sidebar.selectbox("Size", list(EXPORT_SIZES.keys()), index=0)
    dpi = st.sidebar.selectbox("DPI", [100, 150, 200, 300], index=0)
    W, H = EXPORT_SIZES[size_key]

    # ---------------- annotations ----------------
    st.markdown("### Annotations")
    col1, col2, col3 = st.columns(3)

    opts = [game_option_label(int(g), d, m)
            for g, d, m in zip(res["game"], res["Date"], res["Match"])]

    with col1:
        st.markdown("**Horizontal - averages**")
        show_overall = st.checkbox("Overall average", value=False)
        overall_label = ""
        if show_overall:
            overall_label = st.text_input(
                "Overall label", f"Average {fmt_signed(res['diff'].mean())}",
                key="overall_lbl")
        n_seg = int(st.number_input("Segment averages", 0, 10, 0, step=1))
        segments = []
        for i in range(n_seg):
            with st.expander(f"Segment {i + 1}", expanded=(i == 0)):
                fi = st.selectbox("From", range(n), index=0,
                                  format_func=lambda k: opts[k], key=f"sf{i}")
                ti = st.selectbox("To", range(n), index=n - 1,
                                  format_func=lambda k: opts[k], key=f"st{i}")
                lo, hi = sorted((fi, ti))
                val = float(res["diff"].iloc[lo:hi + 1].mean())
                lbl = st.text_input("Label", f"Average {fmt_signed(val)}", key=f"sl{i}")
                full = st.checkbox("Span full width", value=False, key=f"sw{i}")
                sc = st.color_picker("Colour", "#FFFFFF", key=f"sc{i}")
                segments.append({"from": int(res["game"].iloc[lo]),
                                 "to": int(res["game"].iloc[hi]),
                                 "label": lbl, "full_width": full, "color": sc})

    with col2:
        st.markdown("**Horizontal - custom**")
        n_ch = int(st.number_input("Custom horizontals", 0, 10, 0, step=1))
        custom_h = []
        for i in range(n_ch):
            with st.expander(f"Line {i + 1}", expanded=(i == 0)):
                yv = st.number_input("Y value", value=0.0, step=0.05, format="%.4f", key=f"chy{i}")
                lbl = st.text_input("Label", "", key=f"chl{i}")
                cc = st.color_picker("Colour", MUTED, key=f"chc{i}")
                custom_h.append({"y": yv, "label": lbl, "color": cc})

    with col3:
        st.markdown("**Vertical**")
        show_season_lines = st.checkbox("Season boundaries", value=False)
        show_change_lines = st.checkbox("Competition / club changes", value=False)
        n_cv = int(st.number_input("Custom verticals", 0, 10, 0, step=1))
        custom_v = []
        for i in range(n_cv):
            with st.expander(f"Marker {i + 1}", expanded=(i == 0)):
                gi = st.selectbox("At game", range(n), index=0,
                                  format_func=lambda k: opts[k], key=f"cvg{i}")
                lbl = st.text_input("Label", "", key=f"cvl{i}")
                cc = st.color_picker("Colour", TEXT, key=f"cvc{i}")
                custom_v.append({"game": int(res["game"].iloc[gi]),
                                 "label": lbl, "color": cc})

    cfg = {
        "width": W, "height": H, "dpi": dpi,
        "chart_type": chart_type,
        "over_color": over_color, "under_color": under_color,
        "line_color": line_color, "line_width": line_width,
        "ylim_mode": ylim_mode, "ylim_value": ylim_value,
        "xaxis_mode": xaxis_mode, "gridlines": gridlines,
        "title1": title1, "title2": title2, "subtitle": subtitle,
        "footer": footer, "title_fs": title_fs,
        "over_label": over_label, "under_label": under_label,
        "stat_text": stat_text if show_stats else "", "stat_corner": stat_corner,
        "show_overall": show_overall, "overall_label": overall_label,
        "segments": segments, "custom_h": custom_h, "custom_v": custom_v,
        "show_season_lines": show_season_lines,
        "show_change_lines": show_change_lines,
    }

    png = build_chart(res, cfg)

    st.markdown("### Chart")
    st.image(png, width="stretch")
    fname = slugify(f"{title1} {title2}".strip() or f"rolling_{mode}") + f"_{W}x{H}.png"
    st.download_button("Download PNG", data=png, file_name=fname, mime="image/png")

    with st.expander("Game-by-game"):
        table = pd.DataFrame({
            "Game": res["game"],
            "Date": pd.to_datetime(res["Date"]).dt.strftime("%d %b %Y"),
            "Competition": res["Competition"],
            "Match": res["Match"],
        })
        if mode == "Team":
            table[actual_col] = res["__own_v"].round(2)
            table[expected_col] = res["__opp_v"].round(2)
        else:
            table[actual_col] = pd.to_numeric(res[actual_col], errors="coerce").round(2)
            table[expected_col] = pd.to_numeric(res[expected_col], errors="coerce").round(2)
        table["Diff"] = res["diff"].round(3)
        table["Rolling"] = res["rolling"].round(3)
        st.dataframe(table, width="stretch", hide_index=True)


if os.environ.get("ROLLING_XG_NO_UI") != "1":
    main()
