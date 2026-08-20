"""
CIES-style ranking images — copied from the Scouting-Hub ecosystem.

  * player rows : ``cf_make_ranking_image`` + helpers, copied VERBATIM from
                  Matthewduffy23/Scouting-Hub ``pages/05_Strikers.py``
  * team rows   : ``_tri_make_image`` + helpers, copied VERBATIM from
                  Matthewduffy23/TEAM-HQ ``team_hq.py``
                  (the team app lives in its own repo, not in Scouting-Hub)

Nothing here is restyled. Every colour, font size, coordinate, zorder and
export mode is exactly as it is in the source apps.

Four additive deviations were needed to meet this app's brief. Each has a
default that reproduces the original output byte-for-byte, and the test harness
renders both this module and the untouched originals on the same dataframe to
prove the PNGs are identical:

  1. ``cf_make_ranking_image(..., max_rows=10)`` — the original hardcodes
     ``df_show.head(10)``, so a list longer than 10 silently lost rows. The cap
     is now a parameter; ``max_rows=10`` is the original behaviour.
  2. ``_tri_make_image(..., custom_footer_text=None)`` — the team version had no
     custom footer (the player version always did). ``None`` keeps the original
     hardcoded footer lines.
  3. ``_tri_make_image(..., value_label_col=None)`` — the team version always
     renders values through ``_tri_format`` (1 dp, optional %). ``None`` keeps
     that; passing a column name uses pre-formatted strings instead, which is
     how the player version has always worked via ``value_label_col``.
  4. ``cf_make_ranking_image(..., preformatted_labels=False)`` — the player
     version re-formats any label ``float()`` can parse, so a typed "42.50"
     came out as "42.5". ``False`` keeps that; ``True`` prints the label as
     given. Symmetric with 3, and only reachable from this app's own page.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# ``cf_format_value`` reads a page-level global in the Strikers script. It is
# defined here so the copied body below stays byte-identical; left as None it
# takes the composite-formatting branch, and pre-formatted strings (which is
# what this app passes) fall through ``str(v)`` untouched either way.
rank_mode = None


# =========================================================================
# PLAYER VERSION — verbatim from Scouting-Hub/pages/05_Strikers.py
# =========================================================================
def cf_load_remote_png(url: str):
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except Exception:
        return None

CF_BADGE_DIRS = [
    Path(__file__).resolve().parent / "badges",
    Path(__file__).resolve().parent / "crests",
]
for d in CF_BADGE_DIRS:
    d.mkdir(exist_ok=True)

def _cf_clean_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (name or "").lower()).strip("_")

@st.cache_data(show_spinner=False)
def cf_load_local_badge(team: str):
    key = _cf_clean_filename(team)
    if not key:
        return None
    for folder in CF_BADGE_DIRS:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = folder / f"{key}{ext}"
            if p.exists():
                try:
                    return plt.imread(str(p))
                except Exception:
                    continue
    return None

# OPTIONAL: import from your shared team URL map if you have it
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _CF_FOTMOB_TEAM_URLS
except Exception:
    _CF_FOTMOB_TEAM_URLS = {}

def cf_get_fotmob_url(team: str) -> str:
    return (_CF_FOTMOB_TEAM_URLS.get(team) or "").strip()

def cf_fotmob_team_id_from_url(team_url: str) -> str:
    try:
        m = re.search(r"/teams/(\d+)/", str(team_url or ""))
        return m.group(1) if m else ""
    except Exception:
        return ""

def cf_fotmob_crest_url(team: str) -> str:
    team_url = cf_get_fotmob_url(team)
    tid = cf_fotmob_team_id_from_url(team_url)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{tid}.png" if tid else ""

@st.cache_data(show_spinner=False)
def cf_load_fotmob_crest(team: str):
    url = cf_fotmob_crest_url(team)
    if not url:
        return None
    return cf_load_remote_png(url)


def cf_get_team_badge(row: pd.Series):
    team = str(row.get("Team", "")).strip()

    # 1) Local badge first
    img = cf_load_local_badge(team)
    if img is not None:
        return img

    # 2) FotMob crest fallback (team badge)  ✅ ADDED
    crest = cf_load_fotmob_crest(team)
    if crest is not None:
        return crest

    # 3) (Removed) nationality flag fallback – keep None so it doesn't show flags
    return None


# ---------------------------------------------------------
# 6C) Badge size normaliser (make badges fit like flags) ✅ ADDED

def cf_zoom_to_fit(img, target_px: int = 28) -> float:
    """
    Scale any badge image so its largest dimension becomes ~target_px.
    """
    try:
        h, w = img.shape[0], img.shape[1]
        m = max(h, w)
        if m <= 0:
            return 1.0
        return float(target_px) / float(m)
    except Exception:
        return 1.0



def cf_footer_lines_for_metric(metric_label: str, show_ls: bool):
    ls_txt = "(league strength applied)." if show_ls else "(no league-strength adjustment)."

    if metric_label == "Impact Score" or metric_label.startswith("Impact Score"):
        return [
            "Impact Score (CF): combines Carrying, Playmaking, Target Man, Chance Creation and Goal Threat.",
            "Adjusted for minutes played and team context vs league.",
            f"Displayed 0–100 vs the full selected pool {ls_txt}",
        ]
    if metric_label.startswith("Complete Score"):
        return [
            "Complete Score (CF): weighted blend of passing, carrying, progression, xG/xA and box threat.",
            f"Displayed 0–100 vs the full selected pool {ls_txt}",
        ]
    if metric_label.startswith("Custom Combo"):
        return [
            "Custom Combo (CF): equal-weight blend of chosen base scores (Carrying/Playmaking/Target Man/Chance Creation/Goal Threat).",
            f"Displayed vs the selected pool {ls_txt}",
        ]
    return [
        f"{metric_label} (CF): ranks this metric only.",
        f"Displayed 0–100 vs the full selected pool {ls_txt}",
    ]


# ---------------------------------------------------------
# 8) RANKING IMAGE (Standard + 1920×1080)
# ---------------------------------------------------------

def cf_format_value(v):
    if v is None:
        return "—"
    try:
        v = float(v)
    except Exception:
        return str(v)
    if np.isnan(v):
        return "—"

    # 👉 ONLY raw metric mode: always 2dp
    if rank_mode == "Raw metric (any numeric column)":
        return f"{v:.2f}"

    # existing composite formatting
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def cf_make_ranking_image(
    df_show: pd.DataFrame,
    metric_col: str,
    value_label_col: str,
    metric_label: str,
    title_lines,
    brand_logo_url=None,
    show_ls: bool = False,
    show_age: bool = False,
    highlight_players=None,
    export_mode: str = "Standard (auto)",
    theme: str = "Light",
    custom_footer_text: str = None,
    max_rows: int = 10,  # DEVIATION 1: was a hardcoded head(10)
    preformatted_labels: bool = False,  # DEVIATION 4
) -> bytes:

    df_top = df_show.head(max_rows).copy()
    if df_top.empty:
        return b""

    hi_set = set()
    if highlight_players:
        hi_set = {str(x).strip().lower() for x in highlight_players if str(x).strip()}

    def is_hi(row: pd.Series) -> bool:
        return str(row.get("Player", "")).strip().lower() in hi_set

    # theme palette
    if theme == "Dark":
        BG = "#0a0f1c"
        ROW_A, ROW_B = "#0f1628", "#0b1222"
        TXT, SUB, FOOT = "#ffffff", "#b8c0cf", "#9aa6bd"
        DIV = "#23304a"
        BAR_BG, BAR_FG = "#1a2540", "#6b7cff"
        RANK_BG, RANK_EDGE = "#111a2e", "#2b3a5a"
        HILITE, HILITE_EDGE = "#f6d46b", "#d2a100"
    else:
        BG = "#ffffff"
        ROW_A, ROW_B = "#f7f7f7", "#ffffff"
        TXT, SUB, FOOT = "#111111", "#777777", "#9b9b9b"
        DIV = "#e2e2e2"
        BAR_BG, BAR_FG = "#e1e1e1", "#bfbfbf"
        RANK_BG, RANK_EDGE = "#f3f3f3", "#c0c0c0"
        HILITE, HILITE_EDGE = "#f6d46b", "#d2a100"

    scores = pd.to_numeric(df_top[metric_col], errors="coerce")
    max_score = float(scores.max()) if scores.notna().any() else 1.0

    # Footer lines
    if custom_footer_text:
        footer_lines = [ln.strip() for ln in custom_footer_text.split("\n") if ln.strip()]
    else:
        footer_lines = cf_footer_lines_for_metric(metric_label, show_ls)

    # =====================================================
    # 1920×1080 banner
    # =====================================================
    if export_mode == "1920×1080 (banner)":
        DPI = 100
        fig = plt.figure(figsize=(1920.0 / DPI, 1080.0 / DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, color=BG, zorder=0))

        LEFT, RIGHT = 0.045, 0.955

        t1 = title_lines[0].upper() if len(title_lines) > 0 else ""
        t2 = title_lines[1].upper() if len(title_lines) > 1 else ""
        t3 = title_lines[2].upper() if len(title_lines) > 2 else ""

        ax.text(LEFT, 0.972, t1, fontsize=48, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(LEFT, 0.912, t2, fontsize=34, fontweight="bold", color=TXT, ha="left", va="top")
        ax.text(LEFT, 0.870, t3, fontsize=20, color=SUB, ha="left", va="top")

        header_div_y = 0.835
        ax.plot([LEFT, RIGHT], [header_div_y, header_div_y], color=DIV, lw=2.2)

        footer_div_y = 0.040
        ax.plot([LEFT, RIGHT], [footer_div_y, footer_div_y], color=DIV, lw=2.2)

        for i, line in enumerate(footer_lines):
            ax.text(
                LEFT,
                footer_div_y - 0.018 - i * 0.024,
                line,
                fontsize=13,
                color=FOOT,
                ha="left",
                va="top",
                zorder=10,
            )

        ROW_TOP = header_div_y - 0.022
        ROW_BOT = footer_div_y + 0.010
        # DEVIATION 1 (cont.): was a literal / 10.0. max(len, 10) is
        # identical for every input the original could produce (it capped
        # at 10 rows) and only shrinks the band once the cap is lifted.
        row_gap = (ROW_TOP - ROW_BOT) / float(max(len(df_top), 10))
        row_h   = row_gap * 0.99

        RANK_X  = LEFT + 0.024
        CREST_X = LEFT + 0.112
        NAME_X  = LEFT + 0.190

        BAR_L   = LEFT + 0.63
        BAR_R   = RIGHT - 0.155
        BAR_W   = BAR_R - BAR_L
        BAR_H   = row_h * 0.26

        VAL_X   = RIGHT - 0.030

        NAME_FS = 28
        TEAM_FS = 19
        NAME_DY = row_h * 0.20
        TEAM_DY = row_h * 0.26

        for i, (_, row) in enumerate(df_top.iterrows()):
            y = ROW_TOP - (i + 0.5) * row_gap

            ax.add_patch(Rectangle(
                (LEFT, y - row_h / 2),
                RIGHT - LEFT,
                row_h,
                color=(ROW_A if i % 2 == 0 else ROW_B),
                zorder=1,
            ))

            if is_hi(row):
                ax.add_patch(Rectangle(
                    (LEFT, y - row_h / 2),
                    RIGHT - LEFT,
                    row_h,
                    color=HILITE,
                    alpha=0.22,
                    zorder=2,
                ))
                ax.add_patch(Rectangle(
                    (LEFT, y - row_h / 2),
                    RIGHT - LEFT,
                    row_h,
                    fill=False,
                    edgecolor=HILITE_EDGE,
                    lw=2.2,
                    zorder=3,
                ))

            ax.scatter(
                [RANK_X], [y],
                s=1320,
                facecolor=RANK_BG,
                edgecolor=(HILITE_EDGE if is_hi(row) else RANK_EDGE),
                linewidths=2.2,
                zorder=4,
            )
            ax.text(
                RANK_X, y, str(i + 1),
                fontsize=16, fontweight="bold", color=TXT,
                ha="center", va="center", zorder=5
            )

            badge = cf_get_team_badge(row)
            if badge is not None:
                z = cf_zoom_to_fit(badge, target_px=52)
                ax.add_artist(AnnotationBbox(
                    OffsetImage(badge, zoom=z),
                    (CREST_X, y),
                    frameon=False,
                    zorder=5,
                ))

            player = str(row.get("Player", "")).upper()
            team   = str(row.get("Team", ""))
            league = str(row.get("League", ""))

            ax.text(
                NAME_X, y + NAME_DY, player,
                fontsize=NAME_FS, fontweight="bold", color=TXT,
                ha="left", va="center", zorder=6
            )
            ax.text(
                NAME_X, y - TEAM_DY, f"{team} ({league})",
                fontsize=TEAM_FS, color=SUB,
                ha="left", va="center", zorder=6
            )

            v_bar = float(row[metric_col]) if pd.notna(row[metric_col]) else 0.0
            frac  = (v_bar / max_score) if max_score else 0.0
            frac  = max(0.0, min(1.0, frac))

            ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W, BAR_H, color=BAR_BG, zorder=2))
            ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W * frac, BAR_H, color=BAR_FG, zorder=3))

            v_lab = row.get(value_label_col)
            ax.text(
                VAL_X, y, (str(v_lab) if preformatted_labels else cf_format_value(v_lab)),
                fontsize=29, fontweight="bold", color=TXT,
                ha="right", va="center", zorder=6
            )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    # =====================================================
    # Standard (auto height)
    # =====================================================
    N        = len(df_top)
    ROW_H    = 0.82
    HEADER_H = 1.70
    FOOT_H   = 0.70
    TOTAL_H  = HEADER_H + N * ROW_H + FOOT_H

    fig = plt.figure(figsize=(8.3, TOTAL_H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, TOTAL_H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1.0, TOTAL_H, color=BG, zorder=0))

    t1 = title_lines[0].upper() if len(title_lines) > 0 else ""
    t2 = title_lines[1].upper() if len(title_lines) > 1 else ""
    t3 = title_lines[2].upper() if len(title_lines) > 2 else ""
    title_y = TOTAL_H - 0.25
    ax.text(0.04, title_y,        t1, fontsize=19, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, title_y - 0.34, t2, fontsize=14, fontweight="bold", color=TXT, ha="left", va="top")
    ax.text(0.04, title_y - 0.62, t3, fontsize=11, color=SUB, ha="left", va="top")

    base_y = TOTAL_H - HEADER_H
    ax.plot([0.04, 0.96], [base_y + ROW_H/2 + 0.02]*2, color=DIV, lw=1.1, zorder=2)

    LEFT, RIGHT = 0.04, 0.96
    BAR_L, BAR_R = 0.66, 0.82
    BAR_W = BAR_R - BAR_L
    BAR_H = 0.14
    VAL_X = 0.94
    crest_x = 0.14

    for i, (_, row) in enumerate(df_top.iterrows()):
        y = base_y - i * ROW_H

        ax.add_patch(Rectangle((LEFT, y - ROW_H/2), RIGHT - LEFT, ROW_H,
                               color=(ROW_A if i % 2 == 0 else ROW_B), zorder=1))

        if is_hi(row):
            ax.add_patch(Rectangle((LEFT, y - ROW_H/2), RIGHT - LEFT, ROW_H,
                                   color=HILITE, alpha=0.25, zorder=2))
            ax.add_patch(Rectangle((LEFT, y - ROW_H/2), RIGHT - LEFT, ROW_H,
                                   fill=False, edgecolor=HILITE_EDGE, lw=1.3, zorder=3))

        ax.scatter([0.07], [y], s=520, facecolor=RANK_BG,
                   edgecolor=(HILITE_EDGE if is_hi(row) else RANK_EDGE),
                   linewidths=1.2, zorder=4)
        ax.text(0.07, y, str(i+1), fontsize=10, fontweight="bold",
                color=TXT, ha="center", va="center", zorder=5)

        badge = cf_get_team_badge(row)
        if badge is not None:
            z = cf_zoom_to_fit(badge, target_px=40)
            ax.add_artist(AnnotationBbox(
                OffsetImage(badge, zoom=z),
                (crest_x, y),
                frameon=False,
                zorder=5
            ))

        ax.text(0.21, y + 0.12, str(row.get("Player", "")).upper(),
                fontsize=16, fontweight="bold", color=TXT, ha="left", va="center", zorder=5)

        team = str(row.get("Team", ""))
        league = str(row.get("League", ""))
        ax.text(0.21, y - 0.10, f"{team} ({league})",
                fontsize=12, color=SUB, ha="left", va="center", zorder=5)

        v_bar = float(row[metric_col]) if pd.notna(row[metric_col]) else 0.0
        frac = (v_bar / max_score) if max_score else 0.0
        frac = max(0.0, min(1.0, frac))

        ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W, BAR_H, color=BAR_BG, zorder=2))
        ax.add_patch(Rectangle((BAR_L, y - BAR_H/2), BAR_W * frac, BAR_H, color=BAR_FG, zorder=3))

        v_lab = row.get(value_label_col)
        ax.text(VAL_X, y, (str(v_lab) if preformatted_labels else cf_format_value(v_lab)),
                fontsize=16, fontweight="bold", color=TXT, ha="right", va="center", zorder=6)

    ax.plot([LEFT, RIGHT], [0.82]*2, color=DIV, lw=0.9, zorder=2)
    for j, line in enumerate(footer_lines):
        ax.text(LEFT, 0.62 - j*0.18, line, fontsize=9.5, color=FOOT, ha="left", va="top", zorder=4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# =========================================================================
# TEAM VERSION — verbatim from TEAM-HQ/team_hq.py
# =========================================================================
try:
    from team_fotmob_urls import FOTMOB_TEAM_URLS as _FOTMOB_URLS
except Exception:
    _FOTMOB_URLS = {}

try:
    from league_logo_urls import get_league_logo_url as _get_league_logo_url
except Exception:
    def _get_league_logo_url(lg): return ""

@st.cache_data(show_spinner=False)
def load_remote_img(url: str):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        return plt.imread(io.BytesIO(r.content))
    except Exception:
        return None

def fotmob_crest_url(team: str) -> str:
    raw = (_FOTMOB_URLS.get(team) or "").strip()
    if not raw: return ""
    m = re.search(r"/teams/(\d+)/", raw)
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{m.group(1)}.png" if m else ""

@st.cache_data(show_spinner=False)
def get_team_badge(team: str):
    url = fotmob_crest_url(team)
    if url:
        img = load_remote_img(url)
        if img is not None:
            return img
    return None


def _tri_value_text(row, pct_col, value_label_col):
    """DEVIATION 3: pre-formatted label when asked for, else the original."""
    if value_label_col:
        return str(row.get(value_label_col, ""))
    return _tri_format(row["_tri_val"], pct_col)


def _tri_format(val, col_name):
    try:
        v = float(val)
        if np.isnan(v): return "—"
    except: return "—"
    suffix = "%" if "%" in str(col_name) else ""
    return f"{v:.1f}{suffix}"

def _tri_make_image(df_show, rank_col, rank_label, pct_col, is_raw, title_lines, theme, export_mode, top_n, highlight_team=None,
                    custom_footer_text=None,   # DEVIATION 2
                    value_label_col=None):     # DEVIATION 3
    if df_show.empty:
        return b""

    # Theme palette
    if theme == "Dark":
        BG="#0a0f1c"; ROW_A="#0f1628"; ROW_B="#0b1222"
        TXT="#ffffff"; SUB="#b8c0cf"; FOOT="#9aa6bd"; DIV="#23304a"
        BAR_BG="#1a2540"; BAR_FG="#6b7cff"
        RANK_BG="#111a2e"; RANK_EDGE="#2b3a5a"
    else:
        BG="#ffffff"; ROW_A="#f7f7f7"; ROW_B="#ffffff"
        TXT="#111111"; SUB="#777777"; FOOT="#9b9b9b"; DIV="#e2e2e2"
        BAR_BG="#e1e1e1"; BAR_FG="#bfbfbf"
        RANK_BG="#f3f3f3"; RANK_EDGE="#c0c0c0"

    # Highlight colours (gold — same as attacker script)
    HILITE      = "#f6d46b"
    HILITE_EDGE = "#d2a100"

    def is_hi(row):
        return highlight_team and str(row.get("Team", "")) == highlight_team

    scores = pd.to_numeric(df_show["_tri_val"], errors="coerce")
    max_score = float(scores.max()) if scores.notna().any() else 1.0
    if max_score == 0: max_score = 1.0

    # DEVIATION 2: custom footer, matching the player version. None => original.
    if custom_footer_text:
        footer_lines = [ln.strip() for ln in custom_footer_text.split("\n") if ln.strip()]
    else:
        footer_lines = [
            f"Ranked by: {rank_label}.",
            "Scores computed within the selected pool (per-league percentile ranks).",
        ]

    # ── 1920×1080 banner ──
    if export_mode == "1920×1080 (banner)":
        DPI=100; fig=plt.figure(figsize=(19.2, 10.8), dpi=DPI)
        ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.add_patch(Rectangle((0,0),1,1,color=BG,zorder=0))
        LEFT,RIGHT=0.045,0.955

        ax.text(LEFT,0.972,title_lines[0].upper(),fontsize=48,fontweight="bold",color=TXT,ha="left",va="top")
        ax.text(LEFT,0.912,title_lines[1].upper(),fontsize=34,fontweight="bold",color=TXT,ha="left",va="top")
        ax.text(LEFT,0.870,title_lines[2],        fontsize=20,color=SUB,ha="left",va="top")
        ax.plot([LEFT,RIGHT],[0.835,0.835],color=DIV,lw=2.2)
        ax.plot([LEFT,RIGHT],[0.040,0.040],color=DIV,lw=2.2)
        for i,line in enumerate(footer_lines):
            ax.text(LEFT,0.022-i*0.024,line,fontsize=13,color=FOOT,ha="left",va="top",zorder=10)

        ROW_TOP=0.813; ROW_BOT=0.050
        row_gap=(ROW_TOP-ROW_BOT)/float(top_n); row_h=row_gap*0.92
        RANK_X=LEFT+0.024; CREST_X=LEFT+0.105; NAME_X=LEFT+0.175
        BAR_L=LEFT+0.62; BAR_R=RIGHT-0.14; BAR_W=BAR_R-BAR_L; BAR_H=row_h*0.26; VAL_X=RIGHT-0.025

        for i,(_, row) in enumerate(df_show.iterrows()):
            y=ROW_TOP-(i+0.5)*row_gap

            # Row background
            ax.add_patch(Rectangle((LEFT,y-row_h/2),RIGHT-LEFT,row_h,
                                   color=(ROW_A if i%2==0 else ROW_B),zorder=1))

            # Gold highlight overlay
            if is_hi(row):
                ax.add_patch(Rectangle((LEFT,y-row_h/2),RIGHT-LEFT,row_h,
                                       color=HILITE,alpha=0.22,zorder=2))
                ax.add_patch(Rectangle((LEFT,y-row_h/2),RIGHT-LEFT,row_h,
                                       fill=False,edgecolor=HILITE_EDGE,lw=2.2,zorder=3))

            # Rank badge
            ax.scatter([RANK_X],[y],s=1320,facecolor=RANK_BG,
                       edgecolor=(HILITE_EDGE if is_hi(row) else RANK_EDGE),
                       linewidths=2.2,zorder=4)
            ax.text(RANK_X,y,str(i+1),fontsize=16,fontweight="bold",color=TXT,ha="center",va="center",zorder=5)

            badge=get_team_badge(str(row.get("Team","")))
            if badge is not None:
                h,w=badge.shape[0],badge.shape[1]; z=52.0/max(h,w)
                ax.add_artist(AnnotationBbox(OffsetImage(badge,zoom=z),(CREST_X,y),frameon=False,zorder=5))

            ax.text(NAME_X,y+row_h*0.18,str(row.get("Team","")).upper(),
                    fontsize=28,fontweight="bold",color=TXT,ha="left",va="center",zorder=6)
            ax.text(NAME_X,y-row_h*0.22,str(row.get("League","")),
                    fontsize=19,color=SUB,ha="left",va="center",zorder=6)

            frac=max(0.0,min(1.0,float(row["_tri_val"])/max_score))
            ax.add_patch(Rectangle((BAR_L,y-BAR_H/2),BAR_W,BAR_H,color=BAR_BG,zorder=2))
            ax.add_patch(Rectangle((BAR_L,y-BAR_H/2),BAR_W*frac,BAR_H,color=BAR_FG,zorder=3))
            ax.text(VAL_X,y,_tri_value_text(row, pct_col, value_label_col),
                    fontsize=29,fontweight="bold",color=TXT,ha="right",va="center",zorder=6)

        buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=DPI,facecolor=BG); plt.close(fig)
        buf.seek(0); return buf.getvalue()

    # ── Standard ──
    N=len(df_show); ROW_H=0.82; HEADER_H=1.70; FOOT_H=0.55
    TOTAL_H=HEADER_H+N*ROW_H+FOOT_H
    fig=plt.figure(figsize=(8.3,TOTAL_H),dpi=220)
    ax=fig.add_axes([0,0,1,1.0]); ax.set_xlim(0,1.0); ax.set_ylim(0,TOTAL_H); ax.axis("off")
    ax.add_patch(Rectangle((0,0),1.0,TOTAL_H,color=BG,zorder=0))

    title_y=TOTAL_H-0.25
    ax.text(0.04,title_y,      title_lines[0].upper(),fontsize=19,fontweight="bold",color=TXT,ha="left",va="top")
    ax.text(0.04,title_y-0.34, title_lines[1].upper(),fontsize=14,fontweight="bold",color=TXT,ha="left",va="top")
    ax.text(0.04,title_y-0.62, title_lines[2],         fontsize=11,color=SUB,ha="left",va="top")

    base_y=TOTAL_H-HEADER_H
    ax.plot([0.04,0.96],[base_y+ROW_H/2+0.02]*2,color=DIV,lw=1.1,zorder=2)

    LEFT,RIGHT=0.04,0.96
    BAR_L,BAR_R=0.66,0.82; BAR_W=BAR_R-BAR_L; BAR_H=0.14; VAL_X=0.94; crest_x=0.14

    for i,(_, row) in enumerate(df_show.iterrows()):
        y=base_y-i*ROW_H

        # Row background
        ax.add_patch(Rectangle((LEFT,y-ROW_H/2),RIGHT-LEFT,ROW_H,
                               color=(ROW_A if i%2==0 else ROW_B),zorder=1))

        # Gold highlight overlay
        if is_hi(row):
            ax.add_patch(Rectangle((LEFT,y-ROW_H/2),RIGHT-LEFT,ROW_H,
                                   color=HILITE,alpha=0.25,zorder=2))
            ax.add_patch(Rectangle((LEFT,y-ROW_H/2),RIGHT-LEFT,ROW_H,
                                   fill=False,edgecolor=HILITE_EDGE,lw=1.3,zorder=3))

        # Rank badge
        ax.scatter([0.07],[y],s=520,facecolor=RANK_BG,
                   edgecolor=(HILITE_EDGE if is_hi(row) else RANK_EDGE),
                   linewidths=1.2,zorder=4)
        ax.text(0.07,y,str(i+1),fontsize=10,fontweight="bold",color=TXT,ha="center",va="center",zorder=5)

        badge=get_team_badge(str(row.get("Team","")))
        if badge is not None:
            h,w=badge.shape[0],badge.shape[1]; z=40.0/max(h,w)
            ax.add_artist(AnnotationBbox(OffsetImage(badge,zoom=z),(crest_x,y),frameon=False,zorder=5))

        ax.text(0.21,y+0.12,str(row.get("Team","")).upper(),
                fontsize=16,fontweight="bold",color=TXT,ha="left",va="center",zorder=5)
        ax.text(0.21,y-0.10,str(row.get("League","")),
                fontsize=12,color=SUB,ha="left",va="center",zorder=5)

        frac=max(0.0,min(1.0,float(row["_tri_val"])/max_score))
        ax.add_patch(Rectangle((BAR_L,y-BAR_H/2),BAR_W,BAR_H,color=BAR_BG,zorder=2))
        ax.add_patch(Rectangle((BAR_L,y-BAR_H/2),BAR_W*frac,BAR_H,color=BAR_FG,zorder=3))
        ax.text(VAL_X,y,_tri_value_text(row, pct_col, value_label_col),
                fontsize=16,fontweight="bold",color=TXT,ha="right",va="center",zorder=6)

    ax.plot([LEFT,RIGHT],[0.82]*2,color=DIV,lw=0.9,zorder=2)
    for j,line in enumerate(footer_lines):
        ax.text(LEFT,0.62-j*0.18,line,fontsize=9.5,color=FOOT,ha="left",va="top",zorder=4)

    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=220,facecolor=BG); plt.close(fig)
    buf.seek(0); return buf.getvalue()

# =========================================================================
# ADAPTERS — thin call wrappers used by pages_ranking.py.
# These only marshal arguments; no drawing code lives here.
# =========================================================================
def render_player_image(df, *, metric_label, title_lines, theme, export_mode,
                        highlight_players=None, custom_footer_text=None):
    """df needs: Player, Team, League, _MetricForBars, _ValueLabel."""
    return cf_make_ranking_image(
        df_show=df,
        metric_col="_MetricForBars",
        value_label_col="_ValueLabel",
        metric_label=metric_label,
        title_lines=list(title_lines),
        highlight_players=highlight_players,
        export_mode=export_mode,
        theme=theme,
        custom_footer_text=custom_footer_text,
        max_rows=len(df),
        preformatted_labels=True,
    )


def render_team_image(df, *, metric_label, title_lines, theme, export_mode,
                      highlight_team=None, custom_footer_text=None):
    """df needs: Team, League, _tri_val, _ValueLabel."""
    return _tri_make_image(
        df, metric_label, metric_label, metric_label, True,
        list(title_lines), theme, export_mode, max(len(df), 1),
        highlight_team=highlight_team,
        custom_footer_text=custom_footer_text,
        value_label_col="_ValueLabel",
    )
