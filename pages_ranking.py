"""
Custom Ranking — build an ordered list of teams or players by hand, give each
one a value, and render it through the CIES-style ranking image copied from the
Scouting-Hub / TEAM-HQ apps.

The upload exists only so the search knows the entity universe (and so crests
and photos can be looked up). Values are typed in, not derived from the file.
"""

from __future__ import annotations

import io
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

import ranking_image as RI
from photo_utils import get_player_photo_url

EXPORT_MODES = ["Standard (auto)", "1920×1080 (banner)"]
THEMES = ["Light", "Dark"]

VALUE_FORMATS = ["Number", "1 dp", "2 dp", "€m", "£m", "%"]

# The world player export runs to 80k+ rows, so the search never lists the
# universe: it filters server-side and shows only this many matches.
MAX_MATCHES = 30
MIN_QUERY = 2
DASH = "—"

# Wyscout writes the league under either header depending on the export.
LEAGUE_CANDIDATES = ("League", "Competition")
TEAM_CANDIDATES = ("Team", "Team within selected timeframe", "Club")
PLAYER_CANDIDATES = ("Player", "Player name", "Name")


# ==========================================================================
# Upload -> entity universe
# ==========================================================================
def _pick(cols, candidates):
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def normalize(s) -> str:
    """Lowercase, accent-stripped, whitespace-collapsed. 'Thórarinsson' ->
    'thorarinsson'."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())


def _clean_series(s: pd.Series) -> pd.Series:
    """Strings with real blanks, never the string 'nan'."""
    out = s.astype("object").where(s.notna(), "")
    out = out.map(lambda v: "" if v is None else str(v).strip())
    return out.replace({"nan": "", "NaN": "", "None": "", "<NA>": ""})


def read_any(file) -> pd.DataFrame:
    """Read one CSV or XLSX upload."""
    name = str(getattr(file, "name", "") or "").lower()
    try:
        file.seek(0)
    except Exception:
        pass
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file)
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


def _named(name: str, data: bytes):
    """BytesIO that keeps its filename, so read_any can spot .xlsx."""
    buf = io.BytesIO(data)
    buf.name = name
    return buf


def to_payload(files) -> tuple:
    """((filename, bytes), ...) — a hashable cache key for st.cache_data."""
    out = []
    for f in files:
        try:
            f.seek(0)
        except Exception:
            pass
        data = f.getvalue() if hasattr(f, "getvalue") else f.read()
        out.append((str(getattr(f, "name", "upload")), bytes(data)))
    return tuple(out)


def load_entities(files, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """Concat + de-dupe the uploads down to one row per entity."""
    return load_universe(to_payload(files), mode)


@st.cache_data(show_spinner="Indexing upload…")
def load_universe(payload: tuple, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """Parse + index once per set of uploaded bytes.

    Cached on the file bytes, so typing in the search box never re-parses the
    upload — it only filters the indexed frame.
    """
    frames = []
    problems: list[str] = []
    for name, data in payload:
        try:
            frames.append(read_any(_named(name, data)))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: {exc}")
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(), problems

    raw = pd.concat(frames, ignore_index=True, sort=False)

    team_col = _pick(raw.columns, TEAM_CANDIDATES)
    league_col = _pick(raw.columns, LEAGUE_CANDIDATES)
    player_col = _pick(raw.columns, PLAYER_CANDIDATES)

    if team_col is None:
        problems.append("No 'Team' column found.")
        return pd.DataFrame(), problems
    if mode == "Player" and player_col is None:
        problems.append("No 'Player' column found — is this a team export?")
        return pd.DataFrame(), problems

    out = pd.DataFrame()
    if mode == "Player":
        out["Player"] = _clean_series(raw[player_col])
    out["Team"] = _clean_series(raw[team_col])
    out["League"] = (_clean_series(raw[league_col]) if league_col
                     else pd.Series([""] * len(raw)))

    for extra in ("Age", "Position", "Minutes played"):
        col = _pick(raw.columns, (extra,))
        if col is not None:
            out[extra] = raw[col]

    # A player at two clubs is two rows on purpose — the label carries the
    # club, so both stay selectable. Only truly identical rows collapse.
    key = ["Player", "Team", "League"] if mode == "Player" else ["Team", "League"]
    name_col = key[0]
    out = out.drop_duplicates(subset=key, keep="first")
    # The entity itself must have a name; a missing Team is fine and renders
    # as an em dash.
    out = out[out[name_col].str.len() > 0]

    # Search index, built once here so a keystroke only has to filter.
    out["name_key"] = out[name_col].map(normalize)
    out["search_key"] = (
        out["name_key"] + " " +
        (out["Team"].map(normalize) if mode == "Player" else "") + " " +
        out["League"].map(normalize)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    sort_key = key if mode == "Player" else ["Team"]
    return out.sort_values(sort_key).reset_index(drop=True), problems


def entity_label(row, mode: str) -> str:
    """'Coventry City — England 2.' / 'R. Durosinmi — Pisa — Italy 2.'

    Anything missing shows as an em dash rather than 'nan'.
    """
    def bit(name):
        v = str(row.get(name, "") or "").strip()
        return DASH if v in ("", "nan", "None") else v

    parts = [bit("Player"), bit("Team"), bit("League")] if mode == "Player" \
        else [bit("Team"), bit("League")]
    return " — ".join(parts)


# ==========================================================================
# Search
# ==========================================================================
def search_entities(universe: pd.DataFrame, query: str, mode: str,
                    limit: int = MAX_MATCHES) -> pd.DataFrame:
    """Filter the universe server-side; return at most `limit` ranked matches.

    Every whitespace-separated token has to appear somewhere in the row's
    search key, so "ar pisa" finds Durosinmi at Pisa. Matches are ranked
    name-prefix first, then name-substring, then everything else.
    """
    q = normalize(query)
    if universe.empty or len(q) < MIN_QUERY:
        return universe.iloc[0:0]

    keys = universe["search_key"]
    mask = None
    for token in q.split():
        hit = keys.str.contains(token, regex=False, na=False)
        mask = hit if mask is None else (mask & hit)
    if mask is None:
        return universe.iloc[0:0]

    hits = universe.loc[mask]
    if hits.empty:
        return hits

    names = hits["name_key"]
    rank = np.where(names.str.startswith(q), 0,
                    np.where(names.str.contains(q, regex=False, na=False), 1, 2))
    order = np.lexsort((hits.index.to_numpy(), rank))
    return hits.iloc[order[:limit]]


# ==========================================================================
# Value formatting
# ==========================================================================
def format_value(v, fmt: str) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fmt == "1 dp":
        return f"{v:.1f}"
    if fmt == "2 dp":
        return f"{v:.2f}"
    if fmt == "€m":
        return f"€{v:.1f}m"
    if fmt == "£m":
        return f"£{v:.1f}m"
    if fmt == "%":
        return f"{v:.1f}%"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,g}"


# ==========================================================================
# Ordered list held in session state
# ==========================================================================
def _state_key(mode: str) -> str:
    return f"ranking_list_{mode.lower()}"


def get_list(mode: str) -> list[dict]:
    key = _state_key(mode)
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def add_entry(mode: str, row, value: float = 0.0) -> None:
    entries = get_list(mode)
    st.session_state.setdefault("ranking_uid", 0)
    st.session_state["ranking_uid"] += 1
    def field(name):
        v = str(row.get(name, "") or "").strip()
        return DASH if v in ("", "nan", "None") else v

    entry = {
        "id": st.session_state["ranking_uid"],
        "Team": field("Team"),
        "League": field("League"),
        "value": float(value),
    }
    if mode == "Player":
        entry["Player"] = field("Player")
    entries.append(entry)


def move(entries: list[dict], i: int, delta: int) -> None:
    j = i + delta
    if 0 <= j < len(entries):
        entries[i], entries[j] = entries[j], entries[i]


def build_dataframe(entries: list[dict], mode: str, value_fmt: str,
                    sort_desc: bool) -> pd.DataFrame:
    """The exact column contract the copied image functions expect."""
    rows = []
    for e in entries:
        v = float(e.get("value", 0.0) or 0.0)
        row = {
            "Team": e.get("Team", ""),
            "League": e.get("League", ""),
            "_ValueLabel": format_value(v, value_fmt),
        }
        if mode == "Player":
            row["Player"] = e.get("Player", "")
            row["_MetricForBars"] = v
        else:
            row["_tri_val"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if sort_desc:
        col = "_MetricForBars" if mode == "Player" else "_tri_val"
        df = df.sort_values(col, ascending=False, kind="mergesort")
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=8)
def render_cached(mode, df, metric_label, title_lines, theme, export_mode,
                  highlight, footer):
    """The PNG only depends on these inputs, so typing in the search box
    (which changes none of them) reuses the last render instead of
    re-drawing the figure and re-fetching crests on every keystroke."""
    if mode == "Player":
        return RI.render_player_image(
            df, metric_label=metric_label, title_lines=list(title_lines),
            theme=theme, export_mode=export_mode,
            highlight_players=(list(highlight) if highlight else None),
            custom_footer_text=footer,
        )
    return RI.render_team_image(
        df, metric_label=metric_label, title_lines=list(title_lines),
        theme=theme, export_mode=export_mode,
        highlight_team=highlight, custom_footer_text=footer,
    )


# ==========================================================================
# Page
# ==========================================================================
def main() -> None:
    st.title("Custom Ranking")
    st.caption("Build the list by hand, type the values, export the CIES-style "
               "graphic.")

    mode = st.sidebar.radio("Ranking type", ["Team", "Player"], index=0,
                            key="ranking_mode")
    entries = get_list(mode)

    # ---------------- 1. upload the entity universe ----------------
    st.markdown("### 1. Data")
    files = st.file_uploader(
        f"Wyscout {mode.lower()} export(s) — CSV or XLSX",
        type=["csv", "xlsx", "xlsm", "xls"], accept_multiple_files=True,
        key=f"ranking_upload_{mode}",
        help="Only used to populate the search and to look up crests/photos. "
             "Values are typed in below.",
    )

    universe = pd.DataFrame()
    if files:
        universe, problems = load_entities(files, mode)
        for p in problems:
            st.warning(p)
        if not universe.empty:
            st.success(f"{len(universe):,} unique {mode.lower()}s available "
                       f"from {len(files)} file(s).")
    else:
        st.info(f"Upload one or more Wyscout {mode.lower()} exports to search "
                "for entities.")

    # ---------------- 2. build the ordered list ----------------
    st.markdown("### 2. The list")
    if not universe.empty:
        query = st.text_input(
            "Search", "", key=f"ranking_query_{mode}",
            placeholder=("Name, team or league — e.g. \"durosinmi\", "
                         "\"coventry\", \"ar pisa\""),
            help=f"Type at least {MIN_QUERY} characters. The top "
                 f"{MAX_MATCHES} matches are shown.",
        )
        matches = search_entities(universe, query, mode)

        if len(normalize(query)) < MIN_QUERY:
            st.caption(f"Type at least {MIN_QUERY} characters to search "
                       f"{len(universe):,} {mode.lower()}s.")
        elif matches.empty:
            st.caption(f"No {mode.lower()} matches “{query}”.")
        else:
            labels = [entity_label(r, mode) for _, r in matches.iterrows()]
            capped = len(labels) == MAX_MATCHES
            c_pick, c_add = st.columns([6, 1])
            pos = c_pick.selectbox(
                f"Matches for “{query}”", range(len(labels)),
                format_func=lambda k: labels[k],
                key=f"ranking_pick_{mode}", label_visibility="collapsed",
            )
            c_add.button(
                "Add", key=f"ranking_add_{mode}", width="stretch",
                on_click=add_entry, args=(mode, matches.iloc[int(pos)]),
            )
            st.caption(f"{len(labels)} match{'es' if len(labels) != 1 else ''}"
                       + (f" (capped at {MAX_MATCHES} — narrow the search)"
                          if capped else ""))

    show_thumbs = st.checkbox(
        "Show crests / photos in the list", value=False,
        key=f"ranking_thumbs_{mode}",
        help="Fetches each crest or photo from GitHub / FotMob — off by "
             "default so the list stays instant.",
    )

    if not entries:
        st.caption("Nothing added yet.")
    for i, e in enumerate(list(entries)):
        widths = ([0.5] if show_thumbs else []) + [5, 2, 0.7, 0.7, 0.7]
        cols = st.columns(widths)
        c = 0
        if show_thumbs:
            with cols[0]:
                _thumb(e, mode)
            c = 1
        name = e.get("Player") if mode == "Player" else e.get("Team")
        sub = (f"{e.get('Team','')} — {e.get('League','')}" if mode == "Player"
               else e.get("League", ""))
        cols[c].markdown(f"**{i + 1}. {name}**  \n<span style='opacity:.65'>"
                         f"{sub}</span>", unsafe_allow_html=True)
        e["value"] = cols[c + 1].number_input(
            "Value", value=float(e.get("value", 0.0)), step=0.1,
            format="%.4f", key=f"ranking_val_{e['id']}",
            label_visibility="collapsed",
        )
        cols[c + 2].button("↑", key=f"ranking_up_{e['id']}",
                           on_click=move, args=(entries, i, -1),
                           disabled=(i == 0), width="stretch")
        cols[c + 3].button("↓", key=f"ranking_dn_{e['id']}",
                           on_click=move, args=(entries, i, 1),
                           disabled=(i == len(entries) - 1), width="stretch")
        cols[c + 4].button("✕", key=f"ranking_rm_{e['id']}",
                           on_click=lambda eid=e["id"]: entries.remove(
                               next(x for x in entries if x["id"] == eid)),
                           width="stretch")

    if entries:
        st.button("Clear list", key=f"ranking_clear_{mode}",
                  on_click=lambda: entries.clear())

    # ---------------- 3. metric ----------------
    st.markdown("### 3. Metric")
    m1, m2, m3 = st.columns([3, 1, 1])
    metric_label = m1.text_input(
        "Metric name", "Estimated transfer value (€m)",
        key=f"ranking_metric_{mode}")
    value_fmt = m2.selectbox("Value format", VALUE_FORMATS, index=0,
                             key=f"ranking_fmt_{mode}")
    sort_desc = m3.checkbox("Sort by value (desc)", value=False,
                            key=f"ranking_sort_{mode}",
                            help="Off keeps the manual order above.")

    if not entries:
        st.info("Add at least one entry to render the image.")
        return

    # ---------------- 4. image controls ----------------
    st.markdown("### 4. Image")
    i1, i2 = st.columns(2)
    theme = i1.radio("Theme", THEMES, index=0, horizontal=True,
                     key=f"ranking_theme_{mode}")
    export_mode = i2.selectbox("Export format", EXPORT_MODES, index=0,
                               key=f"ranking_export_{mode}")

    t1 = st.text_input("Title line 1", "TOP " + ("TEAMS" if mode == "Team"
                                                 else "PLAYERS"),
                       key=f"ranking_t1_{mode}")
    t2 = st.text_input("Title line 2", str(metric_label).upper(),
                       key=f"ranking_t2_{mode}")
    t3 = st.text_input("Title line 3", "CUSTOM RANKING  |  Wyscout",
                       key=f"ranking_t3_{mode}")

    df = build_dataframe(entries, mode, value_fmt, sort_desc)

    if mode == "Player":
        highlight = st.multiselect(
            "Highlight players", list(df["Player"]), default=[],
            key=f"ranking_hi_{mode}")
        highlight_team = None
    else:
        options = ["— none —"] + list(df["Team"])
        pick = st.selectbox("Highlight team", options, index=0,
                            key=f"ranking_hi_{mode}")
        highlight_team = None if pick == "— none —" else pick
        highlight = None

    use_footer = st.checkbox("Custom footer text", value=False,
                             key=f"ranking_use_footer_{mode}")
    footer_text = ""
    if use_footer:
        footer_text = st.text_area(
            "Footer text (one line each)", value="",
            key=f"ranking_footer_{mode}")

    png = render_cached(
        mode, df, metric_label, (t1, t2, t3), theme, export_mode,
        tuple(highlight or ()) if mode == "Player" else highlight_team,
        footer_text if use_footer else None,
    )

    if not png:
        st.info("Nothing to render.")
        return

    st.image(png, width="stretch")
    st.download_button(
        "Download PNG", data=png,
        file_name=f"custom_ranking_{mode.lower()}.png", mime="image/png",
        key=f"ranking_dl_{mode}",
    )

    with st.expander("List data"):
        st.dataframe(df, width="stretch", hide_index=True)


def _thumb(entry: dict, mode: str) -> None:
    """Crest / photo preview. Never raises — a miss just shows nothing."""
    try:
        if mode == "Player":
            url = get_player_photo_url(entry.get("Player", ""),
                                       entry.get("Team", ""))
        else:
            url = RI.cf_fotmob_crest_url(entry.get("Team", ""))
        if url:
            st.image(url, width=34)
    except Exception:  # noqa: BLE001
        pass
