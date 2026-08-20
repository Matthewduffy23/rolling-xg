# photo_utils.py
# Shared player photo loader for all Scouting-Hub pages.
# Photos are stored in: https://github.com/Matthewduffy23/scouting-photos

import re
import io
import unicodedata
import requests
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

GITHUB_PHOTOS_BASE = "https://raw.githubusercontent.com/Matthewduffy23/scouting-photos/main/photos/"
DEFAULT_AVATAR     = "https://i.redd.it/43axcjdu59nd1.jpeg"


def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.strip().lower().split())


def get_player_photo_url(player: str, team: str) -> str:
    """
    Returns the GitHub raw URL for a player photo.
    e.g. 'B. Cabango', 'Swansea City' → 
    https://raw.githubusercontent.com/.../b_cabango__swansea_city.png
    """
    p = "_".join(re.sub(r"[^a-z0-9 ]", "", _norm(player)).split()) or "unknown"
    t = "_".join(re.sub(r"[^a-z0-9 ]", "", _norm(team)).split()) or "unknown"
    return f"{GITHUB_PHOTOS_BASE}{p}__{t}.png"


@st.cache_data(show_spinner=False, ttl=86400)
def load_player_photo_cached(player: str, team: str):
    """
    Load a player photo as a matplotlib-compatible numpy array.
    Falls back to default avatar if not found.
    Used by ranking images, one-pagers, and polar charts.
    """
    url = get_player_photo_url(player, team)
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
            return plt.imread(io.BytesIO(r.content))
    except Exception:
        pass

    # Fallback to default avatar
    try:
        r = requests.get(DEFAULT_AVATAR, timeout=6)
        if r.status_code == 200:
            return plt.imread(io.BytesIO(r.content))
    except Exception:
        pass

    return None


def get_player_photo_pil(player: str, team: str) -> Image.Image | None:
    """
    Load a player photo as a PIL Image (RGBA).
    Used by Feature Z and other PIL-based rendering.
    """
    url = get_player_photo_url(player, team)
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        pass
    return None