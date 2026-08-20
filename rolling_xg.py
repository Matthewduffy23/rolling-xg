"""
Rolling xG — Streamlit Cloud entry point.

Thin router only: the page bodies live in pages_rolling.py and
pages_ranking.py, and the theme lives in .streamlit/config.toml. Keeping this
file as the entry point means the deploy config never has to change.
"""

from __future__ import annotations

import streamlit as st

# Must be the first Streamlit call in the entry script, before st.navigation.
st.set_page_config(page_title="Rolling xG", layout="wide",
                   initial_sidebar_state="expanded")

import pages_ranking  # noqa: E402
import pages_rolling  # noqa: E402

nav = st.navigation([
    st.Page(pages_rolling.main, title="Rolling xG", icon="📈",
            url_path="rolling-xg", default=True),
    st.Page(pages_ranking.main, title="Custom Ranking", icon="🏆",
            url_path="custom-ranking"),
])
nav.run()
