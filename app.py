"""
Cheops AI - Multi-Channel Fraud Intelligence System.

Main Streamlit entry point. All workflow logic lives in pages/.
This module only applies shared styling and renders the landing content.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_ASSETS_DIR = _SRC_DIR / "apris" / "frontend" / "assets"


def _read_asset(name: str) -> str:
    path = _ASSETS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Asset not found: {path}")
    return path.read_text(encoding="utf-8")


def _render_html(body: str) -> None:
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def _guard_streamlit_entrypoint() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is None:
            print("This is a Streamlit application.")
            print("Run it with: streamlit run app.py")
            raise SystemExit(0)
    except Exception:
        return


def _set_style() -> None:
    css = _read_asset("streamlit_theme.css")
    _render_html(
        f"""
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"> 
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
        <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap\" rel=\"stylesheet\"> 
        <style>{css}</style>
        """
    )


def _apply_matplotlib_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#e5e5e5",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#e5e5e5",
            "axes.titleweight": "bold",
            "font.size": 10,
            "font.family": "sans-serif",
            "text.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#6e6e80",
            "ytick.color": "#6e6e80",
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="Cheops AI - Risk Intelligence",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _set_style()
    _apply_matplotlib_theme()

    _render_html(_read_asset("hero.html"))

    with st.expander("Quick Start", expanded=False):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown(
                """
                1. Open **Scanner** in the sidebar.
                2. Run batch scan (simulation or uploaded file).
                3. Open **Anomaly Dashboard**.
                4. Investigate top-risk entities and dossier details.
                """
            )
        with c2:
            st.markdown("**Reference formulas**")
            st.code(
                "growth_rate = (new_current - new_previous) / max(new_previous, 1)\n"
                "referral_ratio = referred_current / max(new_current, 1)\n"
                "payout_dependency = payouts_total / max(incoming_funds, 1)",
                language="text",
            )
            st.caption("Risk bands: <0.45 low, 0.45-0.70 medium, 0.70-0.85 high, >=0.85 critical")

    st.info(
        "All data in this MVP is synthetic and processed locally. "
        "No real personal records are used."
    )
    st.markdown("---")
    st.markdown("Select a page in the sidebar to continue.")


if __name__ == "__main__":
    _guard_streamlit_entrypoint()
    main()
