from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def render_html(content):
    lines = [line.strip() for line in content.splitlines()]
    clean_content = "\n".join(lines)

    if hasattr(st, "html"):
        st.html(clean_content)
    else:
        st.markdown(clean_content, unsafe_allow_html=True)


# ============================================================
# HEARTWISE - PORTFOLIO APP
# ============================================================

st.set_page_config(
    page_title="HeartWise | AI Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

try:
    MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_random_forest.pkl"
except NameError:
    MODEL_PATH = Path("heart_disease_random_forest.pkl")

FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]


# ============================================================
# VISUAL ASSETS
# ============================================================

LOGO_URL = (
    "https://lh3.googleusercontent.com/aida/"
    "AP1WRLuB5OFKh5QvNiraFF9CfJJAsp9h-EXLtdU7CmVtxXv-mbTaX_HQkGTXKLMoQgynYG1Ush-GQZFeOo3DDwh7HDJRI7VPt3Wrgm5KzElvxchpNn2to9oqwcIkBbUOW9uu5gdluc8EiipDO8jw330_J_Mug8huX6E-iCappl9I0TO4rayOgSAUYjVe0LJpVldIpNdiLiEtTHn5OBug2mUkDYqZq_dC2t67fE1CNDt2RfnY8lDUI-c4Jn_WEQs"
)

HERO_IMAGE_URL = (
    "https://lh3.googleusercontent.com/aida/"
    "AP1WRLuNSFW-zW9iBNQLbl9Imi63fTuzT7KxOepRY0XvsgSkoQuX8GzfURLmzTLNtQFgYIRC6bW-yyIZTfqTpvr6IV8Ea7pAruzqOOPOEGu13K1J4TlORaPSFJvzBkubBtYPpFg2GdR3DxdoPQELY081me8vkBb7ljUb_d3LLjcWG8n7dJoFqg1MQQtNslwezr_be8cAux0I-oBgzmiHrKgSUYYFHrxmFsk3ceGJtW4vJSxd2iGKsq1NMhT9_qk"
)


# ============================================================
# CSS - LIGHT THEME OVERRIDES & CONTRAST FIXES
# ============================================================

def load_css():
    render_html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

        :root {
            --bg: #f4f8fb;
            --white: #ffffff;
            --navy: #0f2438;
            --text: #13283f;
            --muted: #4e6378;
            --line: #d1dfeb;

            --blue: #1463cc;
            --blue-dark: #084396;
            --blue-soft: #e8f2ff;

            --teal: #068a8a;
            --teal-soft: #e2f7f5;

            --green: #118a5d;
            --green-soft: #e4f6ee;

            --coral: #d94349;
            --coral-soft: #fde8e9;

            --orange: #d98218;
            --orange-soft: #fef3df;

            --purple: #5c52cc;
            --purple-soft: #eeedff;

            --shadow: 0 16px 40px rgba(15, 36, 56, 0.09);
            --shadow-sm: 0 8px 24px rgba(15, 36, 56, 0.06);
        }

        html {
            scroll-behavior: smooth;
        }

        html, body, .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--bg) !important;
            color: var(--text) !important;
            font-family: "DM Sans", "Segoe UI", sans-serif !important;
        }

        .block-container {
            max-width: 1240px !important;
            padding: 0.8rem 2rem 4rem !important;
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        h1, h2, h3, h4 {
            font-family: "Manrope", "Segoe UI", sans-serif !important;
            color: var(--navy) !important;
        }

        p, span, label, li {
            color: var(--text) !important;
        }

        /* ---------------- NAVBAR ---------------- */

        .hw-nav {
            position: sticky;
            top: 0;
            z-index: 1000;
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 1.5rem;
            min-height: 72px;
            padding: 0.6rem 0;
            margin-bottom: 2rem;
            background: rgba(244, 248, 251, 0.95);
            border-bottom: 1px solid var(--line);
            backdrop-filter: blur(16px);
        }

        .hw-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-width: 0;
        }

        .hw-brand img {
            width: 44px;
            height: 44px;
            object-fit: contain;
            border-radius: 12px;
            background: #fff;
            border: 1px solid var(--line);
        }

        .hw-brand-name {
            font-family: "Manrope", sans-serif;
            font-size: 1.15rem;
            line-height: 1;
            font-weight: 800;
            color: var(--navy);
        }

        .hw-brand-tag {
            margin-top: 4px;
            font-size: 0.72rem;
            color: var(--muted);
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .hw-nav-links {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2rem;
        }

        .hw-nav-links a {
            color: var(--muted) !important;
            text-decoration: none !important;
            font-size: 0.92rem;
            font-weight: 700;
            padding: 0.45rem 0.2rem;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }

        .hw-nav-links a:hover,
        .hw-nav-links a.active {
            color: var(--blue) !important;
            border-bottom-color: var(--blue);
        }

        .hw-nav-right {
            display: flex;
            justify-content: flex-end;
        }

        .hw-nav-cta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 44px;
            padding: 0.65rem 1.2rem;
            border-radius: 10px;
            background: var(--navy);
            color: #ffffff !important;
            text-decoration: none !important;
            font-size: 0.88rem;
            font-weight: 800;
            box-shadow: 0 6px 18px rgba(15, 36, 56, 0.2);
            transition: transform 0.15s ease, background-color 0.2s ease;
        }

        .hw-nav-cta:hover {
            background: var(--blue);
            color: #ffffff !important;
            transform: translateY(-1px);
        }

        /* ---------------- HERO ---------------- */

        .hw-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 3.2rem;
            background:
                radial-gradient(circle at 90% 8%, rgba(6, 138, 138, 0.14), transparent 30%),
                radial-gradient(circle at 0% 100%, rgba(20, 99, 204, 0.08), transparent 35%),
                linear-gradient(135deg, #ffffff 0%, #f4f9fd 60%, #eef8f6 100%);
            box-shadow: var(--shadow);
        }

        .hw-hero-grid {
            display: grid;
            grid-template-columns: 1.05fr 0.95fr;
            gap: 3.2rem;
            align-items: center;
        }

        .hw-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.85rem;
            border: 1px solid #a8dadc;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.95);
            color: #056a6d !important;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }

        .hw-hero h1 {
            margin: 1.1rem 0 1rem !important;
            max-width: 650px;
            font-size: clamp(2.5rem, 4.8vw, 4rem) !important;
            line-height: 1.08 !important;
            letter-spacing: -0.04em !important;
            font-weight: 800 !important;
        }

        .hw-highlight {
            color: var(--blue);
        }

        .hw-hero-copy p {
            max-width: 650px;
            color: var(--muted) !important;
            font-size: 1.05rem;
            line-height: 1.7;
            margin-bottom: 1.8rem;
        }

        .hw-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.85rem;
        }

        .hw-html-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            padding: 0.75rem 1.25rem;
            border-radius: 11px;
            text-decoration: none !important;
            font-size: 0.88rem;
            font-weight: 800;
            transition: all 0.2s ease;
        }

        .hw-html-btn.primary {
            background: var(--navy);
            color: #ffffff !important;
            box-shadow: 0 8px 20px rgba(15, 36, 56, 0.2);
        }

        .hw-html-btn.primary:hover {
            background: var(--blue);
            color: #ffffff !important;
            transform: translateY(-1px);
        }

        .hw-html-btn.secondary {
            background: #ffffff;
            color: var(--navy) !important;
            border: 1px solid var(--line);
        }

        .hw-html-btn.secondary:hover {
            color: var(--blue) !important;
            border-color: #a3c7e8;
            transform: translateY(-1px);
        }

        .hw-hero-visual {
            position: relative;
            padding: 0.8rem;
            border: 1px solid var(--line);
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 20px 45px rgba(15, 36, 56, 0.1);
        }

        .hw-hero-visual img {
            display: block;
            width: 100%;
            height: 350px;
            object-fit: cover;
            border-radius: 18px;
        }

        .hw-floating {
            position: absolute;
            left: -20px;
            bottom: 25px;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.8rem 1rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 12px 28px rgba(15, 36, 56, 0.12);
        }

        .hw-floating-icon {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 10px;
            background: var(--teal-soft);
            color: var(--teal);
            font-weight: bold;
        }

        .hw-floating strong {
            display: block;
            color: var(--navy);
            font-size: 0.82rem;
        }

        .hw-floating span {
            display: block;
            margin-top: 2px;
            color: var(--muted);
            font-size: 0.72rem;
        }

        /* ---------------- SECTIONS ---------------- */

        .hw-section {
            scroll-margin-top: 90px;
            margin-top: 5rem;
        }

        .hw-section-head {
            max-width: 780px;
            margin: 0 auto 2.2rem;
            text-align: center;
        }

        .hw-kicker {
            display: inline-flex;
            padding: 0.4rem 0.85rem;
            border: 1px solid #bdd5ea;
            border-radius: 999px;
            background: #ffffff;
            color: var(--blue-dark) !important;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }

        .hw-section-head h2 {
            margin: 0.8rem 0 0.6rem !important;
            font-size: clamp(1.8rem, 3vw, 2.5rem) !important;
            letter-spacing: -0.03em !important;
        }

        .hw-section-head p {
            margin: 0;
            color: var(--muted) !important;
            font-size: 0.98rem;
            line-height: 1.65;
        }

        /* ---------------- METRICS ---------------- */

        .hw-metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.2rem;
        }

        .hw-metric {
            position: relative;
            overflow: hidden;
            padding: 1.5rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow-sm);
        }

        .hw-metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.03em;
        }

        .hw-metric-value {
            margin-top: 0.5rem;
            color: var(--navy);
            font-family: "Manrope", sans-serif;
            font-size: 2.3rem;
            line-height: 1;
            font-weight: 800;
        }

        .hw-metric-note {
            margin-top: 0.65rem;
            color: #5d738a !important;
            font-size: 0.78rem;
            font-weight: 500;
        }

        /* ---------------- PROJECT OVERVIEW ---------------- */

        .hw-project {
            display: grid;
            grid-template-columns: 1.08fr 0.92fr;
            gap: 1.5rem;
        }

        .hw-card {
            padding: 1.8rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow-sm);
        }

        .hw-card h3 {
            margin: 0 0 0.75rem !important;
            font-size: 1.4rem !important;
        }

        .hw-card p {
            color: var(--muted) !important;
            line-height: 1.7;
            font-size: 0.92rem;
        }

        .hw-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.2rem;
        }

        .hw-pill {
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: #f0f5fa;
            border: 1px solid var(--line);
            color: var(--navy);
            font-size: 0.78rem;
            font-weight: 700;
        }

        .hw-method-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.85rem;
        }

        .hw-method {
            padding: 1.1rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: #f9fbfd;
        }

        .hw-method-icon {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 10px;
            margin-bottom: 0.65rem;
            background: var(--blue-soft);
            color: var(--blue);
            font-weight: bold;
        }

        .hw-method strong {
            display: block;
            font-size: 0.88rem;
            color: var(--navy);
        }

        .hw-method span {
            display: block;
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.5;
        }

        /* ---------------- PIPELINE ---------------- */

        .hw-pipeline {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.9rem;
            align-items: stretch;
        }

        .hw-step {
            position: relative;
            text-align: center;
            padding: 1.3rem 0.75rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow-sm);
        }

        .hw-step-icon {
            width: 48px;
            height: 48px;
            display: grid;
            place-items: center;
            margin: 0 auto 0.8rem;
            border-radius: 14px;
            font-size: 1.2rem;
        }

        .hw-step:nth-child(1) .hw-step-icon { background: var(--blue-soft); color: var(--blue); }
        .hw-step:nth-child(2) .hw-step-icon { background: var(--teal-soft); color: var(--teal); }
        .hw-step:nth-child(3) .hw-step-icon { background: var(--purple-soft); color: var(--purple); }
        .hw-step:nth-child(4) .hw-step-icon { background: var(--orange-soft); color: var(--orange); }
        .hw-step:nth-child(5) .hw-step-icon { background: var(--green-soft); color: var(--green); }

        .hw-step strong {
            display: block;
            font-size: 0.84rem;
            color: var(--navy);
        }

        .hw-step span {
            display: block;
            margin-top: 0.35rem;
            color: var(--muted);
            font-size: 0.74rem;
        }

        /* ---------------- FORM & STREAMLIT INPUT OVERRIDES (DARK MODE FIX) ---------------- */

        .hw-form-heading {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 1.2rem;
        }

        .hw-form-icon {
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: var(--blue-soft);
            color: var(--blue);
            font-size: 1.2rem;
        }

        .hw-form-heading strong {
            display: block;
            color: var(--navy);
            font-family: "Manrope", sans-serif;
            font-size: 1.15rem;
        }

        .hw-form-heading span {
            display: block;
            margin-top: 2px;
            color: var(--muted);
            font-size: 0.82rem;
        }

        .hw-subsection {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.9rem 0 0.6rem;
            margin: 0.6rem 0 0.8rem;
            border-bottom: 1px solid var(--line);
        }

        .hw-subsection-icon {
            width: 30px;
            height: 30px;
            display: grid;
            place-items: center;
            border-radius: 9px;
            background: var(--teal-soft);
            color: var(--teal);
            font-size: 0.85rem;
        }

        .hw-subsection strong {
            color: var(--navy);
            font-size: 0.92rem;
        }

        /* Streamlit Form Container */
        div[data-testid="stForm"] {
            padding: 2.2rem !important;
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            border-radius: 24px !important;
            box-shadow: var(--shadow) !important;
        }

        /* Input Labels Force Light Theme */
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            color: var(--navy) !important;
            font-size: 0.86rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.35rem !important;
        }

        /* Input Controls Force Light Background & Dark Text */
        div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border-radius: 11px !important;
        }

        div[data-baseweb="input"] input {
            background-color: #ffffff !important;
            color: #0f2438 !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #0f2438 !important;
            border: 1.5px solid #c2d5e5 !important;
            border-radius: 11px !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
        }

        div[data-baseweb="select"] span {
            color: #0f2438 !important;
        }

        /* Plus Minus Buttons Fix */
        div[data-testid="stNumberInput"] button {
            background-color: #f0f5fa !important;
            color: #0f2438 !important;
            border: 0 !important;
        }

        /* Streamlit Selectbox Popover/Dropdown Menu Fix */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"],
        li[role="option"] {
            background-color: #ffffff !important;
            color: #0f2438 !important;
        }

        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: var(--blue-soft) !important;
            color: var(--blue) !important;
        }

        /* Tooltip & Validation Error Messages Contrast Fix */
        div[data-baseweb="tooltip"] {
            background-color: #0f2438 !important;
            color: #ffffff !important;
            font-size: 0.8rem !important;
        }

        /* Submit Button */
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #1463cc, #084396) !important;
            color: #ffffff !important;
            border: 0 !important;
            border-radius: 12px !important;
            min-height: 50px !important;
            font-size: 0.98rem !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 22px rgba(20, 99, 204, 0.28) !important;
            transition: transform 0.15s ease, box-shadow 0.2s ease !important;
            margin-top: 1rem !important;
        }

        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button span {
            color: #ffffff !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 26px rgba(20, 99, 204, 0.35) !important;
        }

        .hw-disclaimer {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            margin-top: 1.2rem;
            padding: 1rem 1.2rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: #f8fafc;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }

        .hw-disclaimer-icon {
            color: var(--orange);
            font-size: 1rem;
        }

        /* ---------------- RESULT ---------------- */

        .hw-result {
            margin-top: 1.8rem;
            padding: 1.6rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
        }

        .hw-result-banner {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1.1rem 1.2rem;
            border-radius: 16px;
            margin-bottom: 1.2rem;
        }

        .hw-result-banner.positive {
            background: var(--coral-soft);
            border: 1px solid #f8babb;
        }

        .hw-result-banner.negative {
            background: var(--green-soft);
            border: 1px solid #b3e6d0;
        }

        .hw-result-icon {
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 900;
        }

        .positive .hw-result-icon { background: #fbc2c4; color: var(--coral); }
        .negative .hw-result-icon { background: #bfead6; color: var(--green); }

        .hw-result-banner strong {
            display: block;
            color: var(--navy);
            font-family: "Manrope", sans-serif;
            font-size: 1.1rem;
        }

        .hw-result-banner span {
            display: block;
            margin-top: 2px;
            color: var(--muted);
            font-size: 0.82rem;
        }

        .hw-prob {
            padding: 1.2rem;
            background: #fbfdff;
            border: 1px solid var(--line);
            border-radius: 16px;
        }

        .hw-prob-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        .hw-prob-value {
            margin-top: 0.3rem;
            color: var(--navy);
            font-family: "Manrope", sans-serif;
            font-size: 2rem;
            font-weight: 800;
        }

        .hw-progress {
            height: 10px;
            overflow: hidden;
            margin: 0.7rem 0 0.6rem;
            border-radius: 999px;
            background: #e2ebf2;
        }

        .hw-progress-fill-red {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #f07b80, #d94349);
        }

        .hw-progress-fill-blue {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #519cf2, #1463cc);
        }

        .hw-prob-row {
            display: flex;
            justify-content: space-between;
            color: var(--muted);
            font-size: 0.78rem;
        }

        /* ---------------- INSIGHTS ---------------- */

        .hw-insights {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.2rem;
        }

        .hw-insight {
            padding: 1.4rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow-sm);
        }

        .hw-insight-icon {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            margin-bottom: 0.85rem;
            border-radius: 12px;
            background: var(--blue-soft);
            color: var(--blue);
            font-weight: bold;
        }

        .hw-insight:nth-child(2) .hw-insight-icon { background: var(--teal-soft); color: var(--teal); }
        .hw-insight:nth-child(3) .hw-insight-icon { background: var(--orange-soft); color: var(--orange); }
        .hw-insight:nth-child(4) .hw-insight-icon { background: var(--purple-soft); color: var(--purple); }

        .hw-insight strong {
            display: block;
            color: var(--navy);
            font-size: 0.92rem;
        }

        .hw-insight span {
            display: block;
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }

        /* ---------------- FOOTER ---------------- */

        .hw-footer {
            margin-top: 5rem;
            padding: 2rem 0 1rem;
            border-top: 1px solid var(--line);
        }

        .hw-footer-grid {
            display: flex;
            justify-content: space-between;
            gap: 2rem;
        }

        .hw-footer strong {
            color: var(--navy);
            font-family: "Manrope", sans-serif;
            font-size: 1rem;
        }

        .hw-footer span {
            display: block;
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.8rem;
        }

        .hw-footer-right {
            text-align: right;
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }

        /* ---------------- RESPONSIVE ---------------- */

        @media (max-width: 1000px) {
            .hw-nav { grid-template-columns: 1fr auto; }
            .hw-nav-links { grid-column: 1 / -1; grid-row: 2; padding-bottom: 0.5rem; }
            .hw-hero-grid, .hw-project { grid-template-columns: 1fr; }
            .hw-metrics, .hw-insights { grid-template-columns: repeat(2, 1fr); }
            .hw-pipeline { grid-template-columns: repeat(5, minmax(145px, 1fr)); overflow-x: auto; padding-bottom: 0.5rem; }
        }

        @media (max-width: 720px) {
            .block-container { padding: 0.5rem 1rem 3rem !important; }
            .hw-nav { grid-template-columns: 1fr; gap: 0.5rem; }
            .hw-brand, .hw-nav-right { justify-content: center; }
            .hw-nav-links { gap: 1rem; }
            .hw-nav-right { display: none; }
            .hw-hero { padding: 1.6rem; border-radius: 20px; }
            .hw-hero h1 { font-size: 2.3rem !important; }
            .hw-hero-visual img { height: 250px; }
            .hw-floating { left: 10px; bottom: 10px; }
            .hw-metrics, .hw-insights, .hw-method-grid { grid-template-columns: 1fr; }
            .hw-footer-grid { flex-direction: column; }
            .hw-footer-right { text-align: left; }
        }
        </style>
        """
    )


# ============================================================
# MODEL LOGIC
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model file could not be found. "
            "Please keep heart_disease_random_forest.pkl in the same folder as app.py."
        )

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError("The trained model could not be loaded.") from exc


def create_input_dataframe(
    age,
    sex,
    chest_pain_type,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    st_slope,
):
    values = [
        age,
        sex,
        chest_pain_type,
        resting_bp,
        cholesterol,
        fasting_bs,
        resting_ecg,
        max_hr,
        exercise_angina,
        oldpeak,
        st_slope,
    ]

    return pd.DataFrame([values], columns=FEATURES)


# ============================================================
# UI COMPONENTS
# ============================================================

def render_nav():
    render_html(
        f"""
        <nav class="hw-nav">
            <div class="hw-brand">
                <img src="{LOGO_URL}" alt="HeartWise">
                <div>
                    <div class="hw-brand-name">HeartWise</div>
                    <div class="hw-brand-tag">AI HEALTH ANALYTICS</div>
                </div>
            </div>

            <div class="hw-nav-links">
                <a href="#about">About</a>
                <a href="#model">Model</a>
                <a href="#prediction">Prediction</a>
                <a href="#insights">Insights</a>
            </div>

            <div class="hw-nav-right">
                <a class="hw-nav-cta" href="#prediction">Start Prediction →</a>
            </div>
        </nav>
        """
    )


def render_hero():
    render_html(
        f"""
        <section class="hw-hero" id="about">
            <div class="hw-hero-grid">

                <div class="hw-hero-copy">
                    <div class="hw-eyebrow">♥ &nbsp; AI-POWERED HEALTH PREDICTION</div>

                    <h1>
                        Heart Disease Prediction,
                        <span class="hw-highlight">Powered by Machine Learning.</span>
                    </h1>

                    <p>
                        HeartWise is an end-to-end machine learning portfolio project
                        that uses clinical and exercise-related health indicators to
                        estimate heart disease risk with a trained Random Forest
                        classification model.
                    </p>

                    <div class="hw-actions">
                        <a class="hw-html-btn primary" href="#prediction">
                            Start Prediction →
                        </a>
                        <a class="hw-html-btn secondary" href="#model">
                            Explore the Model
                        </a>
                    </div>
                </div>

                <div class="hw-hero-visual">
                    <img src="{HERO_IMAGE_URL}" alt="Heart health visualization">

                    <div class="hw-floating">
                        <div class="hw-floating-icon">✚</div>
                        <div>
                            <strong>ML Risk Assessment</strong>
                            <span>Random Forest • Explainable AI</span>
                        </div>
                    </div>
                </div>

            </div>
        </section>
        """
    )


def render_metrics():
    render_html(
        """
        <section class="hw-section" id="model">
            <div class="hw-section-head">
                <div class="hw-kicker">MODEL PERFORMANCE</div>
                <h2>Performance on unseen data</h2>
                <p>
                    Final Random Forest evaluation on the held-out test set.
                    These are the project results used for the portfolio presentation.
                </p>
            </div>

            <div class="hw-metrics">
                <div class="hw-metric">
                    <div class="hw-metric-label">Accuracy</div>
                    <div class="hw-metric-value">92.39%</div>
                    <div class="hw-metric-note">Overall correct predictions</div>
                </div>

                <div class="hw-metric">
                    <div class="hw-metric-label">Recall</div>
                    <div class="hw-metric-value">94.12%</div>
                    <div class="hw-metric-note">Heart-disease cases detected</div>
                </div>

                <div class="hw-metric">
                    <div class="hw-metric-label">F1 Score</div>
                    <div class="hw-metric-value">93.20%</div>
                    <div class="hw-metric-note">Precision + recall balance</div>
                </div>

                <div class="hw-metric">
                    <div class="hw-metric-label">ROC-AUC</div>
                    <div class="hw-metric-value">0.9729</div>
                    <div class="hw-metric-note">Strong class discrimination</div>
                </div>
            </div>
        </section>
        """
    )


def render_project_overview():
    render_html(
        """
        <section class="hw-section">
            <div class="hw-section-head">
                <div class="hw-kicker">PROJECT OVERVIEW</div>
                <h2>More than a prediction model</h2>
                <p>
                    HeartWise demonstrates a complete supervised machine learning
                    workflow, from clinical inputs and preprocessing to evaluation,
                    tuning, prediction, and explainability.
                </p>
            </div>

            <div class="hw-project">
                <div class="hw-card">
                    <h3>What HeartWise demonstrates</h3>
                    <p>
                        The application accepts structured cardiovascular and
                        exercise-related indicators and sends them through the trained
                        inference pipeline. The project includes model evaluation,
                        cross-validation, hyperparameter tuning, Random Forest
                        classification, and SHAP-based explainability.
                    </p>

                    <p>
                        The goal is to demonstrate practical machine learning skills:
                        preparing data, selecting and tuning a model, evaluating
                        performance, interpreting model behavior, and deploying the
                        final model through an interactive Streamlit application.
                    </p>

                    <div class="hw-pills">
                        <span class="hw-pill">Python</span>
                        <span class="hw-pill">Pandas</span>
                        <span class="hw-pill">Scikit-learn</span>
                        <span class="hw-pill">Random Forest</span>
                        <span class="hw-pill">Cross-Validation</span>
                        <span class="hw-pill">Hyperparameter Tuning</span>
                        <span class="hw-pill">SHAP</span>
                        <span class="hw-pill">Streamlit</span>
                    </div>
                </div>

                <div class="hw-card">
                    <h3>ML workflow highlights</h3>

                    <div class="hw-method-grid">
                        <div class="hw-method">
                            <div class="hw-method-icon">◈</div>
                            <strong>Preprocessing</strong>
                            <span>Clinical and categorical features are prepared for model inference.</span>
                        </div>

                        <div class="hw-method">
                            <div class="hw-method-icon">◫</div>
                            <strong>Model Selection</strong>
                            <span>Classification models were evaluated before the final model was selected.</span>
                        </div>

                        <div class="hw-method">
                            <div class="hw-method-icon">⚙</div>
                            <strong>Optimization</strong>
                            <span>Cross-validation and hyperparameter tuning were used during development.</span>
                        </div>

                        <div class="hw-method">
                            <div class="hw-method-icon">✦</div>
                            <strong>Explainability</strong>
                            <span>Feature importance and SHAP analysis help interpret model behavior.</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """
    )


def render_pipeline():
    render_html(
        """
        <section class="hw-section">
            <div class="hw-section-head">
                <div class="hw-kicker">ML PIPELINE</div>
                <h2>How HeartWise works</h2>
                <p>
                    A simple end-to-end view of how user input becomes a machine
                    learning prediction.
                </p>
            </div>

            <div class="hw-pipeline">
                <div class="hw-step">
                    <div class="hw-step-icon">♥</div>
                    <strong>Clinical Inputs</strong>
                    <span>12 model features</span>
                </div>

                <div class="hw-step">
                    <div class="hw-step-icon">⚙</div>
                    <strong>Preprocessing</strong>
                    <span>Feature preparation</span>
                </div>

                <div class="hw-step">
                    <div class="hw-step-icon">▦</div>
                    <strong>Random Forest</strong>
                    <span>Tuned classifier</span>
                </div>

                <div class="hw-step">
                    <div class="hw-step-icon">◉</div>
                    <strong>Prediction</strong>
                    <span>Class + probability</span>
                </div>

                <div class="hw-step">
                    <div class="hw-step-icon">✦</div>
                    <strong>Explainable AI</strong>
                    <span>Feature impact</span>
                </div>
            </div>
        </section>
        """
    )


def render_prediction_form():
    render_html(
        """
        <section class="hw-section" id="prediction">
            <div class="hw-section-head">
                <div class="hw-kicker">PREDICTION</div>
                <h2>Check your prediction</h2>
                <p>
                    Enter the same clinical and exercise-related variables used by
                    the trained model.
                </p>
            </div>

            <div class="hw-form-heading">
                <div class="hw-form-icon">♥</div>
                <div>
                    <strong>Patient Information</strong>
                    <span>Complete the model inputs below to generate a prediction.</span>
                </div>
            </div>
        </section>
        """
    )

    with st.form("prediction_form"):

        render_html(
            """
            <div class="hw-subsection">
                <div class="hw-subsection-icon">●</div>
                <strong>Personal Information</strong>
            </div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=45,
                step=1,
            )

        with col2:
            sex = st.selectbox(
                "Sex",
                options=[("Male", "M"), ("Female", "F")],
                format_func=lambda option: option[0],
            )[1]

        render_html(
            """
            <div class="hw-subsection">
                <div class="hw-subsection-icon">♥</div>
                <strong>Clinical Measurements</strong>
            </div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            resting_bp = st.number_input(
                "Resting Blood Pressure [mmHg]",
                min_value=1,
                max_value=300,
                value=120,
                step=1,
            )

            max_hr = st.number_input(
                "Maximum Heart Rate [bpm]",
                min_value=1,
                max_value=300,
                value=150,
                step=1,
            )

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=[("No", 0), ("Yes", 1)],
                format_func=lambda option: option[0],
            )[1]

        with col2:
            cholesterol = st.number_input(
                "Cholesterol [mg/dL]",
                min_value=0,
                max_value=1000,
                value=200,
                step=1,
            )

            oldpeak = st.number_input(
                "Oldpeak",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
            )

        render_html(
            """
            <div class="hw-subsection">
                <div class="hw-subsection-icon">⌁</div>
                <strong>Clinical Indicators</strong>
            </div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=[
                    ("Typical Angina", "TA"),
                    ("Atypical Angina", "ATA"),
                    ("Non-Anginal Pain", "NAP"),
                    ("Asymptomatic", "ASY"),
                ],
                format_func=lambda option: option[0],
                index=1,
            )[1]

            resting_ecg = st.selectbox(
                "Resting ECG",
                options=[
                    ("Normal", "Normal"),
                    ("ST", "ST"),
                    ("LVH", "LVH"),
                ],
                format_func=lambda option: option[0],
            )[1]

        with col2:
            exercise_angina = st.selectbox(
                "Exercise Angina",
                options=[
                    ("No", "N"),
                    ("Yes", "Y"),
                ],
                format_func=lambda option: option[0],
            )[1]

            st_slope = st.selectbox(
                "ST Slope",
                options=[
                    ("Up", "Up"),
                    ("Flat", "Flat"),
                    ("Down", "Down"),
                ],
                format_func=lambda option: option[0],
            )[1]

        submitted = st.form_submit_button(
            "Predict Heart Disease",
            use_container_width=True,
        )

    render_html(
        """
        <div class="hw-disclaimer">
            <div class="hw-disclaimer-icon">⚠</div>
            <div>
                <strong>Educational use only.</strong>
                This application demonstrates a machine learning classification
                workflow and is not a medical diagnostic tool. Predictions should
                not replace evaluation by a qualified healthcare professional.
            </div>
        </div>
        """
    )

    if submitted:
        input_df = create_input_dataframe(
            age,
            sex,
            chest_pain_type,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope,
        )

        try:
            model = load_model()

            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]

            st.session_state["prediction_result"] = {
                "prediction": prediction,
                "probability": float(probabilities[1] * 100),
                "no_probability": float(probabilities[0] * 100),
                "yes_probability": float(probabilities[1] * 100),
            }

        except FileNotFoundError as exc:
            st.error(str(exc))

        except RuntimeError as exc:
            st.error(str(exc))

        except Exception:
            st.error(
                "Prediction could not be completed. "
                "Please verify the model file and input values."
            )


def render_prediction_result():
    result = st.session_state.get("prediction_result")

    if not result:
        return

    is_positive = result["prediction"] == 1

    if is_positive:
        title = "Heart Disease Detected"
        subtitle = "The model classified this input as class 1."
        icon = "!"
        banner_class = "positive"
    else:
        title = "No Heart Disease Detected"
        subtitle = "The model classified this input as class 0."
        icon = "✓"
        banner_class = "negative"

    render_html(
        f"""
        <div class="hw-result">
            <div class="hw-result-banner {banner_class}">
                <div class="hw-result-icon">{icon}</div>
                <div>
                    <strong>{title}</strong>
                    <span>{subtitle}</span>
                </div>
            </div>
        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        render_html(
            f"""
            <div class="hw-prob">
                <div class="hw-prob-label">HEART DISEASE PROBABILITY</div>
                <div class="hw-prob-value">{result["probability"]:.2f}%</div>
                <div class="hw-progress">
                    <div class="hw-progress-fill-red"
                         style="width:{result["yes_probability"]:.2f}%;">
                    </div>
                </div>
                <div class="hw-prob-row">
                    <span>Predicted class 1</span>
                    <strong>{result["yes_probability"]:.2f}%</strong>
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            f"""
            <div class="hw-prob">
                <div class="hw-prob-label">NO HEART DISEASE PROBABILITY</div>
                <div class="hw-prob-value">{result["no_probability"]:.2f}%</div>
                <div class="hw-progress">
                    <div class="hw-progress-fill-blue"
                         style="width:{result["no_probability"]:.2f}%;">
                    </div>
                </div>
                <div class="hw-prob-row">
                    <span>Predicted class 0</span>
                    <strong>{result["no_probability"]:.2f}%</strong>
                </div>
            </div>
            """
        )

    render_html(
        """
        <div style="
            margin-top:1rem;
            padding:0.85rem 1rem;
            border-radius:12px;
            background:#f8fafc;
            border:1px solid #d1dfeb;
            color:#4e6378;
            font-size:0.78rem;">
            Model probability is an output of the trained classifier and should
            not be interpreted as a clinical diagnosis.
        </div>
        """
    )


def render_insights():
    render_html(
        """
        <section class="hw-section" id="insights">
            <div class="hw-section-head">
                <div class="hw-kicker">MODEL INSIGHTS</div>
                <h2>Built as an explainable ML project</h2>
                <p>
                    The project combines predictive performance with techniques
                    that help explain how the model behaves.
                </p>
            </div>

            <div class="hw-insights">
                <div class="hw-insight">
                    <div class="hw-insight-icon">▦</div>
                    <strong>Random Forest</strong>
                    <span>
                        Final classification model used for the deployed prediction workflow.
                    </span>
                </div>

                <div class="hw-insight">
                    <div class="hw-insight-icon">↔</div>
                    <strong>Cross-Validation</strong>
                    <span>
                        Used during model development to evaluate performance across folds.
                    </span>
                </div>

                <div class="hw-insight">
                    <div class="hw-insight-icon">⚙</div>
                    <strong>Hyperparameter Tuning</strong>
                    <span>
                        Tuning was used to improve the selected Random Forest configuration.
                    </span>
                </div>

                <div class="hw-insight">
                    <div class="hw-insight-icon">✦</div>
                    <strong>SHAP Explainability</strong>
                    <span>
                        SHAP analysis was used to understand feature impact on model output.
                    </span>
                </div>
            </div>
        </section>
        """
    )


def render_footer():
    render_html(
        """
        <footer class="hw-footer">
            <div class="hw-footer-grid">
                <div>
                    <strong>HeartWise</strong>
                    <span>
                        Heart Disease Prediction • Machine Learning Portfolio Project
                    </span>
                </div>

                <div class="hw-footer-right">
                    Python • Pandas • Scikit-learn • Random Forest • SHAP • Streamlit<br>
                    Educational and portfolio demonstration only
                </div>
            </div>
        </footer>
        """
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None

    load_css()

    render_nav()
    render_hero()
    render_metrics()
    render_project_overview()
    render_pipeline()
    render_prediction_form()
    render_prediction_result()
    render_insights()
    render_footer()


if __name__ == "__main__":
    main()