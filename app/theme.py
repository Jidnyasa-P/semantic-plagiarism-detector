import streamlit as st

THEMES = {
    "Light": {
        "background": "#FFFFFF",
        "surface": "#F8FAFC",
        "card": "#FFFFFF",
        "ink": "#0F172A",
        "muted": "#64748B",
        "accent": "#0D9488",
        "border": "#E2E8F0",
        "input": "#FFFFFF",
        "danger": "#FF4B4B",
        "danger_soft": "#FEE2E2",
        "warning": "#FFA500",
        "warning_soft": "#FEF3C7",
        "success": "#21C55D",
        "success_soft": "#DCFCE7",
        "neutral_soft": "#F1F5F9",
    },
    "Dark": {
        "background": "#0E1117",
        "surface": "#161B22",
        "card": "#1F2937",
        "ink": "#F8FAFC",
        "muted": "#CBD5E1",
        "accent": "#2DD4BF",
        "border": "#374151",
        "input": "#111827",
        "danger": "#F87171",
        "danger_soft": "#450A0A",
        "warning": "#FBBF24",
        "warning_soft": "#422006",
        "success": "#4ADE80",
        "success_soft": "#052E16",
        "neutral_soft": "#1E293B",
    },
}
# Backward-compatible default palette used by existing tests and callers.
COLORS = THEMES["Light"]
def initialize_theme() -> None:
    """Initialize the active theme for the current session."""
    if "theme" not in st.session_state:
        st.session_state.theme = "Light"


def get_theme_name() -> str:
    """Return the active theme name."""
    initialize_theme()
    return st.session_state.theme


def set_theme(theme_name: str) -> None:
    """Set the active theme."""
    if theme_name in THEMES:
        st.session_state.theme = theme_name


def get_colors() -> dict:
    """Return the colors for the active theme."""
    return THEMES[get_theme_name()]

def inject_css() -> None:
    """Inject CSS for the currently selected Light or Dark theme."""
    colors = get_colors()

    css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        :root {{
            --background: {colors["background"]};
            --surface: {colors["surface"]};
            --card: {colors["card"]};
            --ink: {colors["ink"]};
            --muted: {colors["muted"]};
            --accent: {colors["accent"]};
            --border: {colors["border"]};
            --input: {colors["input"]};
            --neutral-soft: {colors["neutral_soft"]};
        }}

        html,
        body,
        [class*="css"] {{
            font-family: 'Inter', sans-serif !important;
        }}

        .stApp {{
            background-color: var(--background) !important;
            color: var(--ink) !important;
        }}

        [data-testid="stHeader"] {{
            background-color: var(--background) !important;
        }}

        [data-testid="stToolbar"] {{
            color: var(--ink) !important;
        }}

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {{
            font-family: 'Newsreader', Georgia, serif !important;
            color: var(--ink) !important;
            font-weight: 700 !important;
        }}

        p,
        label,
        span,
        li,
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"] {{
            color: var(--ink);
        }}

        [data-testid="stCaptionContainer"],
        .stCaption {{
            color: var(--muted) !important;
        }}

        .hero-kicker {{
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.25rem;
        }}

        [data-testid="stSidebar"] {{
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--ink);
        }}

        .sidebar-brand-title {{
            font-family: 'Newsreader', serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--ink);
            text-align: center;
            line-height: 1.2;
            margin-top: 0.5rem;
        }}

        .sidebar-brand-kicker {{
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-align: center;
            margin-bottom: 1.5rem;
        }}

        .sidebar-section-label {{
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 2px;
        }}

        div[data-testid="stMetric"] {{
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-top: 4px solid var(--accent) !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
        }}

        div[data-testid="stMetricLabel"] > div {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            color: var(--muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}

        div[data-testid="stMetricValue"] > div {{
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: var(--ink) !important;
        }}

        div[data-testid="stMetricDelta"] > div {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            font-family: 'IBM Plex Mono', monospace;
            text-align: center;
        }}

        .meta-chip {{
            background-color: var(--neutral-soft);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--ink);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .meta-chip code {{
            font-family: 'IBM Plex Mono', monospace !important;
            background: none !important;
            padding: 0 !important;
            color: var(--accent) !important;
            font-weight: 700 !important;
        }}

        .login-container {{
            background-color: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            padding: 2.5rem !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.18) !important;
        }}

        .mono-text {{
            font-family: 'IBM Plex Mono', monospace !important;
        }}

        .legend-container {{
            display: flex;
            gap: 16px;
            align-items: center;
            margin-bottom: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--muted);
        }}

        .legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            display: inline-block;
        }}

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        [data-baseweb="select"] > div {{
            background-color: var(--input) !important;
            color: var(--ink) !important;
            border-color: var(--border) !important;
        }}

        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [role="listbox"] {{
            background-color: var(--card) !important;
            color: var(--ink) !important;
        }}

        .stButton button,
        .stDownloadButton button,
        .stFormSubmitButton button {{
            border-color: var(--border) !important;
        }}

        [data-testid="stExpander"],
        [data-testid="stForm"] {{
            background-color: var(--card) !important;
            border-color: var(--border) !important;
        }}

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {{
            border-color: var(--border) !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            background-color: var(--surface) !important;
            border-color: var(--border) !important;
        }}

        [data-testid="stTabs"] button {{
            color: var(--muted) !important;
        }}

        [data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }}

        hr {{
            border-color: var(--border) !important;
        }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

def severity_tier(score: float, threshold: float) -> str:
    """
    Categorizes the score into a severity tier matching the backend.
    
    High: >= 0.90
    Medium: >= threshold
    Low: < threshold
    """
    if score >= 0.90:
        return "high"
    elif score >= threshold:
        return "medium"
    else:
        return "low"

def tier_from_severity_label(label: str) -> str:
    """Maps existing label string to tier key."""
    clean = label.lower()
    if "high" in clean:
        return "high"
    elif "medium" in clean or "warn" in clean:
        return "medium"
    else:
        return "low"

def tier_color(tier: str) -> str:
    """Returns color hex associated with a tier."""
    if tier == "high":
        return get_colors()["danger"]
    elif tier == "medium":
        return get_colors()["warning"]
    elif tier == "low":
        return get_colors()["success"]
    return get_colors()["neutral_soft"]

def badge_html(tier: str, label: str = None) -> str:
    """Generates standard HTML badge chip for severity."""
    if tier == "high":
        text_color = get_colors()["danger"]
        bg_color = get_colors()["danger_soft"]
        default_label = "🔴 High"
    elif tier == "medium":
        text_color = get_colors()["warning"]
        bg_color = get_colors()["warning_soft"]
        default_label = "🟡 Medium"
    else:
        text_color = get_colors()["success"]
        bg_color = get_colors()["success_soft"]
        default_label = "🟢 Low"
        
    display_label = label if label is not None else default_label
    return f'<span class="badge" style="background-color: {bg_color}; color: {text_color}; border: 1px solid {text_color};">{display_label}</span>'
