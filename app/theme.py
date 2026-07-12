import streamlit as st

COLORS = {
    "background": "#FFFFFF",
    "surface": "#F8FAFC",
    "ink": "#0F172A",
    "accent": "#0D9488",
    "border": "#E2E8F0",
    "danger": "#FF4B4B",
    "danger_soft": "#FEE2E2",
    "warning": "#FFA500",
    "warning_soft": "#FEF3C7",
    "success": "#21C55D",
    "success_soft": "#DCFCE7",
    "neutral_soft": "#F1F5F9"
}

def inject_css():
    """Inject unified custom CSS to enforce Case File aesthetic."""
    css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;0,6..72,700;1,6..72,400&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        /* Base Typography */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Newsreader', Georgia, serif !important;
            color: #0F172A !important;
            font-weight: 700 !important;
        }

        /* Hero kicker */
        .hero-kicker {
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 700;
            color: #0D9488;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.25rem;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        .sidebar-brand-title {
            font-family: 'Newsreader', serif;
            font-size: 1.6rem;
            font-weight: 700;
            color: #0F172A;
            text-align: center;
            line-height: 1.2;
            margin-top: 0.5rem;
        }

        .sidebar-brand-kicker {
            font-family: 'Inter', sans-serif;
            font-size: 0.7rem;
            font-weight: 700;
            color: #0D9488;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .sidebar-section-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            font-weight: 700;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 2px;
        }

        /* Native st.metric Customization (Teal Accent Card) */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-top: 4px solid #0D9488 !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        }

        div[data-testid="stMetricLabel"] > div {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 700 !important;
            color: #64748B !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }

        div[data-testid="stMetricValue"] > div {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
        }

        div[data-testid="stMetricDelta"] > div {
            font-family: 'Inter', sans-serif !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
        }

        /* Badge/Chip Styles */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 700;
            font-family: 'IBM Plex Mono', monospace;
            text-align: center;
        }

        .meta-chip {
            background-color: #F1F5F9;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #334155;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .meta-chip code {
            font-family: 'IBM Plex Mono', monospace !important;
            background: none !important;
            padding: 0 !important;
            color: #0D9488 !important;
            font-weight: 700 !important;
        }

        /* Login page custom styling */
        .login-container {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 2.5rem !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05) !important;
        }

        /* Utility classes */
        .mono-text {
            font-family: 'IBM Plex Mono', monospace !important;
        }

        /* Legend style */
        .legend-container {
            display: flex;
            gap: 16px;
            align-items: center;
            margin-bottom: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
            color: #475569;
        }

        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            display: inline-block;
        }
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
        return COLORS["danger"]
    elif tier == "medium":
        return COLORS["warning"]
    elif tier == "low":
        return COLORS["success"]
    return COLORS["neutral_soft"]

def badge_html(tier: str, label: str = None) -> str:
    """Generates standard HTML badge chip for severity."""
    if tier == "high":
        text_color = COLORS["danger"]
        bg_color = COLORS["danger_soft"]
        default_label = "🔴 High"
    elif tier == "medium":
        text_color = COLORS["warning"]
        bg_color = COLORS["warning_soft"]
        default_label = "🟡 Medium"
    else:
        text_color = COLORS["success"]
        bg_color = COLORS["success_soft"]
        default_label = "🟢 Low"
        
    display_label = label if label is not None else default_label
    return f'<span class="badge" style="background-color: {bg_color}; color: {text_color}; border: 1px solid {text_color};">{display_label}</span>'
