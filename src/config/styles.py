"""
CSS customizado para o dashboard.
"""
from src.config.settings import COLORS


def get_custom_css() -> str:
    """Retorna o CSS customizado para o dashboard."""
    return f"""
/* ══════════════════════════════════════════════════════════════════════════ */
/* RESET & BASE                                                                */
/* ══════════════════════════════════════════════════════════════════════════ */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, #1a1f35 100%);
    color: {COLORS['text_primary']};
    min-height: 100vh;
    line-height: 1.6;
}}

/* ══════════════════════════════════════════════════════════════════════════ */
/* TYPOGRAPHY                                                                  */
/* ══════════════════════════════════════════════════════════════════════════ */
.main-title {{
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, {COLORS['primary_light']} 0%, {COLORS['secondary']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.subtitle {{
    font-size: 0.95rem;
    color: {COLORS['text_secondary']};
    font-weight: 400;
    margin-top: 0.25rem;
}}

.section-title {{
    font-size: 1.125rem;
    font-weight: 600;
    color: {COLORS['text_primary']};
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.section-title::before {{
    content: '';
    width: 4px;
    height: 1.25rem;
    background: linear-gradient(180deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
    border-radius: 2px;
}}

/* ══════════════════════════════════════════════════════════════════════════ */
/* CARDS & CONTAINERS                                                          */
/* ══════════════════════════════════════════════════════════════════════════ */
.dashboard-card {{
    background: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']}33;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}}

.dashboard-card:hover {{
    border-color: {COLORS['primary']}44;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
}}

.metric-card {{
    background: linear-gradient(135deg, {COLORS['bg_card']} 0%, {COLORS['bg_hover']}44 100%);
    border: 1px solid {COLORS['border']}22;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}}

.metric-card:hover {{
    transform: translateY(-2px);
    border-color: {COLORS['primary']}55;
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {COLORS['text_primary']};
    line-height: 1.2;
}}

.metric-label {{
    font-size: 0.8rem;
    color: {COLORS['text_secondary']};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}}

/* ══════════════════════════════════════════════════════════════════════════ */
/* TABS NAVIGATION — Vercel Style                                              */
/* ══════════════════════════════════════════════════════════════════════════ */

/* Main navigation bar */
.nav-pills {{
    display: flex;
    align-items: center;
    gap: 0;
    padding: 0;
    background: transparent;
    border: none;
    border-bottom: 1px solid {COLORS['border']}44;
    border-radius: 0;
    margin-bottom: 2rem;
    padding-bottom: 0;
}}

/* Individual tab buttons */
.nav-pills .nav-link {{
    position: relative;
    color: {COLORS['text_secondary']};
    font-weight: 400;
    font-size: 0.875rem;
    padding: 0.75rem 1rem;
    border: none;
    border-radius: 0;
    background: transparent;
    transition: color 0.15s ease;
    margin: 0;
    margin-bottom: -1px;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    white-space: nowrap;
}}

/* Tab icon styling */
.nav-pills .nav-link i {{
    font-size: 0.875rem;
    opacity: 0.7;
    transition: opacity 0.15s ease;
}}

/* Hover state */
.nav-pills .nav-link:hover {{
    color: {COLORS['text_primary']};
    background: transparent;
}}

.nav-pills .nav-link:hover i {{
    opacity: 1;
}}

/* Active state — Vercel underline style */
.nav-pills .nav-link.active {{
    color: {COLORS['text_primary']} !important;
    background: transparent !important;
    font-weight: 500;
}}

.nav-pills .nav-link.active::after {{
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: {COLORS['text_primary']};
    border-radius: 1px 1px 0 0;
}}

/* Active tab icon */
.nav-pills .nav-link.active i {{
    opacity: 1;
}}

/* Focus state for accessibility */
.nav-pills .nav-link:focus {{
    outline: none;
}}

/* Tab content area */
.tab-content {{
    padding-top: 0.5rem;
}}

/* ══════════════════════════════════════════════════════════════════════════ */
/* FORM CONTROLS                                                               */
/* ══════════════════════════════════════════════════════════════════════════ */
.control-label {{
    font-size: 0.8rem;
    font-weight: 600;
    color: {COLORS['text_secondary']};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    display: block;
}}

.Select-control {{
    background: {COLORS['bg_hover']} !important;
    border: 1px solid {COLORS['border']}44 !important;
    border-radius: 8px !important;
}}

.Select-value-label {{
    color: {COLORS['text_primary']} !important;
}}

.Select-menu-outer {{
    background: {COLORS['bg_card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
}}

/* Slider styling */
.rc-slider-track {{
    background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%) !important;
}}

.rc-slider-handle {{
    border-color: {COLORS['primary']} !important;
    background: {COLORS['text_primary']} !important;
}}

.rc-slider-rail {{
    background: {COLORS['bg_hover']} !important;
}}

/* ══════════════════════════════════════════════════════════════════════════ */
/* UTILITY CLASSES                                                             */
/* ══════════════════════════════════════════════════════════════════════════ */
.text-gradient {{
    background: linear-gradient(135deg, {COLORS['primary_light']} 0%, {COLORS['accent']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.badge-model {{
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}}

.badge-logreg {{
    background: {COLORS['logreg']}22;
    color: {COLORS['logreg']};
    border: 1px solid {COLORS['logreg']}44;
}}

.badge-rf {{
    background: {COLORS['rf']}22;
    color: {COLORS['rf']};
    border: 1px solid {COLORS['rf']}44;
}}

.divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {COLORS['border']}44, transparent);
    margin: 1.5rem 0;
}}

/* ══════════════════════════════════════════════════════════════════════════ */
/* DARK TABLE STYLING                                                          */
/* ══════════════════════════════════════════════════════════════════════════ */
.dark-table-container {{
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid {COLORS['border']}33;
}}

.dark-table-container .table {{
    margin-bottom: 0;
    background: transparent;
    color: {COLORS['text_primary']};
    font-size: 0.8rem;
}}

.dark-table-container .table thead th {{
    background: {COLORS['bg_hover']};
    color: {COLORS['text_secondary']};
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-bottom: 1px solid {COLORS['border']}44;
    border-top: none;
    padding: 0.625rem 0.75rem;
    white-space: nowrap;
}}

.dark-table-container .table tbody td {{
    background: transparent;
    color: {COLORS['text_primary']};
    border-bottom: 1px solid {COLORS['border']}22;
    padding: 0.5rem 0.75rem;
    vertical-align: middle;
}}

.dark-table-container .table tbody tr:last-child td {{
    border-bottom: none;
}}

.dark-table-container .table tbody tr:hover td {{
    background: {COLORS['bg_hover']}44;
}}

.dark-table-container .table-striped tbody tr:nth-of-type(odd) td {{
    background: {COLORS['bg_hover']}22;
}}

.dark-table-container .table-striped tbody tr:nth-of-type(odd):hover td {{
    background: {COLORS['bg_hover']}44;
}}

/* Model badges in table */
.dark-table-container .badge-logreg,
.dark-table-container .badge-rf {{
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
}}

/* ══════════════════════════════════════════════════════════════════════════ */
/* RESPONSIVE                                                                  */
/* ══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {{
    .main-title {{
        font-size: 1.5rem;
    }}
    .dashboard-card {{
        padding: 1rem;
    }}
}}
"""
