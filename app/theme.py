"""Visual system for the PORAF interface.

Two constraints shape this. First, output has to sit alongside existing PORAF
material produced in Word and SPSS, so the look is print-adjacent: a warm
paper surface, thin rules instead of boxes, and typography doing the work that
borders and shadows would otherwise do.

Second, the subgroup palette had to change. The colours previously used for
the four faller subgroups (#2166ac, #4dac26, #d95f02, #d62728) are not
separable: Recovered Function and Declined Function sit at Delta E 9.7 in
normal vision and Maintained vs Recovered collapses to Delta E 2.8 under
deuteranopia. Those are the two clinically opposite outcomes in the study, and
in a printed thesis figure they would be indistinguishable. The palette below
is Okabe-Ito, which is published as colour-vision-deficiency safe and clears
every separation check with the worst adjacent pair at Delta E 11.4.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Surfaces and ink
# --------------------------------------------------------------------------

PAPER = "#FCFCFB"          # chart + page surface
PAPER_SUNK = "#F4F3F0"     # recessed panels
RULE = "#DEDCD6"           # hairline rules
RULE_STRONG = "#C4C1B8"

INK = "#1A1A18"            # primary text
INK_SECONDARY = "#55534C"  # supporting text
INK_MUTED = "#86837A"      # axis labels, captions

NAVY = "#1F4E79"           # PORAF section-header navy, retained
SIGNIFICANT = "#B3261E"    # significant p-values (PORAF convention)

# --------------------------------------------------------------------------
# Categorical palette - Okabe-Ito, validated colour-vision-deficiency safe
# --------------------------------------------------------------------------

SUBGROUP_COLORS: dict[str, str] = {
    "Non-Faller": "#0072B2",           # blue
    "Maintained Function": "#009E73",  # green
    "Recovered Function": "#E69F00",   # amber
    "Declined Function": "#D55E00",    # vermillion
}
SUBGROUP_ORDER = list(SUBGROUP_COLORS)

#: Abbreviation -> colour, for compact plots.
SUBGROUP_COLORS_ABBREV = {
    "NF": "#0072B2", "MF": "#009E73", "RF": "#E69F00", "DF": "#D55E00",
}

ACCENT = "#0072B2"
ACCENT_SOFT = "#CFE3F0"

#: Status colours, reserved - never reused as a series colour.
STATUS = {
    "good": "#1B7F5C",
    "warning": "#B07A00",
    "serious": "#C25A00",
    "critical": "#B3261E",
    "neutral": "#55534C",
}

SERIF = "'Source Serif 4', 'Iowan Old Style', Georgia, 'Times New Roman', serif"
SANS = "'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', 'SF Mono', Menlo, Consolas, monospace"


# --------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------

def plotly_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 420,
    showlegend: bool = True,
) -> dict:
    """Shared layout: recessive axes, no chart junk, hover always on."""
    return dict(
        title=dict(
            text=title,
            font=dict(family=SERIF, size=17, color=INK),
            x=0, xanchor="left", y=0.96,
        ),
        height=height,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=SANS, size=12.5, color=INK_SECONDARY),
        margin=dict(l=8, r=8, t=48 if title else 18, b=8),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=12, color=INK_MUTED)),
            showgrid=False,
            zeroline=False,
            linecolor=RULE_STRONG,
            ticks="outside",
            ticklen=4,
            tickcolor=RULE_STRONG,
            tickfont=dict(size=11.5, color=INK_MUTED),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=12, color=INK_MUTED)),
            showgrid=True,
            gridcolor=RULE,
            gridwidth=1,
            zeroline=False,
            linecolor=RULE_STRONG,
            ticks="outside",
            ticklen=4,
            tickcolor=RULE_STRONG,
            tickfont=dict(size=11.5, color=INK_MUTED),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.0,
            xanchor="right", x=1.0,
            font=dict(size=11.5, color=INK_SECONDARY),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        showlegend=showlegend,
        hoverlabel=dict(
            bgcolor=PAPER,
            bordercolor=RULE_STRONG,
            font=dict(family=SANS, size=12, color=INK),
        ),
        hovermode="closest",
    )


def format_p(p: float | None) -> str:
    """PORAF convention: p < 0.001 rather than a long decimal."""
    if p is None:
        return "n/a"
    try:
        value = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if value != value:  # NaN
        return "n/a"
    return "&lt;0.001" if value < 0.001 else f"{value:.3f}"


# --------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------

def stylesheet() -> str:
    """Global CSS. Typography-led, hairline rules, no shadows or gradients."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --paper: {PAPER};
  --paper-sunk: {PAPER_SUNK};
  --rule: {RULE};
  --rule-strong: {RULE_STRONG};
  --ink: {INK};
  --ink-2: {INK_SECONDARY};
  --ink-3: {INK_MUTED};
  --accent: {ACCENT};
  --navy: {NAVY};
  --sig: {SIGNIFICANT};
}}

.stApp {{ background: var(--paper); }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 5rem; max-width: 1080px; }}

html, body, [class*="css"] {{
  font-family: {SANS};
  color: var(--ink);
  font-feature-settings: 'tnum' 1, 'cv05' 1;
}}

/* Streamlit ships its own heading rules, so these need the .stApp prefix
   to win on specificity rather than relying on source order. */
.stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMarkdownContainer"] h1,
.stApp [data-testid="stMarkdownContainer"] h2,
.stApp [data-testid="stMarkdownContainer"] h3 {{
  font-family: {SERIF} !important;
  color: var(--ink);
  letter-spacing: -0.012em;
  font-weight: 600;
}}
.stApp h1 {{ font-size: 2.15rem; margin: 0 0 .15rem 0; line-height: 1.12; }}
.stApp h2 {{ font-size: 1.32rem; margin: 2.4rem 0 .6rem 0; }}
.stApp h3 {{ font-size: 1.05rem; margin: 1.6rem 0 .4rem 0; }}

/* ---- masthead ---- */
.masthead {{ border-bottom: 1px solid var(--rule-strong); padding-bottom: 1.1rem; margin-bottom: 1.8rem; }}
.masthead .eyebrow {{
  font-size: .70rem; font-weight: 600; letter-spacing: .13em; text-transform: uppercase;
  color: var(--ink-3); margin-bottom: .45rem;
}}
.masthead .sub {{ color: var(--ink-2); font-size: .95rem; margin-top: .3rem; max-width: 62ch; line-height: 1.55; }}

/* ---- section label ---- */
.section-label {{
  font-size: .70rem; font-weight: 600; letter-spacing: .13em; text-transform: uppercase;
  color: var(--ink-3); border-bottom: 1px solid var(--rule); padding-bottom: .4rem;
  margin: 2.2rem 0 1rem 0;
}}

/* ---- verdict: the plain-language answer, always first ---- */
.verdict {{
  border-left: 2px solid var(--accent);
  padding: .1rem 0 .1rem 1.15rem;
  margin: .4rem 0 1.4rem 0;
}}
.verdict .lead {{ font-family: {SERIF}; font-size: 1.16rem; line-height: 1.5; color: var(--ink); }}
.verdict.null {{ border-left-color: var(--ink-3); }}
.verdict.caution {{ border-left-color: {STATUS['warning']}; }}
.verdict.stop {{ border-left-color: var(--sig); }}

/* ---- figures ---- */
.stat-row {{ display: flex; gap: 2.6rem; flex-wrap: wrap; margin: .3rem 0 1.2rem 0; }}
.stat {{ min-width: 92px; }}
.stat .k {{
  font-size: .68rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-3); margin-bottom: .22rem;
}}
.stat .v {{ font-family: {SERIF}; font-size: 1.72rem; font-weight: 600; color: var(--ink); line-height: 1.05; }}
.stat .u {{ font-size: .82rem; color: var(--ink-3); margin-left: .18rem; font-family: {SANS}; }}

/* ---- hero prediction ---- */
.hero {{ border-top: 2px solid var(--ink); border-bottom: 1px solid var(--rule); padding: 1.3rem 0; margin: .6rem 0 1.2rem 0; }}
.hero .v {{ font-family: {SERIF}; font-size: 3.0rem; font-weight: 600; line-height: 1; color: var(--ink); }}
.hero .band {{ font-size: .96rem; color: var(--ink-2); margin-top: .5rem; }}
.hero .band b {{ font-weight: 600; color: var(--ink); }}

/* ---- abstention ---- */
.abstain {{ border: 1px solid var(--rule-strong); background: var(--paper-sunk); padding: 1.15rem 1.3rem; margin: .6rem 0 1.2rem 0; }}
.abstain .t {{
  font-size: .70rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
  color: {STATUS['serious']}; margin-bottom: .55rem;
}}
.abstain ul {{ margin: .3rem 0 0 1.05rem; padding: 0; }}
.abstain li {{ color: var(--ink-2); font-size: .93rem; margin-bottom: .32rem; line-height: 1.5; }}

/* ---- notes ---- */
.note {{ font-size: .87rem; color: var(--ink-2); line-height: 1.58; margin: .5rem 0; }}
.note.tech {{ font-family: {MONO}; font-size: .80rem; color: var(--ink-3); line-height: 1.65; }}
.caveat {{
  font-size: .87rem; color: var(--ink-2); border-left: 1px solid var(--rule-strong);
  padding-left: .85rem; margin: .45rem 0; line-height: 1.55;
}}

/* ---- evidence grade chips ---- */
.grade {{
  display: inline-block; font-size: .66rem; font-weight: 600; letter-spacing: .09em;
  padding: .16rem .46rem; border: 1px solid currentColor; margin-right: .5rem; vertical-align: .09em;
}}
.grade-A {{ color: {STATUS['good']}; }}
.grade-B {{ color: {NAVY}; }}
.grade-C {{ color: {STATUS['warning']}; }}
.grade-D {{ color: var(--ink-3); }}

.rec {{ border-bottom: 1px solid var(--rule); padding: 1.05rem 0; }}
.rec:last-child {{ border-bottom: none; }}
.rec .title {{ font-family: {SERIF}; font-size: 1.08rem; font-weight: 600; color: var(--ink); }}
.rec .body {{ font-size: .91rem; color: var(--ink-2); line-height: 1.6; margin-top: .4rem; }}
.rec .trigger {{ font-family: {MONO}; font-size: .77rem; color: var(--ink-3); margin-top: .45rem; }}
.rec .cite {{ font-size: .80rem; color: var(--ink-3); margin-top: .5rem; font-style: italic; line-height: 1.5; }}
.rec .caution {{ font-size: .86rem; color: {STATUS['serious']}; margin-top: .45rem; line-height: 1.5; }}

/* ---- data-source banner ---- */
.source-tag {{
  display: inline-block; font-size: .70rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; padding: .2rem .5rem; border: 1px solid var(--rule-strong); color: var(--ink-2);
}}
.source-tag.real {{ color: var(--sig); border-color: var(--sig); }}

/* ---- tables ---- */
[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); }}
[data-testid="stTable"] table {{ font-size: .88rem; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{ background: var(--paper-sunk); border-right: 1px solid var(--rule); }}
[data-testid="stSidebar"] .block-container {{ padding-top: 2rem; }}

/* ---- controls ---- */
.stButton > button,
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {{
  font-family: {SANS}; font-weight: 500; font-size: .88rem; border-radius: 2px;
  border: 1px solid var(--ink) !important; background: var(--ink) !important;
  color: var(--paper) !important; padding: .44rem 1.35rem; transition: opacity .12s ease;
  box-shadow: none;
}}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {{
  opacity: .84; border-color: var(--ink) !important;
  background: var(--ink) !important; color: var(--paper) !important;
}}
.stButton > button p, .stFormSubmitButton > button p,
[data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-secondaryFormSubmit"] p {{ color: var(--paper) !important; }}

/* Streamlit derives its header/chrome button background from theme.textColor,
   which here is near-black - so the sidebar collapse control renders as a solid
   block. These are Streamlit's own chrome, not our controls, so reset them. */
[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stBaseButton-header"],
[data-testid="stBaseButton-elementToolbar"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stAppDeployButton"] button {{
  background: transparent !important;
  border: none !important;
  color: var(--ink-2) !important;
  box-shadow: none !important;
}}
[data-testid="stBaseButton-headerNoPadding"] svg,
[data-testid="stBaseButton-header"] svg {{ fill: var(--ink-2) !important; }}
[data-testid="stAppDeployButton"] {{ display: none; }}

/* Forms: a hairline frame, not a raised card. */
[data-testid="stForm"] {{
  border: 1px solid var(--rule); border-radius: 0; background: var(--paper);
  padding: 1.4rem 1.5rem 1.2rem 1.5rem;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{ font-size: .88rem; font-weight: 500; padding: .5rem 0; }}

footer, #MainMenu, header {{ visibility: hidden; }}
</style>
"""


def verdict_block(text: str, tone: str = "neutral") -> str:
    """The plain-language answer, rendered before any technical detail."""
    tone_class = {"neutral": "", "null": "null", "caution": "caution", "stop": "stop"}.get(tone, "")
    return f'<div class="verdict {tone_class}"><div class="lead">{text}</div></div>'


def stat_row(items: list[tuple[str, str, str]]) -> str:
    """A row of key figures: ``(label, value, unit)``."""
    cells = "".join(
        f'<div class="stat"><div class="k">{k}</div>'
        f'<div class="v">{v}<span class="u">{u}</span></div></div>'
        for k, v, u in items
    )
    return f'<div class="stat-row">{cells}</div>'


def section(label: str) -> str:
    return f'<div class="section-label">{label}</div>'
