"""
Centralized Visualization Styles and Color Palettes for Melanoma Assignment 3.
Based on the Okabe-Ito Palette System - The universal gold standard for scientific publications (Cell, Nature, Science)
ensuring 100% colorblind-safe accessibility across deuteranopia, protanopia, and tritanopia.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Study Cohort Color Mappings (Okabe-Ito Gold Standard System)
COHORT_PALETTE = {
    "Liu 2019": "#0072B2",               # Okabe-Ito Blue
    "Liu 2019 (N=104)": "#0072B2",
    "Hugo 2016": "#E69F00",               # Okabe-Ito Orange
    "Hugo 2016 (N=27)": "#E69F00",
    "Riaz 2017": "#CC79A7",               # Okabe-Ito Reddish Purple
    "Riaz 2017 (N=64)": "#CC79A7",
    "TCGA-SKCM": "#37474F",               # Dark Slate Charcoal Reference
    "TCGA-SKCM (N=426)": "#37474F",
    "Pooled Trials": "#009E73",           # Okabe-Ito Bluish Green Benchmark
    "Pooled Trials (N=195)": "#009E73",
}

# Clinical Response Mappings (Okabe-Ito Status Palette)
RESPONSE_PALETTE = {
    "CR/PR": "#009E73",                   # Okabe-Ito Bluish Green (Positive Response)
    "Responder": "#009E73",
    "Response": "#009E73",
    "PD": "#D55E00",                      # Okabe-Ito Vermillion Red (Progressive Disease)
    "Non-responder": "#D55E00",
    "Non-Response": "#D55E00",
    "SD": "#F0E442",                      # Okabe-Ito Yellow (Stable Disease)
}

# Driver Mutation Subtype Mappings (Okabe-Ito Accent Palette)
DRIVER_PALETTE = {
    "BRAF": "#D55E00",                    # Okabe-Ito Vermillion
    "NRAS": "#56B4E9",                    # Okabe-Ito Sky Blue
    "NF1": "#009E73",                     # Okabe-Ito Bluish Green
    "Triple-WT": "#CC79A7",               # Okabe-Ito Reddish Purple
}


def set_presentation_style(font_scale: float = 1.0, dpi: int = 300):
    """
    Applies project-wide Matplotlib and Seaborn style configurations tailored for
    presentation slides and report figures using Okabe-Ito publication standards.
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'font.size': 11 * font_scale,
        'axes.labelsize': 12 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'xtick.labelsize': 11 * font_scale,
        'ytick.labelsize': 11 * font_scale,
        'legend.fontsize': 10 * font_scale,
        'figure.titlesize': 16 * font_scale,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'figure.autolayout': True,
    })


def get_cohort_color(cohort_name: str, default: str = "#37474F") -> str:
    """Returns the standardized hex color code for a given cohort."""
    for key, color in COHORT_PALETTE.items():
        if key.lower() in cohort_name.lower():
            return color
    return default
