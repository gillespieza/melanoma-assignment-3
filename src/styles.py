"""
Centralized Visualization Styles and Color Palettes for Melanoma Assignment 3.
Based on Google Material Design 3 (M3) Color System for optimal contrast and slide readability.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Study Cohort Color Mappings (Google Material Design System)
COHORT_PALETTE = {
    "Liu 2019": "#1976D2",               # Material Blue 700 (Vibrant Slate Indigo)
    "Liu 2019 (N=104)": "#1976D2",
    "Hugo 2016": "#E65100",               # Material Deep Orange 900 (Warm Terracotta)
    "Hugo 2016 (N=27)": "#E65100",
    "Riaz 2017": "#673AB7",               # Material Deep Purple 500 (Rich Violet)
    "Riaz 2017 (N=64)": "#673AB7",
    "TCGA-SKCM": "#37474F",               # Material Blue Grey 800 (Dark Charcoal Reference)
    "TCGA-SKCM (N=426)": "#37474F",
    "Pooled Trials": "#00796B",           # Material Teal 700 (Ocean Emerald Benchmark)
    "Pooled Trials (N=195)": "#00796B",
}

# Clinical Response Mappings (Google Material Status Palette)
RESPONSE_PALETTE = {
    "CR/PR": "#2E7D32",                   # Material Green 800 (Positive Response)
    "Responder": "#2E7D32",
    "Response": "#2E7D32",
    "PD": "#C62828",                      # Material Red 800 (Progressive Disease)
    "Non-responder": "#C62828",
    "Non-Response": "#C62828",
    "SD": "#F57C00",                      # Material Orange 700 (Stable Disease)
}

# Driver Mutation Subtype Mappings (Google Material Accent Palette)
DRIVER_PALETTE = {
    "BRAF": "#D32F2F",                    # Material Red 700
    "NRAS": "#0288D1",                    # Material Light Blue 700
    "NF1": "#388E3C",                     # Material Green 700
    "Triple-WT": "#7B1FA2",               # Material Purple 700
}


def set_presentation_style(font_scale: float = 1.0, dpi: int = 300):
    """
    Applies project-wide Matplotlib and Seaborn style configurations tailored for
    presentation slides and report figures using Material Design typography.
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
