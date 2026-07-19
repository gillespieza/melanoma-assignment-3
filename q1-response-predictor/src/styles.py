"""
Centralized Visualization Styles and Color Palettes for Melanoma Assignment 3.
Use this module across all analysis and plotting scripts to ensure consistent formatting.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Study Cohort Color Mappings
COHORT_PALETTE = {
    "Liu 2019": "#2B5C8F",
    "Liu 2019 (N=104)": "#2B5C8F",
    "Hugo 2016": "#D95F02",
    "Hugo 2016 (N=27)": "#D95F02",
    "Riaz 2017": "#7570B3",
    "Riaz 2017 (N=64)": "#7570B3",
    "TCGA-SKCM": "#333333",
    "TCGA-SKCM (N=426)": "#333333",
    "Pooled Trials": "#1B9E77",
    "Pooled Trials (N=195)": "#1B9E77",
}

# Clinical Response Mappings
RESPONSE_PALETTE = {
    "CR/PR": "#2E7D32",
    "Responder": "#2E7D32",
    "Response": "#2E7D32",
    "PD": "#C62828",
    "Non-responder": "#C62828",
    "Non-Response": "#C62828",
    "SD": "#F57C00",
}

# Driver Mutation Subtype Mappings
DRIVER_PALETTE = {
    "BRAF": "#E41A1C",
    "NRAS": "#377EB8",
    "NF1": "#4DAF4A",
    "Triple-WT": "#984EA3",
}


def set_presentation_style(font_scale: float = 1.0, dpi: int = 300):
    """
    Applies project-wide Matplotlib and Seaborn style configurations tailored for
    presentation slides and report figures.
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


def get_cohort_color(cohort_name: str, default: str = "#555555") -> str:
    """Returns the standardized hex color code for a given cohort."""
    for key, color in COHORT_PALETTE.items():
        if key.lower() in cohort_name.lower():
            return color
    return default
