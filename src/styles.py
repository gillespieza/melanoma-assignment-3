"""Centralized Visualization Styles and Color Palettes for Melanoma Assignment 3.

Based on the Okabe-Ito Palette System - The universal gold standard for scientific publications (Cell, Nature, Science)
ensuring 100% colorblind-safe accessibility across deuteranopia, protanopia, and tritanopia.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Study Cohort Color Mappings (Okabe-Ito Gold Standard System)
COHORT_PALETTE = {
    "Liu 2019": "#0072B2",               # Okabe-Ito Blue
    "Hugo 2016": "#E69F00",               # Okabe-Ito Orange
    "Riaz 2017": "#CC79A7",               # Okabe-Ito Reddish Purple
    "TCGA-SKCM": "#37474F",               # Dark Slate Charcoal Reference
    "Pooled Trials": "#009E73",           # Okabe-Ito Bluish Green Benchmark
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

# Biological Phenotype Subtype Mappings (Okabe-Ito Scientific Standards)
PHENOTYPE_PALETTE = {
    "Immune Hot": "#D55E00",                                                      # Crimson Red (Hot Inflamed)
    "Immune Hot (High TIS & CYT, Inflamed Microenvironment)": "#D55E00",
    "Immune Cold": "#0072B2",                                                     # Okabe-Ito Blue (Cold / Excluded)
    "Immune Cold (Low TIS & Infiltration, Desert)": "#0072B2",
    "Immunosuppressive M2-High": "#CC79A7",                                        # Reddish Purple (M2 Macrophage)
    "M2 Immunosuppressive": "#CC79A7",
    "M2 Immunosuppressive (High M2 Macrophages & CAFs)": "#CC79A7",
    "Mutant-Driven": "#E69F00",                                                   # Orange (MAPK Mutation Driven)
}

from matplotlib.colors import LinearSegmentedColormap


def get_okabe_ito_diverging_cmap():
    """Returns a colorblind-safe Okabe-Ito continuous diverging colormap for heatmaps.

    Maps negative values (e.g. beta < 0) to Okabe-Ito Vermillion Red (#D55E00), neutral values to
    light gray (#FAFAFA), and positive values (e.g. beta > 0) to Okabe-Ito Bluish Green (#009E73).
    """
    colors = ["#D55E00", "#FAFAFA", "#009E73"]
    return LinearSegmentedColormap.from_list("OkabeItoDiverging", colors, N=256)


def set_presentation_style(font_scale: float = 1.0, dpi: int = 300):
    """Applies project-wide Matplotlib and Seaborn style configurations tailored for

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
        'grid.color': '#E5E7EB',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
    })


def get_cohort_color(cohort_name: str, default: str = "#37474F") -> str:
    """Returns the standardized hex color code for a given cohort."""
    for key, color in COHORT_PALETTE.items():
        if key.lower() in cohort_name.lower():
            return color
    return default
