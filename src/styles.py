"""Centralized Visualisation Styles and Colour Palettes for Melanoma Assignment 3.

Based on the Okabe-Ito Palette System - The universal gold standard for scientific publications (Cell, Nature, Science)
ensuring 100% colorblind-safe accessibility across deuteranopia, protanopia, and tritanopia.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Study Cohort Colour Mappings (Okabe-Ito Gold Standard System)
COHORT_PALETTE = {
    "Liu 2019": "#0072B2",               # Okabe-Ito Blue
    "Hugo 2016": "#E69F00",               # Okabe-Ito Orange
    "Riaz 2017": "#CC79A7",               # Okabe-Ito Reddish Purple
    "TCGA-SKCM": "#37474F",               # Dark Slate Charcoal Reference
    "Pooled Trials": "#009E73",           # Okabe-Ito Bluish Green Benchmark
}

# Neutral / Reference Colors
DARK_SLATE_CHARCOAL = "#37474F"           # Dark Slate Charcoal Reference

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
    "Immune Hot": "#D55E00",                                                               # Crimson Red (Hot Inflamed)
    "Immune Hot (High TIS & CYT, Inflamed Microenvironment)": "#D55E00",
    "Immune Cold": "#0072B2",                                                              # Okabe-Ito Blue (Cold / Excluded)
    "Immune Cold (Low TIS & Infiltration, Desert)": "#0072B2",
    "M2-High": "#CC79A7",                                                                  # Reddish Purple (M2 Macrophage)
    "M2": "#CC79A7",
    "Immunosuppressive M2-High": "#CC79A7",
    "Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion)": "#CC79A7",
    "M2 Immunosuppressive": "#CC79A7",
    "M2 Immunosuppressive (High M2 Macrophages & CAFs)": "#CC79A7",
    "Mutant-Driven": "#F0E442",                                                            # Okabe-Ito Yellow (MAPK Mutation Driven)
    "Mutant-Driven (NF1 Loss & High Response Subtype)": "#F0E442",
}


def get_phenotype_color(phenotype_name: str, default: str = "#37474F") -> str:
    """Returns the standardised hex colour code for a given phenotype subtype label (case-insensitive match)."""
    if not isinstance(phenotype_name, str):
        return default
    if phenotype_name in PHENOTYPE_PALETTE:
        return PHENOTYPE_PALETTE[phenotype_name]
    pheno_lower = phenotype_name.lower()
    for key, color in PHENOTYPE_PALETTE.items():
        if key.lower() == pheno_lower:
            return color
    for key, color in PHENOTYPE_PALETTE.items():
        key_lower = key.lower()
        if key_lower in pheno_lower or pheno_lower in key_lower:
            return color
    if "m2" in pheno_lower:
        return "#CC79A7"
    return default


# Full ordered Okabe-Ito palette list for generic sequential categorical encoding
OKABE_ITO = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

# Sex / Gender Data Encoding (Okabe-Ito assignment)
SEX_PALETTE = {
    "Male":    "#56B4E9",  # Okabe-Ito Sky Blue
    "Female":  "#CC79A7",  # Okabe-Ito Reddish Purple
    "Unknown": "#F0E442",  # Okabe-Ito Yellow
}

# Gene Functional Category Encoding (transcriptomic forest plots)
GENE_CATEGORY_PALETTE = {
    "Interferon GTPases":                      "#0072B2",  # Okabe-Ito Blue
    "Chemokines & Cytokines":                  "#009E73",  # Okabe-Ito Bluish Green
    "NK-Cell & T-Cell Receptors & Regulators": "#D55E00",  # Okabe-Ito Vermillion
    "Signalling & Adapters":                    "#E69F00",  # Okabe-Ito Orange
    "Enzymes & Metabolism":                    "#CC79A7",  # Okabe-Ito Reddish Purple
    "Transcription Factors":                   "#56B4E9",  # Okabe-Ito Sky Blue
}

# Analysis Model-Type Encoding (univariate vs multivariate comparisons)
MODEL_TYPE_PALETTE = {
    "Univariate":   "#0072B2",  # Okabe-Ito Blue
    "Multivariate": "#E69F00",  # Okabe-Ito Orange
}

# Feature Selection Method Encoding (AUC comparison plots)
FEATURE_SELECTION_PALETTE = {
    "Curated Signatures":  "#37474F",  # Dark Slate Charcoal
    "SelectKBest (k=20)":  "#F0E442",  # Okabe-Ito Yellow
    "SelectKBest (k=100)": "#E69F00",  # Okabe-Ito Orange
    "SelectKBest (k=200)": "#D55E00",  # Okabe-Ito Vermillion
}

# Threshold Optimisation Strategy Encoding (default vs Youden's J cutoff comparison)
THRESHOLD_STRATEGY_PALETTE = {
    "Default (0.50)":       "#56B4E9",  # Okabe-Ito Sky Blue
    "Optimal (Youden's J)": "#009E73",  # Okabe-Ito Bluish Green
}

# Clinical Utility Strategy Encoding (Q5 decision framework)
STRATEGY_PALETTE = {
    "Phenotype-Stratified (Q5)": "#009E73",  # Okabe-Ito Bluish Green
    "Global Predictor (Q1)":     "#0072B2",  # Okabe-Ito Blue
    "Global Enriched Baseline":  "#0072B2",  # Okabe-Ito Blue (Q1 + Deconvolution)
    "Global Q1 Predictor":       "#0072B2",  # Okabe-Ito Blue (alternate key)
    "Subgroup Specific":         "#009E73",  # Okabe-Ito Bluish Green (alternate key)
    "CD274 (PD-L1+)":            "#E69F00",  # Okabe-Ito Orange
    "High TMB":                  "#CC79A7",  # Okabe-Ito Reddish Purple
    "Treat All":                 "#D55E00",  # Okabe-Ito Vermillion
    "Treat None":                "#F0E442",  # Okabe-Ito Yellow
}

# Treatment Arm Encoding (Q5 treatability scoring)
ARM_PALETTE = {
    "Arm A: Immunotherapy":        "#009E73",  # Okabe-Ito Bluish Green
    "Arm B: Targeted Therapy":     "#E69F00",  # Okabe-Ito Orange
    "Arm C: Combination/Reversal": "#CC79A7",  # Okabe-Ito Reddish Purple
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
    """Returns the standardized hex colour code for a given cohort."""
    for key, color in COHORT_PALETTE.items():
        if key.lower() in cohort_name.lower():
            return color
    return default



