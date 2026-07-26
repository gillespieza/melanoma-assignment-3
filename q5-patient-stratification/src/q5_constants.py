"""Q5 Domain Constants.

Defines feature sets, resistance pathways, and treatability index parameters
specific to Q5 patient stratification and decision impact analysis.
"""

from typing import Dict, List

# Features utilized for patient clustering
PHENOTYPE_FEATURES: List[str] = [
    "IFNG_Signaling",
    "TIS",
    "CYT",
    "CD8_T_cell",
    "IMPRES",
    "PDL1_expression",
    "TMB",
]

# Resistance pathways and key associated genes
RESISTANCE_PATHWAYS: Dict[str, List[str]] = {
    "Antigen_Presentation_Loss": ["B2M", "TAP1", "TAP2", "HLA-A", "HLA-B", "HLA-C"],
    "IFN_Gamma_Resistance": ["JAK1", "JAK2", "STAT1", "STAT3"],
    "Immunosuppressive_Microenvironment": ["IL10", "TGFB1", "IDO1", "ARG1"],
    "High_Proliferation_Glycolysis": ["MYC", "MTOR", "LDHA", "PFKFB3"],
}

# Features evaluated within the treatability index
TREATABILITY_FEATURES: List[str] = [
    "antigen_presentation_intact",
    "ifn_gamma_signaling_intact",
    "low_cna_burden",
    "actionable_mutations_present",
]

# Standard phenotype labels
PHENOTYPE_LABELS: Dict[str, str] = {
    "HOT": "Immune Hot",
    "COLD": "Immune Cold",
    "MUTANT": "Mutant-Driven",
    "RESISTANT": "Therapy Resistant",
}
