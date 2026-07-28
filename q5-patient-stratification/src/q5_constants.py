"""Q5 Domain Constants.

Defines feature sets, resistance pathways, treatability index parameters,
Q3 ODE simulation initial conditions, and Q4 DepMap/LINCS drug target mappings.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

# Features utilized for patient clustering
PHENOTYPE_FEATURES: List[str] = [
    "IFNG_Signaling",
    "TIS",
    "CYT",
    "CD8_T_cell",
    "IMPRES",
    "PDL1_expression",
    "M1_M2_Ratio",
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
    "SUPPRESSIVE": "Immunosuppressive M2-High",
    "MUTANT": "Mutant-Driven",
}

# Q3 ODE initial parameters mapped per Q5 phenotype
Q3_ODE_PHENOTYPE_PARAMS: Dict[str, Dict[str, float]] = {
    "Immune Hot": {"T0": 1.0, "E0": 0.8, "M0": 0.2, "a_kill": 0.15},
    "Immune Cold": {"T0": 1.0, "E0": 0.1, "M0": 0.1, "a_kill": 0.02},
    "Immunosuppressive M2-High": {"T0": 1.0, "E0": 0.3, "M0": 0.8, "a_kill": 0.04},
    "Mutant-Driven": {"T0": 1.0, "E0": 0.5, "M0": 0.3, "a_kill": 0.08},
}

# Q4 DepMap essentiality targets and LINCS perturbagen recommendations per phenotype
Q4_TARGET_NOMINATIONS: Dict[str, Dict[str, List[str]]] = {
    "Immune Cold": {
        "depmap_targets": ["AXL", "MDM2", "CDK4"],
        "lincs_perturbagens": ["Vorinostat", "Entinostat"],
        "strategy": "Priming / Epigenetic Immune Sensitisation + Anti-PD-1",
    },
    "Immunosuppressive M2-High": {
        "depmap_targets": ["CSF1R", "IDO1", "ARG1"],
        "lincs_perturbagens": ["PLX3397", "Epacadostat"],
        "strategy": "M2 Macrophage Depletion / Reprogramming + Anti-PD-1",
    },
    "Mutant-Driven": {
        "depmap_targets": ["BRAF", "MAP2K1", "PIK3CA"],
        "lincs_perturbagens": ["Dabrafenib", "Trametinib"],
        "strategy": "Targeted Kinase Inhibition + Checkpoint Blockade",
    },
}
