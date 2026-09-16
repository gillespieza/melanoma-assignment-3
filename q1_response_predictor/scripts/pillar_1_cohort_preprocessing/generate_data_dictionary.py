"""
Generate Harmonised Data Dictionary for Preprocessed Melanoma Cohorts.

Creates config/data_dictionary.json containing standardized metadata, data types,
units, descriptions, value maps, and cohort-specific overrides for all clinical,
genomic, transcriptomic, and treatment features.
"""

import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd

# Bootstrap project root resolution
_SCRIPT_DIR = Path(__file__).resolve().parent
_SUBPROJECT_ROOT = _SCRIPT_DIR.parents[1]
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR

LOG_DIR = _SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "generate_data_dictionary.log"
CONFIG_DIR = _SUBPROJECT_ROOT / "config"
OUTPUT_JSON = CONFIG_DIR / "data_dictionary.json"


# ---------------------------------------------------------------------------
# Dictionary Metadata Schema Definition
# ---------------------------------------------------------------------------

COLUMN_METADATA_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Identifiers
    "SAMPLE_ID": {
        "display_name": "Sample Identifier",
        "datatype": "STRING",
        "category": "Identifier",
        "standard_unit": "text",
        "harmonised_definition": "Unique cohort-prefixed specimen identifier.",
    },
    "PATIENT_ID": {
        "display_name": "Patient Identifier",
        "datatype": "STRING",
        "category": "Identifier",
        "standard_unit": "text",
        "harmonised_definition": "Unique cohort-prefixed patient identifier.",
    },
    "COHORT": {
        "display_name": "Study Cohort Name",
        "datatype": "CATEGORICAL",
        "category": "Metadata",
        "standard_unit": "text",
        "harmonised_definition": "Harmonised dataset identifier label.",
    },

    # Primary Clinical Outcomes
    "RESPONSE": {
        "display_name": "RECIST Response Label",
        "datatype": "CATEGORICAL",
        "category": "Clinical Outcome",
        "standard_unit": "text",
        "allowed_values": [
            "Complete Response",
            "Partial Response",
            "Stable Disease",
            "Progressive Disease",
        ],
        "harmonised_definition": "Response evaluation criteria in solid tumours (RECIST 1.1) category.",
        "cohort_overrides": {
            "van_allen_2015": {
                "raw_column": "DURABLE_CLINICAL_BENEFIT",
                "notes": "Mapped from abbreviation codes (CR, PR, SD, PD).",
            },
            "skcm_tcga_gdc": {
                "raw_column": "TREATMENT_OUTCOME",
                "notes": "Extracted from GDC treatment timeline data (immunotherapy priority).",
            },
            "skcm_tcga_pan_can_atlas_2018": {
                "raw_column": "MEASURE_OF_RESPONSE",
                "notes": "Extracted from PanCan treatment timeline data.",
            },
        },
    },
    "RESPONSE_BINARY": {
        "display_name": "Binary RECIST Response Indicator",
        "datatype": "CATEGORICAL",
        "category": "Clinical Outcome",
        "standard_unit": "binary flag",
        "allowed_values": [0.0, 1.0],
        "value_map": {
            "1.0": "Responder (Complete Response / Partial Response)",
            "0.0": "Non-Responder (Progressive Disease)",
        },
        "harmonised_definition": "Binary response classification (1 = Responder, 0 = Non-Responder, NaN = Stable Disease/Unmapped).",
    },
    "OS_MONTHS": {
        "display_name": "Overall Survival Duration",
        "datatype": "NUMBER",
        "category": "Survival Endpoint",
        "standard_unit": "months",
        "harmonised_definition": "Duration from diagnosis/treatment to last follow-up or death in months.",
    },
    "OS_STATUS": {
        "display_name": "Overall Survival Event Status",
        "datatype": "CATEGORICAL",
        "category": "Survival Endpoint",
        "standard_unit": "binary flag",
        "allowed_values": [0, 1],
        "value_map": {"1": "Deceased", "0": "Living / Censored"},
        "harmonised_definition": "Binary vital status indicator at last follow-up.",
    },
    "PFS_MONTHS": {
        "display_name": "Progression-Free Survival Duration",
        "datatype": "NUMBER",
        "category": "Survival Endpoint",
        "standard_unit": "months",
        "harmonised_definition": "Time from baseline to progression or death.",
    },
    "PFS_STATUS": {
        "display_name": "Progression-Free Survival Event Status",
        "datatype": "CATEGORICAL",
        "category": "Survival Endpoint",
        "standard_unit": "binary flag",
        "allowed_values": [0, 1],
        "value_map": {"1": "Progressed / Deceased", "0": "Progression-Free / Censored"},
        "harmonised_definition": "Binary progression event indicator.",
    },

    # Demographic & Clinical Baseline Features
    "AGE": {
        "display_name": "Patient Age",
        "datatype": "NUMBER",
        "category": "Demographic",
        "standard_unit": "years",
        "harmonised_definition": "Age at diagnosis or study entry.",
    },
    "SEX": {
        "display_name": "Patient Biological Sex",
        "datatype": "CATEGORICAL",
        "category": "Demographic",
        "standard_unit": "text",
        "allowed_values": ["MALE", "FEMALE"],
        "harmonised_definition": "Reported biological sex.",
    },
    "RACE": {
        "display_name": "Patient Race",
        "datatype": "CATEGORICAL",
        "category": "Demographic",
        "standard_unit": "text",
        "harmonised_definition": "Self-reported or clinical race designation.",
    },
    "SPECIMEN_TYPE": {
        "display_name": "Biopsy Specimen Site Type",
        "datatype": "CATEGORICAL",
        "category": "Specimen",
        "standard_unit": "text",
        "allowed_values": ["Metastatic", "Primary", "Unknown"],
        "harmonised_definition": "Anatomic origin of sequenced tumor specimen.",
    },
    "IMMUNOTHERAPY": {
        "display_name": "Immunotherapy Treatment Flag",
        "datatype": "CATEGORICAL",
        "category": "Treatment History",
        "standard_unit": "binary flag",
        "allowed_values": [0, 1],
        "value_map": {"1": "Treated with Immunotherapy", "0": "No Immunotherapy"},
        "harmonised_definition": "Binary indicator of prior or current immune checkpoint inhibitor therapy.",
    },

    # Genomic Burden & Driver Features
    "TMB_NONSYNONYMOUS": {
        "display_name": "Tumour Mutational Burden (Nonsynonymous)",
        "datatype": "NUMBER",
        "category": "Genomic Burden",
        "standard_unit": "mutations/Mb",
        "harmonised_definition": (
            "Source-provided density of non-silent somatic mutations in protein-coding "
            "regions, harmonised across cohorts using the iAtlas TMB field."
        ),
        "notes": (
            "Values are retained from the source clinical metadata and are not recalculated "
            "from cohort MAF files. The supplied values are consistent with a fixed 30 Mb "
            "exome denominator; cross-cohort comparisons remain subject to source pipeline "
            "differences in callable territory, variant filtering, and mutation calling."
        ),
        "cohort_overrides": {
            "van_allen_2015": {
                "raw_column": "MUTATION_COUNT",
                "raw_unit": "WES non-synonymous mutation count (source-specific)",
                "notes": "Source mutation count is not directly comparable with the harmonised mutations/Mb TMB field.",
            },
            "skcm_tcga_gdc": {
                "raw_column": "TMB_NONSYNONYMOUS",
                "raw_unit": "mutations/Mb",
                "notes": "Filtered from GDC MAF for non-silent variant classifications.",
            },
        },
    },
    "SNV_NEOANTIGEN": {
        "display_name": "SNV Neoantigen Load",
        "datatype": "NUMBER",
        "category": "Genomic Burden",
        "standard_unit": "count",
        "harmonised_definition": "Predicted single nucleotide variant neoepitope burden.",
    },
    "INDEL_NEOANTIGEN": {
        "display_name": "Indel Neoantigen Load",
        "datatype": "NUMBER",
        "category": "Genomic Burden",
        "standard_unit": "count",
        "harmonised_definition": "Predicted insertion/deletion frameshift neoepitope burden.",
    },
    "mut_BRAF": {
        "display_name": "BRAF Somatic Mutation Status",
        "datatype": "BOOLEAN",
        "category": "Driver Mutation",
        "standard_unit": "binary flag",
        "allowed_values": [0, 1],
        "value_map": {"1": "BRAF Mutant", "0": "BRAF Wild-Type"},
        "harmonised_definition": "Binary indicator of non-silent mutation in BRAF gene.",
    },
    "mut_NRAS": {
        "display_name": "NRAS Somatic Mutation Status",
        "datatype": "BOOLEAN",
        "category": "Driver Mutation",
        "standard_unit": "binary flag",
        "allowed_values": [0, 1],
        "value_map": {"1": "NRAS Mutant", "0": "NRAS Wild-Type"},
        "harmonised_definition": "Binary indicator of non-silent mutation in NRAS gene.",
    },
    "mut_NF1": {
        "display_name": "NF1 Somatic Mutation Status",
        "datatype": "BOOLEAN",
        "category": "Driver Mutation",
        "standard_unit": "binary flag",
        "allowed_values": [0, 1],
        "value_map": {"1": "NF1 Mutant", "0": "NF1 Wild-Type"},
        "harmonised_definition": "Binary indicator of non-silent mutation in NF1 gene.",
    },

    # Supplementary & Timeline Features
    "WINTER_HYPOXIA_SCORE": {
        "display_name": "Winter Hypoxia Score",
        "datatype": "NUMBER",
        "category": "Microenvironment Signature",
        "standard_unit": "z-score / index",
        "harmonised_definition": "Gene expression-derived tumor hypoxia signature score (Winter et al.).",
    },
    "TREATMENT_OUTCOME": {
        "display_name": "Overall Treatment Outcome",
        "datatype": "CATEGORICAL",
        "category": "Treatment History",
        "standard_unit": "text",
        "harmonised_definition": "Best overall therapeutic outcome extracted from clinical timeline data.",
    },
    "TX_IMMUNOTHERAPY_OUTCOME": {
        "display_name": "Immunotherapy Specific Outcome",
        "datatype": "CATEGORICAL",
        "category": "Treatment History",
        "standard_unit": "text",
        "harmonised_definition": "Specific outcome recorded for immunotherapy treatment events.",
    },
}


def build_full_data_dictionary(processed_path: Path) -> Dict[str, Any]:
    """Scan processed datasets and assemble complete metadata dictionary."""
    dictionary: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Melanoma Immunotherapy Preprocessing Data Dictionary",
        "version": "1.0.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "project": "HSE ED / Melanoma Response Predictor",
        "description": "Comprehensive data dictionary for harmonised clinical, genomic, transcriptomic, and treatment features across melanoma cohorts.",
        "columns": COLUMN_METADATA_DEFINITIONS.copy(),
    }

    # Discover additional dynamic columns present in processed files
    discovered_cols = set()
    for csv_file in processed_path.rglob("*.csv"):
        if "attrition" in csv_file.name:
            continue
        try:
            df = pd.read_csv(csv_file, nrows=1)
            discovered_cols.update(df.columns)
        except Exception:
            continue

    # Auto-generate schema entries for any unmapped treatment/mutation columns
    for col in sorted(discovered_cols):
        if col not in dictionary["columns"]:
            if col.startswith("TX_TYPE_"):
                treatment_name = col.replace("TX_TYPE_", "").replace("_", " ").title()
                dictionary["columns"][col] = {
                    "display_name": f"Treatment Type: {treatment_name}",
                    "datatype": "BOOLEAN",
                    "category": "Treatment History",
                    "standard_unit": "binary flag",
                    "allowed_values": [0, 1],
                    "harmonised_definition": f"Binary indicator of patient history receiving {treatment_name}.",
                }
            elif col.startswith("TX_AGENT_"):
                agent_name = col.replace("TX_AGENT_", "").replace("_", " ").title()
                dictionary["columns"][col] = {
                    "display_name": f"Therapeutic Agent: {agent_name}",
                    "datatype": "BOOLEAN",
                    "category": "Treatment History",
                    "standard_unit": "binary flag",
                    "allowed_values": [0, 1],
                    "harmonised_definition": f"Binary indicator of treatment administration of agent {agent_name}.",
                }
            elif col.startswith("mut_"):
                gene_symbol = col.replace("mut_", "").split(".")[0]
                dictionary["columns"][col] = {
                    "display_name": f"{gene_symbol} Somatic Mutation Status",
                    "datatype": "BOOLEAN",
                    "category": "Driver Mutation",
                    "standard_unit": "binary flag",
                    "allowed_values": [0, 1],
                    "harmonised_definition": f"Binary indicator of non-silent somatic mutation in {gene_symbol}.",
                }
            else:
                dictionary["columns"][col] = {
                    "display_name": col.replace("_", " ").title(),
                    "datatype": "STRING",
                    "category": "Clinical Feature",
                    "standard_unit": "text / numeric",
                    "harmonised_definition": f"Clinical metadata attribute {col}.",
                }

    return dictionary


def main() -> None:
    """Execute dictionary generation and write output JSON."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    dictionary = build_full_data_dictionary(PROCESSED_DIR)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=2)

    print(f"Successfully generated data dictionary with {len(dictionary['columns'])} features:")
    print(f"  Output path: {OUTPUT_JSON.relative_to(_SUBPROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with (
            contextlib.redirect_stdout(stdout_tee),
            contextlib.redirect_stderr(stderr_tee),
        ):
            main()
