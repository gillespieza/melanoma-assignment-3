"""Single-Patient Immunotherapy Response Predictor.

Provides inference utilities and a command-line interface to predict binary
anti-PD-1 response (CR/PR vs. PD) for a single melanoma patient given either:
1. Raw gene expression values (single patient sample dict/Series or DataFrame)
2. Pre-calculated immune signature scores (IFN-gamma, TIS, CYT, CD8 T-cell, IMPRES, PD-L1)
3. A patient ID lookup from pre-processed cohort data dynamically resolved from config/datasets.yaml

Usage Examples:
    CLI usage:
        python q1-response-predictor/predictor.py --patient RIAZ_PT18 --cohort riaz_2017
        python q1-response-predictor/predictor.py --signatures "IFN_gamma=1.2,TIS=0.8,CYT=1.5,CD8_Tcell=0.9,IMPRES=8.0,PD_L1=2.1"

    Python API usage:
        from q1_response_predictor.predictor import SinglePatientPredictor
        predictor = SinglePatientPredictor()
        res = predictor.predict_from_signatures({
            'IFN_gamma': 1.2, 'TIS': 0.8, 'CYT': 1.5,
            'CD8_Tcell': 0.9, 'IMPRES': 8.0, 'PD_L1': 2.1
        })
        print(res['response_label'], res['predicted_probability'])
"""

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure subproject root is on sys.path
SUBPROJECT_ROOT = Path(__file__).resolve().parent
if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.append(str(SUBPROJECT_ROOT))

from src.utils.paths import PROCESSED_DIR
from src.signatures import extract_all_signatures, zscore_df
from src.config.constants import IMMUNE_SIGNATURES
from src.config.datasets import load_dataset_config_or_empty

MODELS_DIR = SUBPROJECT_ROOT / "models"
CONFIG_PATH = SUBPROJECT_ROOT / "config" / "datasets.yaml"


class SinglePatientPredictor:
    """Predicts immunotherapy response probability for a single melanoma patient using trained SVM."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        """Initialises predictor by loading the trained SVM model and fitted StandardScaler.

        Args:
            models_dir: Directory containing 'final_svm_model.pkl' and 'final_scaler.pkl'.
        """
        self.models_dir = Path(models_dir)
        self.svm_path = self.models_dir / "final_svm_model.pkl"
        self.scaler_path = self.models_dir / "final_scaler.pkl"

        if not self.svm_path.exists():
            raise FileNotFoundError(f"SVM model pickle not found at {self.svm_path}. Run run_pipeline.py first.")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler pickle not found at {self.scaler_path}. Run run_pipeline.py first.")

        with open(self.svm_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.required_signatures = list(IMMUNE_SIGNATURES)

    def predict_from_signatures(
        self,
        signature_dict: dict,
        threshold: float = 0.50,
        patient_id: str = "SINGLE_PATIENT",
    ) -> dict:
        """Predicts anti-PD-1 response given pre-calculated immune signature scores.

        Args:
            signature_dict: Dict mapping signature names to floats.
                Must include: 'IFN_gamma', 'TIS', 'CYT', 'CD8_Tcell', 'IMPRES', 'PD_L1'.
            threshold: Classification probability threshold (default 0.50).
            patient_id: Optional descriptive patient identifier.

        Returns:
            Dict containing predicted_probability, response_label, classification, and raw signatures.
        """
        missing = [s for s in self.required_signatures if s not in signature_dict]
        if missing:
            raise KeyError(f"Missing required immune signatures: {missing}. Must provide {self.required_signatures}.")

        # Order signatures according to trained model schema
        sig_df = pd.DataFrame([signature_dict])[self.required_signatures]

        # Apply model-level StandardScaler transform
        scaled_features = self.scaler.transform(sig_df)
        scaled_features_df = pd.DataFrame(scaled_features, columns=self.required_signatures)

        # Calibrated SVM probability prediction
        prob = float(self.model.predict_proba(scaled_features_df)[0, 1])
        is_responder = prob >= threshold
        label = "Responder (High Response Probability)" if is_responder else "Non-Responder (Low Response Probability)"

        # Calculate prediction confidence score relative to decision threshold
        confidence_score = round(abs(prob - threshold) * 2.0, 4)
        if confidence_score >= 0.40:
            confidence_tier = "High"
        elif confidence_score >= 0.20:
            confidence_tier = "Moderate"
        else:
            confidence_tier = "Low (Borderline)"

        return {
            "patient_id": patient_id,
            "predicted_probability": round(prob, 4),
            "confidence_score": confidence_score,
            "confidence_tier": confidence_tier,
            "classification": "Responder" if is_responder else "Non-Responder",
            "response_label": label,
            "threshold": threshold,
            "raw_signatures": signature_dict,
        }

    def predict_from_expression(
        self,
        expr_df: pd.DataFrame,
        reference_cohort_expr: pd.DataFrame = None,
        patient_id: str = "SINGLE_PATIENT",
    ) -> dict:
        """Predicts anti-PD-1 response from a single patient's raw/log gene expression.

        Args:
            expr_df: Single-row or multi-row DataFrame of gene expression with gene symbols as columns.
            reference_cohort_expr: Optional background reference cohort expression matrix
                to compute robust Z-score standardization.
            patient_id: Optional descriptive patient identifier.

        Returns:
            Dict containing prediction results.
        """
        # Extract 6 immune signatures
        sig_df = extract_all_signatures(expr_df)

        # Standardise signatures
        if reference_cohort_expr is not None:
            ref_sigs = extract_all_signatures(reference_cohort_expr)
            combined_sigs = pd.concat([sig_df, ref_sigs], axis=0)
            scaled_sigs = zscore_df(combined_sigs)
            patient_sig = scaled_sigs.iloc[0].to_dict()
        else:
            patient_sig = sig_df.iloc[0].to_dict()

        return self.predict_from_signatures(patient_sig, patient_id=patient_id)

    def _resolve_cohort_directory(self, cohort: str) -> Path:
        """Resolves cohort name or directory key against active datasets in datasets.yaml.

        Args:
            cohort: Cohort subfolder, cohort name, or study ID (e.g. 'liu_2019', 'Liu 2019', 'gide_2019').

        Returns:
            Path to the cohort's processed directory under PROCESSED_DIR.
        """
        configs = load_dataset_config_or_empty(CONFIG_PATH)
        for cfg in configs:
            if cohort.lower() in (
                cfg.processed_directory.lower(),
                cfg.cohort_name.lower(),
                cfg.study_id.lower(),
                cfg.cohort_name.lower().replace(" ", "_"),
            ):
                return PROCESSED_DIR / cfg.processed_directory

        # Direct path fallback if user provides custom/direct directory under PROCESSED_DIR
        direct_dir = PROCESSED_DIR / cohort
        if direct_dir.exists():
            return direct_dir

        available_dirs = [cfg.processed_directory for cfg in configs]
        available_names = [cfg.cohort_name for cfg in configs]
        raise KeyError(
            f"Cohort '{cohort}' not found in active dataset configuration ({CONFIG_PATH}). "
            f"Available cohort directories: {available_dirs}. Available cohort names: {available_names}."
        )

    def predict_from_saved_patient(
        self,
        patient_id: str,
        cohort: str = "liu_2019",
    ) -> dict:
        """Looks up a processed patient from a cohort dataset and runs SVM prediction.

        Args:
            patient_id: Patient ID (e.g. 'RIAZ_PT18', 'HUGO_PT38', 'LIU_PATIENT125').
            cohort: Cohort name or processed directory (e.g. 'liu_2019', 'riaz_2017', 'gide_2019', 'van_allen_2015', 'skcm_tcga_gdc').

        Returns:
            Dict containing prediction results along with clinical response label if available.
        """
        cohort_dir = self._resolve_cohort_directory(cohort)
        expr_file = cohort_dir / "expr_cleaned.csv"
        clin_file = cohort_dir / "clin_cleaned.csv"

        if not expr_file.exists():
            raise FileNotFoundError(f"Cleaned expression file not found at {expr_file}.")

        df_expr = pd.read_csv(expr_file, index_col=0)
        df_expr.index = df_expr.index.astype(str)

        # Match patient ID (case-insensitive substring match)
        matches = [idx for idx in df_expr.index if patient_id.upper() in idx.upper()]
        if not matches:
            raise KeyError(f"Patient ID '{patient_id}' not found in {cohort}. Available IDs: {list(df_expr.index[:5])}...")

        matched_id = matches[0]
        patient_expr = df_expr.loc[[matched_id]]

        # Compute signatures and Z-score relative to cohort background
        cohort_sigs = extract_all_signatures(df_expr)
        scaled_sigs = zscore_df(cohort_sigs)
        patient_sig = scaled_sigs.loc[matched_id].to_dict()

        result = self.predict_from_signatures(patient_sig, patient_id=matched_id)

        # Attach actual response if clinical metadata exists
        if clin_file.exists():
            df_clin = pd.read_csv(clin_file, index_col=0)
            df_clin.index = df_clin.index.astype(str)
            if matched_id in df_clin.index and "RESPONSE" in df_clin.columns:
                result["actual_recist_response"] = df_clin.loc[matched_id, "RESPONSE"]

        return result


def main():
    """Command-line interface for SinglePatientPredictor."""
    configs = load_dataset_config_or_empty(CONFIG_PATH)
    active_cohort_str = (
        ", ".join(f"'{cfg.processed_directory}'" for cfg in configs)
        if configs
        else "'liu_2019', 'riaz_2017', 'hugo_2016', 'gide_2019', 'van_allen_2015', 'skcm_tcga_gdc'"
    )

    parser = argparse.ArgumentParser(
        description="Single-Patient Anti-PD-1 Immunotherapy Response Predictor (SVM Calibrated Model)"
    )
    parser.add_argument("--patient", type=str, help="Patient ID to look up in processed dataset (e.g., RIAZ_PT18)")
    parser.add_argument(
        "--cohort",
        type=str,
        default="riaz_2017",
        help=f"Cohort name or processed folder for patient lookup ({active_cohort_str})",
    )
    parser.add_argument(
        "--signatures",
        type=str,
        help="Comma-separated immune signature key=value pairs (e.g. 'IFN_gamma=1.2,TIS=0.8,CYT=1.5,CD8_Tcell=0.9,IMPRES=8.0,PD_L1=2.1')",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Classification probability threshold (default: 0.50)",
    )

    args = parser.parse_args()

    predictor = SinglePatientPredictor()

    if args.patient:
        print(f"\n==================================================")
        print(f"Running SVM Inference for Patient: {args.patient} ({args.cohort})")
        print(f"==================================================")
        res = predictor.predict_from_saved_patient(args.patient, cohort=args.cohort)
        print(f"Patient ID:                {res['patient_id']}")
        print(f"Predicted Response Prob:   {res['predicted_probability']:.4f}")
        print(f"Confidence Score:          {res['confidence_score']:.4f} ({res['confidence_tier']} Confidence)")
        print(f"Classification:            {res['classification']}")
        if "actual_recist_response" in res:
            print(f"Actual RECIST Response:    {res['actual_recist_response']}")
        print(f"\nExtracted Immune Signatures (Z-scored):")
        for k, v in res["raw_signatures"].items():
            print(f"  - {k:<12}: {v:.4f}")

    elif args.signatures:
        print(f"\n==================================================")
        print(f"Running SVM Inference from Custom Signatures")
        print(f"==================================================")
        sig_dict = {}
        for pair in args.signatures.split(","):
            k, v = pair.split("=")
            sig_dict[k.strip()] = float(v.strip())

        res = predictor.predict_from_signatures(sig_dict, threshold=args.threshold)
        print(f"Predicted Response Prob:   {res['predicted_probability']:.4f}")
        print(f"Confidence Score:          {res['confidence_score']:.4f} ({res['confidence_tier']} Confidence)")
        print(f"Classification:            {res['classification']}")
        print(f"Response Label:            {res['response_label']}")

    else:
        # Default demo run on a sample patient if no args passed
        print("\n[Demo Run] No arguments passed. Testing prediction on sample patient 'RIAZ_PT18' (Riaz 2017)...")
        res = predictor.predict_from_saved_patient("RIAZ_PT18", cohort="riaz_2017")
        print(f"\n==================================================")
        print(f"SVM Prediction Demo Result:")
        print(f"==================================================")
        print(f"Patient ID:                {res['patient_id']}")
        print(f"Predicted Response Prob:   {res['predicted_probability']:.4f}")
        print(f"Confidence Score:          {res['confidence_score']:.4f} ({res['confidence_tier']} Confidence)")
        print(f"Classification:            {res['classification']}")
        if "actual_recist_response" in res:
            print(f"Actual RECIST Response:    {res['actual_recist_response']}")
        print(f"\nSignature Scores:")
        for k, v in res["raw_signatures"].items():
            print(f"  - {k:<12}: {v:.4f}")


if __name__ == "__main__":
    main()
