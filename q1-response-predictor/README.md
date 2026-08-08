# Melanoma Immunotherapy Response Predictor (`q1-response-predictor`)

A machine learning pipeline designed to predict binary anti-PD-1 immunotherapy response (**Complete Response / Partial Response** vs. **Progressive Disease**) in cutaneous melanoma patients.

The model is trained and evaluated under **Leave-One-Cohort-Out (LOCO) Cross-Validation** across three independent clinical trial cohorts (**Liu 2019**, **Hugo 2016**, **Riaz 2017**) to ensure generalisability to new patient populations and clinical sites across heterogeneous sequencing platforms.

> [!NOTE]
> **Cohort Roles & TCGA Reference Dataset**:
> Model training and LOCO cross-validation are performed exclusively on the 3 anti-PD-1 clinical trial cohorts (**Liu 2019**, **Hugo 2016**, **Riaz 2017**; total $N=248$ evaluable samples). The TCGA-SKCM cohort ($N=471$) is used as an untreated reference dataset for baseline clinical/genomic characterisation (Pillar 1) and Cox proportional hazards survival screening (Pillar 4), but is **not** used to train response prediction models as TCGA-SKCM lacks RECIST immunotherapy response outcome labels.


## 🚀 Single-Patient Predictor (`predictor.py`)

The subproject root contains [`predictor.py`](predictor.py), a dedicated inference tool that predicts a single patient's immunotherapy response probability using our trained, calibrated **Support Vector Machine (SVM)** model (`final_svm_model.pkl`) and StandardScaler (`final_scaler.pkl`).

The predictor requires **12 multimodal features**:
- **8 Transcriptomic / Microenvironment Signatures**: `IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1` (`CD274`), `Macrophage_STV_Score`, `M1_M2_Ratio`
- **3 Driver Mutation Flags**: `mut_BRAF`, `mut_NRAS`, `mut_NF1`
- **1 Genomic Burden Metric**: `TMB_NONSYNONYMOUS`

### Quickstart CLI Examples

#### 1. Predict for a specific patient in a preprocessed cohort:
```bash
python q1-response-predictor/predictor.py --patient RIAZ_PT18 --cohort riaz_2017
```

**Output**:
```text
==================================================
Running SVM Inference for Patient: RIAZ_PT18 (riaz_2017)
==================================================
Patient ID:                RIAZ_PT18_ON
Predicted Response Prob:   0.7559
Confidence Score:          0.5118 (High Confidence)
Classification:            Responder
Actual RECIST Response:    Partial Response

Extracted Immune Signatures (Z-scored):
  - IFN_gamma   : 0.6516
  - TIS         : 0.9526
  - CYT         : 0.9510
  - CD8_Tcell   : 1.0461
  - IMPRES      : 2.4441
  - PD_L1       : 0.7299
  - Macrophage_STV_Score : -0.1240
  - M1_M2_Ratio : 0.8410
  - mut_BRAF    : 1.0000
  - mut_NRAS    : 0.0000
  - mut_NF1     : 0.0000
  - TMB_NONSYNONYMOUS    : 14.2000
```

#### 2. Predict from custom pre-calculated multimodal feature scores:
```bash
python q1-response-predictor/predictor.py --signatures "IFN_gamma=1.2,TIS=0.8,CYT=1.5,CD8_Tcell=0.9,IMPRES=8.0,PD_L1=2.1,Macrophage_STV_Score=0.4,M1_M2_Ratio=1.1,mut_BRAF=1,mut_NRAS=0,mut_NF1=0,TMB_NONSYNONYMOUS=15.0"
```

#### 3. Adjust probability classification threshold (default 0.50):
```bash
python q1-response-predictor/predictor.py --patient HUGO_PT38 --cohort hugo_2016 --threshold 0.45
```


### Python API Usage

You can also import `SinglePatientPredictor` directly in Python:

```python
from q1_response_predictor.predictor import SinglePatientPredictor

predictor = SinglePatientPredictor()

# 1. Predict from a dictionary of 12 multimodal features
res = predictor.predict_from_signatures({
    'IFN_gamma': 1.2,
    'TIS': 0.8,
    'CYT': 1.5,
    'CD8_Tcell': 0.9,
    'IMPRES': 8.0,
    'PD_L1': 2.1,
    'Macrophage_STV_Score': 0.4,
    'M1_M2_Ratio': 1.1,
    'mut_BRAF': 1,
    'mut_NRAS': 0,
    'mut_NF1': 0,
    'TMB_NONSYNONYMOUS': 15.0
})

print(f"Patient ID:                {res['patient_id']}")
print(f"Predicted Response Prob:   {res['predicted_probability']:.4f}")
print(f"Confidence Tier:           {res['confidence_tier']}")
print(f"Classification:            {res['classification']}")

# 2. Predict directly from a patient ID in a cohort dataset
res_cohort = predictor.predict_from_saved_patient("HUGO_PT38", cohort="hugo_2016")

# 3. Predict from a single patient's raw gene expression DataFrame
res_expr = predictor.predict_from_expression(single_patient_gene_expression_df)
```


## 🛠️ Full Pipeline Execution

You can run the pipeline either end-to-end using the master `run_pipeline.py` CLI or step-by-step using individual module scripts.

### 1. Master Pipeline Orchestration (`run_pipeline.py`)

The master pipeline script (`q1-response-predictor/scripts/run_pipeline.py`) supports CLI options to orchestrate downloading, cleaning, merging, and evaluating models:

* **Run Full End-to-End Pipeline (Download → Clean → Merge → LOCO CV & Evaluation)**:
  ```bash
  python q1-response-predictor/scripts/run_pipeline.py --all
  ```

* **Run Specific Preprocessing Steps Before Model Evaluation**:
  ```bash
  # Download, clean, and merge data, then train/evaluate models
  python q1-response-predictor/scripts/run_pipeline.py --download --clean --merge
  ```

* **Run Preprocessing Only (Skip Model Training & Evaluation)**:
  ```bash
  python q1-response-predictor/scripts/run_pipeline.py --download --clean --merge --skip-eval
  ```

* **Run LOCO Cross-Validation & Model Evaluation Only (Assumes Data Prepared)**:
  ```bash
  python q1-response-predictor/scripts/run_pipeline.py
  ```

---

### 2. Step-by-Step Individual Module Execution

Alternatively, individual pillar scripts can be run sequentially:

1. **Download Raw Data**:
   ```bash
   python q1-response-predictor/scripts/pillar-1-cohort-preprocessing/download_data.py
   ```
2. **Clean & Harmonise Datasets**:
   ```bash
   python q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py
   ```
3. **Merge Multi-Cohort Datasets**:
   ```bash
   python q1-response-predictor/scripts/pillar-1-cohort-preprocessing/merge_datasets.py
   ```
4. **Execute LOCO Cross-Validation & Model Training**:
   ```bash
   python q1-response-predictor/scripts/run_pipeline.py
   ```


## 📂 Subproject Directory Hierarchy

```
q1-response-predictor/
├── predictor.py               # Single-patient SVM prediction CLI & API
├── config/                    # Pipeline dataset & constant configuration files
├── logs/                      # Subproject execution log outputs
├── models/                    # Pickled trained models & fitted scalers
│   ├── final_svm_model.pkl    # Final calibrated SVM model
│   ├── final_rf_model.pkl     # Final Random Forest classifier
│   ├── final_lr_model.pkl     # Final Logistic Regression classifier
│   ├── final_xgb_model.pkl    # Final XGBoost classifier
│   ├── final_elasticnet_model.pkl
│   └── final_scaler.pkl       # Fitted StandardScaler
├── plots/                     # High-resolution (300 DPI) publication figures
│   ├── biomarkers/            # Univariate biomarker plots & threshold scans
│   ├── feature_selection/     # Feature selection & TCGA survival heatmaps
│   └── models/                # ROC-AUC, PR, and LOCO CV heatmaps
├── reports/                   # 4-Pillar Markdown analysis reports
│   ├── pillar-1-cohorts-and-preprocessing/
│   ├── pillar-2-clinical-subtyping/
│   ├── pillar-3-transcriptomic-signatures/
│   └── pillar-4-out-of-cohort-benchmarks/
└── scripts/                   # Modular pipeline execution scripts
    ├── pillar-1-cohort-preprocessing/    # Preprocessing, data ingestion & batch correction
    ├── pillar-2-clinical-subtyping/       # Clinical subtyping & survival curves
    ├── pillar-3-transcriptomic-signatures/# Signature computation & ML training
    ├── pillar-4-out-of-cohort-benchmarks/ # LOCO CV & benchmark heatmaps
    ├── exploratory_plots/                # Exploratory visualization generators
    ├── reports/                          # Executive summary generator
    └── run_pipeline.py                   # Master pipeline orchestrator
```

For full analytical reports and methodology details, see [`reports/README.md`](reports/README.md) or [`reports/executive_summary.md`](reports/executive_summary.md).
