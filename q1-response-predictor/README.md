# Melanoma Immunotherapy Response Predictor (`q1-response-predictor`)

A machine learning pipeline designed to predict binary anti-PD-1 immunotherapy response (**Complete Response / Partial Response** vs. **Progressive Disease**) in cutaneous melanoma patients.

The model is specifically engineered and evaluated under **Leave-One-Cohort-Out (LOCO) Cross-Validation** to ensure generalisability to new patient populations and clinical sites across heterogeneous sequencing platforms.


## 🚀 Single-Patient Predictor (`predictor.py`)

The subproject root contains `predictor.py`, a dedicated inference tool that predicts a single patient's immunotherapy response probability using our trained, calibrated **Support Vector Machine (SVM)** model (`final_svm_model.pkl`) and StandardScaler (`final_scaler.pkl`).

### Quickstart CLI Examples

#### 1. Predict for a specific patient in a dataset:
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
Classification:            Responder
Actual RECIST Response:    Partial Response

Extracted Immune Signatures (Z-scored):
  - IFN_gamma   : 0.6516
  - TIS         : 0.9526
  - CYT         : 0.9510
  - CD8_Tcell   : 1.0461
  - IMPRES      : 2.4441
  - PD_L1       : 0.7299
```

#### 2. Predict from custom pre-calculated immune signature scores:
```bash
python q1-response-predictor/predictor.py --signatures "IFN_gamma=1.2,TIS=0.8,CYT=1.5,CD8_Tcell=0.9,IMPRES=8.0,PD_L1=2.1"
```

#### 3. Adjust probability classification threshold (default 0.50):
```bash
python q1-response-predictor/predictor.py --signatures "IFN_gamma=0.5,TIS=0.4,CYT=0.3,CD8_Tcell=0.2,IMPRES=5.0,PD_L1=0.8" --threshold 0.45
```


### Python API Usage

You can also import `SinglePatientPredictor` directly in Python:

```python
from q1_response_predictor.predictor import SinglePatientPredictor

predictor = SinglePatientPredictor()

# 1. Predict from a dictionary of immune signature scores
res = predictor.predict_from_signatures({
    'IFN_gamma': 1.2,
    'TIS': 0.8,
    'CYT': 1.5,
    'CD8_Tcell': 0.9,
    'IMPRES': 8.0,
    'PD_L1': 2.1
})

print(f"Patient ID:                {res['patient_id']}")
print(f"Predicted Response Prob:   {res['predicted_probability']:.4f}")
print(f"Classification:            {res['classification']}")

# 2. Predict directly from a patient ID in a cohort
res_cohort = predictor.predict_from_saved_patient("HUGO_PT38", cohort="hugo_2016")

# 3. Predict from a single patient's raw gene expression DataFrame
res_expr = predictor.predict_from_expression(single_patient_gene_expression_df)
```


## 🛠️ Full Pipeline Execution Order

If you wish to re-run the full data ingestion, preprocessing, training, and report generation pipeline locally:

1. **Download Raw Data**:
   ```bash
   python scripts/download_data.py
   ```
2. **Clean & Harmonise Datasets**:
   ```bash
   python scripts/clean_data.py
   ```
3. **Merge Multi-Cohort Datasets**:
   ```bash
   python scripts/merge_data.py
   ```
4. **Execute Pipeline & Train Models**:
   ```bash
   python q1-response-predictor/scripts/run_pipeline.py
   ```


## 📂 Subproject Directory Hierarchy

```
q1-response-predictor/
├── predictor.py               # Single-patient SVM prediction CLI & API
├── config/                    # Pipeline configuration files
├── logs/                      # Console execution log outputs
├── models/                    # Pickled trained models & fitted scalers
│   ├── final_svm_model.pkl
│   └── final_scaler.pkl
├── plots/                     # High-resolution (300 DPI) publication figures
├── reports/                   # 4-Pillar Markdown analysis reports
└── src/                       # Core python library modules
    ├── signatures.py          # Immune signature computation & Z-scoring
    ├── models.py              # LOCO cross-validation & model tuning
    └── utils/                 # Path, logging, and formatting helpers
```

For full analytical reports and methodology details, see [`reports/README.md`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/README.md) or [`reports/executive_summary.md`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/executive_summary.md).
