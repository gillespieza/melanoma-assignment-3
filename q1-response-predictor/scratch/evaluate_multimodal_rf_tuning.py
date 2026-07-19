import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.signatures import extract_all_signatures


DATA_DIR = BASE_DIR / "data"


def zscore_df(df):
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def load_processed_mutations(mutations_file: Path, target_genes: list, sample_ids: list) -> pd.DataFrame:
    if not mutations_file.exists():
        return pd.DataFrame(0, index=sample_ids, columns=target_genes)

    df_mut = pd.read_csv(mutations_file, index_col="SAMPLE_ID")
    for gene in target_genes:
        if gene not in df_mut.columns:
            df_mut[gene] = 0

    df_mut = (df_mut[target_genes] > 0).astype(int)
    return df_mut.reindex(sample_ids, fill_value=0)


def load_trial_cohort(cohort_dir, cohort_name):
    clin_path = DATA_DIR / f"processed/{cohort_dir}/clin_cleaned.csv"
    expr_path = DATA_DIR / f"processed/{cohort_dir}/expr_cleaned.csv"
    clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")
    expr = pd.read_csv(expr_path, index_col=0)

    for up, low in [
        ("PATIENT_ID", "patient_id"),
        ("RESPONSE_BINARY", "response"),
        ("SEX", "sex"),
        ("AGE", "age"),
        ("OS_STATUS", "os_status"),
        ("OS_MONTHS", "os_months"),
    ]:
        if up in clin.columns and low not in clin.columns:
            clin[low] = clin[up]

    response_map = {
        "Complete Response": 1,
        "Partial Response": 1,
        "Progressive Disease": 0,
        "Stable Disease": np.nan,
        "Mixed Response": np.nan,
    }
    clin["temp_resp"] = clin["RESPONSE"].map(response_map)
    clin = clin.dropna(subset=["temp_resp"]).drop(columns=["temp_resp"])
    expr = expr.loc[clin.index]
    clin["Cohort"] = cohort_name

    if "SEX" in clin.columns:
        clin["SEX"] = clin["SEX"].map({"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"})

    neo_cols = [
        "SNV_NEOANTIGEN",
        "INDEL_NEOANTIGEN",
        "FUSION_NEOANTIGEN",
        "SPLICE_NEOANTIGEN",
        "VIRUS_NEOANTIGEN",
        "ERV_NEOANTIGEN",
    ]
    if "TOTAL_NEOANTIGEN" not in clin.columns:
        available_neo = [col for col in neo_cols if col in clin.columns]
        if available_neo:
            clin["TOTAL_NEOANTIGEN"] = clin[available_neo].fillna(0).sum(axis=1)
            all_nan = clin[available_neo].isna().all(axis=1)
            clin.loc[all_nan, "TOTAL_NEOANTIGEN"] = np.nan
        else:
            clin["TOTAL_NEOANTIGEN"] = np.nan

    if "CNA_PROP" not in clin.columns:
        clin["CNA_PROP"] = np.nan

    return clin, expr


def prepare_multimodal_features():
    df_liu_clin, df_liu_expr = load_trial_cohort("liu_2019", "Liu 2019")
    df_hugo_clin, df_hugo_expr = load_trial_cohort("hugo_2016", "Hugo 2016")
    df_riaz_clin, df_riaz_expr = load_trial_cohort("riaz_2017", "Riaz 2017")

    pathway_genes = [
        "B2M",
        "TAP1",
        "TAP2",
        "JAK1",
        "JAK2",
        "STAT1",
        "PTEN",
        "CDKN2A",
        "PIK3CA",
        "BRAF",
        "NRAS",
        "NF1",
    ]
    cohort_parts = [
        ("liu_2019", df_liu_clin),
        ("hugo_2016", df_hugo_clin),
        ("riaz_2017", df_riaz_clin),
    ]
    for cohort_dir, clin in cohort_parts:
        mutations = load_processed_mutations(
            DATA_DIR / f"processed/{cohort_dir}/mutations_cleaned.csv",
            pathway_genes,
            clin.index.tolist(),
        )
        for col in pathway_genes:
            clin[f"mut_{col}"] = mutations[col]
        clin["mut_Antigen_Presentation"] = (
            clin[["mut_B2M", "mut_TAP1", "mut_TAP2"]].sum(axis=1) > 0
        ).astype(int)
        clin["mut_IFN_gamma_Signaling"] = (
            clin[["mut_JAK1", "mut_JAK2", "mut_STAT1"]].sum(axis=1) > 0
        ).astype(int)
        clin["mut_Survival_Pathways"] = (
            clin[["mut_PTEN", "mut_CDKN2A", "mut_PIK3CA"]].sum(axis=1) > 0
        ).astype(int)

    clin_cols = [
        "Cohort",
        "response",
        "TMB_NONSYNONYMOUS",
        "SEX",
        "TOTAL_NEOANTIGEN",
        "CNA_PROP",
        "mut_BRAF",
        "mut_NRAS",
        "mut_NF1",
        "mut_Antigen_Presentation",
        "mut_IFN_gamma_Signaling",
        "mut_Survival_Pathways",
    ]
    clin_merged = pd.concat(
        [df_liu_clin[clin_cols], df_hugo_clin[clin_cols], df_riaz_clin[clin_cols]]
    )

    sigs_merged = pd.concat(
        [
            zscore_df(extract_all_signatures(df_liu_expr)),
            zscore_df(extract_all_signatures(df_hugo_expr)),
            zscore_df(extract_all_signatures(df_riaz_expr)),
        ]
    )
    sigs_merged = sigs_merged.loc[clin_merged.index]

    features = pd.concat(
        [
            sigs_merged,
            clin_merged[
                [
                    "mut_BRAF",
                    "mut_NRAS",
                    "mut_NF1",
                    "mut_Antigen_Presentation",
                    "mut_IFN_gamma_Signaling",
                    "mut_Survival_Pathways",
                    "TMB_NONSYNONYMOUS",
                    "TOTAL_NEOANTIGEN",
                    "SEX",
                ]
            ],
        ],
        axis=1,
    )
    features["TMB_NONSYNONYMOUS"] = features["TMB_NONSYNONYMOUS"].fillna(
        features["TMB_NONSYNONYMOUS"].median()
    )
    features["TOTAL_NEOANTIGEN"] = features["TOTAL_NEOANTIGEN"].fillna(
        features["TOTAL_NEOANTIGEN"].median()
    )
    features["Sex_Male"] = features["SEX"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)
    features = features.drop(columns=["SEX"])

    clean_idx = clin_merged["response"].dropna().index
    return features.loc[clean_idx], clin_merged.loc[clean_idx, "response"].astype(int).values


def evaluate_rf_grid(X, y, param_grid, label):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    result = cross_validate(
        model,
        X,
        y,
        cv=outer_cv,
        scoring="roc_auc",
        return_estimator=True,
        n_jobs=None,
    )
    scores = result["test_score"]
    best_params = [est.best_params_ for est in result["estimator"]]
    return {
        "search": label,
        "mean_auc": scores.mean(),
        "std_auc": scores.std(),
        "fold_scores": scores,
        "best_params": best_params,
    }


def main():
    features, y = prepare_multimodal_features()

    sig_features = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]
    feature_sets = {
        "Signatures Only": sig_features,
        "Sigs + Drivers + Sex": sig_features + ["mut_BRAF", "mut_NRAS", "mut_NF1", "Sex_Male"],
        "Full Extended": features.columns.tolist(),
    }

    current_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_leaf": [1, 2, 4],
    }
    expanded_grid = [
        {
            "n_estimators": [200],
            "max_depth": [2, 3, 5, None],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", None],
            "class_weight": [None, "balanced"],
        },
        {
            "n_estimators": [500],
            "max_depth": [3, 5],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt"],
            "class_weight": [None, "balanced"],
        },
    ]

    rows = []
    for feature_set_name, cols in feature_sets.items():
        print(f"\nEvaluating {feature_set_name} ({len(cols)} features)")
        X = features[cols].values
        for label, grid in [("current_grid", current_grid), ("expanded_grid", expanded_grid)]:
            outcome = evaluate_rf_grid(X, y, grid, label)
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "search": label,
                    "auc_mean": outcome["mean_auc"],
                    "auc_std": outcome["std_auc"],
                    "fold_scores": ", ".join(f"{score:.3f}" for score in outcome["fold_scores"]),
                    "best_params_by_fold": outcome["best_params"],
                }
            )
            print(
                f"  {label}: {outcome['mean_auc']:.3f} "
                f"(+/-{outcome['std_auc']:.3f}); folds [{rows[-1]['fold_scores']}]"
            )

    results = pd.DataFrame(rows)
    print("\nRandom Forest multimodal 5-fold CV comparison:")
    print(
        results.assign(
            auc=lambda df: df.apply(lambda row: f"{row.auc_mean:.3f} (+/-{row.auc_std:.3f})", axis=1)
        )[["feature_set", "search", "auc", "fold_scores"]].to_string(index=False)
    )

    print("\nBest parameters by outer fold:")
    for _, row in results.iterrows():
        print(f"\n{row['feature_set']} / {row['search']}")
        for fold_idx, params in enumerate(row["best_params_by_fold"], start=1):
            print(f"  fold {fold_idx}: {params}")


if __name__ == "__main__":
    main()
