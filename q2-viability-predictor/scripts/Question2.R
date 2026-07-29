# ============================================================
# QUESTION 2: Cell Viability Predictor for BRAF Inhibitors
# Melanoma cell lines - Dabrafenib and PLX-4720 (a vemurafenib-class drug)
# ============================================================
# ---- STEP 1: Load required packages ----
library(glmnet)  # provides LASSO regression, used to build our predictive model
library(readxl) # allows R to read .xlsx Excel files (GDSC2 data comes as Excel)


# ---- STEP 2: Set the location of our data files ----
data_path <- "~/Downloads/depmap_data/"

# ---- STEP 3: Load cell line metadata and filter to melanoma ----
model_info <- read.csv(paste0(data_path, "Model.csv"), stringsAsFactors = FALSE)
melanoma_models <- model_info[model_info$OncotreeLineage == "Skin" |
                                grepl("Melanoma", model_info$OncotreePrimaryDisease, ignore.case = TRUE), ]
cat("Number of melanoma cell lines found:", nrow(melanoma_models), "\n")
melanoma_ids <- melanoma_models[, c("ModelID", "SangerModelID")]

# ---- STEP 4: Load GDSC2 drug data, filter to our 2 drugs ----
gdsc <- read_excel(paste0(data_path, "GDSC2_fitted_dose_response.csv"))
target_drugs <- c("Dabrafenib", "PLX-4720")
gdsc_braf <- gdsc[gdsc$DRUG_NAME %in% target_drugs, ]
cat("Number of drug-response rows for our 2 drugs (all cancers):", nrow(gdsc_braf), "\n")

# ---- STEP 5: Keep only melanoma cell lines ----
gdsc_melanoma <- merge(gdsc_braf, melanoma_ids, by.x = "SANGER_MODEL_ID", by.y = "SangerModelID")
cat("Number of melanoma-specific drug-response rows:", nrow(gdsc_melanoma), "\n")

# ---- STEP 6: Load gene expression data ----
expression <- read.csv(paste0(data_path, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"),
                       stringsAsFactors = FALSE, check.names = FALSE)

# FIX: column 1 is just a row index, column 4 is the real ModelID
colnames(expression)[1] <- "row_index"
colnames(expression)[4] <- "ModelID"
cat("Columns named 'ModelID' (should be 1):", sum(colnames(expression) == "ModelID"), "\n")

# ---- STEP 7: Build predictor (LASSO Model) for each drug ----
for (drug in target_drugs) {
  cat("\n\n===== Building model for:", drug, "=====\n")
  
  # Filter drug data, remove duplicate cell lines
  drug_data <- gdsc_melanoma[gdsc_melanoma$DRUG_NAME == drug, ]
  drug_data <- drug_data[!duplicated(drug_data$ModelID), ]
  
  # Combine drug response with gene expression
  merged_data <- merge(drug_data[, c("ModelID", "AUC")], expression, by = "ModelID")
  cat("Cell lines with BOTH drug response AND expression data:", nrow(merged_data), "\n")
  
  if (nrow(merged_data) < 10) {
    cat("Not enough data to build a model for", drug, "- skipping.\n")
    next
  }
  
  # Set up predictors (x) and outcome (y)
  y <- merged_data$AUC
  #Exclude ALL non-gene columns, not just ModelID and AUC. The DepMap
  # expression file carries several metadata columns (row_index, SequencingID,
  # ModelConditionID, and two TRUE/FALSE flags) which were previously being
  # passed to the model as if they were genes. The text columns were silently
  # coerced to NA - the source of the "NAs introduced by coercion" warnings.
  non_gene_cols <- c("ModelID", "AUC", "row_index", "SequencingID",
                     "ModelConditionID", "IsDefaultEntryForMC",
                     "IsDefaultEntryForModel")
  x <- as.matrix(merged_data[, !(colnames(merged_data) %in% non_gene_cols)])
  # Split into training and test sets
  set.seed(123)
  train_index <- sample(1:nrow(x), size = floor(0.8 * nrow(x)))
  x_train <- x[train_index, ]
  y_train <- y[train_index]
  x_test  <- x[-train_index, ]
  y_test  <- y[-train_index]
  
  # Train LASSO model
  cv_model <- cv.glmnet(x_train, y_train, alpha = 1)
  
  # Predict on test set
  predictions <- predict(cv_model, newx = x_test, s = "lambda.min")
  correlation <- cor(predictions, y_test)
  cat("Correlation between predicted and actual viability:", round(correlation, 3), "\n")
  
  # Plot actual vs predicted
  plot(y_test, predictions,
       xlab = "Actual viability (AUC)", ylab = "Predicted viability (AUC)",
       main = paste("Predicted vs Actual -", drug))
  abline(0, 1, col = "red")
  
  # Save important genes identified by the model
  coefficients <- coef(cv_model, s = "lambda.min")
  important_genes <- data.frame(
    gene = rownames(coefficients)[which(coefficients[,1] != 0)],
    coefficient = coefficients[which(coefficients[,1] != 0), 1]
  )
  write.csv(important_genes, paste0("important_genes_", gsub("-", "_", drug), ".csv"), row.names = FALSE)
  cat("Saved important genes to important_genes_", gsub("-", "_", drug), ".csv\n", sep = "")
}
# ============================================================
# STABILITY CHECK
# ------------------------------------------------------------
# With ~29 training samples and ~19,000 candidate genes, LASSO gene
# selection can be unstable - a different random train/test split may
# select a different set of genes. To test this, the Dabrafenib model
# is refit across five random seeds and the frequency with which each
# gene is selected is recorded. Genes selected consistently across
# seeds are more likely to reflect real signal than genes appearing
# only once.
# ============================================================

drug_data <- gdsc_melanoma[gdsc_melanoma$DRUG_NAME == "Dabrafenib", ]
drug_data <- drug_data[!duplicated(drug_data$ModelID), ]
merged_data <- merge(drug_data[, c("ModelID", "AUC")], expression, by = "ModelID")

y <- merged_data$AUC
x <- as.matrix(merged_data[, !(colnames(merged_data) %in% non_gene_cols)])

gene_counts <- c()
correlations <- c()

for (s in c(123, 456, 789, 101, 202)) {
  set.seed(s)
  train_index <- sample(1:nrow(x), size = floor(0.8 * nrow(x)))
  
  cv_model <- cv.glmnet(x[train_index, ], y[train_index], alpha = 1)
  preds <- predict(cv_model, newx = x[-train_index, ], s = "lambda.min")
  correlations <- c(correlations, cor(preds, y[-train_index]))
  
  coefs <- coef(cv_model, s = "lambda.min")
  selected <- rownames(coefs)[which(coefs[, 1] != 0)]
  selected <- selected[selected != "(Intercept)"]
  gene_counts <- c(gene_counts, selected)
}

cat("\n--- Correlation across 5 seeds (Dabrafenib) ---\n")
print(round(correlations, 3))
cat("Mean:", round(mean(correlations), 3),
    "| Range:", round(min(correlations), 3), "to", round(max(correlations), 3), "\n")

cat("\n--- Genes selected in 3 or more of 5 seeds ---\n")
freq <- sort(table(gene_counts), decreasing = TRUE)
print(freq[freq >= 3])
# ============================================================
# STEP 8: Apply LASSO models to TCGA-SKCM patients
# ============================================================

# ---- 8a: paths to my files (these live in two different folders) ----
tcga_path <- "~/Downloads/skcm_tcga_pan_can_atlas_2018 2/"   # gene expression data
clin_path <- "~/Downloads/"   # clin_merged.csv is straight in Downloads
# ---- 8b: load the gene coefficients I saved earlier from the LASSO models ----
dabrafenib_coefs <- read.csv("important_genes_Dabrafenib.csv", stringsAsFactors = FALSE)
plx4720_coefs   <- read.csv("important_genes_PLX_4720.csv", stringsAsFactors = FALSE)

# my DepMap gene names look like "TRPM3 (80036)" but TCGA just uses "TRPM3" -
# stripping off the " (entrez_id)" part so the names actually match
dabrafenib_coefs$gene <- gsub(" \\(.*\\)$", "", dabrafenib_coefs$gene)
plx4720_coefs$gene    <- gsub(" \\(.*\\)$", "", plx4720_coefs$gene)


# ---- 8c: load patient gene expression data ----
patient_expr_raw <- read.delim(paste0(tcga_path, "data_mrna_seq_v2_rsem.txt"),
                               stringsAsFactors = FALSE, check.names = FALSE)

# genes are rows and patients are columns in this file - opposite of what I need,
# so flipping it around to match the shape my model expects (patients as rows, genes as columns)
gene_names <- patient_expr_raw$Hugo_Symbol
expr_matrix <- as.matrix(patient_expr_raw[, -(1:2)])  # dropping Hugo_Symbol and Entrez_Gene_Id, not genes
rownames(expr_matrix) <- gene_names

expr_t <- as.data.frame(t(expr_matrix))
expr_t$SAMPLE_ID <- rownames(expr_t)

# this data is raw RSEM values, but my model was trained on log2(TPM+1) from DepMap,
# so I need to log-transform it here or the numbers won't be on the same scale
gene_cols <- setdiff(colnames(expr_t), "SAMPLE_ID")
expr_t[gene_cols] <- log2(expr_t[gene_cols] + 1)

# ---- 8d: only keep patients who did NOT have immunotherapy ----
# TCGA-SKCM is the cohort without immunotherapy response data (unlike Liu/Hugo/Riaz)
clin <- read.csv(paste0(clin_path, "clin_merged.csv"), stringsAsFactors = FALSE)
tcga_patients <- clin[clin$COHORT == "TCGA-SKCM", ]

expr_tcga <- expr_t[expr_t$SAMPLE_ID %in% tcga_patients$SAMPLE_ID, ]
cat("TCGA-SKCM patients with expression data:", nrow(expr_tcga), "\n")

# ---- 8e: function that takes my saved coefficients and scores each patient ----
apply_lasso_score <- function(expr_data, coef_df) {
  intercept <- coef_df$coefficient[coef_df$gene == "(Intercept)"]
  gene_coefs <- coef_df[coef_df$gene != "(Intercept)", ]
  
  # checking how many of my selected genes actually exist in the TCGA file -
  # if this number is low or zero, the gene names probably don't match between datasets
  matched_genes <- intersect(gene_coefs$gene, colnames(expr_data))
  cat("Genes matched:", length(matched_genes), "out of", nrow(gene_coefs), "\n")
  
  gene_coefs <- gene_coefs[gene_coefs$gene %in% matched_genes, ]
  weighted_sum <- as.matrix(expr_data[, matched_genes]) %*% gene_coefs$coefficient
  return(as.vector(intercept + weighted_sum))
} 
expr_tcga$Dabrafenib_Predicted_AUC <- apply_lasso_score(expr_tcga, dabrafenib_coefs)

# checking exactly which gene didn't match
setdiff(dabrafenib_coefs$gene, colnames(expr_tcga))

expr_tcga$Dabrafenib_Predicted_AUC <- apply_lasso_score(expr_tcga, dabrafenib_coefs)
expr_tcga$PLX4720_Predicted_AUC   <- apply_lasso_score(expr_tcga, plx4720_coefs)

# CEP104 didn't match the TCGA file because TCGA uses its older name, KIAA0562 -
# same gene, just a different symbol, so renaming it to recover the match
dabrafenib_coefs$gene[dabrafenib_coefs$gene == "CEP104"] <- "KIAA0562"

# ---- 8f: save the output Amanda asked for ----
output <- merge(expr_tcga[, c("SAMPLE_ID", "Dabrafenib_Predicted_AUC", "PLX4720_Predicted_AUC")],
                clin[, c("SAMPLE_ID", "PATIENT_ID")], by = "SAMPLE_ID")
output <- output[, c("SAMPLE_ID", "PATIENT_ID", "Dabrafenib_Predicted_AUC", "PLX4720_Predicted_AUC")]

write.csv(output, "q2_patient_drug_predictions.csv", row.names = FALSE)
cat("Saved patient predictions to q2_patient_drug_predictions.csv\n")

# ============================================================
# STEP 9: Correlate my predicted viability scores against Amanda's predicted response
# ============================================================

# ---- 9a: load Amanda's predicted immunotherapy response scores ----
amanda_path <- "~/Downloads/"   # change this if her file is saved somewhere else
amanda_scores <- read.csv(paste0(amanda_path, "patient_predicted_response_scores.csv"),
                          stringsAsFactors = FALSE)

# ---- 9b: keep only TCGA-SKCM patients, since that's the non-immunotherapy cohort
# (my 70 patients from Step 8 are already this subset) ----
amanda_tcga <- amanda_scores[amanda_scores$COHORT == "TCGA-SKCM", ]

amanda_tcga <- amanda_tcga[!duplicated(amanda_tcga$PATIENT_ID), ]   # removing duplicate patient rows

# ---- 9c: merge my predictions with her predictions by PATIENT_ID ----
# my "output" data frame from Step 8f already has PATIENT_ID and my two viability scores
merged_results <- merge(output, amanda_tcga[, c("PATIENT_ID", "PREDICTED_RESPONSE_SCORE")],
                        by = "PATIENT_ID")

cat("Patients with both my viability scores AND Amanda's response score:", nrow(merged_results), "\n")

# ---- 9d: run the actual correlation ----
cor_dabrafenib <- cor(merged_results$Dabrafenib_Predicted_AUC,
                      merged_results$PREDICTED_RESPONSE_SCORE)

cor_plx4720 <- cor(merged_results$PLX4720_Predicted_AUC,
                   merged_results$PREDICTED_RESPONSE_SCORE)

cat("\n--- Correlation: predicted BRAF-inhibitor viability vs predicted immunotherapy response ---\n")
cat("Dabrafenib vs immunotherapy response:", round(cor_dabrafenib, 3), "\n")
cat("PLX-4720 vs immunotherapy response:", round(cor_plx4720, 3), "\n")

# ---- 9e: save the merged results ----
write.csv(merged_results, "q2_correlation_results.csv", row.names = FALSE)
cat("Saved merged correlation results to q2_correlation_results.csv\n")
 