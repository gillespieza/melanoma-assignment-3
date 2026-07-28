# ============================================================
# QUESTION 2: Cell Viability Predictor for BRAF Inhibitors
# Melanoma cell lines - Dabrafenib and PLX-4720 (a vemurafenib-class drug)
# ============================================================
# ---- STEP 1: Load required packages ----
library(glmnet)  # provides LASSO regression, used to build our predictive model
library(readxl) # allows R to read .xlsx Excel files 


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
  x <- as.matrix(merged_data[, !(colnames(merged_data) %in% c("ModelID", "AUC"))])
  
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
