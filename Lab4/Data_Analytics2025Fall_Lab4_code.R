# ================================================================
# Lab 4: Principal Component Analysis (PCA) and k-Nearest Neighbors (kNN)
# Dataset: Wine Data
# Author: Elizabeth Rice
# ================================================================

# ------------------------------
# Load Required Libraries
# ------------------------------
library(ggplot2)     # For data visualization
library(readr)       # For reading CSV files
library(class)       # For kNN classification
library(caret)       # For confusion matrix and metrics
library(dplyr)       # For data manipulation
library(ggfortify)   # For PCA visualization
library(e1071)       # For additional ML functions

# ------------------------------
# Load and Prepare the Dataset
# ------------------------------
wine <- read_csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Lab4/wine.data")

# Assign descriptive column names
colnames(wine) <- c(
  "class.wine", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
  "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
  "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"
)

# Inspect the dataset
View(wine)

# Separate class labels and numeric features
Y <- wine[, 1]
wine_numeric <- wine[, -1]

# ------------------------------
# Scale Data for PCA
# ------------------------------
# Standardize numeric columns to have mean = 0 and standard deviation = 1
wine_scaled <- scale(wine_numeric)

# Verify scaling
apply(wine_scaled, 2, mean)  # Should be approximately 0
apply(wine_scaled, 2, sd)    # Should be approximately 1

# ------------------------------
# Perform PCA
# ------------------------------
principal_components <- princomp(wine_scaled, cor = TRUE, score = TRUE)

# View PCA summary (variance explained by each component)
summary(principal_components)

# View loadings (variable contributions to each component)
principal_components$loadings

# Plot PCA scree plot (variance vs component)
plot(principal_components)

# Line plot version for smoother variance visualization
plot(principal_components, type = "l")

# ------------------------------
# Visualize PCA Results (PC1 vs PC2)
# ------------------------------
pca_data <- as.data.frame(principal_components$scores)
pca_data$class <- wine$class.wine  # Add class labels for coloring

# Scatter plot for the first two principal components
ggplot(pca_data, aes(x = Comp.1, y = Comp.2, color = as.factor(class))) +
  geom_point(size = 3, alpha = 0.8) +
  labs(title = "PCA: PC1 vs PC2",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

# ------------------------------
# Identify Least Contributing Variables
# ------------------------------
# Extract absolute loadings for PC1 and sort them
pc1_loadings <- abs(principal_components$loadings[, 1])
pc1_loadings_sorted <- sort(pc1_loadings, decreasing = TRUE)
print(pc1_loadings_sorted)

# Select lowest contributing variables (adjust number as needed)
least_contributing_vars <- names(tail(pc1_loadings_sorted, 2))
least_contributing_vars

# Create a reduced dataset excluding these variables
wine_reduced <- wine_numeric[, !(colnames(wine_numeric) %in% least_contributing_vars)]

# Scale the reduced dataset
wine_scaled_reduced <- scale(wine_reduced)

# ------------------------------
# Re-run PCA on Reduced Dataset
# ------------------------------
pca_reduced <- prcomp(wine_scaled_reduced, center = TRUE, scale. = FALSE)
summary(pca_reduced)

# ------------------------------
# kNN Classification - Model 1 (Original Dataset)
# ------------------------------
n <- 177  # total number of samples
s_data <- sample(n, n * 0.7)  # 70% for training

# Prepare scaled dataset with class labels
wine_scaled_df <- as.data.frame(wine_scaled)
wine_scaled_df$class <- as.factor(wine$class.wine)

# Split data into training and test sets
dataset.train <- wine_scaled_df[s_data, ]
dataset.test  <- wine_scaled_df[-s_data, ]

# Estimate k (number of neighbors)
k <- round(sqrt(n))

# Train and predict using kNN
knn.predicted <- knn(train = dataset.train, test = dataset.test,
                     cl = dataset.train$class, k = 13)

# Confusion matrix
contingency.table <- table(knn.predicted, dataset.test$class, dnn = list('predicted', 'actual'))
contingency.table

# Compute classification accuracy
accuracy <- sum(diag(contingency.table)) / length(dataset.test$class)
accuracy

# ------------------------------
# kNN Classification - Model 2 (Using 1st 3 Principal Components)
# ------------------------------
# Extract PCA scores and retain first 3 PCs
pca_knn_data <- as.data.frame(pca_reduced$x[, 1:3])
pca_knn_data$class <- as.factor(wine$class.wine)

# Split PCA-transformed data
dataset.train <- pca_knn_data[s_data, ]
dataset.test  <- pca_knn_data[-s_data, ]

# Train and predict again
knn.predicted <- knn(train = dataset.train, test = dataset.test,
                     cl = dataset.train$class, k = 13)

# Confusion matrix
contingency.table <- table(knn.predicted, dataset.test$class, dnn = list('predicted', 'actual'))
contingency.table

# Compute classification accuracy
accuracy <- sum(diag(contingency.table)) / length(dataset.test$class)
accuracy

# ------------------------------
# Precision, Recall, and F1-Score Metrics
# ------------------------------
precision <- c()
recall <- c()
f1_score <- c()
classes <- colnames(contingency.table)

# Calculate metrics for each class
for (class in classes) {
  TP <- contingency.table[class, class]
  FP <- sum(contingency.table[class, ]) - TP
  FN <- sum(contingency.table[, class]) - TP
  
  # Precision, Recall, F1-Score formulas
  prec <- TP / (TP + FP)
  rec  <- TP / (TP + FN)
  f1   <- 2 * (prec * rec) / (prec + rec)
  
  # Store results
  precision <- c(precision, prec)
  recall <- c(recall, rec)
  f1_score <- c(f1_score, f1)
}

# Combine metrics into a summary dataframe
metrics_df <- data.frame(
  Class = classes,
  Precision = precision,
  Recall = recall,
  F1_Score = f1_score
)

print(metrics_df)
