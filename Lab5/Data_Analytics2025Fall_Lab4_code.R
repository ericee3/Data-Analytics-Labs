###############################################
#   Lab 5: Classification with SVM and kNN
###############################################

# Load required libraries
library("ggplot2")     # For data visualization
library("readr")       # For reading CSV files
library(class)         # For k-Nearest Neighbors (kNN)
library(caret)         # For confusion matrices and evaluation
library(dplyr)         # For data manipulation
library(ggfortify)     # For PCA plotting (optional visualization)
library(e1071)         # For Support Vector Machines (SVM)

# -----------------------------------------------------------
# Load the Wine dataset and define column names
# -----------------------------------------------------------
wine <- read_csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Lab4/wine.data")

colnames(wine) <- c(
  "class.wine", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash",
  "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols",
  "Proanthocyanins", "Color intensity", "Hue",
  "OD280/OD315 of diluted wines", "Proline"
)

# Separate class label and numeric features
wine_class <- wine[, 1]       # Class label column
wine_numeric <- wine[, -1]    # Feature columns only

# -----------------------------------------------------------
# Data standardization (scaling)
# -----------------------------------------------------------
# Scaling ensures that all features contribute equally to distance-based models
wine_scaled <- scale(wine_numeric)

# Verify scaling results: means should be ~0, standard deviations ~1
apply(wine_scaled, 2, mean)
apply(wine_scaled, 2, sd)

# Convert to dataframe and reattach class labels
wine_scaled <- as.data.frame(wine_scaled)
wine_scaled$wine.class <- factor(wine_class$class.wine)

# -----------------------------------------------------------
# Split dataset into training (70%) and testing (30%)
# -----------------------------------------------------------
n <- nrow(wine)
train.indexes <- sample(n, 0.7 * n)

X_train <- wine_scaled[train.indexes, ]
X_test  <- wine_scaled[-train.indexes, ]

Y_train <- wine_class[train.indexes, ]
Y_test  <- wine_class[-train.indexes, ]

# -----------------------------------------------------------
# Exploratory plots
# -----------------------------------------------------------
# Feature distribution
boxplot(X_train[, 1:13], main = "Wine Feature Distributions (Training Set)")

# Class label distribution
plot(Y_train, main = "Class Distribution in Training Data")

# Visualize two features by class
ggplot(X_train, aes(x = Alcohol, y = Hue, colour = wine.class)) +
  geom_point() +
  ggtitle("Alcohol vs. Hue by Wine Class")

# -----------------------------------------------------------
# Support Vector Machine (SVM) Models
# -----------------------------------------------------------

##################################
# MODEL 1: Linear Kernel SVM
##################################
svm.mod0 <- svm(
  wine.class ~ Alcohol + `Malic acid` + Ash + `Alcalinity of ash` +
    `Total phenols` + Flavanoids + `Color intensity`,
  data = X_train,
  kernel = "linear"
)

# Evaluate on training data
train.pred <- predict(svm.mod0, X_train)
cm <- as.matrix(table(Actual = X_train$wine.class, Predicted = train.pred))
cm

##################################
# MODEL 2: Radial Kernel SVM
##################################
svm.mod1 <- svm(
  wine.class ~ Alcohol + `Malic acid` + Ash + `Alcalinity of ash` +
    `Total phenols` + Flavanoids + `Color intensity`,
  data = X_train,
  kernel = "radial"
)

train.pred <- predict(svm.mod1, X_train)
cm <- as.matrix(table(Actual = X_train$wine.class, Predicted = train.pred))
cm

# Compute evaluation metrics
n <- sum(cm)
diag_vals <- diag(cm)
rowsums <- apply(cm, 1, sum)
colsums <- apply(cm, 2, sum)
recall <- diag_vals / rowsums
precision <- diag_vals / colsums
f1 <- 2 * precision * recall / (precision + recall)
data.frame(precision, recall, f1)

##################################
# Tuned Radial SVM (Hyperparameter Search)
##################################
# Tune SVM hyperparameters using cross-validation
tuned.svm <- tune.svm(
  wine.class ~ Alcohol + `Malic acid` + Ash + `Alcalinity of ash` +
    `Total phenols` + Flavanoids + `Color intensity`,
  data = X_train,
  kernel = "radial",
  gamma = seq(1 / 2^nrow(X_train), 1, 0.01),
  cost = 2^seq(-6, 4, 2)
)

# Build the final tuned model with chosen parameters
svm.mod2 <- svm(
  wine.class ~ Alcohol + `Malic acid` + Ash + `Alcalinity of ash` +
    `Total phenols` + Flavanoids + `Color intensity`,
  data = X_train,
  kernel = "radial",
  gamma = 0.01,
  cost = 1
)

train.pred <- predict(svm.mod2, X_train)
cm <- as.matrix(table(Actual = X_train$wine.class, Predicted = train.pred))
cm

# Calculate precision, recall, and F1 again
n <- sum(cm)
diag_vals <- diag(cm)
rowsums <- apply(cm, 1, sum)
colsums <- apply(cm, 2, sum)
recall <- diag_vals / rowsums
precision <- diag_vals / colsums
f1 <- 2 * precision * recall / (precision + recall)
data.frame(precision, recall, f1)

# -----------------------------------------------------------
# MODEL 3: k-Nearest Neighbors (kNN)
# -----------------------------------------------------------

# Estimate k using heuristic (square root of number of samples)
k <- round(sqrt(n))

# Train and predict with kNN (using training set for simplicity)
knn.predicted <- knn(
  train = X_train,
  test = X_train,
  cl = X_train$wine.class,
  k = 13
)

# Build confusion matrix
cm <- table(Actual = X_train$wine.class, Predicted = knn.predicted)
cm

# Compute evaluation metrics for kNN
n <- sum(cm)
diag_vals <- diag(cm)
rowsums <- apply(cm, 1, sum)
colsums <- apply(cm, 2, sum)
recall <- diag_vals / rowsums
precision <- diag_vals / colsums
f1 <- 2 * precision * recall / (precision + recall)
data.frame(precision, recall, f1)

###############################################
# End of script
# Summary:
# - Built and evaluated three models: Linear SVM, Radial SVM, and kNN
# - Computed precision, recall, and F1-scores for each model
###############################################

