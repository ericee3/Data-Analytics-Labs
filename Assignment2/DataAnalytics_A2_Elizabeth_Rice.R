###############################################
# Assignment 2: Data Analytics (Fall 2025)
# Author: Elizabeth Rice
# Course: Data Analytics
# Description: Exploratory Data Analysis, Linear Models, and kNN Classification
###############################################

# --- Load required libraries ---
library("ggplot2")    # For plotting and visualization
library("readr")      # For reading CSV files
library(class)        # For kNN classification
library(caret)        # For confusion matrix and model evaluation

# --- Load dataset ---
epi_results_2024_pop_gdp <- read_csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Assignment2/epi_results_2024_pop_gdp_v2.csv")
View(epi_results_2024_pop_gdp)

# Rename dataset for simplicity
epi.data <- epi_results_2024_pop_gdp
attach(epi.data)
View(epi.data)

# --- Check for missing values in key variables ---
sum(is.na(epi.data$population))
sum(is.na(epi.data$gdp))
sum(is.na(epi.data$EPI.new))
sum(is.na(epi.data$ECO.new))
sum(is.na(epi.data$BDH.new))
sum(is.na(epi.data$SPI.new))
sum(is.na(epi.data$BER.new))
sum(is.na(epi.data$RLI.new))

# --- Remove rows containing NA values for main predictors ---
epi.data <- epi.data[!is.na(epi.data$population) & 
                       !is.na(epi.data$gdp) & 
                       !is.na(epi.data$SPI.new) & 
                       !is.na(epi.data$BER.new) & 
                       !is.na(epi.data$RLI.new), ]

###############################################
# PART 1: VARIABLE DISTRIBUTION ANALYSIS
###############################################

# --- Display unique regions to select from ---
unique_regions <- unique(epi.data$region)
print(unique_regions)

# --- Select two different regions (for example) ---
region1 <- unique_regions[4]  
region2 <- unique_regions[5]  

# --- Create subsets for each region ---
subset_region1 <- subset(epi.data, region == region1)
subset_region2 <- subset(epi.data, region == region2)

# --- Display first few rows to confirm subsets ---
print(head(subset_region1))
print(head(subset_region2))

# --- Check shape (number of rows and columns) ---
dim(subset_region1)
dim(subset_region2)

# --- Define the variable of interest ---
variable <- "EPI.new"

# --- Plot histograms with density lines for each region ---
p1 <- ggplot(subset_region1, aes(x = EPI.new)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "orange", alpha = 0.5) +
  geom_density(color = "purple", size = 1) +
  ggtitle(paste("Histogram with Density -", region1)) +
  theme_minimal()

p2 <- ggplot(subset_region2, aes(x = EPI.new)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "pink", alpha = 0.5) +
  geom_density(color = "purple", size = 1) +
  ggtitle(paste("Histogram with Density -", region2)) +
  theme_minimal()

# --- Display both histograms ---
print(p1)
print(p2)

# --- Create Q-Q plots for normality check in each region ---
qq_plot <- function(data, region_name, variable) {
  ggplot(data, aes(sample = get(variable))) +
    stat_qq(distribution = qnorm) +
    stat_qq_line(distribution = qnorm, color = "red") +
    ggtitle(paste("Q-Q Plot -", region_name)) +
    theme_minimal()
}

p1 <- qq_plot(subset_region1, region1, variable)
p2 <- qq_plot(subset_region2, region2, variable)

# --- Display Q-Q plots (region vs normal) ---
print(p1)
print(p2)

# --- Additional: Q-Q plot comparing two regions directly ---
qqplot(subset_region1$EPI.new, subset_region2$EPI.new,
       xlab = paste(region1, "Quantiles"),
       ylab = paste(region2, "Quantiles"),
       main = paste("QQ Plot:", region1, "vs", region2))
abline(0, 1, col = "red")

###############################################
# PART 2: LINEAR MODELS
###############################################

# --- Visualize relationships between predictors and response variables ---
ggplot(epi.data, aes(x = log10(gdp), y = ECO.new)) + 
  geom_point() +
  ggtitle("Scatterplot: ECO.new vs log10(GDP)")

ggplot(epi.data, aes(x = log10(population), y = log10(ECO.new))) +
  geom_point() +
  ggtitle("Scatterplot: log10(ECO.new) vs log10(Population)")

# --- Fit linear models for EPI.new with different predictors ---
lmod_11 <- lm(EPI.new ~ log10(population) + log10(gdp), data = epi.data)
lmod_22 <- lm(EPI.new ~ log10(gdp), data = epi.data)
lmod_33 <- lm(EPI.new ~ log10(population), data = epi.data)

# --- Display model summaries ---
summary(lmod_11)
summary(lmod_22)
summary(lmod_33)

# --- Fit linear models for ECO.new as response variable ---
lmod_1 <- lm(ECO.new ~ log10(population) + log10(gdp), data = epi.data)
lmod_2 <- lm(ECO.new ~ log10(gdp), data = epi.data)
lmod_3 <- lm(ECO.new ~ log10(population), data = epi.data)

# --- Display model summaries ---
summary(lmod_1)
summary(lmod_2)
summary(lmod_3)

# --- Fit model using a subset region (for regional comparison) ---
attach(subset_region1)
lmod_s1 <- lm(EPI.new ~ log10(population) + log10(gdp), data = subset_region1)
summary(lmod_s1)

# --- Create new columns for log-transformed predictors ---
epi.data$log10_population <- log10(epi.data$population)
epi.data$log10_gdp <- log10(epi.data$gdp)

# --- Plot regression lines for models (GDP and EPI/ECO) ---
p1 <- ggplot(epi.data, aes(x = log10_gdp, y = EPI.new)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_smooth(method = "lm", color = "orange", se = TRUE) +
  ggtitle("EPI.new vs log10(GDP)") +
  theme_minimal()

p2 <- ggplot(epi.data, aes(x = log10_gdp, y = ECO.new)) +
  geom_point(alpha = 0.6, color = "violet") +
  geom_smooth(method = "lm", color = "orange", se = TRUE) +
  ggtitle("ECO.new vs log10(GDP)") +
  theme_minimal()

# --- Residual plots for model diagnostics ---
residuals_epi <- data.frame(Fitted = fitted(lmod_22), Residuals = resid(lmod_22))
p3 <- ggplot(residuals_epi, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "orange") +
  ggtitle("Residuals Plot for EPI.new Model") +
  theme_minimal()

residuals_eco <- data.frame(Fitted = fitted(lmod_2), Residuals = resid(lmod_2))
p4 <- ggplot(residuals_eco, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.6, color = "violet") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "orange") +
  ggtitle("Residuals Plot for ECO.new Model") +
  theme_minimal()

residuals_sub <- data.frame(Fitted = fitted(lmod_s1), Residuals = resid(lmod_s1))
p5 <- ggplot(residuals_sub, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.6, color = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "orange") +
  ggtitle("Residuals Plot for Sub-Region Model") +
  theme_minimal()

# --- Regional model regression plot ---
p6 <- ggplot(subset_region1, aes(x = log10(gdp), y = EPI.new)) +
  geom_point(alpha = 0.6, color = "violet") +
  geom_smooth(method = "lm", color = "orange", se = TRUE) +
  ggtitle(paste("EPI.new vs log10(GDP) for", region1)) +
  theme_minimal()

# --- Display all plots ---
print(p1); print(p2); print(p3); print(p4); print(p5); print(p6)

###############################################
# PART 3: CLASSIFICATION (kNN)
###############################################

# --- Define the two regions for classification ---
selected_regions <- c("Sub-Saharan Africa", "Latin America & Caribbean")

# --- Subset with relevant predictor variables ---
subset_data <- epi.data[epi.data$region %in% selected_regions, 
                        c("region", "SPI.new", "BER.new", "RLI.new")]

# --- Convert region to factor for classification ---
subset_data$region <- as.factor(subset_data$region)

# --- Visualize data distribution by class ---
ggplot(subset_data, aes(x = BER.new, y = RLI.new, colour = region)) +
  geom_point() +
  ggtitle("Scatterplot of BER.new vs RLI.new by Region")

# --- (Optional) Normalize predictors for better kNN performance ---
# normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }
# subset_data[, 2:4] <- lapply(subset_data[, 2:4], normalize)

# --- Split data into training (80%) and testing (20%) sets ---
set.seed(123)
train_index <- sample(1:nrow(subset_data), 0.8 * nrow(subset_data))
train_data <- subset_data[train_index, ]
test_data <- subset_data[-train_index, ]

# --- Extract features (X) and labels (Y) ---
train_x <- train_data[, 2:4]
train_y <- train_data$region
test_x <- test_data[, 2:4]
test_y <- test_data$region

# --- Estimate k using simple rule of thumb ---
n <- 76
k_v = round(sqrt(n))
k_v

# --- Train and evaluate kNN model for several k values ---
k_values <- c(3, 9, 12)
for (k in k_values) {
  knn_pred <- knn(train_x, test_x, train_y, k = k)
  
  # Compute confusion matrix
  cm <- confusionMatrix(knn_pred, test_y)
  
  # Calculate accuracy
  accuracy <- sum(diag(cm$table)) / sum(cm$table)
  
  print(paste("k =", k, "Accuracy =", round(accuracy, 3)))
  print(cm)
}

# --- Final model with k = 9 ---
knn.predicted <- knn(train_x, test_x, train_y, k = 9)

# --- Create confusion matrix (contingency table) ---
contingency.table <- table(knn.predicted, test_y, dnn = list('Predicted','Actual'))
contingency.table

# --- Calculate overall classification accuracy ---
sum(diag(contingency.table)) / length(test_y)

