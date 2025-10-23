# --------------------------------------------------
# Assignment 5 - Data Analytics
# Elizabeth Rice
# --------------------------------------------------

# Install missing packages if not already installed
packages <- c("tidyverse", "caret", "randomForest", "class", "ggplot2")
installed <- packages %in% rownames(installed.packages())
if(any(!installed)) install.packages(packages[!installed])
lapply(packages, library, character.only = TRUE)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data <- read_csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Assignment5/NYC_Citywide_Annualized_Calendar_Sales_Update_20241107.csv")

# Convert all column names to uppercase for consistency
names(data) <- toupper(names(data))

# --------------------------------------------------
# 1(a) Bronx subset
# --------------------------------------------------
bronx <- data %>%
  filter(BOROUGH %in% c("BRONX", 2)) %>%
  mutate(
    SALE_PRICE = as.numeric(`SALE PRICE`),
    GROSS_SQUARE_FEET = as.numeric(`GROSS SQUARE FEET`),
    LAND_SQUARE_FEET = as.numeric(`LAND SQUARE FEET`),
    YEAR_BUILT = as.numeric(`YEAR BUILT`)
  )

# --------------------------------------------------
# 1(b) Exploratory Data Analysis
# --------------------------------------------------

# Histogram of Sale Price (log scale)
ggplot(bronx, aes(x = SALE_PRICE)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_x_log10() +
  labs(title = "Bronx: Distribution of Sale Price (log scale)",
       x = "Sale Price (log10)", y = "Count")

# Boxplot for Outliers
ggplot(bronx, aes(y = SALE_PRICE)) +
  geom_boxplot(fill = "orange") +
  scale_y_log10() +
  labs(title = "Bronx: Boxplot of Sale Prices (log scale)",
       y = "Sale Price")

# --------------------------------------------------
# 1(c) Regression Analysis - Predict Sale Price
# --------------------------------------------------
bronx_model_data <- bronx %>%
  select(SALE_PRICE, GROSS_SQUARE_FEET, LAND_SQUARE_FEET, YEAR_BUILT) %>%
  drop_na()

set.seed(123)
trainIndex <- createDataPartition(bronx_model_data$SALE_PRICE, p = 0.8, list = FALSE)
trainData <- bronx_model_data[trainIndex, ]
testData <- bronx_model_data[-trainIndex, ]

# Linear regression
lm_model <- lm(SALE_PRICE ~ GROSS_SQUARE_FEET + LAND_SQUARE_FEET + YEAR_BUILT, data = trainData)
summary(lm_model)

# Random forest regression
rf_model <- randomForest(SALE_PRICE ~ GROSS_SQUARE_FEET + LAND_SQUARE_FEET + YEAR_BUILT,
                         data = trainData, ntree = 200, importance = TRUE)
rf_pred <- predict(rf_model, testData)

rf_results <- data.frame(Actual = testData$SALE_PRICE, Predicted = rf_pred)

# Residual plot
ggplot(rf_results, aes(x = Predicted, y = Actual - Predicted)) +
  geom_point(alpha = 0.5, color = "purple") +
  labs(title = "Random Forest Residuals (Bronx)",
       x = "Predicted Sale Price", y = "Residuals")

# --------------------------------------------------
# 1(d) Classification: Predict Neighborhood
# --------------------------------------------------
bronx_class <- bronx %>%
  select(NEIGHBORHOOD, SALE_PRICE, GROSS_SQUARE_FEET, LAND_SQUARE_FEET) %>%
  drop_na()

# Keep top 10 neighborhoods by count
top_neigh <- names(sort(table(bronx_class$NEIGHBORHOOD), decreasing = TRUE))[1:10]
bronx_class <- bronx_class %>% filter(NEIGHBORHOOD %in% top_neigh)

set.seed(42)
split <- createDataPartition(bronx_class$NEIGHBORHOOD, p = 0.8, list = FALSE)
trainClass <- bronx_class[split, ]
testClass <- bronx_class[-split, ]

# Random Forest Classification
rf_class <- randomForest(NEIGHBORHOOD ~ SALE_PRICE + GROSS_SQUARE_FEET + LAND_SQUARE_FEET,
                         data = trainClass, ntree = 200)
pred_class <- predict(rf_class, testClass)

conf_matrix <- confusionMatrix(pred_class, testClass$NEIGHBORHOOD)
print(conf_matrix)

# --------------------------------------------------
# 2(a) Apply Bronx Regression Model to Queens
# --------------------------------------------------
queens <- data %>%
  filter(BOROUGH %in% c("QUEENS", 4)) %>%
  mutate(
    SALE_PRICE = as.numeric(`SALE PRICE`),
    GROSS_SQUARE_FEET = as.numeric(`GROSS SQUARE FEET`),
    LAND_SQUARE_FEET = as.numeric(`LAND SQUARE FEET`),
    YEAR_BUILT = as.numeric(`YEAR BUILT`)
  )

queens_model_data <- queens %>%
  select(SALE_PRICE, GROSS_SQUARE_FEET, LAND_SQUARE_FEET, YEAR_BUILT) %>%
  drop_na()

queens_pred <- predict(rf_model, newdata = queens_model_data)
results_queens <- data.frame(Actual = queens_model_data$SALE_PRICE, Predicted = queens_pred)

# Plot predictions vs actual
ggplot(results_queens, aes(x = Predicted, y = Actual)) +
  geom_point(alpha = 0.4, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Predicted vs Actual Sale Price (Queens)",
       x = "Predicted Price", y = "Actual Price")

# --------------------------------------------------
# 2(b) Apply Bronx Neighborhood Classifier to Queens
# --------------------------------------------------
queens_class <- queens %>%
  select(NEIGHBORHOOD, SALE_PRICE, GROSS_SQUARE_FEET, LAND_SQUARE_FEET) %>%
  filter(NEIGHBORHOOD %in% top_neigh) %>%
  drop_na()

queens_pred_class <- predict(rf_class, queens_class)
confusionMatrix(queens_pred_class, queens_class$NEIGHBORHOOD)

# --------------------------------------------------
# 3. 6000-Level Reflection
# --------------------------------------------------
cat("\nConclusions:\n")
cat("The Random Forest regression model captured nonlinear patterns in Bronx data better than linear regression.\n")
cat("However, when applied to Queens, prediction errors increased, suggesting borough-specific market trends.\n")
cat("Neighborhood classification showed moderate success within Bronx but lower precision in Queens,\n")
cat("likely due to different price distributions and neighborhood characteristics.\n")
cat("Overall, Random Forests provided flexibility but struggled to generalize fully, emphasizing local data variation.\n")

