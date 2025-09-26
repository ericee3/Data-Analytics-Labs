# LAB 2 

# Load required libraries
library("ggplot2")
library("readr")

# Import dataset
NY_House_Dataset <- read_csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Lab2/NY-House-Dataset.csv")
View(NY_House_Dataset)

# Create working copy of dataset
dataset <- NY_House_Dataset
View(dataset)
attach(dataset)

# Scatter plot of log10(PROPERTYSQFT) vs log10(PRICE)
ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point()

# Filter extreme values from PRICE and remove outlier in PROPERTYSQFT
dataset <- dataset[dataset$PRICE < 195000000,]
dataset <- dataset[dataset$PROPERTYSQFT != 2184.207862,]

# Inspect specific PROPERTYSQFT values for a broker
dataset$PROPERTYSQFT[dataset$BROKERTITLE == "Brokered by Douglas Elliman - 575 Madison Ave"][85]

# Plot again after filtering
ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point()

# View column names
names(dataset)


# LAB Exercise 

# Explore relationships of predictors (BEDS, BATH) with PRICE
ggplot(dataset, aes(x = BEDS, y = PRICE)) +
  geom_point()

ggplot(dataset, aes(x = BATH, y = PRICE)) +
  geom_point()


# Additional filtering for outliers in PRICE, BEDS, and BATH
dataset <- dataset[dataset$PRICE < 24500000,]
dataset <- dataset[dataset$BEDS < 15,]
dataset <- dataset[dataset$BATH < 15,]


# Remove rows with missing, zero, or invalid values before taking logs
dataset <- dataset[!is.na(dataset$PRICE) & 
                     !is.na(dataset$PROPERTYSQFT) & 
                     !is.na(dataset$BEDS) & 
                     !is.na(dataset$BATH) & 
                     dataset$PROPERTYSQFT > 0 & 
                     dataset$PRICE > 0, ]


# Check for missing or invalid log values
colSums(is.na(dataset))                      # Count missing values
sum(is.infinite(log10(dataset$PRICE)))       # Check for Inf in PRICE
sum(is.infinite(log10(dataset$PROPERTYSQFT)))# Check for Inf in PROPERTYSQFT
sum(is.infinite((dataset$BEDS)))             # Check for Inf in BEDS
sum(is.infinite((dataset$BATH)))             # Check for Inf in BATH


# Fit multiple linear regression models
lmod_11 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT) + BEDS + BATH, data = dataset)
lmod_22 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT) + BEDS, data = dataset)
lmod_33 <- lm(log10(PRICE) ~ BEDS + BATH, data = dataset)

# Display model summaries
summary(lmod_11)
summary(lmod_22)
summary(lmod_33)


# Define function to plot regression line and residuals for a chosen variable
plot_model <- function(model, data, var_name, model_name) {
  
  # Scatter plot of predictor vs log10(PRICE) with regression line
  ggplot(data, aes_string(x = paste0("log10(", var_name, ")"), y = "log10(PRICE)")) +
    geom_point(alpha = 0.5, color = "green") +  
    stat_smooth(method = "lm", col = "orange", se = FALSE) +  
    labs(title = paste("Scatter plot of log10(", var_name, ") vs log10(PRICE) -", model_name),
         x = paste("log10", var_name), y = "log10(PRICE)") +
    theme_minimal() -> plot1
  
  # Residuals vs Fitted plot
  residuals_data <- data.frame(Fitted = fitted(model), Residuals = resid(model))
  
  ggplot(residuals_data, aes(x = Fitted, y = Residuals)) +
    geom_point(alpha = 0.5, color = "purple") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "orange") +
    labs(title = paste("Residuals vs Fitted -", model_name),
         x = "Fitted Values", y = "Residuals") +
    theme_minimal() -> plot2
  
  # Show both plots
  print(plot1)
  print(plot2)
}

# Generate plots for models using PROPERTYSQFT as the main predictor
plot_model(lmod_11, dataset, "PROPERTYSQFT", "lmod_11")
plot_model(lmod_22, dataset, "PROPERTYSQFT", "lmod_22")


# Repeat plotting for BEDS as the predictor

# Scatter plot of BEDS vs log10(PRICE) with regression line
ggplot(dataset, aes(x = BEDS, y = log10(PRICE))) +
  geom_point(alpha = 0.5, color = "pink") +  
  stat_smooth(method = "lm", col = "grey", se = FALSE) +  
  labs(title = "Regression: BEDS vs log10(PRICE) - Model lmod_33",
       x = "Number of Beds", y = "log10(PRICE)") +
  theme_minimal()

# Residuals vs Fitted plot for lmod_33
residuals_data <- data.frame(Fitted = fitted(lmod_33), Residuals = resid(lmod_33))

ggplot(residuals_data, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(title = "Residuals vs Fitted - Model lmod_33",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()

