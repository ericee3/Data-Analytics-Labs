# LAB 2 

# Load required libraries
library("ggplot2")
library("readr")

# Import dataset
NY_House_Dataset <- read_csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Lab2/NY-House-Dataset.csv")
View(NY_House_Dataset)

# Create working copy of nydataset
nydataset <- NY_House_Dataset
View(nydataset)
attach(nydataset)

# Scatter plot of log10(PROPERTYSQFT) vs log10(PRICE)
ggplot(nydataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point()

# Filter extreme values from PRICE and remove outlier in PROPERTYSQFT
nydataset <- nydataset[nydataset$PRICE < 195000000,]
nydataset <- nydataset[nydataset$PROPERTYSQFT != 2184.207862,]

# Inspect specific PROPERTYSQFT values for a broker
nydataset$PROPERTYSQFT[nydataset$BROKERTITLE == "Brokered by Douglas Elliman - 575 Madison Ave"][85]

# Plot again after filtering
ggplot(nydataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point()

# View column names
names(nydataset)


# LAB Exercise 

# Explore relationships of predictors (BEDS, BATH) with PRICE
ggplot(nydataset, aes(x = BEDS, y = PRICE)) +
  geom_point()

ggplot(nydataset, aes(x = BATH, y = PRICE)) +
  geom_point()


# Additional filtering for outliers in PRICE, BEDS, and BATH
nydataset <- nydataset[nydataset$PRICE < 24500000,]
nydataset <- nydataset[nydataset$BEDS < 15,]
nydataset <- nydataset[nydataset$BATH < 15,]


# Remove rows with missing, zero, or invalid values before taking logs
nydataset <- nydataset[!is.na(nydataset$PRICE) & 
                     !is.na(nydataset$PROPERTYSQFT) & 
                     !is.na(nydataset$BEDS) & 
                     !is.na(nydataset$BATH) & 
                     nydataset$PROPERTYSQFT > 0 & 
                     nydataset$PRICE > 0, ]


# Check for missing or invalid log values
colSums(is.na(nydataset))                      # Count missing values
sum(is.infinite(log10(nydataset$PRICE)))       # Check for Inf in PRICE
sum(is.infinite(log10(nydataset$PROPERTYSQFT)))# Check for Inf in PROPERTYSQFT
sum(is.infinite((nydataset$BEDS)))             # Check for Inf in BEDS
sum(is.infinite((nydataset$BATH)))             # Check for Inf in BATH


# Fit multiple linear regression models
mod1 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT) + BEDS + BATH, data = nydataset)
mod2 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT) + BEDS, data = nydataset)
mod3 <- lm(log10(PRICE) ~ BEDS + BATH, data = nydataset)

# Display model summaries
summary(mod1)
summary(mod2)
summary(mod3)


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
plot_model(mod1, nydataset, "PROPERTYSQFT", "mod1")
plot_model(mod2, nydataset, "PROPERTYSQFT", "mod2")


# Repeat plotting for BEDS as the predictor

# Scatter plot of BEDS vs log10(PRICE) with regression line
ggplot(nydataset, aes(x = BEDS, y = log10(PRICE))) +
  geom_point(alpha = 0.5, color = "pink") +  
  stat_smooth(method = "lm", col = "grey", se = FALSE) +  
  labs(title = "Regression: BEDS vs log10(PRICE) - Model mod3",
       x = "Number of Beds", y = "log10(PRICE)") +
  theme_minimal()

# Residuals vs Fitted plot for mod3
residuals_data <- data.frame(Fitted = fitted(mod3), Residuals = resid(mod3))

ggplot(residuals_data, aes(x = Fitted, y = Residuals)) +
  geom_point(alpha = 0.5, color = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(title = "Residuals vs Fitted - Model mod3",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()

