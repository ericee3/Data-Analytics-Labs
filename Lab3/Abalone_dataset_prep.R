####################################
##### Abalone Data Preparation #####
####################################

# Load required libraries
library(ggplot2)   # For visualization
library(readr)     # For reading CSV files
library(class)     # For kNN classification
library(caret)     # For confusion matrix calculation
library(dplyr)     # For data manipulation

# Read the dataset
abalone.data <- read.csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Lab3/abalone_dataset.csv")
View(abalone.data)

# Create a working copy of the dataset
dataset <- abalone.data

# Add a new column 'age.group' based on number of rings
# young: 0-8 rings, adult: 9-11 rings, old: 12-35 rings
dataset$age.group <- cut(dataset$rings, breaks=c(0,8,11,35), labels=c("young", "adult", "old"))

# Alternative method to assign age groups (overwrites previous column)
dataset$age.group[dataset$rings <= 8] <- "young"
dataset$age.group[dataset$rings > 8 & dataset$rings <= 11] <- "adult"
dataset$age.group[dataset$rings > 11 & dataset$rings <= 35] <- "old"

View(dataset)

####################################
# Exercise 1: kNN Classification
####################################

# Split data into 70% training and 30% testing
n <- nrow(dataset)
s_data <- sample(n, n*0.7)
dataset.train <- dataset[s_data, ]
dataset.test  <- dataset[-s_data, ]

# Estimate a simple k value (square root of n)
k <- round(sqrt(n))

########################
# Train & Evaluate kNN #
########################

# --- Model 1: using length, diameter, and height ---
knn.predicted <- knn(
  train = dataset.train[, 2:4],
  test  = dataset.test[, 2:4],
  cl    = dataset.train$age.group,
  k = 65
)

# Create confusion matrix
contingency.table <- table(knn.predicted, dataset.test$age.group, dnn = list("predicted","actual"))
contingency.table

# Calculate classification accuracy
sum(diag(contingency.table)) / length(dataset.test$age.group)

# --- Model 2: using weight features (whole, shucked, viscera, shell) ---
knn.predicted <- knn(
  train = dataset.train[, 5:8],
  test  = dataset.test[, 5:8],
  cl    = dataset.train$age.group,
  k = 65
)

# Confusion matrix and accuracy
contingency.table <- table(knn.predicted, dataset.test$age.group, dnn = list("predicted","actual"))
contingency.table
sum(diag(contingency.table)) / length(dataset.test$age.group)

##########################
# Find optimal k value
##########################

# Test multiple k values and store accuracies
k.list <- c(45,55,60,65,75,85,105,155,205)
accuracy.list <- c()

for (k in k.list) {
  knn.predicted <- knn(
    train = dataset.train[,5:8],
    test  = dataset.test[,5:8],
    cl    = dataset.train$age.group,
    k = k
  )
  
  contingency.table <- table(knn.predicted, dataset.test$age.group, dnn=list('predicted','actual'))
  accuracy <- sum(diag(contingency.table)) / length(dataset.test$age.group)
  accuracy.list <- c(accuracy.list, accuracy)
}

# Plot accuracy vs k
plot(k.list, accuracy.list, type = "b", xlab = "k", ylab = "Accuracy", main = "kNN Accuracy vs k")

####################################
# Exercise 2: K-Means Clustering
####################################

# Initial scatter plot of dataset colored by age group
ggplot(dataset, aes(x = whole_weight, y = viscera_wieght, colour = age.group)) +
  geom_point()

# Run K-Means clustering with k = 3
k <- 3
data.km <- kmeans(dataset[,5:8], centers = k)

# Store cluster assignments as factor
assigned.clusters <- as.factor(data.km$cluster)

# Plot again for visualization
ggplot(dataset, aes(x = whole_weight, y = viscera_wieght, colour = age.group)) +
  geom_point()

# Print total within-cluster sum of squares
data.km$tot.withinss
data.km$cluster

# Test multiple k values and plot WCSS
k.list <- c(2,3,4,5,6)
wcss.list <- c()

for (k in k.list) {
  data.km <- kmeans(dataset[,5:8], centers = k)
  wcss <- data.km$tot.withinss
  wcss.list <- c(wcss.list, wcss)
  
  # Optional: visualize clusters for each k
  assigned.clusters <- as.factor(data.km$cluster)
  ggplot(dataset, aes(x = whole_weight, y = viscera_wieght, colour = age.group)) +
    geom_point()
}

# Plot WCSS vs number of clusters
plot(k.list, wcss.list, type = "b", xlab = "Number of Clusters (k)", ylab = "WCSS", main = "K-Means WCSS vs k")

