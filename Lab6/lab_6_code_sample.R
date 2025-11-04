################################
# Evaluating Regression Models #
################################

library(readr)
library(ggplot2)
library(e1071)
library(caret)
library(cv)

## read data
NY_House_Dataset <- read_csv("/Users/elizabethrice/Desktop/Data Analytics Labs/Lab6/NY-House-Dataset.csv")

dataset <- NY_House_Dataset

## column names
names(dataset)

## Plot dataset
ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point()

ggplot(dataset, aes(x = PROPERTYSQFT, y = PRICE)) +
  geom_point()

## Linear Regression Model
lin.mod0 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset)

summary(lin.mod0)

ggplot(dataset, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  stat_smooth(method = "lm", col="blue")



## Cleaning
dataset.sub0 <- dataset[-which(dataset$PROPERTYSQFT==2184.207862),]

lin.mod1 <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0)
summary(lin.mod1)
ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  stat_smooth(method = "lm", col="blue")


## SVM - linear
svm.mod0 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="linear")

summary(svm.mod0)

svm.pred <- predict(svm.mod0, dataset.sub0)

ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(x=log10(PROPERTYSQFT), y=svm.pred), col="green")

## SVM - radial
svm.mod1 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="radial")

summary(svm.mod1)

svm.pred <- predict(svm.mod1, dataset.sub0)

ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(x=log10(PROPERTYSQFT), y=svm.pred), col="red")

## SVM - radial - optimized
tuned.svm <- tune.svm(log10(PRICE) ~ log10(PROPERTYSQFT), data=dataset.sub0, kernel="radial", gamma = 10^seq(-3,2,1), cost = 10^seq(-3,2,1), tune.control=tune.control(cross = 5))
tuned.svm

svm.mod1 <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="radial", gamma=0.1, cost=100)

summary(svm.mod1)

svm.pred <- predict(svm.mod1, dataset.sub0)

ggplot(dataset.sub0, aes(x = log10(PROPERTYSQFT), y = log10(PRICE))) +
  geom_point() +
  geom_line(aes(x=log10(PROPERTYSQFT), y=svm.pred), col="red")


## Estimating Model Errors

## split train/test
train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))

train <- dataset.sub0[train.indexes,]
test <- dataset.sub0[-train.indexes,]

## LM
lin.mod <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), train)

summary(cv(lin.mod))

lm.pred <- predict(lin.mod, test)

## err = predicted - real
err <- lm.pred-log10(test$PRICE)

## MAE
abs.err <- abs(err)
mean.abs.err <- mean(abs.err)

## MSE
sq.err <- err^2
mean.sq.err <- mean(sq.err)

## RMSE
sq.err <- err^2
mean.sq.err <- mean(sq.err)
root.mean.sq.err <- sqrt(mean.sq.err)

### Cross Validation ###

### Monte Carlo CV

## Lin Model
k = 10000
mae <- c()
mse <- c()
rmse <- c()

for (i in 1:k) {
  train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))
  
  train <- dataset.sub0[train.indexes,]
  test <- dataset.sub0[-train.indexes,]
  
  lin.mod <- lm(log10(PRICE) ~ log10(PROPERTYSQFT), train)
  
  lm.pred <- predict(lin.mod, test)  
  
  err <- lm.pred-log10(test$PRICE)
  
  abs.err <- abs(err)
  mean.abs.err <- mean(abs.err)
  
  sq.err <- err^2
  mean.sq.err <- mean(sq.err)
  
  root.mean.sq.err <- sqrt(mean.sq.err)  
  
  mae <- c(mae,mean.abs.err)
  mse <- c(mse,mean.sq.err)
  rmse <- c(rmse,root.mean.sq.err)
}

mean(mae)
mean(mse)
mean(rmse)

## Linear SVM Model
k = 100
mae <- c()
mse <- c()
rmse <- c()

for (i in 1:k) {
  train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))
  
  train <- dataset.sub0[train.indexes,]
  test <- dataset.sub0[-train.indexes,]
  
  svm.mod <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="linear")
  
  svm.pred <- predict(svm.mod, test)  
  
  err <- svm.pred-log10(test$PRICE)
  
  abs.err <- abs(err)
  mean.abs.err <- mean(abs.err)
  
  sq.err <- err^2
  mean.sq.err <- mean(sq.err)
  
  root.mean.sq.err <- sqrt(mean.sq.err)  
  
  mae <- c(mae,mean.abs.err)
  mse <- c(mse,mean.sq.err)
  rmse <- c(rmse,root.mean.sq.err)
}

mean(mae)
mean(mse)
mean(rmse)

## Radial SVM Model
k = 100
mae <- c()
mse <- c()
rmse <- c()

for (i in 1:k) {
  train.indexes <- sample(nrow(dataset.sub0),0.75*nrow(dataset.sub0))
  
  train <- dataset.sub0[train.indexes,]
  test <- dataset.sub0[-train.indexes,]
  
  svm.mod <- svm(log10(PRICE) ~ log10(PROPERTYSQFT), dataset.sub0, kernel="radial", gamma=0.1, cost=100)
  
  svm.pred <- predict(svm.mod, test)  
  
  err <- svm.pred-log10(test$PRICE)
  
  abs.err <- abs(err)
  mean.abs.err <- mean(abs.err)
  
  sq.err <- err^2
  mean.sq.err <- mean(sq.err)
  root.mean.sq.err <- sqrt(mean.sq.err)  
  
  mae <- c(mae,mean.abs.err)
  mse <- c(mse,mean.sq.err)
  rmse <- c(rmse,root.mean.sq.err)
}

mean(mae)
mean(mse)
mean(rmse)


#### THE END ####

