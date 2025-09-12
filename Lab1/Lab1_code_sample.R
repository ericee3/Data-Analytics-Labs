library(readr)
library(EnvStats)

# set working directory (relative path)
setwd("~/Courses/Data Analytics/Fall25/labs/lab 1/")

# read data
epi.data <- read_csv("epi_results_2024_pop_gdp.csv")

# view dataframe
View(epi.data)

# print summary of variables in dataframe
summary(epi.data$EPI.new)

# print values in variable
epi.data$EPI.new


######## Optional ########
## If you want to reference the variable without using the dataframe:

# attach dataframe
attach(epi.data)

# print values in variable
EPI.new

########################



### Explore Variable ###

EPI <- epi.data$EPI.new

# find NAs in variable - outputs vector of logical values, true if NA, false otherwise
NAs <- is.na(EPI)

EPI[which(NAs)]

# print values in variable
MHP <- epi.data$MHP.new

MHP

# find NAs inv variavle - outputs vector of logical values, true if NA, false otherwise
NAs <- is.na(MHP)

# print NAs
MHP[which(NAs)]

# take subset of NOT NAs from variable
MHP.noNA <- MHP[!NAs]

MHP.noNA

# filter for only values above 30
MHP.above30 <- MHP.noNA[MHP.noNA>30]

MHP.above30
  
# stats
summary(MHP.above30)

# boxplot of variable(s)
boxplot(EPI, MHP.above30, names = c("EPI","MHP"))


### Histograms ###

# histogram (frequency distribution)
hist(EPI)

# define sequence of values over which to plot histogram
x <- seq(20., 80., 10)
  
# histogram (frequency distribution) over range
hist(EPI, x, prob=TRUE)

# print estimated density curve for variable
lines(density(EPI,na.rm=TRUE,bw=1.)) # or try bw=“SJ”

# print rug
rug(EPI)

x <- seq(20., 80., 5)

# histogram (frequency distribution) over rabge
hist(EPI, x, prob=TRUE) 

# print estimated density curve for variable
lines(density(EPI,na.rm=TRUE, bw="SJ"))

# print rug
rug(EPI)


# histogram (frequency distribution) over rabge
hist(EPI.new, x, prob=TRUE) 

# range
x1<-seq(20,80,1)

# generate probability density values for a normal distribution with given mean and sd
d1 <- dnorm(x1,mean=45, sd=11,log=FALSE)

# print density values
lines(x1,d1)

# generate probability density values for a normal distribution with given mean and sd
d2 <- dnorm(x1,mean=64, sd=11,log=FALSE) 

# print density values
lines(x1,d2) 

# print density values
lines(x1,.5*d2)

### Empirical Cumulative Distribution Function ###

# plot ecdfs
plot(ecdf(EPI), do.points=FALSE, verticals=TRUE) 

plot(ecdf(MHP), do.points=FALSE, verticals=TRUE) 


### Quantile-quantile Plots ###

# print quantile-quantile plot for variable with theoretical normal distribuion
qqnorm(EPI); qqline(EPI)


# print quantile-quantile plot for random numbers from a normal distribution with theoretical normal distribution
x <- rnorm(500)
qqnorm(x); qqline(x)


# print quantile-quantile plot for variable with any theoretical distribution
qqplot(rnorm(180), EPI.new.sub, xlab = "Q-Q plot for norm dsn") 
qqline(EPI.new.sub)

# print quantile-quantile plot for 2 variables
qqplot(EPI, MHP, xlab = "Q-Q plot for EPI vs MHP") 

qqplot(x, EPI, xlab = "Q-Q plot for EPI vs MHP") 
qqline(EPI)

y <- rnorm(500)

qqplot(x, y, xlab = "Q-Q plot for EPI vs MHP") 
qqline(y)


## Statistical Tests

x <- rnorm(500)
y <- rnorm(500)

hist(x)
hist(y)

shapiro.test(x)
shapiro.test(y)

ad.test(x)
ad.test(y)

ks.test(x,y)

wilcox.test(x,y)

var.test(x,y)
t.test(x,y)
