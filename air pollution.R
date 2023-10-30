library(dplyr)  # data manipulation
library(tidyr)  # data manipulation
library(ggplot2)  # data visualization
library(caret)    # machine learning algorithms
library(randomForest) # classification
library(caret)        # evaluating machine learning models
library(xgboost)      # gradient boosting
library(ggplot2)      # data visualization
library(strucchange)  # structural change testing and monitoring in linear regression models.
library(glmnet)       # fitting regularized linear regression models

air <- read.csv("C:\\Users\\visha\\Downloads\\global air pollution dataset.csv")
head(air)

air1 <- air %>% 
  mutate(across(where(is.numeric), ~replace_na(., median(., na.rm=TRUE))))
View(air1)

air2 <- air1 %>% 
  distinct(.keep_all=TRUE)
View(air2)
air2

sum(!complete.cases(air2))

# Calculate mean AQI Value for each city
aira <- aggregate(AQI.Value ~ City, air2, mean)
View(aira)

aira1 <- aggregate(CO.AQI.Value ~ City, air2, mean)
View(aira1)

aira2 <- aggregate(NO2.AQI.Value ~ City, air2, mean)
View(aira2)

aira3 <- aggregate(PM2.5.AQI.Value ~ City, air2, mean)
View(aira3)

aira4 <- aggregate(Ozone.AQI.Value ~ City, air2, mean)
View(aira4)

# Calculate summary statistics for AQI Value
summary(air2$AQI.Value)
summary(air2$CO.AQI.Value)
summary(air2$Ozone.AQI.Value)
summary(air2$NO2.AQI.Value)
summary(air2$PM2.5.AQI.Value)

# Distribution Plots
ggplot(air2, aes(x = air2$AQI.Value)) + stat_function(fun=dnorm,
  args = with(air2, c(mean = mean(air2$AQI.Value, sd = sd(air2$AQI.Value) )))) +
  scale_x_continuous("Value")

ggplot(air2, aes(x = air2$CO.AQI.Value)) + stat_function(fun=dnorm,
  args = with(air2, c(mean = mean(air2$CO.AQI.Value, sd = sd(air2$CO.AQI.Value) )))) +
  scale_x_continuous("Value")

ggplot(air2, aes(x = air2$Ozone.AQI.Value)) + stat_function(fun=dnorm,
  args = with(air2, c(mean = mean(air2$Ozone.AQI.Value, sd = sd(air2$Ozone.AQI.Value) )))) +
  scale_x_continuous("Value")

ggplot(air2, aes(x = air2$NO2.AQI.Value)) + stat_function(fun=dnorm,
  args = with(air2, c(mean = mean(air2$NO2.AQI.Value, sd = sd(air2$NO2.AQI.Value) )))) +
  scale_x_continuous("Value")

ggplot(air2, aes(x = air2$PM2.5.AQI.Value)) + stat_function(fun=dnorm,
  args = with(air2, c(mean = mean(air2$PM2.5.AQI.Value, sd = sd(air2$PM2.5.AQI.Value) )))) +
  scale_x_continuous("Value")


# QQplot
ggplot(aira, aes(sample = aira$AQI.Value)) +
  stat_qq(colour="red") + 
  stat_qq_line(colour="blue") +
  theme(panel.grid.major=element_line(colour="lightblue")
        ,panel.grid.minor=element_line(colour="lightblue"))

ggplot(aira1, aes(sample = aira1$CO.AQI.Value)) +
  stat_qq(colour="red") + 
  stat_qq_line(colour="blue") +
  theme(panel.grid.major=element_line(colour="lightblue")
        ,panel.grid.minor=element_line(colour="lightblue"))

ggplot(aira2, aes(sample = aira2$NO2.AQI.Value)) +
  stat_qq(colour="red") + 
  stat_qq_line(colour="blue")+
  theme(panel.grid.major=element_line(colour="lightblue")
        ,panel.grid.minor=element_line(colour="lightblue"))

ggplot(aira3, aes(sample = aira3$PM2.5.AQI.Value)) +
  stat_qq(colour="red") + 
  stat_qq_line(colour="blue") +
  theme(panel.grid.major=element_line(colour="lightblue")
        ,panel.grid.minor=element_line(colour="lightblue"))

###################### XGBOOST MODEL

### 1

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
parts = createDataPartition(air2$AQI.Value, p = .8, list = F)
traina = air2[parts, ]
testa = air2[-parts, ]

# Define predictor and response variables in the training set
train_xa <- data.matrix(traina[, -which(names(traina) == "AQI.Value")])  # Exclude the target variable
train_ya <- traina$`AQI.Value`

# Define predictor and response variables in the testing set
test_xa <- data.matrix(testa[, -which(names(testa) == "AQI.Value")])    # Exclude the target variable
test_ya <- testa$`AQI.Value`

# Define final training and testing sets as xgboost DMatrix objects
xgb_traina <- xgb.DMatrix(data = train_xa, label = train_ya)
xgb_testa <- xgb.DMatrix(data = test_xa, label = test_ya)

#define watchlist
watchlist = list(train=xgb_traina, test=xgb_testa)

#fit XGBoost model and display training and testing data at each round
model = xgb.train(data = xgb_traina, max.depth = 10, watchlist=watchlist, nrounds = 120)

#define final model
final = xgboost(data = xgb_traina, max.depth = 10, nrounds = 120, verbose = 0)
final

# Make predictions on the testing set
y_pred <- predict(final, xgb_testa)

mean((test_ya - y_pred)^2) #mse
caret::MAE(test_ya, y_pred) #mae
caret::RMSE(test_ya, y_pred) #rmse


### 2
#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
parts1 = createDataPartition(air2$CO.AQI.Value, p = .8, list = F)
traina1 = air2[parts1, ]
testa1 = air2[-parts1, ]

# Define predictor and response variables in the training set
train_xa1 <- data.matrix(traina1[, -which(names(traina1) == "CO.AQI.Value")])  # Exclude the target variable
train_ya1 <- traina1$`CO.AQI.Value`

# Define predictor and response variables in the testing set
test_xa1 <- data.matrix(testa1[, -which(names(testa1) == "CO.AQI.Value")])    # Exclude the target variable
test_ya1 <- testa1$`CO.AQI.Value`

# Define final training and testing sets as xgboost DMatrix objects
xgb_traina1 <- xgb.DMatrix(data = train_xa1, label = train_ya1)
xgb_testa1 <- xgb.DMatrix(data = test_xa1, label = test_ya1)

#define watchlist
watchlist1 = list(train=xgb_traina1, test=xgb_testa1)

#fit XGBoost model and display training and testing data at each round
model1 = xgb.train(data = xgb_traina1, max.depth = 10, watchlist=watchlist1, nrounds = 120)

#define final model
final1 = xgboost(data = xgb_traina1, max.depth = 10, nrounds = 120, verbose = 0)
final1

# Make predictions on the testing set
y_pred1 <- predict(final1, xgb_testa1)

mean((test_ya1 - y_pred1)^2) #mse
caret::MAE(test_ya1, y_pred1) #mae
caret::RMSE(test_ya1, y_pred1) #rmse

### 3

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
partsc = createDataPartition(air2$Ozone.AQI.Value, p = .8, list = F)
trainac = air2[partsc, ]
testac = air2[-partsc, ]

# Define predictor and response variables in the training set
train_xac <- data.matrix(trainac[, -which(names(trainac) == "Ozone.AQI.Value")])  # Exclude the target variable
train_yac <- trainac$Ozone.AQI.Value


# Define predictor and response variables in the testing set
test_xac <- data.matrix(testac[, -which(names(testac) == "Ozone.AQI.Value")])    # Exclude the target variable
test_yac <- testac$Ozone.AQI.Value

# Define final training and testing sets as xgboost DMatrix objects
xgb_trainac <- xgb.DMatrix(data = train_xac, label = train_yac)
xgb_testac <- xgb.DMatrix(data = test_xac, label = test_yac)

#define watchlist
watchlistc = list(train=xgb_trainac, test=xgb_testac)

#fit XGBoost model and display training and testing data at each round
modelc = xgb.train(data = xgb_trainac, max.depth = 12, watchlist=watchlistc, nrounds = 100)

#define final model
finalc = xgboost(data = xgb_trainac, max.depth = 12, nrounds = 100, verbose = 1)
finalc

# Make predictions on the testing set
y_predc <- predict(finalc, xgb_testac)

mean((test_yac - y_predc)^2) #mse
caret::MAE(test_yac, y_predc) #mae
caret::RMSE(test_yac, y_predc) #rmse

### 4

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
partsn = createDataPartition(air2$NO2.AQI.Value, p = .8, list = F)
trainan = air2[partsn, ]
testan = air2[-partsn, ]

# Define predictor and response variables in the training set
train_xan <- data.matrix(trainan[, -which(names(trainan) == "NO2.AQI.Value")])  # Exclude the target variable
train_yan <- trainan$Ozone.AQI.Value


# Define predictor and response variables in the testing set
test_xan <- data.matrix(testan[, -which(names(testan) == "NO2.AQI.Value")])    # Exclude the target variable
test_yan <- testan$NO2.AQI.Value

# Define final training and testing sets as xgboost DMatrix objects
xgb_trainan <- xgb.DMatrix(data = train_xan, label = train_yan)
xgb_testan <- xgb.DMatrix(data = test_xan, label = test_yan)

#define watchlist
watchlistn = list(train=xgb_trainan, test=xgb_testan)

#fit XGBoost model and display training and testing data at each round
modeln = xgb.train(data = xgb_trainan, max.depth = 12, watchlist=watchlistn, nrounds = 100)

#define final model
finaln = xgboost(data = xgb_trainan, max.depth = 12, nrounds = 100, verbose = 1)
finaln

# Make predictions on the testing set
y_predn <- predict(finaln, xgb_testan)

mean((test_yan - y_predn)^2) #mse
caret::MAE(test_yan, y_predn) #mae
caret::RMSE(test_yan, y_predn) #rmse

### 5

#make this example reproducible
set.seed(0)

#split into training (80%) and testing set (20%)
partsp = createDataPartition(air2$PM2.5.AQI.Value, p = .8, list = F)
trainap = air2[partsp, ]
testap = air2[-partsp, ]

# Define predictor and response variables in the training set
train_xap <- data.matrix(trainap[, -which(names(trainap) == "PM2.5.AQI.Value")])  # Exclude the target variable
train_yap <- trainap$PM2.5.AQI.Value


# Define predictor and response variables in the testing set
test_xap <- data.matrix(testap[, -which(names(testap) == "PM2.5.AQI.Value")])    # Exclude the target variable
test_yap <- testap$PM2.5.AQI.Value

# Define final training and testing sets as xgboost DMatrix objects
xgb_trainap <- xgb.DMatrix(data = train_xap, label = train_yap)
xgb_testap <- xgb.DMatrix(data = test_xap, label = test_yap)

#define watchlist
watchlistp = list(train=xgb_trainap, test=xgb_testap)

#fit XGBoost model and display training and testing data at each round
modelp = xgb.train(data = xgb_trainap, max.depth = 12, watchlist=watchlistp, nrounds = 100)

#define final model
finalp = xgboost(data = xgb_trainap, max.depth = 12, nrounds = 100, verbose = 1)
finalp

# Make predictions on the testing set
y_predp <- predict(finalp, xgb_testap)

mean((test_yap - y_predp)^2) #mse
caret::MAE(test_yap, y_predp) #mae
caret::RMSE(test_yap, y_predp) #rmse


#########################  K-Means Cluster Analysis
# Prepare Data
set.seed(42)  # Set a random seed for reproducibility
sample_size <- 1500  # Specify the desired sample size
sampled_data <- air1[sample(1:nrow(air1), size = sample_size), ]
numeric_data <- sampled_data %>% 
  select_if(is.numeric)
scaled_data <- scale(numeric_data)

# Determine number of clusters
wss <- (nrow(scaled_data)-1)*sum(apply(scaled_data,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(scaled_data,
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

fit <- kmeans(scaled_data, 8) # 8 cluster solution
# get cluster means
aggregate(scaled_data,by=list(fit$cluster),FUN=mean)
# append cluster assignment
airc <- data.frame(scaled_data, fit$cluster)

# K-Means Clustering with 5 clusters
fit <- kmeans(scaled_data, 8)


######################### Lasso Regression
yl <- air2$AQI.Value

xl <- data.matrix(air2[, c('CO.AQI.Value','NO2.AQI.Value',
                           'PM2.5.AQI.Value','Ozone.AQI.Value')])

#perform k-fold cross-validation to find optimal lambda value
cv_modell <- cv.glmnet(xl, yl, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambdal <- cv_modell$lambda.min
best_lambdal

#produce plot of test MSE by lambda value
plot(cv_modell) 

#find coefficients of best model
best_modell <- glmnet(xl, yl, alpha = 1, lambda = best_lambdal)
coef(best_modell)

#define new observation
newl = matrix(c(20, 2.0, 3.0, 18.0), nrow=1, ncol=4) 

#use lasso regression model to predict response value
predict(best_modell, s = best_lambdal, newx = newl)

#use fitted best model to make predictions
y_predictedl <- predict(best_modell, s = best_lambdal, newx = xl)

#find SST and SSE
sstl <- sum((yl - mean(yl))^2)
sstl
ssel <- sum((y_predictedl - yl)^2)
ssel

#find R-Squared
rsql <- 1 - ssel/sstl
rsql

######################### Ridge Regression

yl1 <- air2$AQI.Value

xl1 <- data.matrix(air2[, c('CO.AQI.Value','NO2.AQI.Value',
                           'PM2.5.AQI.Value','Ozone.AQI.Value')])

#fit ridge regression model
modell1 <- glmnet(xl1, yl1, alpha = 0)

#view summary of model
summary(modell1)

#perform k-fold cross-validation to find optimal lambda value
cv_modell1 <- cv.glmnet(xl1, yl1, alpha = 0)

#find optimal lambda value that minimizes test MSE
best_lambdal1 <- cv_modell1$lambda.min
best_lambdal1

#produce plot of test MSE by lambda value
plot(cv_modell1) 

#find coefficients of best model
best_modell1 <- glmnet(xl1, yl1, alpha = 0, lambda = best_lambdal1)
coef(best_modell1)

#produce Ridge trace plot
plot(modell1, xvar = "lambda")

#use fitted best model to make predictions
y_predictedl1 <- predict(modell1, s = best_lambdal1, newx = xl1)

#find SST and SSE
sstl1 <- sum((yl1 - mean(yl1))^2)
sstl1
ssel1 <- sum((y_predictedl1 - yl1)^2)
ssel1

#find R-Squared
rsqr <- 1 - ssel1/sstl1
rsqr

######################## One Sample T-test
t.test(x = aira$AQI.Value, mu = 130)
t.test(x = aira1$CO.AQI.Value, mu = 130)
t.test(x = aira2$NO2.AQI.Value, mu = 130)
t.test(x = aira3$PM2.5.AQI.Value, mu = 130)
t.test(x = aira4$Ozone.AQI.Value, mu = 130)


######################## Two Sample T-test
t.test(x = aira$AQI.Value, aira1$AQI.Value)
t.test(x = aira$AQI.Value, aira2$NO2.AQI.Value)
t.test(x = aira$AQI.Value, aira3$PM2.5.AQI.Value)
t.test(x = aira$AQI.Value, aira4$Ozone.AQI.Value)


######################## Chow Test
sctest(aira$AQI.Value ~ aira1$CO.AQI.Value, type = "Chow", point = 10)
sctest(aira$AQI.Value ~ aira2$NO2.AQI.Value, type = "Chow", point = 10)
sctest(aira$AQI.Value ~ aira3$PM2.5.AQI.Value, type = "Chow", point = 10)
sctest(aira$AQI.Value ~ aira4$Ozone.AQI.Value, type = "Chow", point = 10)

####################### Chi-Square Test
chisq.test(aira$City,aira$AQI.Value)
chisq.test(aira1$City,aira1$CO.AQI.Value)
chisq.test(aira2$City,aira2$NO2.AQI.Value)
chisq.test(aira3$City,aira3$PM2.5.AQI.Value)
chisq.test(aira4$City,aira4$Ozone.AQI.Value)
