library(ggplot2)
library(dplyr)
library(statsr)
library(gridExtra)
library(corrplot)

train_data <- read.csv('C:/Users/Aditya/Downloads/Projects/NYC Taxi fare prediction/train_data.csv', header=TRUE, nrows = 10000)
test_data <- read.csv('C:/Users/Aditya/Downloads/Projects/NYC Taxi fare prediction/test_data.csv', header = TRUE, nrows = 3000)

dim(train_data)
dim(test_data)

names(train_data)
names(test_data)

head(train_data)

train_data[train_data=="NULL"] <- NA

train_data[train_data == "0"] <- NA

train_data$fare_amount[train_data$fare_amount < "0"] <- NA

train_data$passenger_count[train_data$passenger_count < "0"] <- NA

train_data$pickup_latitude[train_data$pickup_latitude < "-90"] <- NA
train_data$pickup_latitude[train_data$pickup_latitude > "90"] <- NA


train_data$pickup_longitude[train_data$pickup_longitude < "-180"] <- NA
train_data$pickup_longitude[train_data$pickup_longitude > "180"] <- NA

train_data$dropoff_latitude[train_data$dropoff_latitude < "-90"] <- NA
train_data$dropoff_latitude[train_data$dropoff_latitude > "90"] <- NA


train_data$dropoff_longitude[train_data$dropoff_longitude < "-180"] <- NA
train_data$dropoff_longitude[train_data$dropoff_longitude > "180"] <- NA

apply(train_data, 2, function(x){sum(is.na(x))})
is.na(train_data)
train_data <- na.omit(train_data)
dim(train_data)
apply(train_data, 2, function(x){sum(is.na(x))})
summary(train_data)

train_data$key <- as.Date(train_data$key,'%Y-%m-%d %H:%M:%S') 
train_data$pickup_datetime <- as.Date(train_data$pickup_datetime,'%Y-%m-%d %H:%M:%S')

test_data$key <- as.Date(test_data$key,'%Y-%m-%d %H:%M:%S') 
test_data$pickup_datetime <- as.Date(test_data$pickup_datetime,'%Y-%m-%d %H:%M:%S') 


setwd("C:/Users/Aditya/Downloads/Projects/NYC Taxi fare prediction")
#write.csv(train_data, file = "MyData.csv",row.names=FALSE, na="")
#write.csv(test_data, file = "MyData_test.csv",row.names=FALSE, na="")


library(ggplot2)
ggplot(data=train_data, aes(fare_amount)) + 
  geom_histogram(fill="blue") + 
  labs(title="Histogram of Price") +
  labs(x="Price", y="Count")

quantile(train_data$fare_amount, c(.9, .95, .97, 0.975, 0.98, 0.99, 0.995, 0.999, 0.9999))

train_data$fare_amount <- ifelse(train_data$fare_amount>20,20,train_data$fare_amount)
ggplot(data=train_data, aes(fare_amount)) + 
  geom_histogram(fill="blue") + 
  labs(title="Histogram of fare amount") +
  labs(x="Price", y="Count")

colors = c("red", "yellow", "green", "violet", "orange", "blue", "pink", "cyan")
hist(train_data$fare_amount, col=colors, main = "Histogram for Train score")
summary(train_data$fare_amount)

boxplot(fare_amount~passenger_count, data=train_data, main='Fare amount vs. Passenger Count', xlab='Number of Passenger', ylab='Fare amount')

names(train_data)
train_data <- train_data[ -c(1)]
#train_data$key = as.numeric(train_data$key)
train_data$pickup_datetime = as.numeric(train_data$pickup_datetime)
mydata <- cor(train_data)
mydata_pca <- prcomp(mydata,scale=TRUE)
mydata_pca
summary(mydata_pca)

eigen_data <- mydata_pca$sdev^2
eigen_data

sumlambdas <- sum(eigen_data)
sumlambdas

plot(mydata_pca, type='l')

names(train_data)
set.seed(2017)
split <- sample(seq_len(nrow(train_data)), size = floor(0.75 * nrow(train_data)))
train <- train_data[split, ]
test <- train_data[-split, ]

target_train=train$fare_amount
target_test=test$fare_amount


#Multiple Regression
str(train_data)
fit <- lm(fare_amount ~ ., data=train_data)
summary(fit)
library(MASS)
step <- stepAIC(fit, direction="both")
step$anova # display results

fit <- lm(fare_amount ~ dropoff_longitude+dropoff_latitude+passenger_count, data=train_data)

# build the model
Pred <- predict(fit, test_data) 
Pred
actuals_preds <- data.frame(cbind(actuals=target_test, predicteds=Pred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)

min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max)) 
min_max_accuracy
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)
mape


# Random Forest
#Create model with default parameters
library(randomForest)
rf <- randomForest(fare_amount ~ ., data = train, importance = TRUE)
rf
summary(rf)
rf2 <- randomForest(fare_amount ~ ., data = train, ntree = 500, mtry = 6, importance = TRUE)
rf2
summary(rf2)
predTrain <- predict(rf2, train, type = "class")
importance(rf2) 
predtest <- predict(rf2, test_data) 
predtest
actuals_preds <- data.frame(cbind(actuals=target_test, predicteds=predtest))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max)) 
min_max_accuracy
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)
mape




#XGBOOST
library(xgboost)
#using one hot encoding 
target_train=train$fare_amount
target_test=test$fare_amount
str(target_train)
str(target_test)
train$fare_amount=NULL
feature_names <- names(train)
test$fare_amount=NULL


target_train <- as.numeric(target_train)-1
target_test <- as.numeric(target_test)-1


dtrain <- xgb.DMatrix(as.matrix(sapply(train, as.numeric)),label=target_train, missing=NA)
dtest <- xgb.DMatrix(as.matrix(sapply(test, as.numeric)),label=target_test, missing=NA)


# Set up cross-validation scheme (3-fold)
library(caret)
foldsCV <- createFolds(target_train, k=7, list=TRUE, returnTrain=FALSE)
param <- list(booster = "gblinear"
              , objective = "reg:linear"
              , subsample = 0.7
              , max_depth = 5
              , colsample_bytree = 0.7
              , eta = 0.037
              , eval_metric = 'mae'
              , base_score = 0.012 #average
              , min_child_weight = 100)

xgb_cv <- xgb.cv(data=dtrain,
                 params=param,
                 nrounds=100,
                 prediction=TRUE,
                 maximize=FALSE,
                 folds=foldsCV,
                 early_stopping_rounds = 30,
                 print_every_n = 5)

# Check best results and get best nrounds
print(xgb_cv$evaluation_log[which.min(xgb_cv$evaluation_log$test_mae_mean)])
nrounds <- xgb_cv$best_iteration

xgb <- xgb.train(params = param
                 , data = dtrain
                 # , watchlist = list(train = dtrain)
                 , nrounds = nrounds
                 , verbose = 1
                 , print_every_n = 5
                 #, feval = amm_mae
)



# Feature Importance
importance_matrix <- xgb.importance(feature_names,model=xgb)
xgb.plot.importance(importance_matrix[1:15,])
head(test)
# Predict
preds <- predict(xgb,dtest)
preds

print(xgb)
summary(xgb)


