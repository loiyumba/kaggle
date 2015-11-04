### BIKE SHARING DEMAND

### Finding important features through fscaret package. Use the top 5 features.
### Run two separate models and blend the prediction

## Load the necessary packages
require(fscaret)
require(caret)
data(funcRegPred) # list all the models support by fscaret
funcRegPred

## Set working directory
setwd("D:\\2015\\Kaggle\\Bike Sharing")
## Load the train data
train <- read.csv("train.csv")
str(train)
names(train)
sapply(train, function(x) sum(is.na(x))) # checking missing values
train <- train[, -c(10, 11)]
## Few more features extracted from date feature
train$date <- substr(train$datetime, 1, 10)
train$date <- as.Date(train$date, format = "%Y-%m-%d")
train$year <- as.integer(format(train$date, "%Y"))
train$month <- as.integer(format(train$date, "%m"))
train$day <- as.integer(format(train$date, "%d"))
train$weekday <- weekdays(train$date)
train$weekday <- as.integer(as.factor(train$weekday))
train$hour <- substr(train$datetime, 12, 13)
# Assigning appropriate classes to features
train$season <- as.factor(train$season)
train$holiday <- as.factor(train$holiday)
train$workingday <- as.factor(train$workingday)
train$weather <- as.factor(train$weather)
train$year <- as.factor(train$year)
train$month <- as.factor(train$month)
train$day <- as.factor(train$day)
train$weekday <- as.factor(train$weekday)
train$hour <- as.factor(train$hour)
# Removing useless features
train <- train[, -c(1, 11)]

# Dummyfied all the factor variables
dput(names(train))
train <- train[c("season", "holiday", "workingday", "weather", "temp", "atemp", 
                 "humidity", "windspeed", "year", "month", "day", "weekday", 
                 "hour", "count")] # target variable to be last for fscaret
dummy_train <- dummyVars("~.", data = train, fullRank = FALSE)
train_df <- as.data.frame(predict(dummy_train, train))
head(train_df)
str(train_df)

## Load the test data
test <- read.csv("test.csv")
str(test)
test$count <- 0
sapply(test, function(x) sum(is.na(x)))
test$date <- substr(test$datetime, 1, 10)
test$date <- as.Date(test$date, format = "%Y-%m-%d")
test$year <- as.integer(format(test$date, "%Y"))
test$month <- as.integer(format(test$date, "%m"))
test$day <- as.integer(format(test$date, "%d"))
test$weekday <- weekdays(test$date)
test$weekday <- as.integer(as.factor(test$weekday))
test$hour <- substr(test$datetime, 12, 13)

test$season <- as.factor(test$season)
test$holiday <- as.factor(test$holiday)
test$workingday <- as.factor(test$workingday)
test$weather <- as.factor(test$weather)
test$year <- as.factor(test$year)
test$month <- as.factor(test$month)
test$day <- as.factor(test$day)
test$weekday <- as.factor(test$weekday)
test$hour <- as.factor(test$hour)
str(test)

test <- test[, -c(1, 11)]

dput(names(test))
test <- test[c("season", "holiday", "workingday", "weather", "temp", "atemp", 
               "humidity", "windspeed", "year", "month", "day", "weekday", 
               "hour", "count")] # target variable to be last for fscaret
dummy_test <- dummyVars("~.", data = test, fullRank = FALSE)
test_df <- as.data.frame(predict(dummy_test, test))
head(test_df)
str(test_df)

# data partition for train data set for local validation
split <- createDataPartition(train_df$count, p = .7, list = FALSE, times = 1)
sample_train <- train_df[split, ]
sample_test <- train_df[-split, ]

fsmodels <- c("gbm", "rf")
my_fs <- fscaret(sample_train, sample_test, regPred = TRUE, myTimeLimit = 40, preprocessData = TRUE,
                 Used.funcRegPred = fsmodels, with.labels = TRUE, supress.output = FALSE)
# 2:35 pm

#gbm - weather.1, weather.2, season.1, season.4, month.12
#rf - month.4, month.5, day.19, month.6, humidity

# formula for gbm
gbm_formula <- count ~ weather.1 + weather.2 + season.1 + season.4 + month.12 
rf_formula <- count ~ month.4 + month.5 + day.19 + month.6 + humidity

ctrl <- trainControl(method = "repeatedcv", number = 3, repeats = 3)
# gbm
gbm_model <- train(gbm_formula, data = sample_train, method = "gbm", trControl = ctrl, verbose = TRUE)
gbm_model
summary(gbm_model)
gbm_pred <- predict(gbm_model, newdata = sample_test)
summary(gbm_pred)
gbm_sse <- sum((sample_test$count - gbm_pred)^2)
gbm_sst <- sum((sample_test$count - mean(sample_train$count))^2)
r_squared <- 1 - gbm_sse/gbm_sst

# rf
rf_model <- train(rf_formula, data = sample_train, method = "rf", trControl = ctrl)
summary(rf_model)
plot(rf_model)
rf_pred <- predict(rf_model, newdata = sample_test)
summary(rf_pred)
rf_sse <- sum((sample_test$count - rf_pred)^2)
rf_sst <- sum((sample_test$count - mean(sample_train$count))^2)
r_squared <- 1 - rf_sse/rf_sst

# Blend gbm prediction and rf prediction
blended_pred <- (gbm_pred + rf_pred)/2
summary(blended_pred)
blended_sse <- sum((sample_test$count - blended_pred)^2)
blended_rSquared <- 1 - blended_sse/rf_sst

### Duplicate the same on full train data and see the results.
### First with selected top 5 features and see the LB.
### Then with full features and see the LB.

fsmodels <- c("lm", "gbm", "rf")
my_fs <- fscaret(train_df, test_df, regPred = TRUE, myTimeLimit = 40, preprocessData = TRUE,
                 Used.funcRegPred = fsmodels, with.labels = TRUE, supress.output = FALSE)
# Giving error messages because the days are not matching in train and test.
# Remove all the day variables
train_df <- train_df[, -c(31:49)]
test_df <- test_df[, -c(31:42)]
# Rerun my_fs
# Time took = 10:00 am - 5:30 pm 

## Listing all the outputs
my_fs$VarImp
my_fs$PPlabels

# gbm variables
gbm_formula <- count ~ weather.1 + weather.2 + season.4 + season.1 + weather.3 + month.1
# rf variables
rf_formula <- log(count) ~ month.3 + weekday.1 + workingday.0 + month.9 + month.11 + month.10

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 4)

# gbm model cv
gbm_model <- train(gbm_formula, data = train_df, method = "gbm", trControl = ctrl,
                   verbose = TRUE)
gbm_model
gbm_pred <- predict(gbm_model, newdata = test_df)
summary(gbm_pred)
gbm_submit <- data.frame(datetime = test$datetime, count = gbm_pred) 
summary(gbm_submit)
write.csv(gbm_submit, "gbm_submit.csv", row.names = FALSE) # 1.51801

# Rerun with log(count)
gbm_logform <- log(count) ~ weather.1 + weather.2 + season.4 + season.1 + weather.3 + 
  month.1

gbm_model2 <- train(gbm_logform, data = train_df, method = "gbm", trControl = ctrl,
                    verbose = TRUE)
gbm_model2
gbm_pred2 <- exp(predict(gbm_model2, newdata = test_df)) - 1
summary(gbm_pred2)
gbm_submit2 <- data.frame(datetime = test$datetime, count = gbm_pred2) 
summary(gbm_submit2)
write.csv(gbm_submit2, "gbm_submit2.csv", row.names = FALSE) # 1.36444

# rf model
rf_model <- train(rf_formula, data = train_df, method = "rf", trControl = ctrl,
                  do.trace = TRUE)
rf_model
rf_pred <- exp(predict(rf_model, newdata = test_df)) - 1
summary(rf_pred)
rf_submit <- data.frame(datetime = test$datetime, count = rf_pred)
write.csv(rf_submit, "rf_submit.csv", row.names = FALSE) # 1.41162

# Blended model
blend_model <- (gbm_pred2 + rf_pred)/2
summary(blend_model)
blended_submit <- data.frame(datetime = test$datetime, count = blend_model)
write.csv(blended_submit, "blended.submit.csv", row.names = FALSE) # 1.37587

blend_model2 <- (gbm_pred2*3 + rf_pred)/4
summary(blend_model2)
blended_submit2 <- data.frame(datetime = test$datetime, count = blend_model2)
write.csv(blended_submit2, "blended_submit2.csv", row.names = FALSE) # 1.36603

blend_model3 <- (gbm_pred2*5 + rf_pred)/6
summary(blend_model3)
blended_submit3 <- data.frame(datetime = test$datetime, count = blend_model3)
write.csv(blended_submit3, "blended_submit3.csv", row.names = FALSE) # 1.36432

## Run the model on full features on dummyfied train data
gbm_model <- train(log(count) ~., data = train_df, method = "gbm", trControl = ctrl,
                   verbose = TRUE)
gbm_model
gbm_pred <- exp(predict(gbm_model, newdata = test_df)) - 1
summary(gbm_pred)
submit <- data.frame(datetime = test$datetime, count = gbm_pred)
write.csv(submit, "full_feature_gbm.csv", row.names = FALSE) # 0.51213

# Previous benchmark - 0.48784

rf_model <- train(log(count) ~., data = train_df, method = "rf", trControl = ctrl,
                  do.trace = TRUE)

rf_model
rf_pred <- exp(predict(rf_model, newdata = test_df)) - 1
summary(rf_pred)
rf_submit <- data.frame(datetime = test$datetime, count = rf_pred)
write.csv(rf_submit, "full_feature_rf.csv", row.names = FALSE) # 0.42176

# full feature blended model
blended_model <- (gbm_pred + rf_pred)/2
summary(blended_model)
blended_submit <- data.frame(datetime = test$datetime, count = blended_model)
write.csv(blended_submit, "full_feature_blend.csv", row.names = FALSE) # 0.43436

blended_model2 <- (gbm_pred + rf_pred*6)/7
summary(blended_model2)
blended_submit2 <- data.frame(datetime = test$datetime, count = blended_model2)
write.csv(blended_submit2, "full_feature_blend2.csv", row.names = FALSE) # 0.41570

blended_model4 <- (gbm_pred + rf_pred*7)/8
summary(blended_model4)
blended_submit4 <- data.frame(datetime = test$datetime, count = blended_model4)
write.csv(blended_submit4, "full_feature_blend4.csv", row.names = FALSE) # 0.41589

blended_model5 <- (gbm_pred*2 + rf_pred*6)/8
summary(blended_model5)
blended_submit5 <- data.frame(datetime = test$datetime, count = blended_model5)
write.csv(blended_submit5, "full_feature_blend5.csv", row.names = FALSE) # 0.41715

blended_model6 <- (gbm_pred + rf_pred*5)/6
summary(blended_model6)
blended_submit6 <- data.frame(datetime = test$datetime, count = blended_model6)
write.csv(blended_submit6, "full_feature_blend6.csv", row.names = FALSE) 
# 0.41564 - *396 out of 3252 on LB
# 396/3252 - 0.1217712 top 12 percentile on LB

blended_model7 <- (gbm_pred + rf_pred*4)/5
summary(blended_model7)
blended_submit7 <- data.frame(datetime = test$datetime, count = blended_model7)
write.csv(blended_submit7, "full_feature_blend7.csv", row.names = FALSE) # 0.41594

# xgboost model
require(xgboost)
target <- train_df$count

train_df <- as.matrix(train_df[, -62])
test_df$count <- NULL
test_df <- as.matrix(test_df)

set.seed(500)
xgb_cv <- xgb.cv(data = train_df, 
                 label = log(target + 1),
                 nfold = 5,
                 nrounds = 2500,
                 eta = 0.001,
                 objective = 'reg:linear',
                 eval_metric = 'rmse',
                 early.stop.round = 10,
                 verbose = TRUE)
str(xgb_cv)
xgb_cv
# nround - 1000, eta = 0.003 - 0.560239       0.002784       0.585336      0.009188
# nround - 2000, eta = 0.001 - 0.853929+0.003513	          0.868476+0.014492
# nround - 2000, eta = 0.002 - 0.459845       0.003390       0.493034      0.006664
# nround = 2500, eta = 0.001 - 0.667316       0.004121       0.686660      0.006357

xgb_model <- xgboost(data = train_df, 
                     label = log(target + 1),
                     nrounds = 2000,
                     eta = 0.002,
                     objective = 'reg:linear',
                     verbose = TRUE)
xgb_pred <- exp(predict(xgb_model, newdata = test_df)) - 1
summary(xgb_pred)
submit <- data.frame(datetime = test$datetime, count = xgb_pred)
write.csv(submit, "xgb_submit2.csv", row.names = FALSE) # 0.52287

# 3 blended models
blended_model8 <- (gbm_pred + rf_pred + xgb_pred)/3
summary(blended_model8)
submit_blend <- data.frame(datetime = test$datetime, count = blended_model8)
write.csv(submit_blend, "blend3model.csv", row.names = FALSE) # 0.44959

blended_model9 <- (gbm_pred + rf_pred*2 + xgb_pred)/4
summary(blended_model9)
submit <- data.frame(datetime = test$datetime, count = blended_model9)
write.csv(submit, "blend3model2.csv", row.names = FALSE) #0.43041

blended_model10 <- (gbm_pred + rf_pred*3 + xgb_pred)/5
summary(blended_model10)
submit <- data.frame(datetime = test$datetime, count = blended_model10)
write.csv(submit, "blend3model3.csv", row.names = FALSE) # 0.42218

blended_model11 <- (gbm_pred + rf_pred*4 + xgb_pred)/6
summary(blended_model11)
submit <- data.frame(datetime = test$datetime, count = blended_model11)
write.csv(submit, "blend3model4.csv", row.names = FALSE) # 0.41819

blended_model12 <- (gbm_pred + rf_pred*5 + xgb_pred)/7
summary(blended_model12)
submit <- data.frame(datetime = test$datetime, count = blended_model12)
write.csv(submit, "blend3model5.csv", row.names = FALSE) # 0.41615

blended_model13 <- (gbm_pred + rf_pred*6 + xgb_pred)/8
summary(blended_model13)
submit <- data.frame(datetime = test$datetime, count = blended_model13)
write.csv(submit, "blend3model6.csv", row.names = FALSE) # 0.41508 389* on LB

blended_model14 <- (gbm_pred + rf_pred*7 + xgb_pred)/9
summary(blended_model14)
submit <- data.frame(datetime = test$datetime, count = blended_model14)
write.csv(submit, "blend3model7.csv", row.names = FALSE) # 0.41455 386* on LB

blended_model15 <- (gbm_pred*2 + rf_pred*7 + xgb_pred*2)/11
summary(blended_model15)
submit <- data.frame(datetime = test$datetime, count = blended_model15)
write.csv(submit, "blend3model8.csv", row.names = FALSE) # 0.41985 

blended_model16 <- (gbm_pred + rf_pred*8 + xgb_pred)/10
summary(blended_model16)
submit <- data.frame(datetime = test$datetime, count = blended_model16)
write.csv(submit, "blend3model9.csv", row.names = FALSE) 
# 0.41432 386* on LB
# 386/3252 = 0.1186962 that is top 12%
blended_model17 <- (rf_pred*8 + xgb_pred)/9
summary(blended_model17)
submit <- data.frame(datetime = test$datetime, count = blended_model17)
write.csv(submit, "blend3model10.csv", row.names = FALSE) # 0.41492

blended_model18 <- (rf_pred*7 + xgb_pred)/8
summary(blended_model18)
submit <- data.frame(datetime = test$datetime, count = blended_model18)
write.csv(submit, "blend3model11.csv", row.names = FALSE) # 0.41454









