# DIGIT RECOGNIZER SVM MODEL
# DATE - 30/10/15

# Set working directory
setwd("D:\\2015\\Kaggle\\Digit Recognizer\\Data")

# Load necessary packages
require(readr)
require(kernlab)

# Load the data
train <- read_csv("train.csv")
dim(train)
train$label <- as.factor(train$label)

test <- read_csv("test.csv")

letter.classifier <- ksvm(label ~., data = train, kernel = "rbfdot")
#12:05 pm - 12:30 pm

# Prediction
letter.predict <- predict(letter.classifier, newdata = test)
head(letter.predict)
submission <- data.frame(ImageId = 1:nrow(test), Label = letter.predict)
summary(submission)
head(submission)

write_csv(submission, "svm_model.csv")
# SCORE = 0.97186
