#setwd("C:\\Users\\eravdiv\\OneDrive - Ericsson AB\\Ravi\\PG-Data-Science\\Predictive-Analysis-2\\SVM Dataset")
install.packages("caret")
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages("ggplot2")
install.packages("gridExtra")

############################ SVM Digit Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building with linear SVM
# 5. Hyperparameter tuning and cross validation
# 6. Validating the model results on test data
#####################################################################################

# Business Understanding
# A classic problem in the field of pattern recognition is that of handwritten digit recognition. 
# Suppose that you have an image of a digit submitted by a user via a scanner, a tablet, or other digital devices. 
# The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 
# 
# Objective
# You are required to develop a model using Support Vector Machine which should 
# correctly classify the handwritten digits based on the pixel values given as features.

# 2. Data_number Understanding: 
print(paste("Number of Instances: 60,000"))
print(paste("Number of Attributes: 785"))

#3. Data_number Preparation: 

#Loading Neccessary libraries

library(kernlab)
library(readr)
library(caret)

#Loading Data_number
Data_number <- read.csv("mnist_train.csv", header = F)
colnames(Data_number)
Data_number$V1
Data_number_realdata <- read.csv("mnist_test.csv", header = F)
colnames(Data_number_realdata)
Data_number_realdata$V1

#Understanding Dimensions
dim(Data_number)
dim(Data_number_realdata)

#Structure of the Data_numberset
str(Data_number)
str(Data_number_realdata)

#printing first few rows
head(Data_number)
head(Data_number_realdata)

#Exploring the Data_number
summary(Data_number)
summary(Data_number_realdata)

#checking missing value
sapply(Data_number, function(x) sum(is.na(x)))
sapply(Data_number_realdata, function(x) sum(is.na(x)))

#Making our target class to factor
Data_number$V1<-factor(Data_number$V1)
Data_number_realdata$V1 <- factor(Data_number_realdata$V1)

# Splitting the Data_number between train and test
# Choosing 15% of train data which is 9000 records

set.seed(100)
train.indices_number = sample(1:nrow(Data_number), 0.15*nrow(Data_number))
train_number = Data_number[train.indices_number, ]

# train.indices_number_1 = sample(1:nrow(Data_number), 0.05*nrow(Data_number))
# train_number_1 = Data_number[train.indices_number_1, ]

# train.indices_number_2 = sample(1:nrow(Data_number), nrow(Data_number_realdata))
# test_number_2 = Data_number[-train.indices_number_2, ]
test_number <- Data_number_realdata

# 4. Model Building

#Constructing Model
#Using Linear Kernel
Model_linear_number <- ksvm(V1~ ., data = train_number, scale = FALSE, kernel = "vanilladot")
Model_linear_number
Eval_linear_number<- predict(Model_linear_number, test_number)


#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear_number,test_number$V1)
print(paste("Accuracy for linear kernel = 0.9147"))

#Using RBF Kernel
Model_RBF_number <- ksvm(V1~ ., data = train_number, scale = FALSE, kernel = "rbfdot")
Model_RBF_number
print(paste("Sigma value is 1.6e-7 which can be further tuned to obtain better accuracy"))

Eval_RBF_number<- predict(Model_RBF_number, test_number)


#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF_number,test_number$V1)
print(paste("Accuracy for RBF kernel = 0.9584 which is better than linear kernel indicating the need to tune sigma and C"))

############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation.
trainControl_number <- trainControl(method="cv", number=5, verboseIter = TRUE)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.
metric_number <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
print(paste("Setting the sigma values and C values by trial and error to obtain best possible accuracy"))
set.seed(7)
#grid_number <- expand.grid(.sigma=c(0.0000002, 0.00000022, 0.00000024, 0.00000026), .C=c(2, 2.25, 2.5, 2.75))
grid_number <- expand.grid(.sigma=c(0.01), .C=c(7))
#grid_number <- expand.grid(.sigma=seq(0.0000001, 0.0000002, by=0.00000002), .C=seq(1, 3, by=1))

#train function takes Target ~ Prediction, Data_number, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.
fit.svm_number <- train(V1~., data=train_number, method="svmRadial", metric=metric_number,
                 tuneGrid=grid_number, trControl=trainControl_number, allowParallel = TRUE)

print(fit.svm_number)
print(paste("Best tune at sigma = 2.6e-07 & C=2.75, Accuracy - 0.964"))

plot(fit.svm_number)

# Validating the model results on test data
evaluate_non_linear_number<- predict(fit.svm_number, test_number)
confusionMatrix(evaluate_non_linear_number, test_number$V1)

print(paste("Model evaluation on test data shows an Accuracy = 0.9687"))

