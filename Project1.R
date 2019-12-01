
#Clear the environment 
rm(list=ls(all=TRUE))

setwd("D:/INSOFE/CUTe 2")

#Read the input data that is given

income_data<-read.csv("train_data.csv",header = T)

test_data<-read.csv("test_data.csv",header = T)

#Use head() and tail() functions to get a feel of the data

head(income_data)

tail(income_data)

#Check the structure of the input data

str(income_data)

str(test_data)

#Check the distribution of the input data using the summary function

summary(income_data)

summary(test_data)

#tax_paid column has NA's for more than 80% of the rows so drop that column

income_data<-income_data[,-c(7)]

test_data<-test_data[,-c(7)]

str(income_data)

str(test_data)

#train_data has target attribute(which is the dependant variable) and test_data 
#doesn't have that attribute.We are going to build the model based on independant 
#attributes , group the attributes into numerical attributes and categorical attributes 
#excluding the target variable 

income_data_mod<-income_data[,-c(17)]

#Numerical Attributes - "index","age","financial_weight","years_of_education","gain","loss","working_hours"

#Categorical Attributes - "working_sector","qualification","loan_taken","marital_status","occupation",
#                         "relationship","ethnicity","gender","country"

num_Attr<-c("index","age","financial_weight","years_of_education","gain","loss","working_hours")

cat_Attr<-setdiff(x = colnames(income_data_mod), y = num_Attr)

income_data_cat <- subset(income_data_mod,select =cat_Attr)

income_data[,cat_Attr] <- data.frame(apply(income_data_cat, 2, function(x) as.factor(as.character(x))))

income_data_cat<-income_data[,cat_Attr] 

income_data_num<-income_data[,num_Attr] 

test_data_cat <- subset(test_data,select =cat_Attr)

test_data[,cat_Attr] <- data.frame(apply(test_data_cat, 2, function(x) as.factor(as.character(x))))

test_data_cat<-test_data[,cat_Attr] 

test_data_num<-test_data[,num_Attr] 

#Impute the data for the missing values
# centralImputation on categorical attributes and knnImputation on numerical attributes

library(DMwR)

income_cat_imputed <- centralImputation(data = income_data_cat)

income_num_imputed <- knnImputation(data = income_data_num,k=5)

income_data_final<-cbind(income_cat_imputed,income_num_imputed,target=income_data$target)

sum(is.na(income_data_final))

test_cat_imputed<-centralImputation(data = test_data_cat)

test_num_imputed<-knnImputation(data = test_data_num,k=5)

test_data_final<-cbind(test_cat_imputed,test_num_imputed)

sum(is.na(test_data_final))

#Split the income_data into train and validation sets

library(caret)

set.seed(9999)

train_rows <- createDataPartition(y = income_data_final$target, p = 0.7, list = F) 

train <- income_data_final[train_rows, ]

validation <- income_data_final[-train_rows, ]

#Building a logistic regression model with target as the dependant variable

log_reg<-glm(target ~ .-index,family = binomial(link='logit'), data = train)

summary(log_reg)

#Building a model to remove the insignificant features using stepwise regression
library(MASS)

aic_model <- stepAIC(object = log_reg, direction = "both")

summary(aic_model)

#By doing stepAIC loan_taken variable is removed

#Studying the probability predications and selecting a threshold value for the final classification 
library(ROCR)

prob_train <- predict(aic_model, type = "response")

pred <- prediction(prob_train, train$target)

perf <- performance(pred, measure="tpr", x.measure="fpr")

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

#Checking the area under the curve

perf_auc <- performance(pred, measure="auc")

auc <- perf_auc@y.values[[1]]

print(auc)


#Classifying the elements of train/validation data to positive and negative using the selected threshold value

prob_train <- predict(aic_model, train , type = "response")

preds_train <- ifelse(prob_train > 0.45, 1,0)

prob_validation <- predict(aic_model, validation , type = "response")

preds_validation <- ifelse(prob_validation > 0.45, 1,0)


##Check the confusion matrix for train and validation

library(caret)

confusionMatrix(preds_train, train$target, positive = "1")

confusionMatrix(preds_validation, validation$target, positive = "1")


##########################################################
#Predict the target variable for the test_data.csv using the logistic regression model that is built

prob_test_data <- predict(aic_model, test_data_final , type = "response")

preds_test_data <- ifelse(prob_test_data > 0.45, 1,0)

output<-data.frame(test_data$index,preds_test_data)

colnames(output)<-c("index","target")

write.csv(output,file="Samplesubmission.csv",row.names = F)

boxplot(age~target, data = income_data_final, xlab ="Income", ylab = "Age", main = "Age v/s Income")

barplot(table(income_data_final$qualification))

barplot(table(income_data_final$ethnicity))

barplot(table(income_data_final$marital_status))
