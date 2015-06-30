
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

set.seed(123)
train_ind <- sample(seq_len(nrow(data.input)), size = floor(0.75 * nrow(data.input)))


i <- 2

di <- data.input
di$target <- as.character(di$target)
di$target[di$target != paste('Class_',i,sep='')] <- 'Class_0'
di$target <- as.factor(di$target)

train <- di[train_ind,]
test <- di[-train_ind,]

library(adabag)

adabag <- bagging(target ~ .-id, data = train, mfinal = 10, control = rpart.control(maxdepth = 1))
adabagPredict <- predict.bagging(adabag, newdata = test)

table(adabagPredict$class, test$target)
t <- table(adabagPredict$class, test$target)
(t[1,2] + t[2,1]) / (t[1,1] + t[2,2])


# Class_0 Class_1
# Class_0   11775     434
# Class_2    3247      14
# 
# Misclassification: 0.3122402
# 
# Class_0 Class_2
# Class_0   10086    2123
# Class_2    1426    1835
# 
# Misclassification: 0.2977099


library(randomForest)

rf <- randomForest(target ~ .-id, data = train)
rfPredict <- predict(rf, newdata = test)

table(rfPredict, test$target)
t <- table(rfPredict, test$target)
(t[1,2] + t[2,1]) / (t[1,1] + t[2,2])

# rfPredict Class_0 Class_1
# Class_0   15011     347
# Class_1      11     101
# 
# Misclassification: 0.02368978
# 
# rfPredict Class_0 Class_2
# Class_0   10674     944
# Class_2     838    3014
# 
# Misclassification: 0.130187



library(caret)
library(mlbench)
library(pROC)

ctrl <- trainControl(method = "cv", 
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE)
gbmTune <- train(target ~ .-id, data = train,
                 method = "gbm",
                 metric = "ROC",
                 verbose = FALSE,                    
                 trControl = ctrl)
gbmPredict <- predict(gbmTune, newdata = test)

table(gbmPredict, test$target)
t <- table(gbmPredict, test$target)
(t[1,2] + t[2,1]) / (t[1,1] + t[2,2])

# gbmPredict Class_0 Class_1
# Class_0   14978     361
# Class_1      44      87
#
# Misclassification: 0.0268835
#
# gbmPredict Class_0 Class_2
# Class_0   10379    1248
# Class_2    1133    2710
# 
# Misclassification: 0.1819085
#


library(nnet)

ideal <- class.ind(train$target)
  
ann <- nnet(train[, c(-1,-95)], ideal, size=10, softmax = TRUE, maxit=1000)
annPredict <- predict(ann, test[,c(-1,-95)], type="class")
  
table(annPredict, test$target)
t <- table(annPredict, test$target)
(t[1,2] + t[2,1]) / (t[1,1] + t[2,2])

#annPredict Class_0 Class_1
#Class_0   14863     265
#Class_1     159     183
#
# Misclassification: 0.02818025
#
#annPredict Class_0 Class_2
#Class_0   10348    1059
#Class_2    1164    2899
#
# Misclassification: 0.1678116

