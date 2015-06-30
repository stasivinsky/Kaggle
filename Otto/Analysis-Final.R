
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

di <- data.input[,-c(1,95)]
do <- data.output[,-1]

library(caret)
library(e1071)

trans.i <- preProcess(di, method=c("BoxCox", "center", "scale"))
data.input <- data.frame(data.input[,1], predict(trans.i, di), data.input[,95])
names(data.input)[c(1,95)] <- c('id','target')

trans.o <- preProcess(do, method=c("scale"))
data.output <- data.frame(data.output[,1],predict(trans.o, do))
names(data.output[1]) <- 'id'

sample_size <- floor(0.75 * nrow(data.input))
 
set.seed(123)
train_ind <- sample(seq_len(nrow(data.input)), size = sample_size)
 
train <- data.input[train_ind,-1]
test <- data.input[-train_ind,-1]

calculateLogloss <- function(x){
  logloss = 0
  for (i in 1:nrow(test)) {
    x.norm <- x[i,] / sum(x[i,])
    k = as.integer(substr(test[i,94],7,7))
    logloss = logloss + log(max(min(x.norm[k],1-(10^-15)),10^-15 ))
  }
  logloss = - logloss / nrow(test)
  return(logloss)
}


library(caret)
library(randomForest)

trControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 1,
  classProbs = TRUE)

frGrid <- expand.grid(.mtry=c(40,50,60))

set.seed(1056)
rfFit <- train(train[,-94],train$target,
               method="rf",
               tuneGrid=frGrid,
               ntree=1000,
               trControl=trControl)

rfPredict <- predict(rfFit, newdata = test[,-94], type="prob")

print (calculateLogloss(rfPredict))


# 0.5634538
# mtry = 40, Accuracy = 0.7952941



library(nnet)

trControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 1,
  preProcOptions = list(thresh = 0.95),
  classProbs = TRUE)

nnetGrid <- expand.grid(.size = 6:8,
                        .decay = c(0.1, 1))
#maxSize <- max(nnetGrid$.size)
#numWts <- maxSize * 93

set.seed(1056)
nnFit <- train(train[,-94],train$target,
               method = "nnet",
               metric="Accuracy",
               preProc=c("BoxCox","center","scale","pca"),
               tuneGrid = nnetGrid,
               maxit = 2000,
#               MaxNWts = numWts,
               trControl = trControl)

nnPredict <- predict(nnFit, newdata = test[,-94], type="prob")

print (calculateLogloss(nnPredict))

# 0.6047084
# size = 8, decay = 1, Accuracy = 0.7676264


library(C50)

tuneGrid <- expand.grid(.trials = c((10:10)*10), 
                        .model = "tree",
                        .winnow = FALSE)

set.seed(1056)
c50Fit <- train(train[,-94],train$target,
                             method = "C5.0",
                             metric="Accuracy",
                             tuneGrid = tuneGrid,
                             prob.model = TRUE,
                             trControl = trControl)

c50Predict <- predict(c50Fit, newdata = test[,-94], type="prob")

print (calculateLogloss(c50Predict))

# 0.7191452
# trials = 100, model = tree, winnow = FALSE, Accuracy = 0.7866965



trControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 1,
#  preProcOptions = list(thresh = 0.95),
  classProbs = TRUE)

tuneGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length=40))

set.seed(1056)
glmnFit <- train(train[,-94],train$target,
                method = "glmnet",
                metric="ROC",
                preProc = c("center","scale"),
                tuneGrid = tuneGrid,
#                prob.model = TRUE,
                trControl = trControl)

glmnPredict <- predict(glmnFit, newdata = test[,-94], type="prob")

print (calculateLogloss(glmnPredict))

# 0.758382
# Alpha = 0.1, lambda = 0.01


#install.packages("devtools")
require(devtools)
#devtools::install_github('dmlc/xgboost',subdir='R-package')
library(xgboost)

train.xgb = train
test.xgb = test

y = train.xgb[,ncol(train.xgb)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train.xgb[,-ncol(train.xgb)],test.xgb[,-ncol(test.xgb)])
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train.xgb)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8,
              "eta" = 0.05)

# Run Cross Valication
cv.nround = 50
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

res_min <- 1.0
for (nround in seq(1500,1500,by=500)) {
  for (eta in c(0.05)) {
    # Set necessary parameter
    param <- list("objective" = "multi:softprob",
                  "eval_metric" = "mlogloss",
                  "num_class" = 9,
                  "nthread" = 8,
                  "eta" = eta,
                  "max_depth" = 6,
                  "gamma" = 0.5)
    
    # Train the model
    bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround, verbose = 0)
    
    # Make prediction
    pred = predict(bst,x[teind,])
    pred = matrix(pred,9,length(pred)/9)
    pred = t(pred)
    
    # Output submission
    #pred = format(pred, digits=2,scientific=F) # shrink the size of submission
    pred = data.frame(pred)
    #pred[] <- lapply(pred,as.numeric)
    names(pred) = paste0('Class_',1:9)
    
    res <- calculateLogloss(pred)
    if (res < res_min ) {
      eta_min <- eta
      nround_min <- nround
      res_min <- res
    }
    print(paste(nround, eta, res))
    
  }
}
print (paste("Minimum:", nround_min, eta_min, res_min))


data.input.xgb = data.input[,-1]
data.output.xgb = data.output[,-1]

y = data.input.xgb[,ncol(data.input.xgb)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(data.input.xgb[,-ncol(data.input.xgb)],data.output.xgb)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(data.input.xgb)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Run Cross Valication
cv.nround = 50
bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 250
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
#pred[] <- lapply(pred,as.numeric)
#pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))

write.csv(pred,file='submission_xgb_0511_2.csv', quote=FALSE,row.names=FALSE)

load("save_dump.rda")
output_rf <- data.frame(id = data.output[,1])
output_rf <- cbind(output_rf,rfPredict)
output_nn <- data.frame(id = data.output[,1])
output_nn <- cbind(output_nn,nnPredict)
output_c50 <- data.frame(id = data.output[,1])
output_c50 <- cbind(output_c50,c50Predict)

output <- cbind(output_c50[,1],(3*output_rf[,-1] + 3*pred[,-1] + output_nn[,-1] + output_c50[,-1])/8.0)
names(output)[1] <- "id"

write.csv(output,file='submission_xgb_rf_nn_c50_0511_3.csv', quote=FALSE,row.names=FALSE)



threshold <- 0.1
output <- (nnPredict + rfPredict * 2 + c50Predict) / 3
print (calculateLogloss(output))

save(rfFit, rfPredict, nnFit, glmnFit, nnPredict, c50Fit, c50Predict, glmnPredict, file="save_final.rda")
load("save_final.rda")