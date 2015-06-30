
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

library(caret)
library(mlbench)
library(pROC)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

#output <- data.frame(id = data.output[,1])
output <- readRDS("output.rds")

for (i in 7:9) {
  di <- data.input
  di$target <- as.character(di$target)
  di$target[di$target != paste('Class_',i,sep='')] <- 'Class_0'
  di$target <- as.factor(di$target)

  ctrl <- trainControl(method = "cv", 
                       summaryFunction = twoClassSummary, 
                       classProbs = TRUE)
  gbmTune <- train(target ~ .-id, data = di,
                   method = "gbm",
                   metric = "ROC",
                   verbose = FALSE,                    
                   trControl = ctrl)
  gbmPredict <- predict(gbmTune, newdata = data.output, type="prob")
  
  output <- cbind(output,gbmPredict[,2])
  names(output)[i+1] <- paste('Class_',i,sep='')
}

saveRDS(output, file="output.rds")

write.csv(output, file = "submission_05.csv", quote=FALSE, row.names=FALSE)

# Best score so far! 0.67608

#gbmPredict Class_0 Class_1
#   Class_0   14985     353
#   Class_1      37      95
