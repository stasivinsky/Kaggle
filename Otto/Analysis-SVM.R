
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

library(caret)

set.seed(1056)

system.time( rfFit <- randomForest(target ~ .-id, data = data.input, strata = data.input$target, sampsize=1000))


trControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 2,
  classProbs = TRUE)

system.time (rfFit <- train(target ~ .-id, 
               data = data.input,
               method = "rf",
               tuneLength = 4,
#               strata = data.input$target,
#               sampsize = 1000,
               ntree = 1000,
               trControl = trControl))

rfPredict <- predict(rfFit, newdata = data.output, type="prob")

output_rf <- data.frame(id = data.output[,1])
output_rf <- cbind(output_rf,rfPredict)

write.csv(output_rf, file = "submission_randomforest_0527_1.csv", quote=FALSE, row.names=FALSE)


library(nnet)

set.seed(1056)

nnetGrid <- expand.grid(.size = 1:8,
                        .decay = c(0, 0.1, 1,2))
maxSize <- max(nnetGrid$.size)
numWts <- maxSize * 93

system.time( nnFit <- nnet(target ~ .-id, data = data.input, size = 5))

system.time (nnFit <- train(target ~ .-id, 
                            data = data.input,
                            method = "nnet",
                            metric="ROC",
                            preProc=c("center","scale","spatialSign"),
                            tuneGrid = nnetGrid,
                            maxit = 2000,
                            MaxNWts = numWts,
                            trControl = trControl))

nnPredict <- predict(nnFit, newdata = data.output, type="prob")

output_nn <- data.frame(id = data.output[,1])
output_nn <- cbind(output_nn,nnPredict)

write.csv(output_nn, file = "submission_nnet_0528_1.csv", quote=FALSE, row.names=FALSE)


library(C50)

set.seed(1056)

tuneGrid <- expand.grid(.trials = c((10:10)*10), 
                        .model = "tree",
                        .winnow = FALSE)

set.seed(1056)
system.time (c50Fit <- train(data.input[,-c(1,95)], data.input$target,
                            method = "C5.0",
                            metric="ROC",
                            tuneGrid = tuneGrid,
                            prob.model = TRUE,
                            trControl = trControl))

c50Predict <- predict(c50Fit, newdata = data.output[,-1], type="prob")

output_c50 <- data.frame(id = data.output[,1])
output_c50 <- cbind(output_c50,c50Predict)

write.csv(output_c50, file = "submission_c50_0429_2.csv", quote=FALSE, row.names=FALSE)




library(deepnet)

set.seed(1056)

dnnGrid <- expand.grid(.layer1 = 9:9,
                       .layer2 = 4:5,
                       .layer3 = 5:10,
                       .hidden_dropout = 0,
                       .visible_dropout = 0)

# fix for caret
class2ind <- function(cl)
{
  n <- length(cl)
  cl <- as.factor(cl)
  x <- matrix(0, n, length(levels(cl)) )
  x[(1:n) + n*(unclass(cl)-1)] <- 1
  dimnames(x) <- list(names(cl), levels(cl))
  x
}

set.seed(1056)

system.time (dnnFit <- train(data.input[,-c(1,95)], data.input$target,
                             method = "dnn",
                             preProcess = c("BoxCox","center","scale"),
                             tuneGrid = dnnGrid,
                             trControl = trControl))


library(gbm)

set.seed(1056)


gbmGrid <- expand.grid(.interaction.depth = 3,
                       .n.trees = c(150),
                       .shrinkage = 0.1)

gbmFit <- train(target ~ .-id, 
                data = data.input,
                method = "gbm",
                preProc=c("center","scale"),
                tuneGrid = gbmGrid,
                trControl = trControl)

gbmPredict <- predict(gbmFit, newdata = data.output, type="prob")

output_gbm <- data.frame(id = data.output[,1])
output_gbm <- cbind(output_gbm,gbmPredict)

write.csv(output_gbm, file = "submission_gbm_0501_1.csv", quote=FALSE, row.names=FALSE)

install.packages("RWeka")
library(RWeka)

wekaGrid <- expand.grid(.committees = seq(10, 100, 10), 
                       .neighbors = 0)
set.seed(1056)
wekaFit <- train(x = data.input[,-c(1,95)], y = data.input$target,
                method = "Weka",
                tunelength = 2,
                trControl = trControl)





allResamples <- resamples(list("rf" = rfFit, "nn" = nnFit, "c50" = c50Fit, "gbm" = gbmFit))

summary(allResamples)



# The best model so far!

output <- cbind(output_c50[,1],(3*output_rf[,-1] + output_nn[,-1] + output_c50[,-1])/5.0)
names(output)[1] <- "id"

write.csv(output, file = "submission_3rf_nn_c50_0502_3.csv", quote=FALSE, row.names=FALSE)





actual.p <- colSums(output[,2:10])/nrow(output)

totals <- table (data.input$target)
theoretical.p <- as.data.frame.matrix(t(totals / sum(totals)))

coeff <- theoretical.p / actual.p

output.adj <- data.frame(id = output[,1])
for (i in 1:9) {
  output.adj <- cbind(output.adj, output[,i+1]*coeff[[i]])
}
names(output.adj) <- c("id","Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9")

head(output,5)
head(output.adj,5)

write.csv(output.adj, file = "submission_rf_gbm_c50_calibrated_0502_2.csv", quote=FALSE, row.names=FALSE)











rfPredict2 <- predict(rfFit, newdata = data.input[,-95], type="prob")
new.input <- data.frame(id = data.input[,1], rfPredict2, target = data.input[,95])
new.output <- data.frame(id = data.output[,1], rfPredict)
  
library(nnet)

set.seed(1056)

nnetGrid <- expand.grid(.size = 1:8,
                        .decay = c(0, 0.1, 1,2))
maxSize <- max(nnetGrid$.size)
numWts <- maxSize * 9

nnFit2 <- train(target ~ .-id, 
                data = new.input,
                method = "nnet",
                metric="ROC",
                preProc=c("center","scale","spatialSign"),
                tuneGrid = nnetGrid,
                maxit = 2000,
                MaxNWts = numWts,
                trControl = trControl)

nnPredict2 <- predict(nnFit2, newdata = new.output, type="prob")

output_nn2 <- data.frame(id = data.output[,1])
output_nn2 <- cbind(output_nn2,nnPredict2)

write.csv(output_nn2, file = "submission_fp_nnet_0430_1.csv", quote=FALSE, row.names=FALSE)


data.input2 <- data.frame(data.input[,2:94] > 0, stringsAsFactors = TRUE)
data.input2[] <- lapply(data.input2,factor)
data.output2 <- data.frame(data.output[,2:94] > 0, stringsAsFactors = TRUE)
data.output2[] <- lapply(data.output2,factor)
n <- paste(names(data.input)[2:94], "_f", sep="")
names(data.input2) <- n
names(data.output2) <- n

data.input2 <- cbind(data.input[,-95], data.input2, target = data.input$target)
data.output2 <- cbind(data.output[,-95], data.output2)

library(randomForest)

set.seed(1056)

system.time( rfFit2 <- randomForest(target ~ .-id, data = data.input2, strata = data.input2$target, sampsize=1000))

trControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 2,
  classProbs = TRUE)

rfFit2 <- train(target ~ .-id, 
                data = data.input2,
                method = "rf",
                tuneLength = 4,
                ntree = 500,
                trControl = trControl)

rfPredict2 <- predict(rfFit2, newdata = data.output2, type="prob")

output_rf2 <- data.frame(id = data.output2[,1])
output_rf2 <- cbind(output_rf2,rfPredict2)

write.csv(output_rf, file = "submission_randomforest_0527_1.csv", quote=FALSE, row.names=FALSE)



rfGrid <- expand.grid(.mtry = 62)
set.seed(1056)
rfFit3 <- train(data.input[,-c(1,95)], data.input$target,
                method = "rf",
                tuneGrid = rfGrid,
#                strata = data.input$target,
#                sampsize = 1000,
                ntree = 1000)

rfPredict3 <- predict(rfFit3, newdata = data.output[,-1], type="prob")

output_rf3 <- data.frame(id = data.output[,1])
output_rf3 <- cbind(output_rf3,rfPredict3)

write.csv(output_rf, file = "submission_randomforest_0502_1.csv", quote=FALSE, row.names=FALSE)





library(caret)

nearZeroVar <- nearZeroVar(d)
d <- d[,-nearZeroVar]

correlations <- cor(d)
highCorrelations <- findCorrelation(correlations, cutoff = .75)
d <- d[,-highCorrelations]

trans <- preProcess(d, method=c("BoxCox", "center", "scale", "pca"))
d.trans <- predict(trans, d)

library(randomForest)

rf <- randomForest(di.class ~ ., data = d.trans[1:nrow(di),])
rfPredict <- predict(rf, newdata = d.trans[(nrow(di)+1):nrow(d.trans),], type="prob")


save(nnPredict, rfPredict, c50Predict, gbmPredict, nnFit, rfFit, c50Fit, gbmFit, file="save_dump.rda")
load("save_dump.rda")