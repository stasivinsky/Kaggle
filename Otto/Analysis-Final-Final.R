
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

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

install.packages("h2o")
library(h2o)

localH2O <- h2o.init(nthread=4,Xmx="4g")

train <- read.csv("train.csv")

for(i in 2:94){
  train[,i] <- as.numeric(train[,i])
  train[,i] <- sqrt(train[,i]+(3/8))
}


test <- read.csv("test.csv")

for(i in 2:94){
  test[,i] <- as.numeric(test[,i])
  test[,i] <- sqrt(test[,i]+(3/8))
}



train.hex <- as.h2o(localH2O,train)
test.hex <- as.h2o(localH2O,test[,2:94])

predictors <- 2:(ncol(train.hex)-1)
response <- ncol(train.hex)

submission <- read.csv("sampleSubmission.csv")
submission[,2:10] <- 0

for(i in 1:20){
  print(i)
  model <- h2o.deeplearning(x=predictors,
                            y=response,
                            data=train.hex,
                            classification=T,
                            activation="RectifierWithDropout",
                            hidden=c(1024,512,256),
                            hidden_dropout_ratio=c(0.5,0.5,0.5),
                            input_dropout_ratio=0.05,
                            epochs=50,
                            l1=1e-5,
                            l2=1e-5,
                            rho=0.99,
                            epsilon=1e-8,
                            train_samples_per_iteration=2000,
                            max_w2=10,
                            seed=1)
  submission[,2:10] <- submission[,2:10] + as.data.frame(h2o.predict(model,test.hex))[,2:10]
  print(i)
  write.csv(submission,file="submission_h2o_0517_1.csv",row.names=FALSE) 
}      


library(caret)
require(devtools)
#devtools::install_github('dmlc/xgboost',subdir='R-package')
library(xgboost)

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
              "nthread" = 8,
              "eta" = 0.05,
              "max_depth" = 6,
              "gamma" = 0.5)

# Train the model
nround = 1500
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

write.csv(pred,file='submission_xgb_0514_1.csv', quote=FALSE,row.names=FALSE)


output <- cbind(submission[,1],(submission[,-1] + pred[,-1])/2.0)
names(output)[1] <- "id"
write.csv(output,file='submission_xgb_h2o_0517_2.csv', quote=FALSE,row.names=FALSE)


load("save_dump.rda")
output_rf <- data.frame(id = data.output[,1])
output_rf <- cbind(output_rf,rfPredict)
output_nn <- data.frame(id = data.output[,1])
output_nn <- cbind(output_nn,nnPredict)
output_c50 <- data.frame(id = data.output[,1])
output_c50 <- cbind(output_c50,c50Predict)

output <- cbind(output_c50[,1],(output_rf[,-1] + pred[,-1])/2.0)
names(output)[1] <- "id"

write.csv(output,file='submission_xgb_rf_0512_2.csv', quote=FALSE,row.names=FALSE)

output <- cbind(output_c50[,1],(3*output_rf[,-1] + 5*pred[,-1] + output_nn[,-1] + output_c50[,-1])/10.0)
names(output)[1] <- "id"

write.csv(output,file='submission_xgb_rf_nn_c50_0512_3.csv', quote=FALSE,row.names=FALSE)


threshold <- 0.1
output <- (nnPredict + rfPredict * 2 + c50Predict) / 3
print (calculateLogloss(output))

save(rfFit, rfPredict, nnFit, glmnFit, nnPredict, c50Fit, c50Predict, glmnPredict, file="save_final.rda")
load("save_final.rda")