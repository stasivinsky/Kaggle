
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

library(randomForest)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

output <- data.frame(id = data.output[,1])
#output <- readRDS("output.rds")

for (i in 1:9) {
  di <- data.input
  di$target <- as.character(di$target)
  di$target[di$target != paste('Class_',i,sep='')] <- 'Class_0'
  di$target <- as.factor(di$target)

  rf <- randomForest(target ~ .-id, data = di)
  rfPredict <- predict(rf, newdata = data.output, type="prob")
  
  output <- cbind(output,rfPredict[,2])
  names(output)[i+1] <- paste('Class_',i,sep='')

  saveRDS(output, file="output.rds")
}


head(output, 10)

write.csv(output, file = "submission_06.csv", quote=FALSE, row.names=FALSE)

# Once again the best score! 0.56396

#rfPredict Class_0 Class_1
#  Class_0   15011     347
#  Class_1      11     101