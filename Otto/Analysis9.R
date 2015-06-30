
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

library(randomForest)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

output <- data.frame(id = data.output[,1])
#output <- readRDS("output.rds")

  di <- data.input

  rf <- randomForest(target ~ .-id, data = di)
  rfPredict <- predict(rf, newdata = data.output, type="prob")

  rfPredict.matrix <- as.data.frame(rfPredict)
  output <- cbind(output,rfPredict.matrix)

head(output, 10)

write.csv(output, file = "submission_09.csv", quote=FALSE, row.names=FALSE)

# Once again the best score! 0.56396

#rfPredict Class_0 Class_1
#  Class_0   15011     347
#  Class_1      11     101