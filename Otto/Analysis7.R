
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

library(kernlab)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

output <- data.frame(id = data.output[,1])
#output <- readRDS("output.rds")

for (i in 1:9) {
  di <- data.input
  di$target <- as.character(di$target)
  di$target[di$target != paste('Class_',i,sep='')] <- 'Class_0'
  di$target <- as.factor(di$target)

  rbf <- rbfdot(sigma=0.1)
  svm <- ksvm(target ~ .-id, data = di,type="C-bsvc",kernel=rbf,C=10, prob.model=TRUE)
  svmPredict <- predict(svm, newdata = data.output, type="probabilities")
  
  output <- cbind(output,svmPredict[,2])
  names(output)[i+1] <- paste('Class_',i,sep='')

  saveRDS(output, file="output.rds")
}

head(output, 10)

write.csv(output, file = "submission_07.csv", quote=FALSE, row.names=FALSE)

# Not so good! Result: 0.76870


