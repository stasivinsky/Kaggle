
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

di.id <- data.input$id
di.class <- data.input$target
di <- data.input[,-c(1,95)]

do.id <- data.output$id
do <- data.output[,-1]

d <- rbind(di, do)

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

rfPredict.matrix <- as.data.frame(rfPredict)
output <- cbind(do.id,rfPredict.matrix)
names(output)[1] <- "id"

head(output, 10)

max_vector <- as.vector(apply(output[,-1],1,max))

for (i in 1:9) {
  output[,i+1] <- output[,i+1] == max_vector
}

output[output == FALSE] <- 0
output[output == TRUE] <- 1

write.csv(output, file = "submission_randomforest_02.csv", quote=FALSE, row.names=FALSE)

