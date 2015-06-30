
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

di.id <- data.input$id
di.class <- data.input$target
di <- data.input[,-c(1,95)]

library(e1071)

skewValues <- apply(di,2,skewness)
head(skewValues)

library(caret)

di <- di[,-nearZeroVar(data.output)]

correlations <- cor(di)
di <- di[,-findCorrelation(correlations, cutoff = .75)]

trans <- preProcess(di, method=c("BoxCox", "center", "scale", "pca"))
di.trans <- predict(trans, di)

set.seed(123)
train_ind <- sample(seq_len(nrow(di.trans)), size = floor(0.75 * nrow(di.trans)))

train <- di.trans[train_ind,]
test <- di.trans[-train_ind,]

train.class <- di.class[train_ind]
test.class <- di.class[-train_ind]

library(randomForest)

rf <- randomForest(train.class ~ ., data = train)
rfPredict <- predict(rf, newdata = test)

table(rfPredict, test.class)
