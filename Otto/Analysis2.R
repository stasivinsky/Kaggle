
library(class)

HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

# Find highly correlated features
set.seed(7)
install.packages("mlbench")
library(mlbench)
install.packages("caret")
library(caret)

correlationMatrix <- cor(data.input[,2:94])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
print(highlyCorrelated)

# OUtput: [1]  3 15 39
# These features can be removed as redundant

#Rank Features by importance

library(mlbench)
library(caret)
library(e1071)

set.seed(7)
control <- trainControl(method="repeatedcv", number=10, repeats=3)
model <- train(target~.-id, data=data.input, method="lvq", preProcess="scale", trControl=control)
importance <- varImp(model, scale=FALSE)

# Low performance. Did not get back any results in reasonable time

library(caret)
install.packages("pls")
library(pls)
install.packages("klaR")
library(klaR)


fit <- plsda(data.input[,2:94], data.input[,95], probMethod="Bayes")
predictions <- predict(fit, data.input[,2:94])

table(predictions, data.input$target)

# Low performance

library(MASS)

# fit model
fit <- lda(target~., data=data.input[,-1])
# summarize the fit
summary(fit)
# make predictions
predictions <- predict(fit, data.input[,2:94])$class
# summarize accuracy
table(predictions, data.input$target)

# predictions Class_1 Class_2 Class_3 Class_4 Class_5 Class_6 Class_7 Class_8 Class_9
# Class_1     952      45       6       3       0     231     152     435     498
# Class_2     433   13375    4904    1931     485     967     840     831     673
# Class_3       2    2384    2807     375      10      46     195      62       1
# Class_4       0      78      66     271       0      24      12       0       3
# Class_5       0     116      50      13    2239       1       3       2       1
# Class_6      52       9       1      40       0   12222      77     139      95
# Class_7      22      77     149      52       0     157    1392      58      23
# Class_8     223      33      17       5       5     291     163    6842     172
# Class_9     245       5       4       1       0     196       5      95    3489

predictions <- predict(fit, data.output[,2:94])$class

output <- data.frame(data.output[1],predictions, predictions, predictions, predictions, predictions, predictions, predictions, predictions, predictions)
names(output) <- c('id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9')
output2 <- output

output <- output2

output.data <- output[,-1]
output.data[] <- lapply(output.data, as.character)
output.data[output.data$Class_1 == 'Class_1',1] <- '1'
output.data[output.data$Class_1 != '1',1] <- '0'
output.data[output.data$Class_2 == 'Class_2',2] <- '1'
output.data[output.data$Class_2 != '1',2] <- '0'
output.data[output.data$Class_3 == 'Class_3',3] <- '1'
output.data[output.data$Class_3 != '1',3] <- '0'
output.data[output.data$Class_4 == 'Class_4',4] <- '1'
output.data[output.data$Class_4 != '1',4] <- '0'
output.data[output.data$Class_5 == 'Class_5',5] <- '1'
output.data[output.data$Class_5 != '1',5] <- '0'
output.data[output.data$Class_6 == 'Class_6',6] <- '1'
output.data[output.data$Class_6 != '1',6] <- '0'
output.data[output.data$Class_7 == 'Class_7',7] <- '1'
output.data[output.data$Class_7 != '1',7] <- '0'
output.data[output.data$Class_8 == 'Class_8',8] <- '1'
output.data[output.data$Class_8 != '1',8] <- '0'
output.data[output.data$Class_9 == 'Class_9',9] <- '1'
output.data[output.data$Class_9 != '1',9] <- '0'

output.data[] <- lapply(output.data, as.numeric)
output <- data.frame(output[,1],output.data)
names(output)[1] <- 'id'

write.csv(output, file = "submission_03.csv", quote=FALSE, row.names=FALSE)

# Bad result. Performance is extremely poor
