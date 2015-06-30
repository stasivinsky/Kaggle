
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

table(data.input$target)

data.input.1 <- data.input
data.input.1$target <- as.character(data.input.1$target)
data.input.1$target[data.input.1$target != 'Class_1'] <- 'Class_0'
data.input.1$target <- as.factor(data.input.1$target)

#data.input.1[data.input.1 == 0] <- NA

for (i in 2:94) {
  data.input.1[,i] <- as.factor(data.input.1[,i])
}

set.seed(123)
train_ind <- sample(seq_len(nrow(data.input.1)), size = floor(0.75 * nrow(data.input.1)))

train <- data.input.1[train_ind,]
test <- data.input.1[-train_ind,]

glm <- glm(target ~ feat_1, data=data.input.1, family="binomial")
pred <- predict(model, data.input.1, type = "response")
contrasts(data.input.1$target)
glm.pred=rep ("Class_0" ,nrow(data.input.1))
glm.pred[pred > 0.5] <- "Class_1"
table(glm.pred, data.input.1$target)


library(boot)
k    <- 3
kfCV <- cv.glm(data=data.input.1, glmfit=glm, K=k)
kfCV$delta
summary(glm$fitted.values[1:10])

table(glm$fitted.values,
pred <- predict(model, data.input.1, type = "response")
table(pred,data.input.1$target)

for (i in 1:2) {
  print(paste("feat_",i,sep=""))
  model <- glm(as.formula(paste("target ~ feat_",i,sep="")), data = train, family = "binomial")  
  pred <- predict(model, test$target)
  table(pred, test$target)
}

i <- 1
model <- glm(as.formula(paste("target ~ feat_",i,sep="")), data = train, family = "binomial")  

print(model)
summary(model)
pred <- predict(model, test)
table(pred, test$target)

summary(test$feat_2)



library(stats)
pc.cr <- princomp(data.input.1[,2:94], na.action=na.pass)

summary(pc.cr)
loadings(pc.cr)

library(lattice)
pc.cr$scores
pca.plot <- xyplot(pc.cr$scores[,2] ~ pc.cr$scores[,1])
pca.plot$xlab <- "First Component"
pca.plot$ylab <- "Second Component"
pca.plot

class1 <- data.input.1[data.input.1$target == 'Class_1',]


plot(data.input.1$target ~ data.input.1$feat_1)


#Rank features by importance

# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(target~.-id, data=data.input.1, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)
