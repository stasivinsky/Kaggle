
HOMEDIR <- "C:\\Work\\GitHub\\Kaggle\\CrowdFlower"
setwd(HOMEDIR)

library(readr)

data.train <- read_csv("data\\train.csv")
data.test <- read_csv("data\\test.csv")

library(tm)
library(SnowballC)

clean_column <- function(v) {
  corpus <- Corpus(VectorSource(c(v)))
  
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)

  return(unlist(sapply(corpus, '[', "content")))
}

data.train$query <- clean_column(data.train$query)
data.train$product_title <- clean_column(data.train$product_title)
data.train$product_description <- clean_column(data.train$product_description)

data.test$query <- clean_column(data.test$query)
data.test$product_title <- clean_column(data.test$product_title)
data.test$product_description <- clean_column(data.test$product_description)


count_matches <- function(query, text) {
  matches <- 0
  
  for (word.q in unlist(strsplit(query, split=" "))) {
    for (word.t in unlist(strsplit(text, split=" "))) {
      if ( word.q == word.t) {
        matches <- matches + 1
        break
      }
    }
  }
  return (matches / length(unlist(strsplit(query, split=" "))))
}

data.tr <- cbind(data.train,
                 matches_title = apply(data.train[,c('query','product_title')], 1, function(x) count_matches(x[1],x[2])),
                 matches_desc = apply(data.train[,c('query','product_description')], 1, function(x) count_matches(x[1],x[2])))
data.tr$median_relevance <- as.factor(data.tr$median_relevance)

data.tst <- cbind(data.test,
                  matches_title = apply(data.test[,c('query','product_title')], 1, function(x) count_matches(x[1],x[2])),
                  matches_desc = apply(data.test[,c('query','product_description')], 1, function(x) count_matches(x[1],x[2])))

features <- c('matches_title','matches_desc')


sample_size <- floor(0.75 * nrow(data.tr))

set.seed(123)
train_ind <- sample(seq_len(nrow(data.tr)), size = sample_size)

train <- data.tr[train_ind,-1]
test <- data.tr[-train_ind,-1]


library(caret)
library(randomForest)

trControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2)

set.seed(1056)
rfFit <- train(train[,features],train$median_relevance,
               method="rf",
               tunelength=2,
               ntree=1000,
               trControl=trControl)

rfPredict <- predict(rfFit, newdata = test[,features])

table(as.numeric(rfPredict),as.numeric(test$median_relevance))


pred <- as.numeric(rfPredict)
output <- data.frame(data.tr[-train_ind,1],prediction = pred)

library(Metrics)
kappa <- ScoreQuadraticWeightedKappa(output$prediction, test$median_relevance, 1, 4)
print(kappa)



trControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2)

set.seed(1056)
rfFit <- train(data.tr[,features],data.tr$median_relevance,
               method="rf",
               tunelength=2,
               ntree=1000,
               trControl=trControl)

rfPredict <- predict(rfFit, newdata = data.tst[,features])

pred <- as.numeric(rfPredict)
output <- data.frame(id = data.tst[,1],prediction = pred)

write.csv(output,file='submission_rf_0527_1.csv', quote=FALSE,row.names=FALSE)
