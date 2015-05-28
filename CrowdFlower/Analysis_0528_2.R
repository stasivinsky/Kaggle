
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

library(stringdist)

count_dist <- function(query, text) {
  dist = stringdist(query, text, method = "cosine", q=4)
  if (dist == Inf) dist = 1000
  return(dist)
}

data.tr <- cbind(data.train,
                 matches_title = apply(data.train[,c('query','product_title')], 1, function(x) count_matches(x[1],x[2])),
                 matches_desc = apply(data.train[,c('query','product_description')], 1, function(x) count_matches(x[1],x[2])),
                 dist_title = apply(data.train[,c('query','product_title')], 1, function(x) count_dist(x[1],x[2])),
                 dist_desc = apply(data.train[,c('query','product_description')], 1, function(x) count_dist(x[1],x[2]))
)

data.tr$median_relevance <- as.factor(data.tr$median_relevance)

data.tst <- cbind(data.test,
                  matches_title = apply(data.test[,c('query','product_title')], 1, function(x) count_matches(x[1],x[2])),
                  matches_desc = apply(data.test[,c('query','product_description')], 1, function(x) count_matches(x[1],x[2])),
                  dist_title = apply(data.test[,c('query','product_title')], 1, function(x) count_dist(x[1],x[2])),
                  dist_desc = apply(data.test[,c('query','product_description')], 1, function(x) count_dist(x[1],x[2]))
)

features <- c('matches_title','matches_desc', 'dist_title', 'dist_desc')

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

# method = "osa"     : 0.2846907
# method = "lv"      : 0.2897751
# method = "dl"      : 0.2873493
# method = "hamming" : N/A
# method = "lcs"     : 0.3056846
# method = "qgram", q = 1    : 0.2899353
# method = "qgram", q = 2    : 0.301495
# method = "qgram", q = 3    : 0.3127208
# method = "qgram", q = 4    : 0.2988106
# method = "qgram", q = 5    : 0.3066426
# method = "cosine", q = 1   : 0.350478
# method = "cosine", q = 2   : 0.3468017
# method = "cosine", q = 3   : 0.3798038
# method = "cosine", q = 4   : 0.391223
# method = "cosine", q = 5   : 0.3733624
# method = "jaccard", q = 1  : 0.3377043
# method = "jaccard", q = 2  : 0.3408622
# method = "jaccard", q = 3  : 0.3558975
# method = "jaccard", q = 4  : 0.3575334
# method = "jaccard", q = 5  : 0.3337584
# method = "jw", p=0.00      : 0.3307299
# method = "jw", p=0.05      : 0.3301669
# method = "jw", p=0.1       : 0.3341208
# method = "jw", p=0.15      : 0.3395542
# method = "jw", p=0.2       : 0.3343029
# method = "jw", p=0.25      : 0.3284874
# method = "soundex" : N/A


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

write.csv(output,file='submission_rf_0528_2.csv', quote=FALSE,row.names=FALSE)

# 0.42515
