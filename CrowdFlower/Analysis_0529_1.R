
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


library(Matrix)

data.combi <- data.train[,-((ncol(data.train)-1):ncol(data.train))]

all_text <- Corpus(VectorSource(data.combi$query))
dtm <- DocumentTermMatrix(all_text,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.98)
df_q<-Matrix(as.matrix(dtm),sparse=T)
df_q<-as.data.frame(as.matrix(dtm))
colnames(df_q)=paste("q_",colnames(df_q),sep="")

all_text <- Corpus(VectorSource(data.combi$product_title))
dtm<-DocumentTermMatrix(all_text,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.98)
df_pt<-Matrix(as.matrix(dtm),sparse=T)
df_pt<-as.data.frame(as.matrix(dtm))
colnames(df_pt)=paste("pt_",colnames(df_pt),sep="")

all_text <- Corpus(VectorSource(data.combi$product_description))
dtm<-DocumentTermMatrix(all_text,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.98)
df_pd<-as.data.frame(as.matrix(dtm))
colnames(df_pd)=paste("pd_",colnames(df_pd),sep="")

combi=cbind(df_q,df_pt,df_pd)

combi<-Matrix(as.matrix(combi),sparse=T)
combi<-as.data.frame(as.matrix(combi))

data.tr <- cbind(data.tr,combi)
features <- c(features, colnames(combi))


sample_size <- floor(0.75 * nrow(data.tr))

set.seed(123)
train_ind <- sample(seq_len(nrow(data.tr)), size = sample_size)

train <- data.tr[train_ind,-1]
test <- data.tr[-train_ind,-1]


library(caret)
library(randomForest)

trControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 1)

set.seed(1056)
rfFit <- train(train[,features],train$median_relevance,
               method="rf",
               tunelength=2,
               ntree=500,
               trControl=trControl)

rfPredict <- predict(rfFit, newdata = test[,features])

table(as.numeric(rfPredict),as.numeric(test$median_relevance))


pred <- as.numeric(rfPredict)
output <- data.frame(data.tr[-train_ind,1],prediction = pred)

library(Metrics)
kappa <- ScoreQuadraticWeightedKappa(output$prediction, test$median_relevance, 1, 4)
print(kappa)

# 0.4091397


data.combi <- rbind(data.train[,-((ncol(data.train)-1):ncol(data.train))],data.test)

all_text <- Corpus(VectorSource(data.combi$query))
dtm <- DocumentTermMatrix(all_text,control=list(tolower=TRUE,removePunctuation=TRUE,
                                                removeNumbers=TRUE,stopwords=TRUE,
                                                stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.98)
df_q<-Matrix(as.matrix(dtm),sparse=T)
df_q<-as.data.frame(as.matrix(dtm))
colnames(df_q)=paste("q_",colnames(df_q),sep="")

all_text <- Corpus(VectorSource(data.combi$product_title))
dtm<-DocumentTermMatrix(all_text,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.98)
df_pt<-Matrix(as.matrix(dtm),sparse=T)
df_pt<-as.data.frame(as.matrix(dtm))
colnames(df_pt)=paste("pt_",colnames(df_pt),sep="")

all_text <- Corpus(VectorSource(data.combi$product_description))
dtm<-DocumentTermMatrix(all_text,control=list(tolower=TRUE,removePunctuation=TRUE,
                                              removeNumbers=TRUE,stopwords=TRUE,
                                              stemming=TRUE,weighting=function(x) weightTfIdf(x,normalize=T)))
dtm <- removeSparseTerms(dtm,0.98)
df_pd<-as.data.frame(as.matrix(dtm))
colnames(df_pd)=paste("pd_",colnames(df_pd),sep="")

combi=cbind(df_q,df_pt,df_pd)

combi<-Matrix(as.matrix(combi),sparse=T)
combi<-as.data.frame(as.matrix(combi))

data.tr <- cbind(data.tr[,1:10],combi[1:nrow(data.tr),])
data.tst <- cbind(data.tst,combi[(nrow(data.tr)+1):nrow(combi),])
features <- c(features[1:4], colnames(combi))

trControl <- trainControl(
  method = "repeatedcv",
  number = 2,
  repeats = 1)

set.seed(1056)
rfFit <- train(data.tr[,features],data.tr$median_relevance,
               method="rf",
               tunelength=2,
               ntree=500,
               trControl=trControl)

rfPredict <- predict(rfFit, newdata = data.tst[,features])

pred <- as.numeric(rfPredict)
output <- data.frame(id = data.tst[,1],prediction = pred)

write.csv(output,file='submission_rf_0529_1.csv', quote=FALSE,row.names=FALSE)

# 0.47979
