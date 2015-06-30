
HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

data.input.1 <- data.input
data.input.1$target <- as.character(data.input.1$target)

library(plyr)

class.prob <- count(data.input.1, 'target')
class.prob <- data.frame(class.prob, class.prob$freq / sum(class.prob$freq))
names(class.prob) <- c('class','freq','prob')
class.prob$class <- as.character(class.prob$class)

feature.prob <- data.frame(feature=character(), value=integer(), freq=integer(), prob=numeric())
names(feature.prob) <- c('feature', 'value', 'freq', 'prob')

for (i in 1:93) {
  fp <- count(data.input.1[data.input.1[,i+1]>0,], paste('feat_',i,sep=''))
  fp <- data.frame(paste('feat_',i,sep=''), fp, fp$freq / sum(fp$freq))
  names(fp) <- c('feature', 'value', 'freq', 'prob')
  feature.prob <- rbind(feature.prob,fp)
}
feature.prob$feature <- as.character(feature.prob$feature)

feature_given_class.prob <- data.frame(class = character(), feature=character(), value=integer(), freq=integer(), prob=numeric())
names(feature_given_class.prob) <- c('class', 'feature', 'value', 'freq', 'prob')

for (c in 1:9) {
  for (i in 1:93) {
    fcp <- count(data.input.1[data.input.1$target == paste('Class_',c,sep='') & data.input.1[,i+1]>0,], paste('feat_',i,sep=''))
    fcp <- data.frame(paste('Class_',c,sep=''), paste('feat_',i,sep=''), fcp, fcp$freq / sum(fcp$freq))
    names(fcp) <- c('class', 'feature', 'value', 'freq', 'prob')
    feature_given_class.prob <- rbind(feature_given_class.prob,fcp)
  }
}
feature_given_class.prob$class <- as.character(feature_given_class.prob$class)
feature_given_class.prob$feature <- as.character(feature_given_class.prob$feature)


class_given_feature.prob <- data.frame(feature=character(), class <- character(), value=integer(), prob=numeric())
names(class_given_feature.prob) <- c('feature', 'class', 'value', 'prob')

for (i in 1:nrow(feature_given_class.prob)) {
  row <- feature_given_class.prob[i,]
  
  value <- row$value
  feature <- row$feature
  class <- row$class
  
  prob <- row$prob
  prob.class <- class.prob[class.prob$class == class,]$prob
  prob.feature <- feature.prob[(feature.prob$feature == feature) & (feature.prob$value == value),]$prob
    
  new.prob <- prob * prob.class / prob.feature
  
  
  class_given_feature.prob <- rbind(class_given_feature.prob, data.frame(feature, class, value, new.prob))
}
class_given_feature.prob$class <- as.character(class_given_feature.prob$class)
class_given_feature.prob$feature <- as.character(class_given_feature.prob$feature)
names(class_given_feature.prob) <- c('feature', 'class', 'value', 'prob')


output <- data.frame(data.output$id)

class.prob <- data.frame(id=integer(), Class_1=numeric(), Class_2=numeric(), Class_3=numeric(), Class_4=numeric(), Class_5=numeric(),
                         Class_6=numeric(), Class_7=numeric(), Class_8=numeric(), Class_9=numeric())
for(i in 50001:nrow(data.output)) {
  row <- data.output[i,]
  
  prob <- data.frame(c('Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'))
  names(prob)[1] <- 'class'
  prob[,1] <- as.character(prob[,1])
  
  for (f in 1:93) {
    value <- as.integer(row[,f+1])
    prob <- merge(prob, class_given_feature.prob[class_given_feature.prob$feature == paste('feat_',f,sep='') &
                                     class_given_feature.prob$value == value,c('class','prob')], by='class', all.x = TRUE, incomparables=NA)
    names(prob)[f+1] <- paste('prob_',f,sep='')
  }  
  
  #means <- data.frame(class=prob[,1],prob=rowMeans(prob[,-1], na.rm=TRUE))
  means <- data.frame(class=prob[,1], apply(prob[,-1],1,max, na.rm=TRUE))
  means.row <- data.frame(row$id,means[means$class == 'Class_1',2],means[means$class == 'Class_2',2],means[means$class == 'Class_3',2],
                          means[means$class == 'Class_4',2],means[means$class == 'Class_5',2],means[means$class == 'Class_6',2],
                          means[means$class == 'Class_7',2],means[means$class == 'Class_8',2],means[means$class == 'Class_9',2])
  class.prob <- rbind(class.prob, means.row)
}
names(class.prob) <- c('id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9')
write.csv(class.prob, file = "submission_04_60k.csv", quote=FALSE, row.names=FALSE)
