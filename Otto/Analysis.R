
library(class)

HOMEDIR <- "C:\\Work\\Kaggle\\Otto\\R"
setwd(HOMEDIR)

data.input <- read.csv("train.csv", header=TRUE, sep=",")
data.output <- read.csv("test.csv", header=TRUE, sep=",")

#---------------------
# Baseline model based on the probabilities of the occurances of the specific classes 
# in the training set. The model calculates the probability of every class and assigns these 
# same probabilities to every observation. 
#
# Score: 1.94999
#---------------------

totals <- table (data.input$target)
totals.df <- as.data.frame.matrix(t(totals / sum(totals)))

output <- data.frame(data.output$id, totals.df)
names(output)[1] <- "id"

write.csv(output, file = "submission_01.csv", quote=FALSE, row.names=FALSE)


# smp_size <- floor(0.75 * nrow(data.input))
# 
# set.seed(123)
# train_ind <- sample(seq_len(nrow(data.input)), size = smp_size)
# 
# train <- data.input[train_ind,-1 ]
# test <- data.input[-train_ind, -1]

# library(kernlab)
# rbf <- rbfdot(sigma=0.1)
# SVM <- ksvm(target~.,data=train,type="C-bsvc",kernel=rbf,C=10,prob.model=TRUE)
# fitted(SVM)
# SVM.predicted <- predict(SVM, test[,-94])
# table(SVM.predicted, test[,94])

# SVM.predicted Class_1 Class_2 Class_3 Class_4 Class_5 Class_6 Class_7 Class_8 Class_9
# Class_1     158       1       3       1       0      12       7      18      44
# Class_2       6    3181     892     197      17      15      45      13      19
# Class_3       6     453     946     102       3       1      50       9       0
# Class_4       1      64      38     271       0       8       1       0       1
# Class_5       0       3       0       1     654       1       7       1       1
# Class_6     159     213     158      83       7    3373     316     703     367
# Class_7      16      23      26       6       0      18     249      20       3
# Class_8      46      13       6       5       0      32      32    1364      41
# Class_9      56       7       1       0       1      24       3      19     829

# SVM.predicted <- predict(SVM, test[,-94], type="probabilities")
# output <- data.frame(data.input[-train_ind, 1], SVM.predicted)

#---------------------
# SVM model. Did not beat the previous results :) 
#
# Score: 1.99397
#---------------------


library(kernlab)
rbf <- rbfdot(sigma=0.1)
SVM <- ksvm(target~.,data=data.input[,-1],type="C-bsvc",kernel=rbf,C=10,prob.model=TRUE)

SVM.predicted <- predict(SVM, data.output[,-1], type="probabilities")
output <- data.frame(data.output[, 1], SVM.predicted)
output[output < 0] <- 0.0
output[output > 1] <- 1.0
output <- data.frame(data.output[, 1], output[,-1])
names(output)[1] <- "id"

write.csv(output, file = "submission_02.csv", quote=FALSE, row.names=FALSE)

