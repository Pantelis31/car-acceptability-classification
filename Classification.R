rm(list=ls())
if (!require("e1071"))
{install.packages("e1071")
  library("e1071")
}
if (!require("ggplot2"))
{install.packages("ggplot2")
  library("ggplot2")
}
if (!require("gridExtra"))
{install.packages("gridExtra")
  library("gridExtra")
}
if (!require("ROCR"))
{install.packages("ROCR")
  library("ROCR")
}
if (!require("DMwR"))
{install.packages("DMwR")
  library("DMwR")
}
if (!require("MASS"))
{install.packages("MASS")
  library("MASS")
}


#Read in the data
raw <- read.table("car.data.txt", sep = ",")
data <- data.frame(raw[, 1:6], eval = ifelse(raw[, 7] == "unacc", "Negative",
                                               "Positive"))

#Exploratory analysis
p1 <- ggplot(data, aes(V3, ..count..)) + geom_bar(aes(fill = eval), position = "dodge") + xlab("Number of doors") + ylab("Count") +
  labs(title = "Number of doors")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p2 <- ggplot(data, aes(V1, ..count..)) + geom_bar(aes(fill = eval), position = "dodge") + xlab("Buying price") + ylab("Count") +
  labs(title = "Buying price")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p3 <- ggplot(data, aes(V2, ..count..)) + geom_bar(aes(fill = eval), position = "dodge") + xlab("Maintainance price") + ylab("Count") +
  labs(title = "Maintainance price")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p4 <- ggplot(data, aes(V4, ..count..)) + geom_bar(aes(fill = eval), position = "dodge") + xlab("Person capacity") + ylab("Count") +
  labs(title = "Person capacity")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p5 <- ggplot(data, aes(V5, ..count..)) + geom_bar(aes(fill = eval), position = "dodge") + xlab("Luggage boot size") + ylab("Count") +
  labs(title = "Luggage boot size")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p6 <- ggplot(data, aes(V6, ..count..)) + geom_bar(aes(fill = eval), position = "dodge") + xlab("Safety") + ylab("Count") +
  labs(title = "Safety")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

grid.arrange(p1,p2,p3,p4,p5,p6,nrow = 3)

colnames(data) <- c("Buying_price", "Maint_price", "Num_doors", "Person_capacity", "Lugg_boot_size", "Safety", "Eval")

#Change variables to factors
data$Eval <- as.factor(data$Eval)
data$Buying_price <- as.factor(data$Buying_price)
data$Maint_price <- as.factor(data$Maint_price)
data$Num_doors <- as.factor(data$Num_doors)
data$Person_capacity <- as.factor(data$Person_capacity)
data$Lugg_boot_size <- as.factor(data$Lugg_boot_size)
data$Safety <- as.factor(data$Safety)

#We can visualize the classes on the training and test set
p7 <- ggplot(data, aes(Eval, col = Eval , fill = Eval)) + geom_bar() + labs(title="Level Count",y="Count", x="Levels")+
  ggtitle("Class numbers")
p7

#Splitting the data to training and test
set.seed(1993)
split_size <- floor(0.75*nrow(data))
train_index <- sample(1:nrow(data), size = split_size)

train_set <- data[train_index, ]
test_set <- data[-train_index, ]

#USE SMOTE FOR BALANCING THE CLASSES
#Class numbers in the train set before balancing
p8 <- ggplot(train_set, aes(Eval, col = Eval , fill = Eval)) + geom_bar() + labs(title="Level Count",y="Count", x="Levels")+
  ggtitle("Class numbers in train set")
p8

balanced_train <- SMOTE(Eval~., data = train_set, perc.over = 1000, k = 4, perc.under = 120)

#Class numbers after balancing
p9 <- ggplot(balanced_train, aes(Eval, col = Eval , fill = Eval)) + geom_bar() + labs(title="Level Count",y="Count", x="Levels")+
  ggtitle("Class numbers in train set (SMOTE)")
p9


#Histograms after balancing
p10 <- ggplot(balanced_train, aes(Num_doors, ..count..)) + geom_bar(aes(fill = Eval), position = "dodge") + xlab("Number of doors") + ylab("Count") +
  labs(title = "Number of doors")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p11 <- ggplot(balanced_train, aes(Buying_price, ..count..)) + geom_bar(aes(fill = Eval), position = "dodge") + xlab("Buying price") + ylab("Count") +
  labs(title = "Buying price")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p12 <- ggplot(balanced_train, aes(Maint_price, ..count..)) + geom_bar(aes(fill = Eval), position = "dodge") + xlab("Maintainance price") + ylab("Count") +
  labs(title = "Maintainance price")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p13 <- ggplot(balanced_train, aes(Person_capacity, ..count..)) + geom_bar(aes(fill = Eval), position = "dodge") + xlab("Person capacity") + ylab("Count") +
  labs(title = "Person capacity")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p14 <- ggplot(balanced_train, aes(Lugg_boot_size, ..count..)) + geom_bar(aes(fill = Eval), position = "dodge") + xlab("Luggage boot size") + ylab("Count") +
  labs(title = "Luggage boot size")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

p15 <- ggplot(balanced_train, aes(Safety, ..count..)) + geom_bar(aes(fill = Eval), position = "dodge") + xlab("Safety") + ylab("Count") +
  labs(title = "Safety")+
  scale_fill_manual("Acceptability", values = c("Negative" = "orange", "Positive" = "darkblue"))

grid.arrange(p10,p11,p12,p13,p14,p15,nrow = 3)

## FITTING NAIVE BAYES AND LR CLASSIFIERS

# Computing metrics of performance
metrics_table <- function(confusion_matrix){
  sum_cols <- apply(confusion_matrix, 2, sum)
  sum_rows <- apply(confusion_matrix, 1, sum)
  precision <- diag(confusion_matrix)/sum_cols
  recall <- diag(confusion_matrix)/sum_rows
  f1_score <- 2*((precision*recall)/(precision+recall))
  result <- as.data.frame(rbind(precision, recall, f1_score))
  row.names(result) <- c("Precision", "Recall", "F1")
  return(result)
}

## Check performance of the models in unbalanced data

#NAIVE BAYES
nb0 <- naiveBayes(Eval ~., data = train_set, laplace = 0.1)

#Make predictions on train and test sets to check for overfitting 
nb0_pred <- predict(nb0, test_set[, -7])
nb0_pred_train <- predict(nb0, train_set[, -7])

#Confusion matrix
cmat_nb0 <- table(Actual = test_set$Eval, Predicted = nb0_pred)
cmat_nb0_train <- table(Actual = train_set$Eval, Predicted = nb0_pred_train)

#Metrics
metrics_table(cmat_nb0)
metrics_table(cmat_nb0_train)



#LOGISTIC REGRESSION
lr0 <- glm(Eval ~ ., data = train_set, family = binomial(link = "logit"))

#Make predictions on train and test sets to check for overfitting 
lr0_probs <- predict.glm(lr0, newdata = test_set , type = "response")
lr0_pred <- ifelse(lr0_probs > 0.5, "Positive", "Negative")

lr0_probs_train <- predict.glm(lr0, newdata = train_set , type = "response")
lr0_pred_train <- ifelse(lr0_probs_train > 0.5, "Positive", "Negative")

#Confusion matrix
cmat_lr0 <- table(Actual = test_set$Eval, Predicted = lr0_pred)
cmat_lr0_train <- table(Actual = train_set$Eval, Predicted = lr0_pred_train)

#Metrics
metrics_table(cmat_lr0)
metrics_table(cmat_lr0_train)



## Fit models on the balanced training data

#NAIVE BAYES MODEL
nb1 <- naiveBayes(Eval ~., data = balanced_train, laplace = 0.1)

#Make predictions on the train and test sets to check for overfitting
nb1_pred <- predict(nb1, test_set[, -7])
nb1_pred_train <- predict(nb1, balanced_train[, -7])


#Confusion matrices
cmat_nb1 <- table(Actual = test_set$Eval, Predicted = nb1_pred)
cmat_nb1_train <- table(Actual = balanced_train$Eval, Predicted = nb1_pred_train)

#Metrics
metrics_table(cmat_nb1)
metrics_table(cmat_nb1_train)





#FULL LOGISTIC REGRESSION MODEL
lrfull <- glm(Eval ~ ., data = balanced_train, family = binomial(link = "logit"))

#Predictions on the test set
lrfull_probs <- predict.glm(lrfull, newdata = test_set , type = "response")
lrfull_pred <- ifelse(lrfull_probs > 0.5, "Positive", "Negative")

#Predictions on the training set
lrfull_probs_train <- predict.glm(lrfull, newdata = balanced_train , type = "response")
lrfull_pred_train <- ifelse(lrfull_probs_train > 0.5, "Positive", "Negative")

#Confusion matrix
cmat_lrfull <- table(Actual = test_set$Eval, Predicted = lrfull_pred)
cmat_lrfull_train <- table(Actual = balanced_train$Eval, Predicted = lrfull_pred_train)

#Metrics
metrics_table(cmat_lrfull)
metrics_table(cmat_lrfull_train)















