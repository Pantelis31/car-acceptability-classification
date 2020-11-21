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


raw <- read.table("car.data.txt", sep = ",")
data <- data.frame(raw[, 1:6], eval = ifelse(raw[, 7] == "unacc", "Negative",
                                             "Positive"))

colnames(data) <- c("Buying_price", "Maint_price", "Num_doors", "Person_capacity", "Lugg_boot_size", "Safety", "Eval")

#Change variables to factors
data$Eval <- as.factor(data$Eval)
data$Buying_price <- as.factor(data$Buying_price)
data$Maint_price <- as.factor(data$Maint_price)
data$Num_doors <- as.factor(data$Num_doors)
data$Person_capacity <- as.factor(data$Person_capacity)
data$Lugg_boot_size <- as.factor(data$Lugg_boot_size)
data$Safety <- as.factor(data$Safety)

#Splitting the data to training and test
set.seed(1993)
split_size <- floor(0.75*nrow(data))
train_index <- sample(1:nrow(data), size = split_size)

train_set <- data[train_index, ]
test_set <- data[-train_index, ]

#Loss
Loss <- function(predictions, truth){
  return(mean(predictions != truth))
}


#Initialize loss vectors
loss0 <- numeric(0)
loss1 <- numeric(0)
loss2 <- numeric(0)
loss_lr <- numeric(0)

#Fit the models for 1:n data vectors in the training set 
for (rows in 25:nrow(train_set)){
  
  train <- train_set[1:rows, ]
  #Fit Naive Bayes models for different Laplace parameters
  nb0 <- naiveBayes(Eval ~., data = train, laplace = 0.1)
  nb1 <- naiveBayes(Eval ~., data = train, laplace = 1)
  nb2 <- naiveBayes(Eval ~., data = train, laplace = 10)
  
  #Make predictions
  nb0_pred <- predict(nb0, test_set[, -7])
  nb1_pred <- predict(nb1, test_set[, -7])
  nb2_pred <- predict(nb2, test_set[, -7])
  
  #Compute loss
  loss0[rows - 24] <- Loss(nb0_pred, test_set$Eval)
  loss1[rows - 24] <- Loss(nb1_pred, test_set$Eval)
  loss2[rows - 24] <- Loss(nb2_pred, test_set$Eval)
  
  #Fit full logistic regression model
  lr <- tryCatch(glm(Eval ~ ., data = train, family = binomial(link = "logit")),
                 error = function(e) e)
  
  if (!inherits(lr, "error")){
    #Make predictions
    lr_probs <- predict.glm(lr, newdata = test_set[,-7] , type = "response")
    lr_pred <- ifelse(lr_probs > 0.5, "Positive", "Negative")
    #Compute loss
    loss_lr[rows - 24] <- Loss(lr_pred, test_set$Eval)
  }
  else{
    loss_lr[rows - 24] <- NA
  }

  #Results
  res <- as.data.frame(cbind(loss0, loss1, loss2, loss_lr))
  colnames(res) <- c("NB a=0.1", "NB a=1", "NB a=10", "LR")
}


ggplot() + 
  geom_line( aes(x = log(25:1296), y = res[ ,4], colour = "black")) +
  geom_line( aes(x = log(25:1296), y = res[ ,3], colour = "blue")) +
  geom_line( aes(x = log(25:1296), y = res[ ,1], colour = "green")) +
  geom_line( aes(x = log(25:1296), y = res[ ,2], colour = "purple")) +
  xlab('Data Vectors') +
  ylab('Loss') +
  scale_colour_manual(values = c("black" ,"purple","green","blue"),
                      name = "Classifiers" ,
                      labels = c("LR full","NB a=1","NB a=0.1","NB a=10")
  ) +
  labs(title = "Model Comparison")











