library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)

#Loading train & test data
mnist <- read.csv("train.csv")
mnist_test <- read.csv("test.csv")
View(mnist)
ncol(mnist)
nrow(mnist)

#----------------------------------------Data Prepration & EDA------------------------------------------------------
#Checking dataset dimensions
dim(mnist)
dim(mnist_test)

#Checking Structure & Exploring the dataset
str(mnist)
summary(mnist)

#Distribution percentage of each digit
table(mnist$label)/nrow(mnist) *100
#Note: Both test & train datasets have uniform & similar distribution of each digit

#checking missing value
sapply(mnist, function(x) sum(is.na(x)))  #Have 1 missing value row in train dataframe
sapply(mnist_test, function(x) sum(is.na(x))) #No missing values found in test dataframe

#Removing Missing values from training(mnist) dataframe
mnist[is.na(mnist)] <- 0

#Visualising a digit in data
digit <- matrix(as.numeric(mnist[7,-1]), nrow = 28)
image(digit, col = grey.colors(255))

##Exploratory Data Analysis
mnist_copy<-mnist
mnist_copy$intensity <- apply(mnist_copy[,-1], 1, mean) #takes the mean of each row in training set

intbylabel <- aggregate(mnist_copy$intensity, by = list(mnist_copy$label), FUN = mean)

plot <- ggplot(data=intbylabel, aes(x=Group.1, y = x)) + geom_bar(stat="identity")
plot + scale_x_discrete(limits=0:9) + xlab("digit label") + ylab("average intensity")

#Checking distribution of few digits
mnist_copy$label <- as.factor(mnist_copy$label)
p1 <- qplot(subset(mnist_copy, label ==2)$intensity, binwidth = .75, xlab = "Intensity Hist for 2")
p2 <- qplot(subset(mnist_copy, label ==5)$intensity, binwidth = .75, xlab = "Intensity Hist for 5")
p3 <- qplot(subset(mnist_copy, label ==7)$intensity, binwidth = .75, xlab = "Intensity Hist for 7")
p4 <- qplot(subset(mnist_copy, label ==9)$intensity, binwidth = .75, xlab = "Intensity Hist for 9")
grid.arrange(p1, p2, p3,p4, ncol = 2)

#Distribution of 7 is less 'normal' with multiple peaks, perhaps there are different ways people tend to write seven
mnist_copy_7 <- mnist_copy[mnist_copy$label == 7, ]
flip <- function(matrix){
  apply(matrix, 2, rev)
}
#Shows 9 diffrent ways people write digit 7 
par(mfrow=c(3,3))
for (i in 10:18){
  digit <- flip(matrix(rev(as.numeric(mnist_copy_7[i,-c(1, 786)])), nrow = 28)) #shows different styles of digit 
  image(digit, col = grey.colors(255))
}

#Making our target class to factor
mnist$label <-factor(mnist$label)
str(mnist$label)

#-----------------------------------------------Model Prepration----------------------------------------------------
#Split the data into train and test set
#I have used provided test set for testing instead of spliting train

#Note: Few Model prepration steps below may take time, took 2-3 minutes on my PC, I have optimized this using PCA later  

##Construting Model Using RBF Kernel
#creating train test split
set.seed(100)
indices = sample.split(mnist$label, SplitRatio = 0.2)
train = mnist[indices,]
test = mnist[-indices,]

Model_RBF <- ksvm(label~., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test)
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test$label)

#------------------------------------------------Cross-Validation---------------------------------------------------
#Cross validation was done small homogeneous sample(20%) as its Compute-intensive
library(caTools)
table(train$label) #sample have a even distribution of all digits   
trainControl <- trainControl(method="cv", number=5) #cv is Cross Validation, number is folds
metric <- "Accuracy"
#Expand.grid functions takes set of hyperparameters, that we shall pass to our model
grid <- expand.grid(.C=c(0.1,0.5,1,2))
fit.svm <- train(label~., data=train, method="svmRadialCost", metric=metric,
                 tuneGrid=grid, trControl=trainControl)

#Checking model efficiency for different Cost values
print(fit.svm)
plot(fit.svm)

#Evaluating model created on sampled data on test dataset
Eval_fit.svm<- predict(fit.svm, test)
confusionMatrix(Eval_fit.svm,test$label)
#Model on sample dataset gives a decent accuracy of 94.34% on test dataset

#Note: Cross-Validation on subset of 20% train data & gives best Accuracy of only 94.62% for C=2
#Cross-Vaidation provided optimal value of C = 2 which I use for better solution using PCA explained below

##--------------------------------------Principal Component Analysis(PCA)--------------------------------------------
#Reducing features using PCA
mnist_norm<-as.matrix(mnist[,-1])/255
mnist_norm_cov <- cov(mnist_norm)
pca <- prcomp(mnist_norm_cov)
trainlabel <- mnist[,1]

#Checking relationship between number of Pricipal Components & Variance
vexplained <- as.data.frame(pca$sdev^2/sum(pca$sdev^2))
vexplained <- cbind(c(1:784),vexplained,cumsum(vexplained[,1]))
colnames(vexplained) <- c("No_of_Principal_Components","Individual_Variance_Explained","Cumulative_Variance_Explained")

#Plot between Cumulative Variance & Principal Components
plot(vexplained$No_of_Principal_Components,vexplained$Cumulative_Variance_Explained, xlim = c(0,150),type='b',pch=16,xlab = "Principal Componets",ylab = "Cumulative Variance Explained",main = 'Principal Components vs Cumulative Variance Explained')

#Table showing Cumulative Variance & Principal Components
vexplainedsummary <- vexplained[seq(0,150,5),]
vexplainedsummary
#Note: Variance till Number of Principal Components 45 is 0.9909898

##Applying SVM on training set and calculating accuracy
library(e1071)
mnist_final <- as.matrix(mnist[,-1]) %*% pca$x[,1:45]
trainlabel <- as.factor(trainlabel)
svm.model.final <- svm(mnist_final,trainlabel,cost = 2)
predictionfinaltrain <- predict(svm.model.final,mnist_final)
correcttrainfinal <- predictionfinaltrain==trainlabel
Accuracytrainfinal <- (sum(correcttrainfinal)/nrow(mnist_final))*100
Accuracytrainfinal #99.70 %

#Predicting mnist test labels using above model
mnist_test_pca<-as.matrix(mnist_test) %*% pca$x[,1:45]
mnist_predictions<-predict(svm.model.final,mnist_test_pca)
mnist_test$predicted_labels<-mnist_predictions
mnist_test$predicted_labels

write.csv(mnist_predictions, file = "my_submission.csv")

#Testing few predicted labels manually
digit <- matrix(as.numeric(mnist_test[8,-785]), nrow = 28)
image(digit, col = grey.colors(255))

#Note: Prediction using PCA saves time by idetifying 45 components covering 99.77% variance, 
#we got a fair accuracy of 97.25% on test data using PCA with C=2
