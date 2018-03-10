---
title: "Mushroom Classification"
output: 
  html_document:
    code_folding: hide
---

<br>
<center><img src="mushroom.jpg"></center>
<br>

#### About dataset:
I took this dataset from kaggle(https://www.kaggle.com/mig555/mushroom-classification/data) though it was originally contributed to the UCI Machine Learning repository nearly 30 years ago.

This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

<br>

#### Problem & Approach: 
To develop a binary classifier to predict which mushroom is poisonous & which is edible. I will build a Naive Bayes classifier for prediction followed by basic EDA of data. Later I will also test Decision Tree & Random Forest models on this dataset.


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(dplyr)
library(ggplot2)
library(e1071)
library(caret)
library(cowplot)
library(purrr)
library(rpart)
library(rpart.plot)
library(randomForest) 
library(knitr)
library(kableExtra)

mushroom<-read.csv("mushrooms.csv")
```


<br>

#### Lets check data structure 
```{r}
str(mushroom)
```

<br>

#### Renaming all entities
_As you might have noticed all data entities are named by initials only. Lets convert these to proper names for clarity & also convert all attributes to factors as all attributes are categorical here._ 
```{r}
colnames(mushroom) <- c("Edibility", "CapShape", "CapSurface", 
                        "CapColor", "Bruises", "Odor", 
                        "GillAttachment", "GillSpacing", "GillSize", 
                        "GillColor", "StalkShape", "StalkRoot", 
                        "StalkSurfaceAboveRing", "StalkSurfaceBelowRing", "StalkColorAboveRing", 
                        "StalkColorBelowRing", "VeilType", "VeilColor", 
                        "RingNumber", "RingType", "SporePrintColor", 
                        "Population", "Habitat")

mushroom <- mushroom %>% map_df(function(.x) as.factor(.x))

levels(mushroom$Edibility) <- c("edible", "poisonous")
levels(mushroom$CapShape) <- c("bell", "conical", "flat", "knobbed", "sunken", "convex")
levels(mushroom$CapColor) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                "green", "purple", "white", "yellow")
levels(mushroom$CapSurface) <- c("fibrous", "grooves", "scaly", "smooth")
levels(mushroom$Bruises) <- c("no", "yes")
levels(mushroom$Odor) <- c("almond", "creosote", "foul", "anise", "musty", "none", "pungent", "spicy", "fishy")
levels(mushroom$GillAttachment) <- c("attached", "free")
levels(mushroom$GillSpacing) <- c("close", "crowded")
levels(mushroom$GillSize) <- c("broad", "narrow")
levels(mushroom$GillColor) <- c("buff", "red", "gray", "chocolate", "black", "brown", "orange", 
                                 "pink", "green", "purple", "white", "yellow")
levels(mushroom$StalkShape) <- c("enlarging", "tapering")
levels(mushroom$StalkRoot) <- c("missing", "bulbous", "club", "equal", "rooted")
levels(mushroom$StalkSurfaceAboveRing) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushroom$StalkSurfaceBelowRing) <- c("fibrous", "silky", "smooth", "scaly")
levels(mushroom$StalkColorAboveRing) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                "green", "purple", "white", "yellow")
levels(mushroom$StalkColorBelowRing) <- c("buff", "cinnamon", "red", "gray", "brown", "pink", 
                                "green", "purple", "white", "yellow")
levels(mushroom$VeilType) <- "partial"
levels(mushroom$VeilColor) <- c("brown", "orange", "white", "yellow")
levels(mushroom$RingNumber) <- c("none", "one", "two")
levels(mushroom$RingType) <- c("evanescent", "flaring", "large", "none", "pendant")
levels(mushroom$SporePrintColor) <- c("buff", "chocolate", "black", "brown", "orange", 
                                        "green", "purple", "white", "yellow")
levels(mushroom$Population) <- c("abundant", "clustered", "numerous", "scattered", "several", "solitary")
levels(mushroom$Habitat) <- c("wood", "grasses", "leaves", "meadows", "paths", "urban", "waste")
```
<br>

#### Lets check few records from dataset now
```{r}
head(mushroom) %>% kable("html") %>%
  kable_styling()
```

<br>

#### Lets check the structure of data now
```{r}
str(mushroom)
```
<br>

#### Lets find out more about each category of each attribute
```{r}
summary(mushroom)
```
<br>

#### Checking Frequency of mushroom classes
```{r}
freq <- function(x){table(x)/length(x)*100} 
freq(mushroom$Edibility)
```
_Poisonous & Edible classes are almost balanced._

<br>

#### Bar Charts comparing Edibility across all mushroom features
```{r, echo=FALSE}
bar_theme1<- theme(axis.text.x = element_text(hjust = 1, vjust = 0.5, angle = 90), 
                   legend.position="none")

plot_grid(ggplot(mushroom, aes(x=VeilType,fill=Edibility))+ geom_bar(), 
          ggplot(mushroom, aes(x=VeilColor ,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=RingNumber,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=RingType,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=SporePrintColor,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=Population,fill=Edibility))+ geom_bar()+bar_theme1,
          align = "h")  
```
<br>
```{r, echo=FALSE}
plot_grid(ggplot(mushroom, aes(x=GillAttachment,fill=Edibility))+ geom_bar()+bar_theme1, 
          ggplot(mushroom, aes(x=GillSize,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=GillSpacing,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=GillColor,fill=Edibility))+ geom_bar()+bar_theme1,
          align = "h")  
```
<br>
```{r, echo=FALSE}
plot_grid(ggplot(mushroom, aes(x=StalkShape,fill=Edibility))+ geom_bar()+bar_theme1, 
          ggplot(mushroom, aes(x=StalkRoot,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=StalkSurfaceAboveRing,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=StalkSurfaceBelowRing,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=StalkColorAboveRing,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=StalkColorBelowRing,fill=Edibility))+ geom_bar()+bar_theme1,
          align = "h")   
```
<br>
```{r, echo=FALSE}
plot_grid(ggplot(mushroom, aes(x=CapShape,fill=Edibility))+ geom_bar()+bar_theme1, 
          ggplot(mushroom, aes(x=CapSurface,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=CapColor,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=Bruises,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=Odor,fill=Edibility))+ geom_bar()+bar_theme1,
          ggplot(mushroom, aes(x=Habitat,fill=Edibility))+ geom_bar()+bar_theme1,
          align = "h")   
```
<br>

### Classifying Mushrooms:

<br>

#### Creating Train Test Splits
_I will take 70% (5386 mushrooms) sample data for training & 30% (2438 mushrooms) for testing._
```{r}
set.seed(2)
s=sample(1:nrow(mushroom),0.7*nrow(mushroom))
mush_train=mushroom[s,]
mush_test=mushroom[-s,]
mush_test1<- mush_test[, -1]
```
<br>

#### Creating Model using Naive Bayes Classifier
_Naive Bayes classifier is based on Bayes Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature._
```{r}
model <- naiveBayes(Edibility ~. , data = mush_train)
```
<br>

#### Predicting Mushroom Class on Testset 
_Lets test our model on remaining 30% test data_
```{r}
pred <- predict(model, mush_test1)
```
<br>

#### Model Evaluation

```{r}
confusionMatrix(pred,mush_test$Edibility)
```
_In case of mushroom classification few False Negatives are tolrable but even a single False Positive can take someones life. We measure these as Senstivity & Specificity._

_We are getting Senstivity(True Positive Rate) of 99.28% which is good as it represent our prediction for edible mushrooms & only .7% False negatives(9 Mushrooms)._

_But Specificity(True Negative Rate) or our ability to classify Poisnous mushrooms is 87.58%, which is not  so good as more then 10% Poisnous mushrooms may get identified as Edible. This model have 147 False Positves which is not acceptable. so tets try a decision tree based model now._

<br>

#### Creating a Decision Tree based Classifier
_Decision tree is a type of supervised learning algorithm works for both categorical and continuous input and output variables. In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on most significant splitter / differentiator in input variables._
```{r}
tree.model <- rpart(Edibility ~ .,data = mush_train,method = "class",parms = list(split ="information"))
#prp(tree.model)
#summary(tree.model)
rpart.plot(tree.model,extra =  3,fallen.leaves = T)

tree.predict <- predict(tree.model, mush_test[,-c(1)], type = "class")
confusionMatrix(mush_test$Edibility, tree.predict)
```

_This model gives us an ideal Specificity of 1 but Decision Trees are prone to overfitting & so model may perform poorly on test data. Also algorithm have chosen only 2 attributes Order & SporePrintColor for making this prediction which might be questionable. Lets also try a final model called Random Forest which is robust & often gives a more generalisable model._

<br>

#### Creating Random Forest Classifier
_Random forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees._
```{r}
set.seed(1234)
rf_model <- randomForest(Edibility~.,data=mush_train, importance = TRUE, ntree = 1000)
rf_model
varImpPlot(rf_model)
```

_Random forest of 1000 trees have also identified Odor & SporePrintColor as most important variables. Lets check its predictions on test data now._

```{r}
rf_prediction <- predict(rf_model, mush_test[,-c(1)])
confusionMatrix(mush_test$Edibility, rf_prediction)
```

_We are getting 100% Accuracy, Sensitivity & Specificity now. I am not sure if this is correct. Please share your feedback on errors & improvemnts. Thanks for reading this notebook._
<br>
