library(dplyr)
library(ggplot2)
library(leaflet)

setwd("/media/newhd/Data Science/DataSets/kaggle_datasets/crime-prediction")
crime<-read.csv("train.csv")

str(crime)
summary(crime)
View(crime)
dim(crime) #878049 records & 9 features

#Checking missing values
nrow(crime[(is.na(crime) | crime==""), ])
sum(is.na(crime))

#Checking for duplicates
crime[!(duplicated(crime[]) | duplicated(crime[], fromLast = TRUE)), ] %>% nrow()

#Exploratory Data Analysis
#Univariate
plot_theme<-theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))
plot_text<-geom_text(stat='count',aes(label=..count..),vjust=-1)

ggplot(crime,aes(x=Category)) + geom_bar() + plot_theme + xlab("Crime Category") + ylab("Number of Crimes")
#Larcenecy/Theft, Other Offences, Assault & Non Criminal were major crimes

ggplot(data=crime,aes(x=DayOfWeek)) + geom_bar() + plot_text + xlab("Day of Week") + ylab("Number of Crimes")
#Most crimes were committed on Friday(133734) & Least on Sunday(116707) but not much variation overall

ggplot(data=crime,aes(x=PdDistrict)) + geom_bar() + plot_text + xlab("District") + ylab("Number of Crimes")
#Most crimes were committed in Southern district(157182) and least in Richmond(45209)

ggplot(data=crime,aes(x=Resolution)) + geom_bar() + plot_theme + plot_text + xlab("Resolution") + ylab("Count")
#Arret Booked & Arrest Cited are most commom resolutions followed by None

#Bivariate Analysis
#Relationships Among Category & Other variables
#Lets Explore data using Dates 
#Convering Date from factor to POSIXlt format & deriving Year, Month, Hour data
crime$Dates<-strptime(crime$Dates, "%Y-%m-%d %H:%M:%OS")
crime$Year<-crime$Dates$year+1900
crime$Month<-crime$Dates$mon+1
crime$Hour<-format(crime$Dates, format="%H") 
summary(crime$Year) #Data span from year 2003 to 2015

ggplot(data=crime,aes(x=Category,fill=as.factor(Month))) + geom_bar(position = position_fill()) + plot_theme +
  xlab("Crime Category") + ylab("Relative Frequency by Month") + labs(fill = "Months")
#Treason only happened in months 9,10,11,12 & 2!
table(crime$Month)

ggplot(data=crime,aes(x=Category,fill=as.factor(Year))) + geom_bar(position = position_fill()) + plot_theme +
  xlab("Crime Category") + ylab("Relative Frequency by Year") + labs(fill = "Years")
#Treason data is only from past 5 years

ggplot(data=crime,aes(x=Category,fill=as.factor(Hour))) + geom_bar(position = position_fill()) + plot_theme +
  xlab("Crime Category") + ylab("Relative Frequency by Hour") + labs(fill = "Hours")
#But this is cluttered, Lets further aggregate hours in 4 buckets
Hour_Bucket<-as.numeric(crime[,"Hour"])
Hour_Bucket<-Hour_Bucket+1
#crime$Hour_Bucket<-cut(Hour_Bucket, 4, include.lowest=TRUE, labels=c("Late Night", "Morning", "Day","Night"))
crime$Hour_Bucket<-cut(Hour_Bucket, 5, labels=c("Late Night","Morning","Day","Till Evening","Night"))
ggplot(data=crime,aes(x=Category,fill=as.factor(Hour_Bucket))) + geom_bar(position = position_fill()) + plot_theme +
  xlab("Crime Category") + ylab("Frequency by Time Window") + labs(fill = "Time Window")
#Incidents like 'Diving Under Influence' & 'Prostitution' happens after dark while 'Bad Checks' & 'Fraud' usually happend in daylight

ggplot(data=crime,aes(x=Category,fill=DayOfWeek)) + geom_bar(position = position_fill()) + plot_theme +
  xlab("Crime Category") + ylab("Relative Frequency by WeekDay") + labs(fill = "Week Day")
#Trea crime occured more often on Saturday
ggplot(data=crime,aes(x=Category,fill=DayOfWeek)) + geom_bar(position = position_dodge()) + plot_theme +
  xlab("Crime Category") + ylab("Crime Frequency") + labs(fill = "Week Day")
ggplot(data=crime,aes(x=Category,fill=DayOfWeek)) + geom_bar(position = position_stack()) + plot_theme +
  xlab("Crime Category") + ylab("Crime Frequency") + labs(fill = "Week Day")

ggplot(data=crime,aes(x=Category,fill=PdDistrict)) + geom_bar(position = position_fill()) + plot_theme +
  xlab("Crime Category") + ylab("Relative Frequency by District") + labs(fill = "District")
#For most Driving,Drunk & Drug cases resolution was Arrest Booked, for most Theft including Vehicle Theft resolution was None

ggplot(data=crime,aes(x=Category,fill=Resolution)) + geom_bar(position = position_fill()) + plot_theme +
  xlab("Crime Category") + ylab("Relative Frequency by Resolution")

#Spatial Data Aanalysis of few data subsets
crime_kidnapping<-subset(crime,crime$Category=="KIDNAPPING")

crime_kidnapping %>%
  leaflet() %>%
  addTiles() %>%
  addMarkers(lng=crime_kidnapping$X,lat=crime_kidnapping$Y,clusterOptions = markerClusterOptions(maxClusterRadius=40))

crime_kidnapping %>%
  leaflet() %>%
  addTiles() %>%
  addCircles(lng=crime_kidnapping$X,lat=crime_kidnapping$Y,radius = 20,color = "red")

crime_kidnapping %>%
  leaflet() %>%
  addTiles() %>%
  addCircles(lng=crime_kidnapping$X,lat=crime_kidnapping$Y,radius = 60,
             color = ~ifelse(DayOfWeek=="Sunday" | DayOfWeek=="Saturdy" | DayOfWeek=="Friday", "blue", "red"),
             stroke = FALSE, fillOpacity = 1)

crime_kidnapping %>%
  leaflet() %>%
  addTiles() %>%
  addCircles(lng=crime_kidnapping$X,lat=crime_kidnapping$Y,radius = 60,
             color = ~ifelse(Hour_Bucket=="Night", "blue", "red"),
             stroke = FALSE, fillOpacity = 1)

#Regionwise Analysis
#colnames(crime)[8]<-"longitude"
#colnames(crime)[9]<-"latitude"
#invisible(lapply(crime$PdDistrict, function(x) {assign(x$PdDistrict,x,pos = .GlobalEnv)}))

df_list[1] %>%
  leaflet() %>%
  addTiles() %>%
  addCircles(lng=BAYVIEW.X,lat=BAYVIEW.Y,radius = 20,color = "red")

#Classification Trees
library(rpart)
library(rpart.plot)
library(caret)
library(caTools)

set.seed(100)
indices = sample.split(crime$Category, SplitRatio = 0.1)
crime_sample = crime[indices,]
table(crime_sample$Category)
#test = telecom_final[!(indices),]

#Building tree model- default hyperparameters
tree.model <- rpart(Category ~ ., data = crime_sample, method = "class")

#Display decision tree
prp(tree.model)

#Make predictions on the test set
tree.predict <- predict(tree.model, test, type = "class")

#Evaluate the results
confusionMatrix(test$income, tree.predict, positive = ">50K")  # 0.8076
