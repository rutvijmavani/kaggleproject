train<-read_csv("D:/google downloads/train.csv/train.csv")
head(train)
str(train)
summary(train)
colSums(is.na(train))
table(train$Embarked)
table(is.na(train$Embarked))
train$Embarked[is.na(train$Embarked)]<-"S"
table(train$Embarked)
table(is.na(train$Embarked))
table(is.na(train$Age))[2]/table(is.na(train$Age))[1]
summary(train$Age)
train$is_age_missing<-ifelse(is.na(train$Age),1,0)
train$is_fare_missing<-ifelse(is.na(train$Fare),1,0)
train$travelers<-train$SibSp+train$Parch+1
train$Survived<-as.factor(train$Survived)
train$Pclass<-as.factor(train$Pclass)
train$is_age_missing<-as.factor(train$is_age_missing)
train$is_fare_missing<-as.factor(train$is_fare_missing)
train2<-subset(train,select = c(Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,is_age_missing,is_fare_missing,travelers))
dummy<-dummyVars(~.,data = train2[,-1])
dummy_train<-predict(dummy,train2[,-1])
head(dummy_train)
pre.process<-preProcess(dummy_train,method = "bagImpute")
imputed.data<-predict(pre.process,dummy_train)
head(imputed.data)
train$Age<-imputed.data[,6]
head(train$Age,20)
train$Fare<-imputed.data[,9]
head(train$Fare,20)
table(is.na(train2))
training<-subset(train,select = c(Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,travelers))
set.seed(123)
partition_indexes<-createDataPartition(training$Survived,times = 1,p=0.7,list = FALSE)
titanic.train<-training[partition_indexes,]
titanic.test<-training[-partition_indexes,]
head(titanic.train)
#train.y<-titanic.train$Survived
#test.y<-titanic.test$Survived
#titanic.train<-titanic.train[,-1]
#titanic.test<-titanic.test[,-1]
titanic.train$Embarked<-factor(titanic.train$Embarked,levels = c("S","C","Q"),labels = c(1,2,3))
titanic.train$Sex<-factor(titanic.train$Sex,levels = c("male","female"),labels = c(1,2))
#titanic.train$Sex<-as.numeric(as.factor(titanic.train$Sex))
#titanic.train$Embarked<-as.numeric(as.factor(titanic.train$Embarked))
#titanic.train$Pclass<-as.numeric(as.factor(titanic.train$Pclass))
#titanic.train$Survived<-as.numeric(as.factor(titanic.train$Survived))
#titanic.test$Survived<-as.numeric(as.factor(titanic.test$Survived))
head(titanic.train)
titanic.train[,c(4,5,6,7,9)]<-scale(titanic.train[,c(4,5,6,7,9)])
head(titanic.train)
titanic.test$Embarked<-factor(titanic.test$Embarked,levels = c("S","C","Q"),labels = c(1,2,3))
titanic.test$Sex<-factor(titanic.test$Sex,levels = c("male","female"),labels = c(1,2))
#titanic.test$Sex<-as.numeric(as.factor(titanic.test$Sex))
#titanic.test$Embarked<-as.numeric(as.factor(titanic.test$Embarked))
#titanic.test$Pclass<-as.numeric(as.factor(titanic.test$Pclass))
head(titanic.test)
titanic.test[,c(4,5,6,7,9)]<-scale(titanic.test[,c(4,5,6,7,9)])
#titanic.train$Survived<-train.y
#titanic.test$Survived<-test.y
train.control <- trainControl(method = "repeatedcv", number = 10,repeats = 3, search = "grid")
tune.grid <- expand.grid(eta = 0.05,
                         nrounds = 150,
                         max_depth = 8,
                         min_child_weight = 2.0,
                         colsample_bytree = 0.3,
                         gamma = 0,
                         subsample = 1
)
head(tune.grid)
library(doSNOW)
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
caret.cv <- train(Survived ~ ., data = titanic.train, method = "xgbTree",
                  tuneGrid = tune.grid, trControl = train.control)
stopCluster(cl)
preds<-predict(caret.cv,titanic.test)
head(preds)
confusionMatrix(preds,titanic.test$Survived)
#caret.rf <- train(Survived ~ .,
 #                 data = titanic.train,
  #                method = "gbm",
                 # tuneGrid = tune.grid,
   #               trControl = train.control,vebose=FALSE)
my_control <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=TRUE,
  index=createResample(titanic.train$Survived, 25),
  summaryFunction=twoClassSummary
)
#levels(titanic.train$Survived) <- make.names(levels(factor(titanic.train$Survived)))
#model_list <- caretList(
 # Survived~., data=titanic.train,
  #trControl=my_control,
  #methodList=c("xgbTree", "adaboost")
#)
#p<-predict(model_list,newdata = titanic.test)
#head(p)
#confusionMatrix(p,titanic.test$Survived)
# Example of Boosting Algorithms
#control <- trainControl(method="repeatedcv", number=10, repeats=3)
#seed <- 7
#metric <- "Accuracy"
# C5.0
#tune<-expand.grid(trials=1,model="tree",winnow=FALSE)
#set.seed(seed)
#fit.c50 <- train(Survived~., data=titanic.train, method="C5.0",trControl=control,tuneGrid=tune)
#fit.c50
#p<-predict(fit.c50,titanic.test)
# Stochastic Gradient Boosting
#tune1<-expand.grid(n.trees=150,interaction.depth=3,shrinkage=0.1,n.minobsinnode=10)
#set.seed(seed)
#fit.gbm <- train(Survived~., data=titanic.train, method="gbm",trControl=control, verbose=FALSE,tuneGrid=tune1)
#fit.gbm
# summarize results
#boosting_results <- resamples(list(c5.0=fit.c50, gbm=fit.gbm))
#summary(boosting_results)
#dotplot(boosting_results)
#boosting_results
#p<-predict(boosting_results,titanic.test)
test_set<-read_csv("D:/google downloads/test.csv/test.csv")
colSums(is.na(test_set))
table(test_set$Embarked)
test_set$Embarked[is.na(test_set$Embarked)]<-"S"
colSums(is.na(test_set))
test_set$travelers<-test_set$SibSp+test_set$Parch+1
test_set$is_age_missing<-ifelse(is.na(test_set$Age),1,0)
test_set$is_fare_missing<-ifelse(is.na(test_set$Fare),1,0)
test_set$is_age_missing<-as.factor(test_set$is_age_missing)
test_set$Pclass<-as.factor(test_set$Pclass)
test_set$is_fare_missing<-as.factor(test_set$is_fare_missing)
testing<-subset(test_set,select= c(Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,is_age_missing,is_fare_missing,travelers))
dummy2<-dummyVars(~.,data = testing[,])
dummy_test<-predict(dummy2,testing[,])
head(dummy_test)
pre.process1<-preProcess(dummy_test,method = "bagImpute")
imputed.data1<-predict(pre.process1,dummy_test)
head(imputed.data1)
test_set$Age<-imputed.data1[,6]
test_set$Fare<-imputed.data1[,9]
test_final<-subset(test_set,select = c(Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,travelers))
colSums(is.na(test_final))
test_final$Embarked<-factor(test_final$Embarked,levels = c("S","C","Q"),labels = c(1,2,3))
test_final$Sex<-factor(test_final$Sex,levels = c("male","female"),labels = c(1,2))
head(test_final)
test_final[,c(3,4,5,6,8)]<-scale(test_final[,c(3,4,5,6,8)])
pred_test<-predict(caret.cv,test_final)
#pred_test$PassengerId<-test_set$PassengerId
#pred_test
test_set$Survived<-pred_test
head(test_set)
head(test_set$Survived)
output_final<-subset(test_set,select =c(PassengerId,Survived))
write.csv(output_final,"D:\\google downloads\\output.csv",row.names = FALSE)
