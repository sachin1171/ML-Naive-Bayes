############################################Problem 1######################################
# Import the salary dataset
library(readr)
salary_train <- read.csv(file.choose())
salary_test <- read.csv(file.choose())

#pre processing training dataset
x <- salary_train$Salary
salary_train$Salary <- ifelse(x == " <=50K",1,2)
View(x)
cols <- c("age" ,"workclass" , "education" ,"educationno" , "maritalstatus" , "occupation","relationship","race" ,"sex", "native" ,"Salary")
salary_train[cols] <- lapply(salary_train[cols] , factor)
str(salary_train)

#pre processing testing dataset
y <- salary_test$Salary
salary_test$Salary <- ifelse(y == " <=50K",1,2)
View(y)
cols <- c("age" ,"workclass" , "education" ,"educationno" , "maritalstatus" , "occupation","relationship","race" ,"sex", "native" ,"Salary")
salary_test[cols] <- lapply(salary_test[cols] , factor)
str(salary_test)

# examine the Salary variable more carefully
str(salary_train$Salary)
table(salary_train$Salary)

# proportion of salary 0(salary = <=50K ) and 1 (salary = >50K )
prop.table(table(salary_train$Salary))

#Training a model on the data
install.packages("e1071")
library(e1071)
## building naiveBayes classifier.
model <- naiveBayes(Salary~., data = salary_train)
model
### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.
model2 <- naiveBayes(Salary~., data = salary_train,laplace = 3)
model2

##  Evaluating model performance with out laplace
salary_test_pred <- predict(model, salary_test)

# Evaluating model performance after applying laplace smoothing
salary_test_pred_lap <- predict(model2, salary_test)

## crosstable without laplace
install.packages("gmodels")
library(gmodels)

CrossTable(salary_test_pred, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(salary_test_pred == salary_test$Salary)
test_acc

## crosstable of laplace smoothing model
CrossTable(salary_test_pred_lap, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(salary_test_pred_lap == salary_test$Salary)
test_acc_lap

# On Training Data without laplace 
salary_train_pred <- predict(model, salary_train)

# train accuracy
train_acc = mean(salary_train_pred == salary_train$Salary)
train_acc


# prediction on train data for laplace model
salary_train_pred_lap <- predict(model2, salary_train)
salary_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(salary_train_pred_lap == salary_train$Salary)
train_acc_lap

#################################Problem 2#################################
# Import the car_ad dataset
library(readr)
car_ad <- read.csv(file.choose())
car_ad <- car_ad[c(-1)]

#pre processing dataset
cols <- c("Gender" , "Age" ,"Purchased")
cols <- c("Purchased")
car_ad[cols] <- lapply(car_ad[cols] , factor)
car_ad[c(3)] <- scale(car_ad[c(3)])
view

# examine the Purchased variable more carefully
str(car_ad$Purchased)
table(car_ad$Purchased)

# proportion of Purchased
prop.table(table(car_ad$Purchased))

#data partition
set.seed(1234)
ind <- sample(2 , nrow(car_ad) , replace = TRUE , prob = c(0.8 , 0.2))
train <- car_ad[ind == 1 , ]
test <- car_ad[ind == 2 , ]

#Training a model on the data
#install.packages("e1071")
library(e1071)

## building naiveBayes classifier without laplace
model <- naiveBayes(Purchased~., data = train)
model

##  Evaluating model performance without laplace
test_pred <- predict(model, test)

## crosstable
#install.packages("gmodels")
library(gmodels)

CrossTable(test_pred, test$Purchased,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(test_pred == test$Purchased)
test_acc

# On Training Data 
train_pred <- predict(model, train)

# train accuracy
train_acc = mean(train_pred == train$Purchased)
train_acc

## building naiveBayes classifier with laplace
model <- naiveBayes(Purchased~., data = train , laplace =3)
model

##  Evaluating model performance with laplace
test_pred <- predict(model, test)


## test accuracy for laplace model
test_acc <- mean(test_pred == test$Purchased)
test_acc

# On Training Data 
train_pred <- predict(model, train)

# train accuracy
train_acc = mean(train_pred == train$Purchased)
train_acc
################################Problem 3################################
# Import the salary dataset
library(readr)
tweet <- read.csv(file.choose())

tweet$target <- factor(tweet$target)

# examine the target variable more carefully
str(tweet$target)
table(tweet$target)

# proportion of fake or not fake tweets
prop.table(table(tweet$target))

# build a corpus using the text mining (tm) package
install.packages("tm")
library(tm)

str(tweet$text)

tweet_corpus <- Corpus(VectorSource(tweet$text))
tweet_corpus <- tm_map(tweet_corpus, function(x) iconv(enc2utf8(x), sub='byte'))

# clean up the corpus using tm_map()
corpus_clean <- tm_map(tweet_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# create a document-term sparse matrix
tweet_dtm <- DocumentTermMatrix(corpus_clean)
head(tweet_dtm[1:10, 1:30])

# To view DTM we need to convert it into matrix first
dtm_matrix <- as.matrix(tweet_dtm)
str(dtm_matrix)

View(dtm_matrix[1:10, 1:20])

colnames(tweet_dtm)[1:50]

# creating training and test datasets
tweet_train <- tweet[1:5329, ]
tweet_test  <- tweet[5330:7613, ]

tweet_corpus_train <- corpus_clean[1:5329]
tweet_corpus_test  <- corpus_clean[5330:7613]

tweet_dtm_train <- tweet_dtm[1:5329, ]
tweet_dtm_test  <- tweet_dtm[5330:7613, ]

# check that the proportion of fake tweet is similar
prop.table(table(tweet$target))

prop.table(table(tweet_train$target))
prop.table(table(tweet_test$target))

# indicator features for frequent words
# dictionary of words which are used more than 5 times
tweet_dict <- findFreqTerms(tweet_dtm_train, 5)

tweet_dtm_train <- DocumentTermMatrix(tweet_corpus_train, list(dictionary = tweet_dict))
tweet_dtm_test  <- DocumentTermMatrix(tweet_corpus_test, list(dictionary = tweet_dict))

tweet_test_matrix <- as.matrix(tweet_dtm_test)
View(tweet_test_matrix[1:10,1:10])

# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
tweet_dtm_train <- apply(tweet_dtm_train, MARGIN = 2, convert_counts)
tweet_dtm_test  <- apply(tweet_dtm_test, MARGIN = 2, convert_counts)

View(tweet_dtm_train[1:10,1:10])

#Training a model on the data
install.packages("e1071")
library(e1071)
## building naiveBayes classifier.
tweet_classifier <- naiveBayes(tweet_dtm_train, tweet_train$target)
tweet_classifier

### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
tweet_lap <- naiveBayes(tweet_dtm_train, tweet_train$target,laplace = 2)
tweet_lap

##  Evaluating model performance with out laplace
tweet_test_pred <- predict(tweet_classifier, tweet_dtm_test)

# Evaluating model performance after applying laplace smoothing
tweet_test_pred_lap <- predict(tweet_lap, tweet_dtm_test)

## crosstable without laplace
library(gmodels)

CrossTable(tweet_test_pred, tweet_test$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(tweet_test_pred == tweet_test$target)
test_acc

## crosstable of laplace smoothing model
CrossTable(tweet_test_pred_lap, tweet_test$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(tweet_test_pred_lap == tweet_test$target)
test_acc_lap

# On Training Data without laplace 
tweet_train_pred <- predict(tweet_classifier, tweet_dtm_train)
tweet_train_pred

# train accuracy
train_acc = mean(tweet_train_pred == tweet_train$target)
train_acc

# prediction on train data for laplace model
tweet_train_pred_lap <- predict(tweet_lap, tweet_dtm_train)
tweet_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(tweet_train_pred_lap == tweet_train$target)
train_acc_lap
