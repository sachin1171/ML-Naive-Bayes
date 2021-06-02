############################Problem 1###################################

import pandas as pd
import numpy as np

# Loading the data set
salary_train = pd.read_csv("C:/Users/usach/Desktop/Naive byes/SalaryData_Train.csv",encoding = "ISO-8859-1")

salary_test = pd.read_csv("C:/Users/usach/Desktop/Naive byes/SalaryData_Test.csv",encoding = "ISO-8859-1")

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB

x = {' <=50K' : 0 , ' >50K' : 1}
salary_train.Salary = [x[item] for item in salary_train.Salary]
salary_test.Salary = [x[item] for item in salary_test.Salary]

#getting dummy values for train dataset
salary_train_dummies = pd.get_dummies(salary_train) 
salary_train_dummies.drop(['Salary'] , axis = 1 , inplace= True)
salary_train_dummies.head(3)
#checking for na values
salary_train_dummies.columns[salary_train_dummies.isna().any()]

#getting dummy values for test dataset
salary_test_dummies = pd.get_dummies(salary_test) 
salary_test_dummies.drop(['Salary'] , axis = 1 , inplace= True)
salary_test_dummies.head(3)

#checking for na values
salary_train_dummies.columns[salary_test_dummies.isna().any()]
salary_test_dummies.columns[salary_test_dummies.isna().any()]

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(salary_train_dummies, salary_train.Salary)

# Evaluation on Test Data
test_pred = classifier_mb.predict(salary_test_dummies)
accuracy_test = np.mean(test_pred == salary_test.Salary)
accuracy_test

from sklearn.metrics import accuracy_score
accuracy_score(test_pred, salary_test.Salary) 

pd.crosstab(test_pred, salary_test.Salary)

# Training Data accuracy
train_pred = classifier_mb.predict(salary_train_dummies)
accuracy_train = np.mean(train_pred == salary_train.Salary)
accuracy_train

# Multinomial Naive Bayes changing default alpha for laplace smoothing
classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(salary_train_dummies, salary_train.Salary)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(salary_test_dummies)
accuracy_test_lap = np.mean(test_pred_lap == salary_test.Salary)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, salary_test.Salary) 

pd.crosstab(test_pred_lap, salary_test.Salary)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(salary_train_dummies)
accuracy_train_lap = np.mean(train_pred_lap == salary_train.Salary)
accuracy_train_lap

####################################Problem 2##################################
import pandas as pd
import numpy as np

# Loading the data set
car_data = pd.read_csv("C:/Users/usach/Desktop/Naive byes/NB_Car_Ad.csv",encoding = "ISO-8859-1")
#droping first column which is of nominal type
car_data = car_data.iloc[:,1:]

#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
car_data [['Age','EstimatedSalary']] = scaler.fit_transform(car_data [['Age','EstimatedSalary']])

#splitting data into train and test
from sklearn.model_selection import train_test_split
car_train, car_test = train_test_split(car_data, test_size = 0.2)

#getting dummy values for train dataset
car_train_dummies = pd.get_dummies(car_train) 
car_train_dummies.drop(['Purchased'] , axis = 1 , inplace= True)
car_train_dummies.head(3)

#getting dummy values for test dataset
car_test_dummies = pd.get_dummies(car_test) 
car_test_dummies.drop(['Purchased'] , axis = 1 , inplace= True)
car_test_dummies.head(3)

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB(alpha = 3)
classifier_mb.fit(car_train_dummies, car_train.Purchased)

# Evaluation on Test Data
test_pred = classifier_mb.predict(car_test_dummies)
accuracy_test = np.mean(test_pred == car_test.Purchased)
accuracy_test

from sklearn.metrics import accuracy_score
accuracy_score(test_pred, car_test.Purchased) 

pd.crosstab(test_pred, car_test.Purchased)

# Training Data accuracy
train_pred = classifier_mb.predict(car_train_dummies)
accuracy_train = np.mean(train_pred == car_train.Purchased)
accuracy_train

#since the model is giving less accuracy it is considered to be not a efficient model
#so we go for Gaussian model

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB as GB
classifier_mb_lap = GB()
classifier_mb_lap.fit(car_train_dummies, car_train.Purchased)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(car_test_dummies)
accuracy_test_lap = np.mean(test_pred_lap == car_test.Purchased)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, car_test.Purchased) 

pd.crosstab(test_pred_lap, car_test.Purchased)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(car_train_dummies)
accuracy_train_lap = np.mean(train_pred_lap == car_train.Purchased)
accuracy_train_lap


##################################Problem 3########################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
tweet = pd.read_csv("C:/Users/usach/Desktop/Naive byes/Disaster_tweets_NB.csv",encoding = "ISO-8859-1")
tweet = tweet.iloc[:,3:5]
# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open("C:/Users/usach/Desktop/Naive byes/stop.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

tweet.text = tweet.text.apply(cleaning_text)

# removing empty rows
tweet = tweet.loc[tweet.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

tweet_train, tweet_test = train_test_split(tweet, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of tweet texts into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer = split_into_words).fit(tweet.text)

# Defining BOW for all tweets
all_tweet_matrix = tweet_bow.transform(tweet.text)

# For training messages
train_tweet_matrix = tweet_bow.transform(tweet_train.text)

# For testing messages
test_tweet_matrix = tweet_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire tweet
tfidf_transformer = TfidfTransformer().fit(all_tweet_matrix)

# Preparing TFIDF for train tweet
train_tfidf = tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweet_test.target) 

pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, tweet_test.target) 

pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap
