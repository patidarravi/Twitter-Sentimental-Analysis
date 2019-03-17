# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 23:22:02 2019

@author: Rvi
"""

#################### Importing the Libraries ############################
import numpy as np                          #For Array
import pandas as pd                         #For DataFrame
import re                                   #For text cleaning Regular Expersion
import nltk                                 # For Natural Language Processing
nltk.download("stopwords")                  # Downloading Stopwords
from nltk.corpus import stopwords           # # Importing Stopwords
from nltk.stem.porter import PorterStemmer  # Importing Stemmer for Exchanging words with root words
from sklearn.feature_extraction.text import CountVectorizer  # For Creating Bag of Words Model
from sklearn.model_selection import train_test_split         # For Splitting the dataset into Training and Test Set
from sklearn.linear_model import LogisticRegression     #Logistic Regression model

# ******************** Importing Datasets for Training ************************
cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv(r"C:\Users\Rvi\spyder\twitter_sentimental.csv",header=None,names=cols, encoding='latin-1')


# ********************Droppig the Unwanted Columns ************************
data.drop(['id','date','query_string','user'],axis=1,inplace=True)

# ******************** function to clean tweet text ***************************
def clean_tweets(data1):
    review_list=[]
    review= re.sub(r'https?://[A-Za-z0-9./]+','',data1)
    review= re.sub(r'@[a-zA-Z0-9]+',"",review)
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    review_list.append(review)
    return review_list
	
# ******************** Cleaning Tweets of Training Dataset ********************	
review_list=[]
for i in range(790000,810001):
    review_list+=clean_tweets(data.text[i])
    if(i%100==0):
        print("%d tweets cleaned"%(i-790000))

# ******************** Creating Bag Of Words Model ****************************    

cv = CountVectorizer(max_features =17000 )
X = cv.fit_transform(corpus).toarray()  
y = l.loc[790000:810000,"sentiment"].values

# ******************** Splitting the dataset into Training and Test Set *******

X_train,X_test,y_train,ytest = train_test_split(X,y,test_size= 0.30,random_state =0)

# ******************** Fitting Machine Learning Model to Training Test *******************

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0) 
classifier.fit(X_train,y_train)

# ******************** Predict The Test Result *****************************

y_pred = classifier.predict(X_test)

# ******************** Lets Making A Confusion Matrix *************************
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(ytest,y_pred)
ac= accuracy_score(ytest,y_pred)
print("accuracy:",ac)

# ******************** TWITTER SENTIMENTAL ANALYSIS STARTS ********************

import tweepy 
from tweepy import OAuthHandler

apiKey="BENxu9HYv9TFCR9xvS9a8fIiS"
apiSecret="HsfY7LVoAG45Q92fGFBcPmpcnwLIWwqHiZgAqEp9GMyhr3si4Y"

accessKey="2976539238-K2uy5Mhz9p4gylw2IQEp6q2TGLXkbXEb5WgQk4e"
accessSecret="kWZPLaWz4iKOGEKCwrIVRNLjigt9bCWC0UTIZo15LDmdS"

auth=tweepy.auth.OAuthHandler(apiKey,apiSecret)
auth.set_access_token(accessKey,accessSecret)
api=tweepy.API(auth)

SEARCH=input("Enter the tweet to be search  :")
tweet_fetch=api.search(q=SEARCH,lang="en",count = 200)
#print("No. of tweet fetch",len(tweet_fetch))


# ******************** Parsing Tweets *************************

tweets= []
for tweet in tweet_fetch:
                parsed_tweet = {} 
                parsed_tweet['text'] = tweet.text 
                if tweet.retweet_count > 0: 
                    if parsed_tweet not in tweets: 
                        tweets.append(parsed_tweet) 
                else: 
                    tweets.append(parsed_tweet)




