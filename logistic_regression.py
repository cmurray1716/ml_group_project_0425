import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import FeatureHasher
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

#reading in the data
df = pd.read_csv("C:/Users/Colm/Downloads/genre_full.csv")
#converting lyrics to lowercase
df["Lyrics"] = df["Lyrics"].str.lower()

#Creating a 70/30 test/train split
train_df, val_df = train_test_split(df, test_size=0.30, random_state=1000)
trainx = train_df["Lyrics"].values
testx = val_df["Lyrics"].values
trainy = train_df['Genre'].values
testy = val_df['Genre'].values

#Using the logistic regression function in scikit
log_model = LogisticRegression(max_iter=10000)
#Using the TFID vectorizer to convert the lyrics to numbers
vectorizer =  TfidfVectorizer(max_features=500)
trainx_v = vectorizer.fit_transform(trainx)
testx_v = vectorizer.fit_transform(testx)

#Running the model
log_model.fit(trainx_v, trainy)
#Testing the model
predictions = log_model.predict(testx_v)

print(confusion_matrix(testy, predictions))
print(accuracy_score(testy, predictions))