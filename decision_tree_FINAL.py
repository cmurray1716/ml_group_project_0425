import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import FeatureHasher
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from skopt import BayesSearchCV
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, CondensedNearestNeighbour
from imblearn.pipeline import Pipeline

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
dt = DecisionTreeClassifier(criterion = "log_loss" , max_depth = 12, min_samples_leaf = 5)
#Using the TFID vectorizer to convert the lyrics to numbers
vectorizer =  TfidfVectorizer(min_df =  0.3194736842105263, max_features = 924)
trainx_v = vectorizer.fit_transform(trainx)
testx_v = vectorizer.fit_transform(testx)

#Running the model
dt.fit(trainx_v, trainy)
#Testing the model
predictions = dt.predict(testx_v)

print("Confusion Matrix:")
print(confusion_matrix(testy, predictions))
print("Accuracy Score:")
print(accuracy_score(testy, predictions))
print("Precision Score:")
print(precision_score(testy, predictions, average = "weighted"))
print("Recall Score:")
print(recall_score(testy, predictions, average = "weighted"))
print("F1 Score:")
print(f1_score(testy, predictions, average = "weighted"))