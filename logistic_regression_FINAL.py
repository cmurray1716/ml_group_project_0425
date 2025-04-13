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
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks, CondensedNearestNeighbour

#reading in the data
df = pd.read_csv("C:/Users/Colm/Downloads/genre_full.csv")
#converting lyrics to lowercase
df["Lyrics"] = df["Lyrics"].str.lower()

#Creating a 70/30 test/train split
train_df, val_df = train_test_split(df, test_size=0.30, random_state=47)
trainx = train_df["Lyrics"].values
testx = val_df["Lyrics"].values
trainy = train_df['Genre'].values
testy = val_df['Genre'].values

#Using the logistic regression function in scikit
log_model = LogisticRegression(max_iter=10000, solver = "sag", penalty = "l2", C = 36.84842105263157)
#Using the TFID vectorizer to convert the lyrics to numbers
vectorizer =  TfidfVectorizer(min_df = 0.3968421052631579, max_features= 588)
trainx_v = vectorizer.fit_transform(trainx)
testx_v = vectorizer.fit_transform(testx)

#Random undersampling to 
rus = RandomUnderSampler(random_state=54)

trainx_v_resampled, trainy_resampled = rus.fit_resample(trainx_v, trainy)

#Running the model
log_model.fit(trainx_v_resampled, trainy_resampled)
#Testing the model
predictions = log_model.predict(testx_v)

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