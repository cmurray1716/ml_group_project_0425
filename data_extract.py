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
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0


#Data came already separated into a test and train file but will combine these so I have more control over the split.
df_1 = pd.read_csv("C:/Users/Colm/Downloads/genre_train_test/train.csv")
df_2 = pd.read_csv("C:/Users/Colm/Downloads/genre_train_test/test.csv")

#the data contains a few variables: Artist name, song name, lyrics and genre. 
#the model will only use the lyrics for classification so the other columns will be removed.

#For some reason only the training data has a language value so I will add this to the test file.
#Only english lyrics will be used for this project as they are by far the most common language in the dataset.
#Including other languages in the same model could result in underfitting.
df_1 = df_1[df_1["Language"] == "en"][["Lyrics","Genre"]]

#Using langdetect to add language to the test file, and returning only english lyrics.
df_2["Language"] = df_2["Lyrics"].apply(detect)
df_2 = df_2[df_2["Language"] == "en"][["Lyrics","Genre"]]

#combining both files
df = pd.concat([df_1,df_2])

df.to_csv("C:/Users/Colm/Downloads/genre_full.csv")

