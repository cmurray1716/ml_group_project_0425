#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the combined dataset
df = pd.read_csv(r"C:/Users/miche/Documents/CA4 Machine Learning/combinedDF.csv")

# Preprocess: lowercase lyrics
df["Lyrics"] = df["Lyrics"].str.lower()


# In[2]:


# Split into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.30, random_state=1000)
trainx = train_df["Lyrics"].values
testx = val_df["Lyrics"].values
trainy = train_df['Genre'].values
testy = val_df['Genre'].values



# In[3]:


# Vectorize with TF-IDF
vectorizer = TfidfVectorizer(min_df=0.01)  # You can tune this
trainx_v = vectorizer.fit_transform(trainx)
testx_v = vectorizer.transform(testx)

# Train Naive Bayes model
nb = MultinomialNB()
nb.fit(trainx_v, trainy)

# Make predictions
predictions = nb.predict(testx_v)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(testy, predictions))
print("Accuracy Score:")
print(accuracy_score(testy, predictions))
print("Precision Score:")
print(precision_score(testy, predictions, average="weighted"))
print("Recall Score:")
print(recall_score(testy, predictions, average="weighted"))
print("F1 Score:")
print(f1_score(testy, predictions, average="weighted"))


# In[4]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Create pipeline
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Define hyperparameter grid aligned with Random Forest search structure
param_dist_nb = dict(
    nb__alpha=np.linspace(0.01, 1.0, num=10),                  # Naive Bayes smoothing
    tfidf__min_df=np.linspace(0.01, 0.5, num=10),              # min document frequency
    tfidf__max_features=np.arange(200, 1100, 200)              # TF-IDF max features
)

# Random SearchCV
random_search_nb = RandomizedSearchCV(
    estimator=pipeline_nb,
    param_distributions=param_dist_nb,
    n_iter=3,
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

# Fit model
random_search_nb.fit(trainx, trainy)

# Output best results
print("Best Parameters:", random_search_nb.best_params_)
print("Best Score:", random_search_nb.best_score_)


# In[5]:


from sklearn.metrics import classification_report

# Extract best TF-IDF vectorizer and model from search
best_tfidf_nb = random_search_nb.best_estimator_.named_steps['tfidf']
best_nb_model = random_search_nb.best_estimator_.named_steps['nb']

# Transform the data
trainx_nb_transformed = best_tfidf_nb.fit_transform(trainx)
testx_nb_transformed = best_tfidf_nb.transform(testx)

# Fit the best model on full training data
best_nb_model.fit(trainx_nb_transformed, trainy)

# Predict on validation data
nb_predictions = best_nb_model.predict(testx_nb_transformed)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(testy, nb_predictions))

print("\nClassification Report:")
print(classification_report(testy, nb_predictions))

print("Accuracy Score:", accuracy_score(testy, nb_predictions))
print("Precision Score:", precision_score(testy, nb_predictions, average='weighted'))
print("Recall Score:", recall_score(testy, nb_predictions, average='weighted'))
print("F1 Score:", f1_score(testy, nb_predictions, average='weighted'))


# In[ ]:




