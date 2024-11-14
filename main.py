import numpy as np
import pandas as pd

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("train.csv") # tweets and disaster labels
test_df = pd.read_csv("test.csv") # tweets to predict

# example tweets for reference
non_disaster_tweet = train_df[train_df["target"] == 0]["text"].values[1]
disaster_tweet = train_df[train_df["target"] == 1]["text"].values[1]

# convert text to numerical vectors
count_vectorizer = feature_extraction.text.CountVectorizer()

# transform tweets into a matrix
# each column - word, each cell - word frequency
train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df["text"])

# Creates a Ridge Classifier (a linear classifier with regularization)
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

clf.fit(train_vectors, train_df["target"])

sample_submission = pd.read_csv("sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.to_csv("submission.csv", index=False)
