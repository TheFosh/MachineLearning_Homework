import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
df = pd.read_csv("card_transdata.csv")
fraud = df["fraud"]
df = df.iloc[:, :-1]
X = df.copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, fraud, test_size=0.001)

std = np.std(X_test)

new_df = X_test / std

clt = kMeans(n_clusters=2)
clt.fit(new_df)

plt.scatter(new_df[:, 1], new_df[:, 2], c=clt.labels_, s=10.0)
plt.show()

for i in range(2):
    correct_count = 0
    incorrect_count = 0
    for f in y_test[clt.labels_ == i]:
        if f == i:
            correct_count += 1
        else:
            incorrect_count += 1
    size = len(y_test[clt.labels_ == i])
    print(f"For label {i}: ")
    print(f"Category Size: {size}")
    print(f"Correct ratio: {correct_count / len(y_test[clt.labels_ == i])}")
    print(f"Incorrect ratio: {incorrect_count / len(y_test[clt.labels_ == i])}\n")
