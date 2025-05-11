import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# https://www.kaggle.com/datasets/asinow/car-price-dataset
df = pd.read_csv("car_price_dataset.csv")

X = df[["Engine_Size", "Mileage", "Doors", "Owner_Count", "Price"]].copy().to_numpy()

y = df.loc[:, "Brand"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

###### Decision Tree ######
clf = DecisionTreeClassifier(class_weight="balanced")
parameters = {"max_depth": range(2, 16)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
max_depth = grid_search.best_params_["max_depth"]
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])
clf = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced")
clf.fit(X_train, y_train)

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
###########################################

##### Random Forest #####
clf = RandomForestClassifier(oob_score=True, verbose=3, class_weight="balanced")
parameters = {"max_depth": range(2, 10)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
max_depth = grid_search.best_params_["max_depth"]
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])
clf = RandomForestClassifier(max_depth=max_depth, class_weight="balanced")
clf.fit(X_train, y_train)

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
