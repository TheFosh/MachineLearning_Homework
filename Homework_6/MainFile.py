import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# https://www.kaggle.com/datasets/udaymalviya/bank-loan-data
df = pd.read_csv("cleaned_data.csv")

# Select variables
y = df.iloc[:, -1].copy().to_numpy()
X = df.iloc[:, :-1].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# parameters = {"C": np.linspace(5, 1, num=10)}
# grid_search = GridSearchCV(reg, param_grid=parameters, cv=5)
# grid_search.fit(X_train, y_train)
# score_df = pd.DataFrame(grid_search.cv_results_)
# print(score_df[['param_C', 'mean_test_score', 'rank_test_score']])

# bag_clf = BaggingClassifier(reg)
# parameters = {"n_estimators": range(30, 40)}
# grid_search = GridSearchCV(bag_clf, param_grid=parameters, cv=5)
# grid_search.fit(X_train, y_train)
# score_df = pd.DataFrame(grid_search.cv_results_)
# print(score_df[['param_n_estimators', 'mean_test_score', 'rank_test_score']])

best_C = 4.5
best_n = 32
reg = LinearSVC(C=best_C)
bag_clf = BaggingClassifier(reg, max_features=1.0, n_estimators=best_n, verbose=3, oob_score=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

# Score (Train): ~ 0.817
# Score (Test): ~ 0.803
# OOB Score: ~ 0.814
print(f"Score (Train): {bag_clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {bag_clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {bag_clf.oob_score_:.3f}")

cm = confusion_matrix(y_test, bag_clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=bag_clf.classes_)
disp_cm.plot()
plt.show()

clf = RandomForestClassifier()
parameters = {"max_depth": range(2, 16)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
max_depth = grid_search.best_params_["max_depth"]

clf = RandomForestClassifier(max_depth=max_depth, oob_score=True, verbose=3)
clf.fit(X_train, y_train)

importances = pd.DataFrame(clf.feature_importances_, index=df.columns[1:])
importances.plot.bar()
plt.show()

print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

### All data
# df = pd.read_csv("cleaned_data.csv")
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
# reg = LinearSVC(C=best_C)
# bag_clf = BaggingClassifier(reg, max_features=1.0, n_estimators=best_n, verbose=3, oob_score=True, n_jobs=-1)
# bag_clf.fit(X_train, y_train)
#
# print(f"Score (Train): {bag_clf.score(X_train, y_train):.3f}")
# print(f"Score (Test): {bag_clf.score(X_test, y_test):.3f}")
# print(f"OOB Score: {bag_clf.oob_score_:.3f}")
#
# clf = RandomForestClassifier(max_depth=max_depth, oob_score=True, verbose=3)
# clf.fit(X_train, y_train)
#
# print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
# print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
# print(f"OOB Score: {clf.oob_score_:.3f}")
