import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

import SVCPlot

df = pd.read_csv("Heart_Disease_Prediction.csv")
df = df[["Max HR", "Cholesterol", "Heart Disease"]]
y = df.iloc[:, -1].copy().to_numpy()
X = df.iloc[:, :-1].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

C = 1.0
clf = SVC(C=C, kernel="rbf")

parameters = {"C": np.linspace(0.01, 2, 20)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_C', 'mean_test_score', 'rank_test_score']])

best_C = grid_search.best_params_["C"]
print(best_C)

clf = SVC(C=best_C, kernel="rbf")
clf.fit(X_train, y_train)
SVCPlot.plot(X, y, clf)

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
