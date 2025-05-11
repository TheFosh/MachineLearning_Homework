import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style
df = pd.read_csv("student_performance_large_dataset.csv")

X = df[["Age", "Study_Hours_per_Week", "Assignment_Completion_Rate (%)", "Sleep_Hours_per_Night", "Attendance_Rate (%)",
        "Time_Spent_on_Social_Media (hours/week)"]].copy().to_numpy()

y = df.loc[:, "Final_Grade"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier(class_weight="balanced")
parameters = {"max_depth": range(2, 16)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
max_depth = grid_search.best_params_["max_depth"]
print(max_depth)

clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced")
clf.fit(X_train, y_train)
print(f"Score: {clf.score(X_test, y_test):.3f}")

plt.figure(figsize=(10, 10))
plot_tree(clf, filled=True,
          feature_names=["Age", "Study_Hours_per_Week", "Assignment_Completion_Rate (%)", "Sleep_Hours_per_Night",
                         "Attendance_Rate (%)",
                         "Time_Spent_on_Social_Media (hours/week)"], class_names=clf.classes_, fontsize=7)
plt.show()

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
