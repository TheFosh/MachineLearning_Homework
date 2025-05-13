# Homework for Machine Leanring Course
## Homework 4
For this homework, we are to make a support vector classifier with a kernel of a radial basis function. 

The dataset must be a classification type of data set with quantifiable variables. I found this dataset on [heart disease](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction) to be good for this task.

For this, I trained the algorithm on only two columns of data so the classifier can be displayed. An example diagram of this is shown here.
![Class_Diagram](https://github.com/TheFosh/MachineLearning_Homework/blob/main/Images/Homework4_Display.png)

As well, this produces the following confusion matrix.
![Class_CM](https://github.com/TheFosh/MachineLearning_Homework/blob/main/Images/Homework4_CM.png)

## Homework 5
For this homework, we were to create a decison tree and find an optimal max depth. As well, try to see the results of a random forest trained on a data set of 1000 - 10000 observations.

### Desicion Tree
For the decision tree, I found this dataset on [car brands](https://www.kaggle.com/datasets/asinow/car-price-dataset). Here is an example of confusion matrix produced after performing a grid search to find a good max depth.
![DecTree_Ex](https://github.com/TheFosh/MachineLearning_Homework/blob/main/Images/Homework5_Display.png).

### Forrest
In order to train the random forrest, I found this dataset on [student grade proformance](https://www.kaggle.com/datasets/adilshamim8/student-performance-and-learning-style). Using this and a grid seach for the optimal parameters, an exmaple run of the algorithm produced the following confusion matrix.
![RanFor_CM](https://github.com/TheFosh/MachineLearning_Homework/blob/main/Images/Homework5_CM.png)

## Homework 6
For this homework, we were to create a random forest and bagging support vector classifier.

Our data for this should for a type of classification problem and only using quantitaive features. I found this dataset on [loan data](https://www.kaggle.com/datasets/udaymalviya/bank-loan-data) that has information about a person and where are not their loan was accepted or rejected. This accecpted and rejeted data was used as the classification label.

### Random Forrest

Using all the quantitative features in the dataset for a random forrest, the following confusion matrix is produced.
![RandomForrest_CM](https://github.com/TheFosh/MachineLearning_Homework/blob/main/Images/Homework6_RanForCM.png)

### Bagging SVC

Using all the quantitative features in the dataset for a Linear Bagging SVC, the following confusion matrix is produced.
![BaggingSVC_CM](https://github.com/TheFosh/MachineLearning_Homework/blob/main/Images/Homework6_CM.png)

## Homework 7
For this homework we were tasked with simply trying something interesting with clustering.

For this I wanted to try to make a classification type of problem but with cluster as a means of classification.
The data I used for this was [card fraud](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud) data. Data was on different transactions and they would be labeled as fraud or not.

The results of this were seen in this chart.

![Homework7_Display](https://github.com/TheFosh/MachineLearning_Homework/blob/main/Images/Homework7_Display.png)

This shows the different groups that the clustering algorithm found. They are being hardcoded into 2 dimmensions for the sake of visuallizing the data.

Using the data for actually labeling these points, we can figure out how correct the method is.
The last bit of code in the main file simulates a confusion matrix by given a the ratios of correct and incorrect.
Example run:

For label 0:

Category Size: 980, Correct ratio: 0.926530612244898, Incorrect ratio: 0.07346938775510205

For label 1:

Category Size: 20, Correct ratio: 0.45, Incorrect ratio: 0.55
