# Homework for Machine Leanring Course
## Homework 4

## Homework 5

## Homework 6

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
Category Size: 980
Correct ratio: 0.926530612244898
Incorrect ratio: 0.07346938775510205

For label 1: 
Category Size: 20
Correct ratio: 0.45
Incorrect ratio: 0.55
