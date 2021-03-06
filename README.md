# PredictingActivityFitbitApplewatch


*This repository contains all files used for predicting activity from Fitbit and Apple Watch Data.*

## Project Description
This project is an exercise in supervised learning which uses experimental data collected from 46 participants who engaged in various levels of strenuous activity while wearing a Fitbit and Apple Watch. From the data collected I explore how well various classifiers perform at being able to predict the activity of the participants given the data recorded from the device. Ultimately, the ensemble learners Random Forest and Gradient Boosting perform the best recording a 90% accuracy on validation data. 

Additionally I perform Principle Component Analysis and K means Clustering as unsupervised methods to help further understand the predictor space.

### Methods Used
* Inferential Statistics
* Unsupervised Learning PCA and K-Means Clustering
* Ensemble Learners (Stochastic Gradient Boosting and Random Forests)
* Naive Bayes
* K Nearest Neighbors
* Hyper parameter tuning
* Cross Validation
* Data Visualization
* Predictive Modeling

### Technology Used
* python
* sklearn
* seaborn
* pandas
* numpy
* Jupyter Lab

## Files
* aw_fb_data.csv (Data for analysis from Kaggle.com)
* Study_Predict_Behavior.pdf (Paper From Kaggle.com)
* helper_class_methods.py (python module with classes and functions used to aid analysis)
* Fitbit_Activity_Classifier.ipynb (Jupyter Notebook With EDA and Modelling)

## Project Takeaways
* The EDA showed a lot of noise in many of the variables in the predictor space

* Interestingly, the ensemble models Random Forest and Gradient Boosting performed well on accurately predicting the activity in the validation data set. On both the Fitbit and Apple Watch data the best performance was from the Gradient Boost model with around 90 and 84 percent accuracy respectively. The plus side of using the Stochastic Gradient Boost (when each weak classifier is limited in the predictors it can choose) is that we get to look at the variable importance feature. This also true of the Random Forest model.

* We saw that the feature importance varied between the data collected on Apple Watch and Fitbit. It seems the ensemble models preferred the steps and calories on the Fitbit data and the heart rate related variables on the Apple Watch data. 

* It would be interesting to see how these models would fair if the same data was collected using more current versions of these devices.
 


## Kaggle Link
https://www.kaggle.com/code/eigenvalue42/fitbit-vs-apple-watch-classifying-activity