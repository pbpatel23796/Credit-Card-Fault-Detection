# Credit-Card-Fault-Detection
This project classifies various credit card transactions into fraudulent and non-fraudulent transactions using various machine learning algorithms and data pre-processing techniques in python.

# Dataset Description
The dataset used in this project is imported from a CSV file, which was publicly available on kaggle. The dataset was collected and analysed during a research collaboration of Worldline and the Machine Learning Group of ULB (University Libre de Bruxelles) on big data mining and fraud detection by Andrea Dal Pozzolo and his peers. It contains a total of 284,807 transactions made in September 2013 by European cardholders and the data set contains 492 fraud transactions, which is highly imbalanced. Due to the confidentiality issue, only 28 features obtained after principal components analysis of actual attributes are provided in the dataset. Only the time and the amount data are not transformed and are provided as such. The feature ’Time’ contains the seconds elapsed between each transaction and the first transaction in the dataset. The attribute ’Amount’ is the transaction Amount. Finally, attribute ’Class’ is the type of transaction label and it takes value 1 in case of fraud and 0 otherwise. The dataset does not contain any inconsistency of missing values. The figures below provide some useful insights into the dataset. They show how often the fraudulent transactions occur and the range of amount in fraudulent transactions.

![image](https://user-images.githubusercontent.com/28703328/116040388-f8a76a00-a639-11eb-8ae1-01bdc57dae29.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/28703328/116040388-f8a76a00-a639-11eb-8ae1-01bdc57dae29.png" width="350">
  <img src="https://user-images.githubusercontent.com/28703328/116040564-2d1b2600-a63a-11eb-8735-3f6e1b4e74f0.png" width="350">
</p>

![image](https://user-images.githubusercontent.com/28703328/116040564-2d1b2600-a63a-11eb-8735-3f6e1b4e74f0.png)

# Source of Dataset
https://www.kaggle.com/mlg-ulb/creditcardfraud

# Machine Learning Algorithms used
1. Random Forest Classifier
2. XGB Classifier
3. LGBM Classifier
4. SMOTE
5. BorderlineSMOTE
6. SVMSMOTE
7. ADASYN

# Proposed Model
![image](https://user-images.githubusercontent.com/28703328/116039870-47a0cf80-a639-11eb-921e-78fce28331b2.png)

# Code Execution
There are various ways by which you can execute the python code
1. Use Python IDE: Download the code and the dataset on your computer and use the python IDE to execute the code
2. Use Google Colab: Copy the code and import the dataset into Google Colab and execute the code
