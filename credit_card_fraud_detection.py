# Commented out IPython magic to ensure Python compatibility.
# -------------------------------------------------------------------------------------------------- #
# This project below classifies the fraudulent credit card transactions from the valid credit card   #
# transactions that are present in the dataset. Various machine learning algorithm used in this      #
# project such as Random Forest Classifier, XGB Classifier, LGBM Classifier, SMOTE, BorderlineSMOTE, #
# SVMSMOTE and ADASYN. The results are then compared by using normal dataset, under-sampled dataset  #
# and over-sampled dataset for each and every machine learning algorithm mentioned earlier.          #
# -------------------------------------------------------------------------------------------------- #
import pandas as pd
import numpy as np

# import sklearn library for Random Forest Classifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, roc_curve, auc, average_precision_score

# import xgboost library for XGBoost Classifier
from xgboost import XGBClassifier

# import lightgbm library for LGB Classifier
from lightgbm import LGBMClassifier

# import libraries for over sampling 
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN

# import matplotlib library to plot the graphs
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# This function imports the dataset file and stores it in dataset variable
# and returns the dataset variable
def import_dataset():
  dataset = pd.read_csv("creditcard.csv")
  dataset.head()

  return dataset

# This function visualizes dataset and shows useful information about the transactions
def visualize_dataset(dataset):
  fig, (ax1, ax2) = plt.subplots(2,1, sharex = True, figsize = [6,4])

  # plots the histogram of transactions versus time
  ax1.hist(dataset.Time[dataset.Class == 1], bins = 50)
  ax1.set_title("Fraudulent")
  ax2.hist(dataset.Time[dataset.Class == 0], bins = 50)
  ax2.set_title("Non-Fraudulent")

  plt.xlabel('Time (in Seconds)')
  plt.ylabel('Number of Transactions')
  plt.show()

  fig, (ax1, ax2) = plt.subplots(2,1, sharex = True, figsize = [6,4])

  # plots the histogram of amount versus transactions
  ax1.hist(dataset.Amount[dataset.Class == 1], bins = 30)
  ax1.set_title("Fraudulent")
  ax2.hist(dataset.Amount[dataset.Class == 0], bins = 30)
  ax2.set_title("Non-Fraudulent")

  plt.xlabel('Amount in $')
  plt.ylabel('Number of Transactions')
  plt.yscale('log')
  plt.show()

  # plots the bar graph to show imbalance dataset
  plt.bar(['Valid','Fraud'], list(dataset['Class'].value_counts()))
  plt.show()

  Class = [len(dataset.loc[dataset.Class == 1]), len(dataset.loc[dataset.Class == 0])]
  pd.Series(Class, index = ['Fraudulent', 'Non-fraudulent'], name = 'target')

  #Percentage of minority(fraudulent) class
  print('% of Fraudulent Class = {:.3f}%'.format(len(dataset[dataset.Class == 1])*100 / len(dataset)))

# This function normalizes the values of 'Amount' column to check whether that column holds any 
# high correlated values to other columns which can be useful to predict the fraudulent transactions
# This function returns the normalized dataset
def normalize_dataset(dataset):
  dataset["Normalized_Amount"] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1, 1))
  #Drop time & amount variable
  dataset = dataset.drop(['Time', 'Amount'], axis = 1)
  dataset.head()

  return dataset

# This function plots the correlation matrix of the normalized dataset which shows the correlation among
# the columns of the dataset
def plot_correlation_matrix(dataset):
  corrmat = dataset.corr() 
  fig = plt.figure(figsize = (12, 9)) 
  sns.heatmap(corrmat, vmax = .8, square = True)
  plt.show()

# This function splits the dataset into 70% training set and 30% testing set and returns the 
# parameters used to train the models
def split_dataset(dataset):
  X = dataset.drop(columns = 'Class')
  y = dataset['Class']
  #Split data into train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

  return X_train, X_test, y_train, y_test

# This function contains all the 3 models used in this project. This single function is called everytime 
# we need to train the model. Only the parameter changes, which are passed to this function. This 
# function outputs the results of all the models from RFC, XGBoost, and LGB in that order.
def results(balancing_technique, X_train, X_test, y_train, y_test):
    print(balancing_technique)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize = (12,6))
    model_name = ["RF", "XGB", "LGB"]
    RFC = RandomForestClassifier(random_state = 0)
    XGBC = XGBClassifier(random_state = 0)
    LGBC = LGBMClassifier(random_state = 0)

    for clf,i in zip([RFC, XGBC, LGBC], model_name):
        model = clf.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:,1]
        print("#"*25,i,"#"*25)
        print("Training Accuracy = {:.3f}".format(model.score(X_train, y_train)))
        print("Test Accuracy = {:.3f}".format(model.score(X_test, y_test)))
        print("ROC_AUC_score : %.6f" % (roc_auc_score(y_test, y_pred)))
        #Confusion Matrix
        print(confusion_matrix(y_test, y_pred))
        print("-"*15,"CLASSIFICATION REPORT","-"*15)
        print(classification_report(y_test, y_pred))
        
        #precision-recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
        avg_pre = average_precision_score(y_test, y_pred_prob)
        ax1.plot(precision, recall, label = i+ " average precision = {:0.2f}".format(avg_pre), lw = 3, alpha = 0.7)
        ax1.set_xlabel('Precision', fontsize = 14)
        ax1.set_ylabel('Recall', fontsize = 14)
        ax1.set_title('Precision-Recall Curve', fontsize = 18)
        ax1.legend(loc = 'best')
        #find default threshold
        close_default = np.argmin(np.abs(thresholds_pr - 0.5))
        ax1.plot(precision[close_default], recall[close_default], 'o', markersize = 8)

        #roc-curve
        fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr,tpr)
        ax2.plot(fpr,tpr, label = i+ " area = {:0.2f}".format(roc_auc), lw = 3, alpha = 0.7)
        ax2.plot([0,1], [0,1], 'r', linestyle = "--", lw = 2)
        ax2.set_xlabel("False Positive Rate", fontsize = 14)
        ax2.set_ylabel("True Positive Rate", fontsize = 14)
        ax2.set_title("ROC Curve", fontsize = 18)
        ax2.legend(loc = 'best')
        #find default threshold
        close_default = np.argmin(np.abs(thresholds_roc - 0.5))
        ax2.plot(fpr[close_default], tpr[close_default], 'o', markersize = 8)
        plt.tight_layout()

# This function performs Down-Sampling of the majority class in the dataset
# And returns new training parameters which are then passed to the results function to train the models
def down_sample_dataset(dataset):
  train_majority = dataset[dataset.Class == 0]
  train_minority = dataset[dataset.Class == 1]
  # down-sample the majority class of valid transactions to match the 492 fraudulent transactions
  train_majority_downsampled = resample(train_majority, replace = False, n_samples = 492, random_state = 0)
  train_downsampled = pd.concat([train_majority_downsampled, train_minority])

  X = train_downsampled.drop(columns = 'Class')
  y = train_downsampled['Class']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

  return X_train, X_test, y_train, y_test

# This function performs up-sampling of the minority class in the dataset and returns the new training
# parameters which are then passed to the results function
def up_sample_dataset(dataset):
  #Note in up sampling, first split the minority class data into train and test set and then up-sample the train data and test it with test data
  X = dataset.drop(columns = 'Class')
  y = dataset['Class']
  #First split data into train and test
  X_train_us, X_test, y_train_us, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  #Now resample the train data
  dataset_us = pd.concat([X_train_us, y_train_us], axis = 1)
  train_majority = dataset_us[dataset_us.Class == 0]
  train_minority = dataset_us[dataset_us.Class == 1]
  train_majority.shape, train_minority.shape

  train_minority_upsampled = resample(train_minority, replace = True, n_samples = 199019, random_state = 0)
  print(train_majority.shape, train_minority_upsampled.shape)
  train_upsampled = pd.concat([train_minority_upsampled, train_majority])
  X_train = train_upsampled.drop(columns = 'Class')
  y_train = train_upsampled['Class']

  return X_train, X_test, y_train, y_test

# This function performs regular SMOTE on the normalized dataset and returns new training parameters
def smote_regular(dataset):
  sm = SMOTE(random_state = 0)
  X = dataset.drop(columns = 'Class')
  y = dataset['Class']
  X_train_sm, X_test, y_train_sm, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  X_train, y_train = sm.fit_sample(X_train_sm, y_train_sm)
  X_test = np.array(X_test)
  y_test = np.array(y_test)

  return X_train, X_test, y_train, y_test

# This function performs borderLine SMOTE on the normalized dataset and returns new training parameters
def borderline_smote(dataset):
  sm = BorderlineSMOTE(random_state = 0)
  X = dataset.drop(columns = 'Class')
  y = dataset['Class']
  X_train_sm, X_test, y_train_sm, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  X_train, y_train = sm.fit_sample(X_train_sm, y_train_sm)
  X_test = np.array(X_test)
  y_test = np.array(y_test)

  return X_train, X_test, y_train, y_test

# This function performs ADASYN SMOTE on the normalized dataset and returns new training parameters
def adasyn(dataset):
  adasyn = ADASYN(random_state = 0)
  X = dataset.drop(columns = 'Class')
  y = dataset['Class']
  X_train_as, X_test, y_train_as, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  X_train, y_train = adasyn.fit_sample(X_train_as, y_train_as)
  X_test = np.array(X_test)
  y_test = np.array(y_test)

  return X_train, X_test, y_train, y_test

# Call the import_dataset() function and get the dataset
dataset = import_dataset()

# Visualize the dataset by plotting the histograms and bar chart
visualize_dataset(dataset)

# Normalize the dataset
dataset = normalize_dataset(dataset)

# Check for correlation among column values for feature extraction
plot_correlation_matrix(dataset)

# Split the dataset and fetch the training parameters
X_train, X_test, y_train, y_test = split_dataset(dataset)

# Train the models first with imbalanced dataset
results("Without Balancing", X_train, X_test, y_train, y_test)

# Fetch new training parameters after performing down-sampling on the dataset
X_train, X_test, y_train, y_test = down_sample_dataset(dataset)

# Train the models with the down-sampled dataset and check results
results("Down Sampling", X_train, X_test, y_train, y_test)

# Fetch new training parameters after performing up-sampling on the dataset
X_train, X_test, y_train, y_test = up_sample_dataset(dataset)

# Train the model with the up-sampled dataset and check results
results("Up Sampling", X_train, X_test, y_train, y_test)

# Perform SMOTE on the dataset and fetch new training parameters
X_train, X_test, y_train, y_test = smote_regular(dataset)

# Train the model with the new parameters and check results
results("SMOTE Regular", X_train, X_test, y_train, y_test)

# Perform BorderLine SMOTE on the dataset and fetch new training parameters
X_train, X_test, y_train, y_test = borderline_smote(dataset)

# Train the model with the new parameters and check results
results("Borderline SMOTE", X_train, X_test, y_train, y_test)

# Perform ADASYN SMOTE on the dataset and fetch new training parameters
X_train, X_test, y_train, y_test = adasyn(dataset)

# Train the model with the new parameters and check results
results("ADASYN SMOTE", X_train, X_test, y_train, y_test)