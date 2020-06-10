# Trabalho Final - SVM - Abalone
# Aluno: Bruno H. Hjort

# Data Set: http://archive.ics.uci.edu/ml/datasets/Abalone
# 1 - Sex / nominal / -- / M, F, and I (infant)
# 2 - Length / continuous / mm / Longest shell measurement
# 3 - Diameter / continuous / mm / perpendicular to length
# 4 - Height / continuous / mm / with meat in shell
# 5 - Whole weight / continuous / grams / whole abalone
# 6 - Shucked weight / continuous / grams / weight of meat
# 7 - Viscera weight / continuous / grams / gut weight (after bleeding)
# 8 - Shell weight / continuous / grams / after being dried
# 9 - Rings / integer / -- / +1.5 gives the age in years

import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt


""" SK-Learn Library Import"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.utils import Bunch
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

"""Scipy, Stats Library"""
import scipy
from scipy.stats import skew
from scipy.stats import uniform

import joblib
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import os


def load_data():
    """
    loading the dataset into a scikit learn Bunch object
    returns:
    ----
    data,target,feature_names,target_names
    """
    # Load the data from this file
    data_dir = os.path.join(os.getcwd(), "knn_svm_proj\\abalone-dataset\\abalone.data")

    # x data labels
    xnlabs = ["Sex"]
    xqlabs = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
    ]
    xlabs = xnlabs + xqlabs

    # y data labels
    ylabs = ["Rings"]

    # Load data to dataframe
    df = pd.read_csv(data_dir, header=None, sep=",", names=xlabs + ylabs)

    """LabelEnconding the Categorical Data"""
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"].tolist())

    # Filter zero values of height/length/diameter
    df = df[df["Height"] > 0.0]
    df = df[df["Length"] > 0.0]
    df = df[df["Diameter"] > 0.0]
    # df["Volume"] = df["Height"] * df["Length"] * df["Diameter"]
    # xqlabs.append("Volume")
    ## Filter values that Rings>11
    # df = df[df["Rings"] < 12]

    dummies = pd.get_dummies(df[xnlabs], prefix="Sex")

    dfdummies = df[xqlabs + ylabs].join(dummies)

    xqlabs = xqlabs + dummies.columns.tolist()

    return Bunch(
        data=dfdummies[xqlabs],
        target=df[ylabs],
        feature_names=xqlabs,
        target_names=ylabs,
    )


def load_data_classes_reduced():
    """
    loading the dataset into a scikit learn Bunch object
    returns:
    ----
    data,target,feature_names,target_names
    """
    # Load the data from this file
    data_dir = os.path.join(os.getcwd(), "knn_svm_proj\\abalone-dataset\\abalone.data")

    # x data labels
    xnlabs = ["Sex"]
    xqlabs = [
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
    ]
    xlabs = xnlabs + xqlabs

    # y data labels
    ylabs = ["Rings"]

    # Load data to dataframe
    df = pd.read_csv(data_dir, header=None, sep=",", names=xlabs + ylabs)

    """LabelEnconding the Categorical Data"""
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"].tolist())

    # Filter zero values of height/length/diameter
    df = df[df["Height"] > 0.0]
    df = df[df["Length"] > 0.0]
    df = df[df["Diameter"] > 0.0]
    # df["Volume"] = df["Height"] * df["Length"] * df["Diameter"]
    # xqlabs.append("Volume")
    ## Filter values that Rings>11
    # df = df[df["Rings"] < 12]

    dummies = pd.get_dummies(df[xnlabs], prefix="Sex")

    dfdummies = df[xqlabs + ylabs].join(dummies)

    xqlabs = xqlabs + dummies.columns.tolist()

    """Feature Engineering , class 1 : 1-8, class 2 : 9-8, class 3 : 11 >"""
    df["Rings_1"] = np.where(df["Rings"] <= 8, 1, 0)
    df["Rings_2"] = np.where(((df["Rings"] > 8) & (df["Rings"] <= 10)), 2, 0)
    df["Rings_3"] = np.where(df["Rings"] > 10, 3, 0)
    df["Rings_new"] = df["Rings_1"] + df["Rings_2"] + df["Rings_3"]
    ylabs = ["Rings_new"]

    return Bunch(
        data=dfdummies[xqlabs],
        target=df[ylabs],
        feature_names=xqlabs,
        target_names=ylabs,
    )


if __name__ == "__main__":

    # Loading data
    abl_dataset = load_data()

    """Learning Features and Predicting Features"""
    X_train = abl_dataset["data"].drop(["Sex"], axis=1)
    y_train = abl_dataset["target"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0
    )

    """Creating Object of SVM"""
    # svmModel = svm.SVC(kernel="linear", C=1, gamma=1) # result_acc = 0.24980047885075818
    svmModel = svm.SVC(kernel="rbf", C=1, gamma=100)  # result_acc = 0.26256983240223464
    """Learning from Training Set"""
    svmModel.fit(X_train, y_train)
    """Predicting for Training Set"""
    y_pred = svmModel.predict(X_test)
    """Accuracy Score"""
    result_acc = accuracy_score(y_test, y_pred)

    # Loading data("Rings" has only 3 classes)
    abl_dataset_new = load_data_classes_reduced()
    """Learning Features and Predicting Features"""
    X_train_new = abl_dataset_new["data"].drop(["Sex"], axis=1)
    y_train_new = abl_dataset_new["target"]
    # Train Test Split
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_train_new, y_train_new, test_size=0.3, random_state=0
    )
    """Creating Object of SVM"""
    svmModel_new = svm.SVC(
        kernel="rbf", C=1, gamma=100
    )  # result_acc = 0.26256983240223464
    """Learning from Training Set"""
    svmModel_new.fit(X_train_new, y_train_new)
    """Predicting for Training Set"""
    y_pred_new = svmModel_new.predict(X_test_new)
    """Accuracy Score"""
    result_acc_new = accuracy_score(y_test_new, y_pred_new)  # 0.6480446927374302

    # Plot result figure
    x = list(range(1, X_test_new.shape[0] + 1))
    y1 = y_test_new["Rings_new"].values
    y2 = y_pred_new
    plt.ylim(0.5, 3.5)
    plt.plot(x, y1, "b.", label="Actual Rings of test set")
    plt.plot(x, y2, "rx", label="Predicted Rings of test set")
    plt.title("Support Vector Machines Classification (Test set)")
    plt.xlabel("Index")
    plt.ylabel("Rings")
    plt.legend()
    plt.show()

    input()  # for debug console
