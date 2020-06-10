# Data Set: http://archive.ics.uci.edu/ml/datasets/Abalone
# 1 - Sex / nominal / -- / M, F, and I (infant)
# 2 - Length / continuous / mm / Longest shell measurement
# 3 - Diameter / continuous / mm / perpendicular to length  # 直径
# 4 - Height / continuous / mm / with meat in shell
# 5 - Whole weight / continuous / grams / whole abalone
# 6 - Shucked weight / continuous / grams / weight of meat
# 7 - Viscera weight / continuous / grams / gut weight (after bleeding)
# 8 - Shell weight / continuous / grams / after being dried
# 9 - Rings / integer / -- / +1.5 gives the age in years

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" SK-Learn Library Import"""
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn import svm
from sklearn.model_selection import train_test_split
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
    # 数据集存放目录
    data_dir = os.path.join(os.getcwd(), "knn_svm_proj\\abalone-dataset\\abalone.data")

    # 定义数据集标签
    xnlabs = ["Sex"]  # 非数值字段
    xqlabs = [  # 数值字段
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
    ]
    xlabs = xnlabs + xqlabs

    # 要预测的属性
    ylabs = ["Rings"]

    # 将数据加载到pandas模块的DataFrame对象中
    df = pd.read_csv(data_dir, header=None, sep=",", names=xlabs + ylabs)

    """将非数值字段的项值数值化为0,1,2"""
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"].tolist())

    # 过滤除去无意义的字段
    df = df[df["Height"] > 0.0]
    df = df[df["Length"] > 0.0]
    df = df[df["Diameter"] > 0.0]

    # dummies = pd.get_dummies(df[xnlabs], prefix="Sex")
    # dfdummies = df[xqlabs + ylabs].join(dummies)
    # xqlabs = xqlabs + dummies.columns.tolist()

    # 返回Bunch对象
    return Bunch(
        data=df[xqlabs + xnlabs],
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

    """Feature Engineering , class 1 : 1-8, class 2 : 9-10, class 3 : 11 >"""
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

    # 加载数据集
    abl_dataset = load_data()

    """训练、预测特征"""
    X_train = abl_dataset["data"].drop(
        ["Sex"], axis=1
    )  # result_acc = 0.26256983240223464
    # X_train = abl_dataset["data"] # 0.2601755786113328
    y_train = abl_dataset["target"]

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0
    )

    """创建SVM训练模型对象"""
    # kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    # gamma: “ rbf”，“ poly”和“ Sigmoid”的kernel系数。
    # svmModel = svm.SVC(kernel="linear", C=1, gamma=1) # result_acc = 0.24980047885075818
    svmModel = svm.SVC(kernel="rbf", C=1, gamma=100)  # result_acc = 0.26256983240223464
    """根据给定的训练数据拟合SVM模型。"""
    svmModel.fit(X_train, y_train)
    """使用训练后的模型，基于测试集进行目标参数的预测"""
    y_pred = svmModel.predict(X_test)
    """计算训练结果精度"""
    result_acc = accuracy_score(y_test, y_pred)

    # 生成分类结果汇报
    svm_report = "Classification Report:\n{}".format(
        classification_report(y_test, y_pred, zero_division=1)
    )

    #######

    # 将Rings属性划分为3类，重复上面的过程

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

    # Classification Report
    svm_report_new = "Classification Report:\n{}".format(
        classification_report(y_test_new, y_pred_new, zero_division=1)
    )

    # Plot result figure
    plt.subplot(122)
    x = list(range(1, X_test_new.shape[0] + 1))
    y1 = y_test_new["Rings_new"].values
    y2 = y_pred_new
    plt.xlim(0, 150)
    plt.ylim(0.5, 3.5)
    plt.plot(x, y1, "g.", label="Actual Rings of test set")
    plt.plot(x, y2, "rx", label="Predicted Rings of test set")
    plt.title("Support Vector Machines Classification (Test set)")
    plt.xlabel("Index")
    plt.ylabel("Rings")
    plt.legend()
    # plt.show()

    # Plot result figure
    plt.subplot(121)
    x = list(range(1, X_test.shape[0] + 1))
    y1 = y_test["Rings"].values
    y2 = y_pred
    plt.xlim(150, 550)
    plt.ylim(0, 25)
    plt.plot(x, y1, "g.", label="Actual Rings of test set")
    plt.plot(x, y2, "rx", label="Predicted Rings of test set")
    plt.title("Support Vector Machines Classification (Test set)")
    plt.xlabel("Index")
    plt.ylabel("Rings")
    plt.legend()
    plt.show()

    input()  # for debug console
