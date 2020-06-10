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
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
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

    """创建决策树训练模型对象"""
    clsModel = DecisionTreeClassifier()
    """根据给定的训练数据拟合Decision Tree模型。"""
    clsModel.fit(X_train, y_train)
    """使用训练后的模型，基于测试集进行目标参数的预测"""
    y_pred = clsModel.predict(X_test)
    """计算训练结果精度"""
    result_acc = accuracy_score(y_test, y_pred)  # 0.19632881085395051

    # 生成分类结果汇报
    cls_report = "Classification Report:\n{}".format(
        classification_report(y_test, y_pred, zero_division=1)
    )

    # 使用参数设置画决策树图, 内存上限缘故设置最大深度为5
    plt.figure(figsize=(20, 10))
    tree.plot_tree(
        clsModel.fit(X_train, y_train),
        filled=True,
        fontsize=4,
        class_names=[
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
        ],
        max_depth=5,
    )
    plt.show()
    plt.clf()

    # Plot result figure
    x = list(range(1, X_test.shape[0] + 1))
    y1 = y_test["Rings"].values
    y2 = y_pred
    plt.xlim(150, 550)
    plt.ylim(0, 25)
    plt.plot(x, y1, "g.", label="Actual Rings of test set")
    plt.plot(x, y2, "rx", label="Predicted Rings of test set")
    plt.title("Decision Tree Regression (Test set)")
    plt.xlabel("Index")
    plt.ylabel("Rings")
    plt.legend()
    plt.show()

    input()  # for debug console
