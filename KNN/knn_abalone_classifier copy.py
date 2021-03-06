import os

# numbers, stats, plots
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import scipy.stats as stats

# sklearn support
from sklearn import metrics, preprocessing
from sklearn.model_selection import cross_validate
from sklearn.utils import Bunch
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# machine learning algorithm of interest
from sklearn.neighbors import KNeighborsClassifier

import mglearn


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

    # Filter zero values of height/length/diameter
    df = df[df["Height"] > 0.0]
    df = df[df["Length"] > 0.0]
    df = df[df["Diameter"] > 0.0]
    # df["Volume"] = df["Height"] * df["Length"] * df["Diameter"]
    # xqlabs.append("Volume")
    ## Filter values that Rings>11
    df = df[df["Rings"] < 12]

    dummies = pd.get_dummies(df[xnlabs], prefix="Sex")

    dfdummies = df[xqlabs + ylabs].join(dummies)

    xqlabs = xqlabs + dummies.columns.tolist()

    return Bunch(
        data=dfdummies[xqlabs],
        target=df[ylabs],
        feature_names=xqlabs,
        target_names=ylabs,
    )


if __name__ == "__main__":
    # load abalone dataset, which is a Bunch object
    abl_dataset = load_data()

    # Training and Testing Data

    # splitting the labeled data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(
        abl_dataset["data"], abl_dataset["target"], test_size=0.3, random_state=0
    )
    """Debug
        # print("X_train shape: {}".format(X_train.shape))  # X_train shape: (2922, 11)
        # print("y_train shape: {}".format(y_train.shape))  # y_train shape: (2922, 1)
        # print("X_test shape: {}".format(X_test.shape))    # X_test shape: (1253, 11)
        # print("y_test shape: {}".format(y_test.shape))    # y_test shape: (1253, 1)
    """

    """ 
    Visualizing the original data
    # create dataframe from data in X_train
    # label the columns using the strings in abl_dataset.feature_names
    abl_df = pd.DataFrame(X_train, columns=abl_dataset.feature_names)
    # create a scatter matrix from the dataframe, color by y_train["Rings"].values
    grr = scatter_matrix(
        abl_df,
        c=y_train["Rings"].values,
        figsize=(30, 30),
        marker=".",
        hist_kwds={"bins": 20},
        s=1,
        alpha=0.8,
        cmap=mglearn.cm3,
    )
    plt.show()
    """

    # Building k-Nearest Neighbors model

    # store the training set using KNeighborsClassifier class
    knn = KNeighborsClassifier(
        n_neighbors=48
        # n_neighbors=35
    )  # fixed number k of neighbors in the training set to 5

    # build the model on the training set
    knn.fit(X_train, y_train["Rings"].values)  # returns the knn object itself

    # Making Predictions

    # call the predict method of the knn object
    y_pred = knn.predict(X_test)
    print("Test set predictions:\n {}".format(y_pred))

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n{}".format(cm))

    # evaluating the Model
    # print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))  # use np.mean()
    score = 0
    i = 0

    for yp in y_pred:
        if yp == y_test["Rings"].values[i]:
            score += 1
        i += 1

    print("Test set score: {:.2f}".format(score / len(y_pred)))  # use np.mean()
    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))  # use knn.score()

    print(
        "Classification Report:\n{}".format(
            classification_report(y_test, y_pred, zero_division=1)
        )
    )

    # Plot result figure
    plt.subplot(122)
    # x = X_test["Volume"]
    x = list(range(1, X_test.shape[0] + 1))
    y1 = y_test["Rings"].values
    y2 = y_pred
    plt.plot(x, y1, "b.", label="Actual Rings of test set")
    plt.plot(x, y2, "rx", label="Predicted Rings of test set")
    plt.title("K-Nearest Neighbors Classification (Test set)")
    plt.xlabel("Index")
    plt.ylabel("Rings")
    plt.legend()
    # plt.show()

    # predict on training set
    y_pred_on_trainset = knn.predict(X_train)
    print("Train set predictions:\n {}".format(y_pred_on_trainset))
    # evaluating the Model
    # print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))  # use np.mean()
    score = 0
    i = 0
    for yp in y_pred_on_trainset:
        if yp == y_train["Rings"].values[i]:
            score += 1
        i += 1
    print(
        "Test set score: {:.2f}".format(score / len(y_pred_on_trainset))
    )  # use np.mean()

    # Plot result figure
    plt.subplot(121)
    # x = X_train["Volume"]
    x = list(range(1, X_train.shape[0] + 1))
    y1 = y_train["Rings"].values
    y2 = y_pred_on_trainset
    plt.plot(x, y1, "bo", label="Actual Rings of train set")
    plt.plot(x, y2, "r+", label="Predicted Rings of train set")
    plt.title("K-Nearest Neighbors Classification (Train set)")
    plt.xlabel("Index")
    plt.ylabel("Rings")
    plt.legend()
    plt.show()
