# Algorithm

Svm（support Vector Mac）又称为支持向量机，是一种二分类的模型(也可以用于多类别问题的分类)。

其主要思想为找到空间中的一个能将所有数据样本划开的超平面，并且使得本本集中所有数据到这个超平面的距离最短。

# Virtualenv Preparation

```powershell
> virtualenv machine-learning-venv
> machine-learning-venv\Scripts\activate
```

# Packages Installing

1. [Scikit-learn](https://scikit-learn.org/stable/install.html)
   
   Scikit-learn(sklearn)是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，包括回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法。 

   ```python
    """ SK-Learn Library Import"""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import Bunch
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
   ```
   - `sklearn.preprocessing.LabelEncoder` : to transform non-numerical labels to numerical labels
     ```python
     df["Sex"] = LabelEncoder().fit_transform(df["Sex"].tolist())
     # M, F, I -> 2, 0, 1 
     ```
   - `sklearn.utils.Bunch` : container object exposing keys as attributes
     ```python
     b = Bunch(
        data=dfdummies[xqlabs],
        target=df[ylabs],
        feature_names=xqlabs,
        target_names=ylabs,
     )
     b.data
     b["data"]
     ```

   - `sklearn.svm` : The sklearn.svm module includes Support Vector Machine algorithms.

   - `sklearn.model_selection.train_test_split` : Split arrays or matrices into random train and test subsets.
      
   - `sklearn.svm.SVC` : C-Support Vector Classification based on libsvm.
     ```python
     sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
     # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
     # C: float, default=1.0 。超参数，作为误差的惩罚项。
     # 当C比较小时，可以接受一些数据点的误分类,模型的偏差高而方差低(high bias low variance)
     # 当C比较大时，对错误分类的点惩罚严重，分类器会尽力避免错误的分类，得到的模型偏差小但方差高(low bias high variance)
     ```
     
   - `sklearn.metrics.classification_report` : 建立一个显示主要分类指标的文本报告。
   - `sklearn.metrics.accuracy_score` : Caculate accuracy classification score.

# 运行结果
![](images\svm_output.png)


