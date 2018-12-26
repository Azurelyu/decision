import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("E:/train.csv",dtype={"Age": np.float64},)
train.head(10)


def harmonize_data(titanic):
# 填充空数据以及把string数据转成interger表示
# 对于年龄字段发生缺失，用所有年龄的均值替代
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# 性别男 用0来替代
    titanic.loc[titanic["sex"] == "male", "sex"] = 0
# 性别女 用1来替代
    titanic.loc[titanic["sex"] == "female", "sex"]= 1

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    return titanic



