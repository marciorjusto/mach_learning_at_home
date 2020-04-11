# -*- coding: utf-8 -*-
"""
09/04/2020 - Marcio Justo
1a experiência com Machine Learning
https://dev.to/educationecosystem/a-simple-machine-learning-project-in-python-5d11
"""
from sklearn import datasets

# Este é um comentário
iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)
#rint(digits.target)

#--------
print(iris.data)
print(iris.target)
print(iris.target_names)

#--------
import seaborn as sns

iris_data = iris.data
iris_target = iris.target
sns.boxplot(data = iris_data, width=0.5, fliersize=5)
sns.set(rc={'figure.figsize':(1,10)})

#--- Training and Testing
import numpy as np
from sklearn import tree

iris_test_ids = np.random.permutation(len(iris_data))

iris_train_one = iris_data[iris_test_ids[:-15]]
iris_test_one = iris_data[iris_test_ids[-15:]]

iris_train_two = iris_target[iris_test_ids[:-15]]
iris_test_two = iris_target[iris_test_ids[-15:]]

iris_classify = tree.DecisionTreeClassifier()
iris_classify.fit(iris_train_one, iris_train_two)

iris_predict = iris_classify.predict(iris_test_one)

from sklearn.metrics import accuracy_score
print('----')
print(iris_predict)
print(iris_test_two)
print(accuracy_score(iris_predict, iris_test_two)* 100)