# -*- coding: utf-8 -*-
"""
09/04/2020 - Marcio Justo
Created on Thu Apr  9 00:45:52 2020

@author: marciorjusto

Importar base de dados Iris de um CSV
https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
"""
# load libraries
from pandas import read_csv

from matplotlib import pyplot

# load dataset
file = 'iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(file, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(150))

# descriptions - statistical summary
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())


# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
