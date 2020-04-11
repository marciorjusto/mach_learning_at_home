# -*- coding: utf-8 -*-
"""
11/04/2020 - Marcio Justo
Created on Thu Apr  11 00:45:52 2020

@author: marciorjusto

Importar base de dados Iris de uma URL
https://medium.com/@jayasagar/python-hello-world-machine-learning-60137af9d3bf
"""
# Abordagem 3 - Usar dataset carregado de uma URL

# load libraries
from pandas import read_csv

from matplotlib import pyplot

# load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

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
