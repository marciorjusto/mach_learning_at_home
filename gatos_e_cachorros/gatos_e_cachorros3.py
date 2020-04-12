# -*- coding: utf-8 -*-
"""
11/04/2020 - Marcio Justo
2a experiência com Machine Learning
https://medium.com/ciencia-descomplicada/machine-learning-classificando-gatos-e-cachorros-d45f1fddbff
https://towardsdatascience.com/beginners-guide-to-machine-learning-with-python-b9ff35bc9c51
https://www.digitalocean.com/community/tutorials/como-construir-um-classificador-de-machine-learning-em-python-com-scikit-learn-pt
"""

"""
NumPy - Arrays e Matrizes multidimensionais

https://docs.scipy.org/doc/numpy/user/
https://docs.scipy.org/doc/numpy/user/quickstart.html

NumPy is shortened from Numerical Python, it is the most universal and versatile 
library both for pros and beginners. Using this tool you are up to operate with 
multi-dimensional arrays and matrices with ease and comfort. Such functions like 
linear algebra operations and numerical conversions are also available.
"""
# import numpy as np

"""
Pandas - Carregar e manipular dados

https://pandas.pydata.org/pandas-docs/stable/
https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

Pandas is a well-known and high-performance tool for presenting data frames.
Using it you can load data from almost any source, calculate various functions
and create new parameters, build queries to data using aggregate functions akin
to SQL. What is more, there are various matrix transformation functions, a sliding
window method and other methods for obtaining information from data. So it’s totally
an indispensable thing in the arsenal of a good specialist.
"""
from pandas              import read_csv

"""
Scikit-Learn - Algoritmos de Aprendizado de Máquina

I can say it’s the most well-designed ML package I’ve observed so far. 
It implements a wide-range of machine-learning algorithms and makes it 
comfortable to plug them into actual applications. You can use a whole 
slew of functions here like regression, clustering, model selection, 
preprocessing, classification and more. So, it’s totally worth learning 
and using. The great advantage here is the high speed of work. So it’s 
not surprising why such leading platforms like Spotify, Booking.com, 
J.P.Morgan are using scikit-learn.
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics     import accuracy_score
from sklearn.model_selection import train_test_split

"""
Matplotlib

Matplotlib is a flexible library for creating graphs and visualization.
It is powerful but somewhat heavy-weight. At this point, you can skip 
Matplotlib and use Seaborn to get started (see Seaborn below).
"""
from matplotlib          import pyplot

"""
1. Importar dados de uma planilha CSV
"""
file = 'gatos_e_cachorros.csv'
rotulos = ['EH_FOFINHO', 'TEM_ORELHA_PEQ', 'FAZ_MIAU', 'BICHO_EH']
dataset = read_csv(file, names=rotulos, sep=';', header=0)

"""
2. Separar dados em dados de treinamento e dados de teste
"""
# Separa a coluna da classificação das colunas de características
atributos = dataset.drop('BICHO_EH', axis=1)
resultados = dataset.BICHO_EH

atributos_train    \
, atributos_test   \
, resultados_train \
, resultados_test  = train_test_split( atributos       \
                                     , resultados      \
                                     , test_size=0.3   \
                                     , random_state=42 \
                                     )

"""
TREINANDO ALGORITMO
Informa atributos de bichos conhecidos
"""
print(atributos_train.head(5))
# Informa quais bichos devem resultar a partir das características conhecidas
print(resultados_train.head(5))

"""
TESTANDO
Informa atributos de bichos desconhecidos
"""
print(atributos_test.head(5))
#Quais são os bichos com os atributos acima?
print(resultados_test.head(5))


"""
3. Fazer previsões
"""
# Fazer predições
atributos_previsao = [0, 0, 0]
dados_previsao = [atributos_previsao]

# Criação do modelo
modelo = MultinomialNB()
modelo.fit(atributos_train, resultados_train)
#resultado_previsao = modelo.predict(dados_previsao)

#print("Bicho previsto:")
#print(resultado_previsao)

#print("Acurácia de " + str(accuracy_score(, resultado_previsao) * 100) + "%")

"""
A variável data representa um objeto Python que funciona como um dicionário. 
As chaves importantes do dicionário a considerar são:
 
    - os nomes dos rótulos de classificação (target_names)
 - os rótulos reais (target)                               
 
 - os nomes de atributo/característica (feature_names)
 - os atributos (data).

Atributos são uma parte crítica de qualquer classificador. Os atributos capturam
características importantes sobre a natureza dos dados. Dado o rótulo que estamos
tentando prever (tumor maligno versus benigno), os possíveis atributos úteis
incluem o tamanho, raio, e a textura do tumor.

Crie novas variáveis para cada conjunto importante de informações e atribua os dados:

# Organizar nossos dados
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
"""

# Forma do dataset (linhas x colunas)
#print(dataset.shape)

# N primeiros registros
#print(dataset.head(150))

# descriptions - statistical summary
#print(dataset.describe())

# class distribution - Agrupa quantitativamente pelo valor de uma coluna (característica)
#print(dataset.groupby('BICHO_EH').size())


# box and whisker plots
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

"""

# Gato ou Cachorro? 
k_EH_GATO = 1
k_EH_CACHORRO = -1

# Características
# - É fofinho?
# - Tem orelhinha pequena?
# - Faz miau?
k_EH_FOFINHO = 1
k_NAO_EH_FOFINHO = 0

k_TEM_ORELHA_PEQ = 1
k_NAO_TEM_ORELHA_PEQ = 0

k_FAZ_MIAU = 1
k_NAO_FAZ_MIAU = 0

# Características dos bichos
bichinho1 = [k_EH_FOFINHO    , k_TEM_ORELHA_PEQ    , k_FAZ_MIAU]
bichinho2 = [k_EH_FOFINHO    , k_NAO_TEM_ORELHA_PEQ, k_FAZ_MIAU]
bichinho3 = [k_NAO_EH_FOFINHO, k_TEM_ORELHA_PEQ    , k_FAZ_MIAU]
bichinho4 = [k_EH_FOFINHO    , k_TEM_ORELHA_PEQ    , k_NAO_FAZ_MIAU]
bichinho5 = [k_NAO_EH_FOFINHO, k_TEM_ORELHA_PEQ    , k_NAO_FAZ_MIAU]
bichinho6 = [k_NAO_EH_FOFINHO, k_TEM_ORELHA_PEQ    , k_NAO_FAZ_MIAU]

dados_treino = [bichinho1, bichinho2, bichinho3, bichinho4, bichinho5, bichinho6]

# Labels (convenções) dos datasets
#  1 = Gato
# -1 = Cachorro
# 3 primeiros são gatos e 3 últimos são cachorros
rotulos_treino = [k_EH_GATO, k_EH_GATO, k_EH_GATO, k_EH_CACHORRO, k_EH_CACHORRO, k_EH_CACHORRO]

# Criação do modelo
modelo = MultinomialNB()
modelo.fit(dados_treino, rotulos_treino)

# Fazer predições
bicho_misterioso1 = [k_EH_FOFINHO    , k_TEM_ORELHA_PEQ    , k_FAZ_MIAU]
bicho_misterioso2 = [k_EH_FOFINHO    , k_NAO_TEM_ORELHA_PEQ, k_NAO_FAZ_MIAU]
bicho_misterioso3 = [k_NAO_EH_FOFINHO, k_NAO_TEM_ORELHA_PEQ, k_FAZ_MIAU]

dados_previsao = [bicho_misterioso1, bicho_misterioso2, bicho_misterioso3]

# Resultados
rotulos_previstos = modelo.predict(dados_previsao)
rotulos_esperados = [k_EH_GATO, k_EH_CACHORRO, k_EH_GATO]

print ("Rótulos previstos pelo modelo: ")
print(rotulos_previstos)

print("Rótulos esperados: ")
print(rotulos_esperados)
"""
#print("Acurácia de " + str(accuracy_score(rotulos_previstos, rotulos_esperados) * 100) + "%")
