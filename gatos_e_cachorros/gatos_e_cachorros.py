# -*- coding: utf-8 -*-
"""
11/04/2020 - Marcio Justo
2a experiência com Machine Learning
https://medium.com/ciencia-descomplicada/machine-learning-classificando-gatos-e-cachorros-d45f1fddbff
"""
# Gato ou Cachorro? 
# Características
# - É fofinho?
# - Tem orelhinha pequena?
# - Faz miau?

# Características dos bichos
bichinho1 = [1, 1, 1]
bichinho2 = [1, 0, 1]
bichinho3 = [0, 1, 1]
bichinho4 = [1, 1, 0]
bichinho5 = [0, 1, 0]
bichinho6 = [0, 1, 0]

dados = [bichinho1, bichinho2, bichinho3, bichinho4, bichinho5, bichinho6]

# Labels (convenções) dos datasets
#  1 = Gato
# -1 = Cachorro
# 3 primeiros são gatos e 3 últimos são cachorros
marcacoes = [1, 1, 1, -1, -1, -1]

# Criação do modelo
from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados,marcacoes)

# Fazer predições
bicho_misterioso1 = [1, 1, 1]
bicho_misterioso2 = [1, 0, 0]
bicho_misterioso3 = [0, 0, 1]

teste = [bicho_misterioso1, bicho_misterioso2, bicho_misterioso3]

# Resultados
resultado = modelo.predict(teste)
marcacoes_teste = [1,-1, 1]

print("Resultado: ")
print(resultado)

print ("Marcacoes: ")
print(marcacoes_teste)

print("Acurácia de 66,66% (base muito pequena)")