# -*- coding: utf-8 -*-
"""
11/04/2020 - Marcio Justo
2a experiência com Machine Learning
https://medium.com/ciencia-descomplicada/machine-learning-classificando-gatos-e-cachorros-d45f1fddbff
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

print("Acurácia de " + str(accuracy_score(rotulos_previstos, rotulos_esperados) * 100) + "%")