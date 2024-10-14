"""
Rede Neural Artificial - Base de dados Breast Cancer
Previsões com a rede neural
Determinação da precisão da rede neural
Carregar rede neural compilada baseada nos pesos e classificar se um novo registro é um tumor maligno ou benigno
"""


import numpy as np
import pandas as pd
from keras.models import model_from_json
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Carregar o arquivo do disco
arquivo = open('classificador_breast.json', 'r')
# Salvar em uma variável
estrutura_rede = arquivo.read()
arquivo.close()

# Cria a estrtutura da rede neural com as configurações lidas
classificador = model_from_json(estrutura_rede)
# Carrega o arquivo de pesos da rede
classificador.load_weights('classificador_breast.h5')


# Novo registro para a classificação de tumor maligno ou benigno
novo_registro = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                          0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05,
                          0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                          0.84, 158, 0.363]])

# Previsão do tumor
previsao = classificador.predict(novo_registro)
print(previsao)
previsao = (previsao > 0.5)
print(previsao)


""" Parte didatica (Para validar que pode-se utilizar base de dados maiores para fazer a avaliação) -> Avaliação de uma 
base de dados de teste (utilizando a mesma base de dados na qual foi feito o treinamento) """
previsores = pd.read_csv('Entradas_Breast_cancer.csv')
classes = pd.read_csv('Saidas_classes.csv')

classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Mostrar o valor da loss function e o valor da precisão (accuracy) da rede
resultado = classificador.evaluate(previsores, classes)

print(resultado)