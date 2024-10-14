"""
Rede Neural Artificial - Base de dados Breast Cancer
Previsões com a rede neural
Determinação da precisão da rede neural
Salvar rede neural compilada (pesos)
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

previsores = pd.read_csv('Entradas_Breast_cancer.csv')
classes = pd.read_csv('Saidas_classes.csv')


## Substituir os parametros ideias conforme a classificação Tuning da aula anterior
# Criação de uma nova rede neural
classificador = Sequential()
# Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 30 + 1 / 2 = 15.5
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
# Adicionado o Dropout para análise
classificador.add(Dropout(0.2))
# Criação da segunda camada oculta
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
classificador.add(Dropout(0.2))
# Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid
classificador.add(Dense(units=1, activation='sigmoid'))


classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(previsores, classes, batch_size=10, epochs=100)


""" Criação de um arquivo utilizando o json e salvar os pesos no formato h5 """
classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('classificador_breast.h5')