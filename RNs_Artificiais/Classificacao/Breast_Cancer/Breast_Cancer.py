"""
Rede Neural Artificial - Base de dados Breast Cancer
Previsões com a rede neural
Determinação da precisão da rede neural
"""


import pandas as pd
import numpy as np
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Importação da base de dados """
previsores = pd.read_csv('Entradas_Breast_cancer.csv')
classes = pd.read_csv('Saidas_classes.csv')

# lib que já faz a divisão da base de dados entre treinamento e teste
# test_size signifca que será utilizado 25% dos registros para teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classes, test_size=0.25)


#import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense


""" Criação de uma rede neural modelo sequencial e compilação """
classificador = Sequential()
# Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 30 + 1 / 2 = 15.5
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
# Criação da segunda camada oculta
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
# Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid
classificador.add(Dense(units=1, activation='sigmoid'))

otimizador = keras.optimizers.legacy.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])
# classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=100)


""" Obtem os pesos que a rede utilizou """
#pesos0 = classificador.layers[0].get_weights()
#print(pesos0)
#print(len(pesos0))
#pesos1 = classificador.layers[1].get_weights()
#pesos2 = classificador.layers[2].get_weights()


""" Avaliação da base de dados de teste para analisar a precisão do algoritimo com o sklearn """
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

""" Avaliação da base de dados de teste para analisar a precisão do algoritimo com o keras. 
    Loss e binary accuracy """
resultado = classificador.evaluate(previsores_teste, classe_teste)

print('Precisão de: ', precisao)
print('Matriz de confusão: ', matriz)
print('Resultado: ', resultado)



print("fim")
# print(previsores_treinamento)
# print(previsores_teste)
# print(classe_treinamento)
# print(classe_teste)

# print(tensorflow.__version__)
# print(keras.__version__)



