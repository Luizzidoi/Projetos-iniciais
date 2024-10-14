"""
Rede Neural Artificial - Base de dados Breast Cancer
Previsões com a rede neural
Determinação da precisão da rede neural
Tuning = encontrar as melhores configurações para a rede
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


previsores = pd.read_csv('Entradas_Breast_cancer.csv')
classes = pd.read_csv('Saidas_classes.csv')


def criarRede(optimizer, loss, kernel_initializer, activation, neurons):

    # Criação de uma nova rede neural
    classificador = Sequential()
    # Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 30 + 1 / 2 = 15.5
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    # Adicionado o Dropout para análise
    classificador.add(Dropout(0.2))
    # Criação da segunda camada oculta
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    # Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid
    classificador.add(Dense(units=1, activation='sigmoid'))

    # otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])

    return classificador

""" Lista de parametros para que a rede possa escolher a melhor configuração """
classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, scoring='accuracy', cv=5)
grid_search = grid_search.fit(previsores, classes)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_


print(melhores_parametros)
print(melhor_precisao)
