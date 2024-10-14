"""
Rede Neural Artificial - Base de dados Breast Cancer
Previsões com a rede neural
Determinação da precisão da rede neural
Validação Cruzada -> A base é dividida em k partes e ela mesma faz 10 treinamentos. Cada treinamento
uma parte da base de dados é usada para teste. A Precisão da rede neural é a media dos 10 treinamentos.
"""

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

""" Importação da base de dados """
previsores = pd.read_csv('Entradas_Breast_cancer.csv')
classes = pd.read_csv('Saidas_classes.csv')

# Função de para a criação da rede neural
def criarRede():

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

    otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics=['binary_accuracy'])

    return classificador


classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)

resultados = cross_val_score(estimator = classificador, X = previsores, y = classes, cv = 10, scoring = 'accuracy')
media = resultados.mean()        # função que cálcula a média da variável 'resultados'
desvio = resultados.std()        # Função que cálcula o desvio padrão da variável 'resultados'

print('  ')
print(resultados)
print(media)
print(desvio)