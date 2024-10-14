import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       # Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)

previsores = pd.read_csv('Entradas_Breast_cancer.csv')
classes = pd.read_csv('Saidas_classes.csv')

## Substituir os parametros ideias conforme a classificação Tuning da aula anterior
classificador = Sequential()                                                                                    # Criação de uma nova rede neural
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))        # Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 30 + 1 / 2 = 15.5
classificador.add(Dropout(0.2))  # Adicionado o Dropout para análise
classificador.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))                      # Criação da segunda camada oculta
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation='sigmoid'))                                                         # Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid

# otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores, classes, batch_size=10, epochs=100)

novo_registro = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                          0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05,
                          0.015, 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                          0.84, 158, 0.363]])

previsao = classificador.predict(novo_registro)                                                                 # Previsão do tumor
print(previsao)
previsao = (previsao > 0.5)
print(previsao)