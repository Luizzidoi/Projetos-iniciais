import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base = pd.read_csv('iris.csv')

# iloc é um função que pode ser utilizada para fazer a divisão de um arquivo csv
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


# Importação da classe que será utilizada para a transformação em atributos numéricos
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Faz a transformação do atributo (classes) categórico para atributo numérico
classe = label_encoder.fit_transform(classe)
# Variável que vai receber a transformação em 3 dimensões
classe_dummy = np_utils.to_categorical(classe)
# Iris Setosa       1 0 0
# Iris Virginica    0 1 0
# Iris Versicolor   0 0 1


# Lib que já faz a divisão da base de dados entre treinamento e teste
# test_size signifca que será utilizado 25% dos registros para teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)



""" Criação de uma nova rede neural """
classificador = Sequential()
# Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 4 + 3 / 2 = 3.5
classificador.add(Dense(units=4, activation='relu', input_dim=4))
# Criação da segunda camada oculta
classificador.add(Dense(units=4, activation='relu'))
# Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid
classificador.add(Dense(units=3, activation='softmax'))

classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Treinamento da rede neural, encontra uma relação entre previsores_treinamento e classe_treinamento
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)


""" RESULTADOS """
# Metodo especifico do Keras para uma avaliação automática. Compara 'previsores_teste' com 'classe_teste' e mostra a % de acertos
resultado = classificador.evaluate(previsores_teste, classe_teste)

# Realiza a avaliação manual dos previsores_teste para uma possível avaliação de onde teve os erros de classificação
previsoes = classificador.predict(previsores_teste)
import numpy as np
# Essa funçao (np.argmax()) percorre a variável classe_teste e retorna apenas o valor de qual tipo de planta a rede classificou. Utilizado em problemas com mais de 2 classes na saída
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

# Cria a matriz de confusão para indicar os acertos e erros da rede
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)

# print(classe_dummy)
# print(resultado)
# print(previsoes)
print(matriz)