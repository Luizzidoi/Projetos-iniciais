import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.models import model_from_json
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" SALVAR E CARREGAR REDE NEURAL EM DISCO """

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



"""Criação da Rede Neural para o treinamento e posteriormente salvar em disco """

# # Criação de uma nova rede neural
# classificador = Sequential()
# # Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 4 + 3 / 2 = 3.5
# classificador.add(Dense(units=8, activation='tanh', kernel_initializer='random_uniform', input_dim=4))
# classificador.add(Dropout(0.3))
# # Criação da segunda camada oculta
# classificador.add(Dense(units=8, activation='tanh', kernel_initializer='random_uniform'))
# classificador.add(Dropout(0.3))
# # Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid
# classificador.add(Dense(units=3, activation='softmax'))
#
# classificador.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # Treinamento da rede neural, encontra uma relação entre previsores e classe_dummy
# classificador.fit(previsores, classe, batch_size=10, epochs=2000)


""" Salvar a rede neural e os pesos em disco """
# classificador_json = classificador.to_json()
# with open('classificador_Iris.json', 'w') as json_file:
#     json_file.write(classificador_json)
# classificador.save_weights('classificador_Iris.h5')


""" Carregar estrutura da Rede Neural em disco """

arquivo = open('classificador_Iris.json', 'r')
# Salvar em um variável
estrutura_rede = arquivo.read()
arquivo.close()

# Cria a estrtutura da rede neural com as configurações lidas
classificador = model_from_json(estrutura_rede)
# Carrega o arquivo de pesos da rede
classificador.load_weights('classificador_Iris.h5')


""" Classificação de um novo registro """
novo_registro = np.array([[3.2, 4.5, 0.9, 1.1]])
# Previsão do tumor
previsao = classificador.predict(novo_registro)
print(previsao)
previsao = (previsao > 0.5)
print(previsao)

if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')


