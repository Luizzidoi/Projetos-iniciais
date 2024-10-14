"""
Redes Neurais - Autoencoder
Aprendizagem não supervisionada
Redução de dimensionalidade
Base de dados MNIST
Autoencoder e classificação
Comparação utilizando a base de dados com e sem redução de dimensionalidade

"""


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Carregando a base de dados mnist """
(previsores_treinamento, classe_treinamento), (previsores_teste, classe_teste) = mnist.load_data()
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

""" Transformação dos valores das classes em variáveis do tipo dummy (para não gerar erros com keras) """
classe_dummy_treinamento = np_utils.to_categorical(classe_treinamento)
classe_dummy_teste = np_utils.to_categorical(classe_teste)

""" Reshape da imagem original """
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))


""" Algoritmo para o treinamento e geração das imagens codificadas - Rede neural densa """
print('ALGORITMO INICIAL PARA A CRIAÇÃO DAS VARIÁVEIS CODIFICADAS\n\n')
autoencoder = Sequential()
autoencoder.add(Dense(units=32, activation='relu', input_dim=784))
autoencoder.add(Dense(units=784, activation='sigmoid'))
print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=50, batch_size=256,
                validation_data=(previsores_teste, previsores_teste))
print('\n\nETAPA 1 CONCLUÍDA!\n\n\n')


dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))

previsores_treinamento_codificado = encoder.predict(previsores_treinamento)
previsores_teste_codificado = encoder.predict(previsores_teste)



########################################################################################################################
""" Comparar os resultados entre a rede com e sem redução de dimensionalidade """

""" ## Rede neural sem redução de dimensionalidade ## """
# units = (784 + 10)/2 = 397
print('TREINAMENTO DA REDE NEURAL SEM REDUÇÃO DE DIMENSIONALIDADE\n\n')
c1 = Sequential()
c1.add(Dense(units=397, activation='relu', input_dim=784))
c1.add(Dense(units=397, activation='relu'))
c1.add(Dense(units=10, activation='softmax'))

c1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
c1.fit(previsores_treinamento, classe_dummy_treinamento, epochs=100, batch_size=256,
                validation_data=(previsores_teste, classe_dummy_teste))
print('\n\nETAPA 2 CONCLUÍDA!\n\n\n')


""" ## Rede neural com redução de dimensionalidade ## """
# units = (32 + 10)/2 = 21
print('TREINAMENTO DA REDE NEURAL COM REDUÇÃO DE DIMENSIONALIDADE\n\n')
c2 = Sequential()
c2.add(Dense(units=21, activation='relu', input_dim=32))
c2.add(Dense(units=21, activation='relu'))
c2.add(Dense(units=10, activation='softmax'))

c2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
c2.fit(previsores_treinamento_codificado, classe_dummy_treinamento, epochs=100, batch_size=256,
                validation_data=(previsores_teste_codificado, classe_dummy_teste))
print('\n\nETAPA 3 CONCLUÍDA!\n')



########################################################################################################################
""" 
Conclusão: Apesar da rede neural com redução de dimensionalidade resultar em uma precisão um pouco inferior (98 contra 
95% - utilizando 100 epocas nas duas redes), o processo de treinamento é bem mais rápido, o que traz uma vantagem nesse 
tipo de processo. Portanto depende muito do sistema e do que queremos.
"""