"""
Redes Neurais - Autoencoder
Aprendizagem não supervisionada
Redução de dimensionalidade
Base de dados MNIST
Redimensionamento, codificação e decodificação de imagens

"""


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Carregando a base de dados mnist """
# _ é que não há necessidade do preeenchimento das classes treinamento e teste
(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()

# Normalização
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

""" Reshape da imagem original """
# np.prod faz o produto de 28x28 (pixels)
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))

""" Definição da redução de dimensionalidade """
# 784 - 32 - 784 (entradas - camada escondida - saídas)
fator_compactacao = 784 / 32


""" Algoritmo para o treinamento - Rede neural densa """
autoencoder = Sequential()
autoencoder.add(Dense(units=32, activation='relu', input_dim=784))
autoencoder.add(Dense(units=784, activation='sigmoid'))
# Visualização da estrutura
# Param: dense_1 = 784*32 + 32(bayes)
# Param: dense_2 = 32*784 + 784(bayes)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Como não estamos trabalhando com problema de classificação, então não é preciso passar a 'classe treinamento'
# Previsores treinamento é comparado com previsores treinamento -> x e y
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=50, batch_size=256,
                validation_data=(previsores_teste, previsores_teste))


########################################################################################################################
""" Visualização das imagens (original, codificada e reconstruída) """
dimensao_original = Input(shape=(784,))
# Buscar a camada de codificação
camada_encoder = autoencoder.layers[0]
# Buscar a camada codificada (oculta)
# Model utilizado para criação manual da rede
encoder = Model(dimensao_original, camada_encoder(dimensao_original))
print(encoder.summary())

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)

plt.figure(figsize=(18, 18))
# i é um contador de 0 a 10 (numero_imagens)
# indice_image é o indice da imagem escolhida randomicamente entre as 10 mil imagens de previsores_teste
for i, indice_imagem in enumerate(imagens_teste):

    # Imagem original
    eixo = plt.subplot(10, 10, i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())

    # Imagem codificada
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8, 4))
    plt.xticks(())
    plt.yticks(())

    # Imagem reconstruída
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())

plt.show()



print('\nFim')