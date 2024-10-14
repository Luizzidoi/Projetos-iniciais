"""
Redes Neurais - Autoencoder
Aprendizagem não supervisionada
Redução de dimensionalidade
Base de dados MNIST
Redimensionamento, codificação e decodificação de imagens
Convolutional Autoencoder - Unição de tecnicas de convolução com autoencoder

"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
# UpSampling2D é o inverso do MaxPooling
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Carregando a base de dados mnist """
(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()

""" Reshape da imagem original """
# Modelagem das variáveis para utilizar as práticas de convolução
# 1 no final = número de canais -> para a escala de cinza
previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), 28, 28, 1))
previsores_teste = previsores_teste.reshape((len(previsores_teste), 28, 28, 1))

previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255


""" Algoritmo para o treinamento - com redes neurais convolucionais """
autoencoder = Sequential()

##### Encoder
autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
# Max pooling que pega as melhores caracteristicas para reduzir a imagem
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

# Padding = como a imagem será passada. É necessário indicar para não gerar erro
autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Strides = indicar de quantos em quantos pixels as imagens deve andar
# (4, 4, 8) = dimensões da matriz resultante
autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2)))

autoencoder.add(Flatten())

# Retornar o vetor em matriz (4, 4, 8)
autoencoder.add(Reshape((4, 4, 8)))


##### Decoder
autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
autoencoder.add(UpSampling2D(size=(2, 2)))

autoencoder.add(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=50, batch_size=256,
                validation_data=(previsores_teste, previsores_teste))


""" Extração da camada codificada -> É a camada Flatten """
# inputs = extrai todos os dados de entrada
# outputs = extrair a camada codificada
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
print(encoder.summary())

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)


""" Visualização das imagens """
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)
plt.figure(figsize=(18, 18))
for i, indice_imagem in enumerate(imagens_teste):

    # Imagem original
    eixo = plt.subplot(10, 10, i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())

    # Imagem codificada
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(16, 8))
    plt.xticks(())
    plt.yticks(())

    # Imagem reconstruída
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28, 28))
    plt.xticks(())
    plt.yticks(())

plt.show()
