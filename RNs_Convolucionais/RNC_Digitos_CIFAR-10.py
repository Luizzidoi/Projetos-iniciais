"""
Redes Neurais Convolucionais
Base de dados CIFAR-10 (objetos)
Desafio: Analisar a precisão dessa rede neural

"""


import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
# Flatten é a etapa 3 da rede neural convolucional
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
# Conv2D é a etapa 1 (operador de convolução) da rede neural e MaxPooling2D é a etapa 2 (pooling)
from keras.layers import Conv2D, MaxPooling2D
# Função utilizada para normalização dos mapas de caracteristicas. Melhorias da rede neural
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(X_treinamento, y_treinamento), (X_teste, y_teste) = cifar10.load_data()
# Alguns exemplos:
# Avião - 650
# Pássaro - 6
# Gato - 9
# Veado - 3
# Cachorro - 813
# Sapo - 651
# Cavalo - 652
# Barco - 811
# Caminhão - 970
# Automóvel - 4
## Busca a imagem da base de dados. cmap = 'gray' -> Mostrar a imagem sem cor, para diminuir o processamento
# plt.imshow(X_treinamento[650])
# plt.title('Classe' + str(y_treinamento[650]))
# plt.show()

# As dimensões dessas imagens é 32x32 e o número de canais é 3 pois vamos utilizar as imagens coloridas
# reshape = mudar o formato dos dados
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 32, 32, 3)
previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)

# Conversão para float para podermos aplicar a normalização
# Modificar o tipo da variável para float32
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# Normalização para os dados ficarem na escala entre 0 e 1 e agilizar o processamento
# '/= 255' -> recebe ele mesmo e divide por 255
previsores_treinamento /= 255
previsores_teste /= 255

# Criação de variáveis do tipo dummy, pois tem-se 10 saídas
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)



""" Criação da rede neural com duas camadas de convolução """
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Conv2D(32, (3, 3), activation='relu'))
# Apenas para melhorias da rede, normalização do mapa de caracteristica
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())

# Criação da rede neural densa
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))

" Sem Augumentation "
# classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# validation_data = já treina a rede e faz os testes com as bases 'previsores_teste' e 'classe_teste'
# classificador.fit(previsores_treinamento, classe_treinamento, batch_size=128, epochs=10,
#                   validation_data=(previsores_teste, classe_teste))

# Sem augumentation = 68,77% de precisão


" Com augumentation "

classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

gerador_treinamento = ImageDataGenerator(rotation_range=7, horizontal_flip=True,
                                         shear_range=0.2, height_shift_range=0.07,
                                         zoom_range=0.2)
gerador_teste = ImageDataGenerator()

base_treinamento = gerador_treinamento.flow(previsores_treinamento, classe_treinamento, batch_size=128)
base_teste = gerador_teste.flow(previsores_teste, classe_teste, batch_size=128)

classificador.fit_generator(base_treinamento, steps_per_epoch=50000 / 128, epochs=10,
                            validation_data=base_teste, validation_steps=10000 / 128)

# Com augumentation = 63,47% de precisão.
# Porém pode-se colocar o 'steps_per_epochs' igual a 50.000. Nesse caso o gerador passará por todas as imagens para
# fazer a transformação (maior processamento) e provavelmente obteremos maior % de precisão da rede.


""" Printa a precisão dessa Rede Neural """
resultado = classificador.evaluate(previsores_teste, classe_teste)

print(resultado)

