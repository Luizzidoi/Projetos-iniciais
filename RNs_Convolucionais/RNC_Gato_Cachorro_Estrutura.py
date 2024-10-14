"""
Redes Neurais Convolucionais
Base de dados de gatos e cachorros (pasta 'dataset')
Carrega uma imagem da base e informa se a imagem é um gato ou cachorro
Obs.: Não é necessário carregar as imagens manualmente utilizando essa classe (ImageDataGenerator)
"""


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# Função utilizada para normalização dos mapas de caracteristicas. Melhorias da rede neural
from keras.layers.normalization import BatchNormalization
# Classe geradora do Augumentation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# Faz a leitura de uma imagem
from keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


""" Estrutura da Rede neural """
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Conv2D(32, (3, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))

classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=1, activation='sigmoid'))

classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


""" Gerar as imagens utilizadas para o treinamento, normalizá-las e redimensioná-las """
# rescale = normalização dos dados para diminuir a escala (entre 0 e 1)
gerador_treinamento = ImageDataGenerator(rescale=1./255, rotation_range=7, horizontal_flip=True,
                                         shear_range=0.2, height_shift_range=0.07, zoom_range=0.2)
gerador_teste = ImageDataGenerator(rescale=1./255)


""" Buscar a pasta da base de dados """
base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size=(64, 64),
                                                           batch_size=32, class_mode='binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set', target_size=(64, 64),
                                               batch_size=32, class_mode='binary')

# 4000 e 1000 é a quantidade de imagens que a base de dados tem para treinamento e teste, respectivamente
classificador.fit_generator(base_treinamento, steps_per_epoch=4000 / 32, epochs=5,
                            validation_data=base_teste, validation_steps=1000 / 32)



""" Previsões de uma imagem """
# Carregar uma imagem para teste
imagem_teste = image.load_img('dataset/test_set/gato/cat.3500.jpg', target_size=(64, 64))
# Transforma a imagem para uma matriz de dados dos pixels.
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
# Expande as dimensões da imagem (matriz) pq é a forma que o Tensorflow trabalha com as imagens. Axis = 0 para adc uma coluna
# (1, 64, 64, 3) = (Quantidade de imagens (batch), pixels de altura, pixels de largura, quantidade de canais)
imagem_teste = np.expand_dims(imagem_teste, axis=0)

previsao = classificador.predict(imagem_teste)
# Como é uma função sigmoid, o valor da previsao é uma probabilidade

""" Printa qual o valor de cada classe. Cachorro = 0 e Gato = 1 """
print(base_treinamento.class_indices)
print(previsao)
if previsao > 0.5:
    print("A imagem é um gato!")
else:
    print("A imagem é um cachorro!")