"""
Redes Neurais Convolucionais
Base de dados MNIST (Digitos)
Augumentation
Um método que aumenta a quantidade das imagens; evita overfitting; utilizado quando a base de dados é poquena
"""


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
# Classe geradora do Augumentation
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()


previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')
previsores_treinamento /= 255
previsores_teste /= 255
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
classificador.add(MaxPooling2D(pool_size=(2, 2)))
classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modifica as imagens de acordo com os parametros passados para aumentar a base de dados
gerador_treinamento = ImageDataGenerator(rotation_range=7, horizontal_flip=True,
                                         shear_range=0.2, height_shift_range=0.07,
                                         zoom_range=0.2)
# Não é necessário aumentar a base de dados de teste pq apenas as mil imagens já é o suficiente pra testar
# Por isso não passa nenhuma confg
gerador_teste = ImageDataGenerator()

# Cria nova base de dados com mais imagens após as modificações
base_treinamento = gerador_treinamento.flow(previsores_treinamento, classe_treinamento, batch_size=128)
base_teste = gerador_teste.flow(previsores_teste, classe_teste, batch_size=128)

classificador.fit_generator(base_treinamento, steps_per_epoch=60000 / 128, epochs=5,
                            validation_data=base_teste, validation_steps=10000 / 128)



print('Fim')

# Observação: Como essa base de dados já é grande, então não é tão necessário utilizar o Augumentation. Porém em
#  bases de dados menores esse tipo de resolução se torna muito útil, melhorando bastante os resultados.
#  Tivemos uma precisão de 98% (val_accuracy) aproximadamente.
#  A rede neural já cria a nova base de dados e treina logo em seguida. Com os códigos aprendidos não é possível
#  visualizar a quantidade de imagens novas que a rede neural criou.