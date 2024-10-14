"""
Redes Neurais Convolucionais
Base de dados MNIST (Digitos)
1 - Analisar a precisão dessa rede neural
2 - Buscar um número aleatório da base MNIST e classificar o digíto correspondente
"""


import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
# Flatten é a etapa 3 da rede neural convolucional
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
# Conv2D é a etapa 1 (operador de convolução) da rede neural e MaxPooling2D é a etapa 2 (pooling)
from keras.layers import Conv2D, MaxPooling2D
# Função utilizada para normalização dos mapas de caracteristicas. Melhorias da rede neural
from keras.layers.normalization import BatchNormalization
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



""" Carrega a base de dados em variáveis: x (previsores) e y (classes) """
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
# Busca a imagem da base de dados. cmap = 'gray' -> Mostrar a imagem sem cor, para diminuir o processamento
plt.imshow(X_treinamento[0], cmap='gray')
plt.title('Classe ' + str(y_treinamento[0]))

## Viasualizador da imagem na tela
# plt.show()


""" Transformação nos dados para que o Tensorflow consiga fazer a leitura """
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)  # reshape = mudar o formato dos dados
print('X_treinamento.shape[0] significa a posição 0 de (60000, 28, 28):',  X_treinamento.shape[0])
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')  # Modificar o tipo da variável para uint32
previsores_teste = previsores_teste.astype('float32')

# Modificação da escala dos valores para que o processamento seja mais rápido. Ao invés de 0 a 255, será de 0 a 1
# Técnica do min max normalization = normalização dos dados para diminuir a escala
previsores_treinamento /= 255    # '/= 255' -> recebe ele mesmo e divide por 255
previsores_teste /= 255

# Modificação das classe para tipo dummy
classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)


""" Estrutura da Rede Neural """
########################################################################################################################
# Etapa 1 = Operador de convolução
classificador = Sequential()
classificador.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
# Apenas para melhorias da rede, normalização do mapa de caracteristica
classificador.add(BatchNormalization())
# Etapa 2 = Pooling
classificador.add(MaxPooling2D(pool_size=(2, 2)))
#Etapa 3 = Flattening
# Ao adicionar mais uma camada de convolução a etapa Flattening precisar ser implementado apenas uma vez no final
# classificador.add(Flatten())

# Adição de mais uma camada de convolução para melhorar os resultados - melhorias da rede neural
classificador.add(Conv2D(32, (3, 3), activation='relu'))
# Apenas para melhorias da rede, normalização do mapa de caracteristica
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))
# Etapa 3
classificador.add(Flatten())

# Etapa 4 = Criação da rede neural densa
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# validation_data = já treina a rede e faz os testes com as bases 'previsores_teste' e 'classe_teste'
classificador.fit(previsores_treinamento, classe_treinamento, batch_size=128, epochs=10,
                  validation_data=(previsores_teste, classe_teste))

""" Printa o valor da precisão da rede -. O mesmo que usar o parametro 'validaiton data' anteriormente """
resultado = classificador.evaluate(previsores_teste, classe_teste)
print(resultado)



""" Previsão de uma imagem """
# Utilizando uma imagem da base de dados de teste com o valor 7
plt.imshow(X_teste[0], cmap='gray')
plt.title('Classe' + str(y_teste[0]))
plt.show()

# Variável que armazenará a imagem a ser classificada e
# também fazemos a transformação na dimensão para o tensorflow processar
imagem_teste = X_teste[0].reshape(1, 28, 28, 1)

imagem_teste = imagem_teste.astype('float32')
imagem_teste /= 255

# Como temos um problema multiclasse e a função de ativação softmax, será gerada uma probabilidade para
# cada uma das classes. A variável previsão terá a dimensão 1x10, sendo que em cada coluna estará o
# valor de probabilidade de cada classe
previsao = classificador.predict(imagem_teste)
print(previsao)

# Como cada índice do vetor representa um número entre 0 e 9, basta agora
# buscarmos qual é o maior índice e o retornarmos. Executando o código abaixo
# você terá o índice 7 que representa a classe 7
import numpy as np
numero = np.argmax(previsao)
print('O número é:', numero)


" Salvar rede neural "

# Salva como String todos os dados que foram passados para a rede (Estrutura da rede neural)
classificador_json = classificador.to_json()

# salvando a estrutura da rede neural em disco
with open('classificador_RNC_MNIST.json', 'w') as json_file:
    json_file.write(classificador_json)

# Salvando os pesos da rede neural
classificador.save_weights('classificador_RNC_MNIST.h5')
