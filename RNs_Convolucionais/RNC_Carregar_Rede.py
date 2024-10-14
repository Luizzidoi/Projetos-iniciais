"""
Rede Neural Artificial - Base de dados MNIST
Previsões com a rede neural
Carregar uma redeu neural salva e classificar uma nova imagem
Código desenvolvido por mim
Essa rede neural está com aproximadamente 99% de precisão

"""


import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import pandas as pd
from keras.models import model_from_json
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



""" Carrega a base de dados em variáveis: x (previsores) e y (classes) """
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()



# Carregar o arquivo do disco
arquivo = open('classificador_RNC_MNIST.json', 'r')
# Salvar em uma variável
estrutura_rede = arquivo.read()
arquivo.close()

# Cria a estrtutura da rede neural com as configurações lidas
classificador = model_from_json(estrutura_rede)
# Carrega o arquivo de pesos da rede
classificador.load_weights('classificador_RNC_MNIST.h5')



" Classificação de uma nova imagem "
# Utilizando uma imagem da base de dados de teste com o valor 7
plt.imshow(X_teste[10], cmap='gray')
plt.title('Classe' + str(y_teste[10]))
plt.show()

# Variável que armazenará a imagem a ser classificada e
# também fazemos a transformação na dimensão para o tensorflow processar
imagem_teste = X_teste[10].reshape(1, 28, 28, 1)

imagem_teste = imagem_teste.astype('float32')
imagem_teste /= 255

# Como temos um problema multiclasse e a função de ativação softmax, será gerada uma probabilidade para
# cada uma das classes. A variável previsão terá a dimensão 1x10, sendo que em cada coluna estará o
# valor de probabilidade de cada classe
previsao = classificador.predict(imagem_teste)
print('Matriz previsão:', previsao)

# Como cada índice do vetor representa um número entre 0 e 9, basta agora
# buscarmos qual é o maior índice e o retornarmos. Executando o código abaixo
# você terá o índice 7 que representa a classe 7
import numpy as np
numero = np.argmax(previsao)
print('O número é:', numero)