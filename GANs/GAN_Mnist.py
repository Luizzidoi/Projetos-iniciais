"""
Redes Neurais - GANs (Generative Adversarial Networks)
Geração de imagens
Base de dados MNIST
Geração de digitos manuscritos
Após o aprendizado, a rede neural consegue gerar novos digitos

"""


###### Enquanto as outras redes neurais utiliza o Tensorflow 2.1.0, a GAN é necessário a utilização do Tensorflow 1.14  #######
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Flatten, Reshape
# Conceito de regularização
from keras.regularizers import L1L2
# Keras adversarial para utilização das redes neurais do tipo GAN
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling



""" Carregamento da base de dados """
(previsores_treinamento, _), (_, _) = mnist.load_data()

previsores_treinamento = previsores_treinamento.astype('float32') / 255



########################################################################################################################
""" GERADOR (responsavel por gerar as imagens) """
# units = 500. Foi feito o Tuning para descobrir qual a quantidade de units (neuronios) é melhor
# iinput dim = a rede pega de 100 em 100 imagens para fazer o treinamento para o discriminador
# kernel_regularizer = valores extraídos da documentação da lib. Adionada uma função de penalidade na aprendizagem para evitar o overfitting
# units da saída = 784, pois são 28x28 pixels que o Gerador precisa passar na saída
gerador = Sequential()
gerador.add(Dense(units=500, input_dim=100, activation='relu',
                  kernel_regularizer=L1L2(1e-5, 1e-5)))
gerador.add(Dense(units=500, input_dim=100, activation='relu',
                  kernel_regularizer=L1L2(1e-5, 1e-5)))
gerador.add(Dense(units=784, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)))
# Transformar os 784 pixels em imagens (28x28)
gerador.add(Reshape((28, 28)))


########################################################################################################################
""" DISCRIMINADOR (responsável por acessar as imagens geradas e informar se são parecidas com as originais) """
# InputLayer é para que a camada oculta receba a imagem (28, 28)
discriminador = Sequential()
discriminador.add(InputLayer(input_shape=(28, 28)))
# Tranformar novamente em vetor para que a rede consiga trabalhar
discriminador.add(Flatten())

discriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminador.add(Dense(units=500, activation='relu', kernel_regularizer=L1L2(1e-5, 1e-5)))
discriminador.add(Dense(units=1, activation='sigmoid', kernel_regularizer=L1L2(1e-5, 1e-5)))

########################################################################################################################

""" Contrução da GAN """
# normal_latent_sampling = geração de 100 imagens, input_dim configurado na camada oculta do gerador
gan = simple_gan(gerador, discriminador, normal_latent_sampling((100,)))

# player_params = pegar os pesos que a rede gerou/utilizou no gerador e discriminador
model = AdversarialModel(base_model=gan, player_params=[gerador.trainable_weights, discriminador.trainable_weights])

# AdversarialOptimizerSimultaneous = vai atualizar cada uma das redes neurais simultaneamente em cada um dos
    # batchs (quantidade de registros utilizados para fazer a atualização dos pesos)
# player_optimizers = otimizador utilizado em cada uma das redes
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(), player_optimizers=['adam', 'adam'],
                          loss='binary_crossentropy')

# x = base de dados com os 60mil digitos, ou seja, imagens dos objetos reais
model.fit(x=previsores_treinamento, y=gan_targets(60000), epochs=1, batch_size=256)


""" Visualizar as imagens """
# 10 = imagens; 100 = input_dim do gerador
amostras = np.random.normal(size=(10, 100))
previsao = gerador.predict(amostras)
# previsao.shape[0] retorna o valor 10. Ou seja, um for de 0 a 10
for i in range(previsao.shape[0]):
    plt.imshow(previsao[i, :], cmap='gray')

plt.show()


print('\n Fim')

# Para a visualização das imagens utilizamos somente o gerador, pois toda a parte do 
    # treinamento (descriminador + treinamento) é para programar os pesos do gerador para que ele
    # receba como paramêtro números aleatórios e consiga fazer a geração das imagens

