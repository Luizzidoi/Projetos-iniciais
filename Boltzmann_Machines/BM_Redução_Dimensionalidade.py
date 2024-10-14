""""
Redes Neurais - Boltzmann Machines
Aprendizagem não supervisionada
Redução de dimensionalidade
Algoritmo Naive Bayes
Redução de dimensionalidade em imagens

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Implementação das RBM do sklearn
from sklearn.neural_network import BernoulliRBM
# Algoritmo que faz estatisticas de probabilidade
from sklearn.naive_bayes import GaussianNB
# Classe que executa vários processos em conjunto
from sklearn.pipeline import Pipeline


""" Carregar a base de dados do sklearn """
base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = base.target


""" Normalização dos dados """
normalizador = MinMaxScaler(feature_range=(0, 1))
previsores = normalizador.fit_transform(previsores)


""" Divisão da base de dados em treinamento e teste """
# Random_state = 0 -> indica que sempre que executado a base será divida da mesma forma, com resultados semelhantes
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.2,
                                                                                              random_state=0)

""" Implementação do RBM para redução de dimensaionalidade"""
rbm = BernoulliRBM(random_state=0)
# n_iter = número de epochs
# n_components = numero de neuronios na camada escondida
rbm.n_iter = 25
rbm.n_components = 50

""" Algoritmo naive_Bayes """
naive_rbm = GaussianNB()


""" Execução do treinamento """
# Pipeline = maneira de executar dois processos subsequentes, ou então dois processos ao mesmo tempo
# Pipeline primeiro executa o algorítimo rbm e depois os resultados passam para o algorítimo Naive Bayes
classificador_rbm = Pipeline(steps=[('rbm', rbm), ('naive', naive_rbm)])
classificador_rbm.fit(previsores_treinamento, classe_treinamento)


""" Visualização das imagens que sofreu redução de dimensionalidade """
# Nesse for será percorrido os neuronios da camada escondida para analisar as imagens criadas
# rbm.components_ = valores do 50 neuronios definidos na camada escondida
# cmap=plt.cm.gray_r -> cor da imagem preto e branco
# xticks e y ticks = retirar os valores que aparecem nos eixos x e y (caption)
plt.figure(figsize=(20, 20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()


""" Cpmparação da tecnica rbm com a não utilização de rbm """
# Precisão com RBM
previsoes_rbm = classificador_rbm.predict(previsores_teste)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)
print('A precisão usando a técnica de RBM é: ', precisao_rbm)

# Precisão sem RBM
naive_simples = GaussianNB()
naive_simples.fit(previsores_treinamento, classe_treinamento)
previsoes_naive = naive_simples.predict(previsores_teste)
precisao_naive = metrics.accuracy_score(previsoes_naive, classe_teste)
print('A precisão sem utilizar a técnica de RBM é: ', precisao_naive)


#### Observação: a imagem reduz de 64 caracteristicas para 50, conforme adotado (rbm.n_components = 50)
print("\nFim")