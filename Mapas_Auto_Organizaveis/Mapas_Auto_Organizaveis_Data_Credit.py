"""
Redes Neurais - Mapas Auto Organizáveis
Aprendizagem não supervisionada
Detecção de características e agrupamentos
Base de dados de crédito
Detecção de suspeitos de fraude

"""


from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


""" Carregar nas variáveis os atributos da base de dados """
base = pd.read_csv('credit_data.csv')
base = base.dropna()
# Média da idade
print('A média das idades são: ', base.age.mean())
# Preenchimento com o valor da média para algumas idades que estão com valores negativos
base.loc[base.age < 0, 'age'] = 40.80

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

normalizador = MinMaxScaler(feature_range=(0, 1))
X = normalizador.fit_transform(X)


""" Construção do mata auto organizável """
# tamanho do som = 5*sqrt(1997) = 223,43
som = MiniSom(x=15, y=15, input_len=4, sigma=1.0, learning_rate=0.5, random_seed=0)
# Inicialização dos pesos e treinamento
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)


""" Visualização do mapa """
import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, plot
# MID - mean inter neuron distance -> traz quanto parecido um neuronio é de seus vizinhos
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
color = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor='None', markersize=10,
         markeredgecolor=color[y[i]], markeredgewidth=2)

plt.show()


########################################################################################################################
""" 
Detecção de fraudes -> outliers
Buscar no mapa se os registros em amarelo (maior MID) são clientes suspeitos e ganharam o empréstimo (classificação 0),
ou seja, se os registros em amarelo foram escolhidos como BMUs
"""

""" Buscar quais registros estão asssociados a cada um dos neuronios """
mapeamento = som.win_map(X)
# Concatenar dois neuronios escolhidos no mapa (que estão em cor amarela - suspeitos)
# Axix = 0  para concatenação um abaixo do outro
suspeitos = np.concatenate((mapeamento[(4, 5)], mapeamento[(6, 13)]), axis=0)
suspeitos = normalizador.inverse_transform(suspeitos)


""" Buscar as classes de cada um dos suspeitos para avaliação """
classe = []
# i percorre todos os registros da base (1997 registrios)
# j percorre todos os registros da lista de suspeitos
for i in range(len(base)):
    for j in range(len(suspeitos)):
        # procurar se o id encontra na base é igual ao encontrado na lista dos suspeitos
        # round = arrendamento do valor e int para utilizar o inteiro
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i, 4])

classe = np.asarray(classe)
# Concatenar os dados dos suspeitos com as suas respectivasa classe
suspeitos_final = np.column_stack((suspeitos, classe))
# Ordenação dos dados para melhor visualização (conforme às clases)
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]
print(suspeitos_final)


print('Fim')