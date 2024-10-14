"""
Redes Neurais - Mapas Auto Organizáveis
Aprendizagem não supervisionada
Detecção de características e agrupamentos
Base de dados de vinhos
Classificação e agrupamento dos tipos de vinhos

"""


from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


""" Carregar nas variáveis os atributos da base de dados """
base = pd.read_csv('wines.csv')
X = base.iloc[:, 1:14]. values
y = base.iloc[:, 0].values

normalizador = MinMaxScaler(feature_range=(0, 1))
X = normalizador.fit_transform(X)


""" Contrução e configuração do mapa auto organizável """
# SOM = Self Organizing Map
# x = linhas
# y = colunas
som = MiniSom(x=8, y=8, input_len=13, sigma=1.0, learning_rate=0.5, random_seed=2)

# Inicialicação dos pesos
som.random_weights_init(X)
# Treinamento
som.train_random(data=X, num_iteration=100)

# Mostrar os pesos utilizados
som._weights
# Valores do mapa auto organizável
som._activation_map
# Mostra quantas vezes cada neuronio foi utilizado como BMU
q = som.activation_response(X)


""" Visualização do mapa """
import matplotlib.pyplot as plt
from pylab import pcolor, colorbar, plot
# MID - mean inter neuron distance -> traz quanto parecido um neuronio é de seus vizinhos -> e é isso que o gráfico plota
# T é para matriz transposta
pcolor(som.distance_map().T)
colorbar()

# Mostrar o neuronio vencedor para um determinado registro
w = som.winner(X[1])
# Criação dos marcadores
markers = ['o', 's', 'D']
# criação das cores
color = ['r', 'g', 'b']

# modificação da classe para 0,1 ou 2 ao invés de 1,2 ou 3 (para que o algoritmo consiga fazer a comparação)
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2


""" Adicionar ao mapa a classificação com os marcadores """
for i, x in enumerate(X):
    # print(i)
    # print(x)
    w = som.winner(x)
    # print(w)
    # plot(posição x do marcador para um quadradinho da matriz, posição y do marcador, variável com as classe (0 a 177),
    # cor da fonte, tamanho do marcador, preenchimento da cor dos simbolos, configuração da borda)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor='None', markersize=10,
         markeredgecolor=color[y[i]], markeredgewidth=2)
plt.show()


print('Fim')