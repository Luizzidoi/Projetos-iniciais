"""
Redes Neurais Recorrentes
Base de dados Bolsa de valores
Previsão do preços de ações
Carregar uma rede salva em disco
Geração de um gráfico para comparação entre o preço real e o previsto pela rede neural

"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



""" Carregar o arquivo do disco """

arquivo = open('regressor_1Previsor_1Saida.json', 'r')
# Salvar em uma variável
estrutura_rede = arquivo.read()
arquivo.close()

# Cria a estrtutura da rede neural com as configurações lidas
regressor = model_from_json(estrutura_rede)
# Carrega o arquivo de pesos da rede
regressor.load_weights('regressor_1Previsor_1Saida.h5')


""" Carregando as variáveis com os atributos da base de dados """

base = pd.read_csv('petr4_treinamento.csv')
# Retirando os valores Nan da base de dados
base = base.dropna()
# Utilizando o parametro "Open" da base de dados para o treinamento
base_treinamento = base.iloc[:, 1:2].values

# Normalização dos valores para valores entre 0 e 1. Minimizar o processamento
normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)



""" Utilizando a base de dados de teste  """

base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
# Concatenar a base de dados de treinamento com a base de dados de teste para conseguir buscar/utilizar os 90 valors anteriores.
# Axis=0 para concatenção por coluna
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
# Buscar 90 valores anteriores
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
# Deixar no formato do numpy: (112, 1)
entradas = entradas.reshape(-1, 1)
# Normalização das entradas
entradas = normalizador.transform(entradas)


""" Utilizando a base de teste """

X_teste = []
# 90 é o início da base de teste e 112 o fim
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])

X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

diferenca = previsoes.mean() - preco_real_teste.mean()
print(diferenca)


########################################################################################################################
""" Geração do gráfico para análise """

plt.plot(preco_real_teste, color='red', label='Preço Real')
plt.plot(previsoes, color='blue', label='Previsões')
plt.title('Previsão do preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor')
plt.legend()
plt.show()



print("Fim")