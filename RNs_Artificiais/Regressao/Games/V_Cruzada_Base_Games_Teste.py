"""
Redes Neurais - Regressão
Base de dados de video games
Regressão com múltiplas saídas
Previsão do valor de venda de alguns jogos em países diferentes
Teste: Previsão do valor total das vendas dos jogos utilizando validação Cruzada
"""

import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
import numpy as np
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import StratifiedKFold

""" PRÉ PROCESSAMENTO """
base = pd.read_csv('games.csv')

base = base.drop('Other_Sales', axis=1)
base = base.drop('Global_Sales', axis=1)
base = base.drop('Developer', axis=1)

base = base.dropna(axis=0)

base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

nome_jogos = base.Name
base = base.drop('Name', axis=1)

previsores = base.iloc[:, [0, 1, 2, 3, 7, 8, 9, 10, 11]].values
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

vendas = (np.vstack((venda_na, venda_eu, venda_jp))).T

# Tranformação dos atributos categóricos em atributos numéricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

# Linhas de código atualizada para a criação da variável do tipo dummy
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(sparse=False), [0, 2, 3, 8])], remainder='passthrough')
previsores = np.array(ct.fit_transform(previsores))


seed = 5
np.random.seed(seed)

# Variável kfold para controlar a validação cruzada
kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)

venda_teste_na = []
np.ndarray(venda_teste_na)
venda_teste_eu = []
np.ndarray(venda_teste_eu)
venda_teste_jp = []
np.ndarray(venda_teste_jp)

matriz_previsoes_na = []
matriz_previsoes_eu = []
matriz_previsoes_jp = []

""" ESTRUTURA DA REDE NEURAL COM VALIDAÇÃO CRUZADA """

for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(vendas.shape[0], 1))):
    print('Indices utilizados: Indice treinamento: ', indice_treinamento, '\nIndice teste: ', indice_teste)
    print('\n\n')

    venda_teste_na.append(venda_na[indice_teste])
    venda_teste_eu.append(venda_eu[indice_teste])
    venda_teste_jp.append(venda_jp[indice_teste])

    camada_entrada = Input(shape=(61,))
    camada_oculta1 = Dense(units=32, activation='sigmoid')(camada_entrada)
    camada_oculta2 = Dense(units=32, activation='sigmoid')(camada_oculta1)
    camada_saida1 = Dense(units=1, activation='linear')(camada_oculta2)
    camada_saida2 = Dense(units=1, activation='linear')(camada_oculta2)
    camada_saida3 = Dense(units=1, activation='linear')(camada_oculta2)

    regressor = Model(inputs=camada_entrada, outputs=[camada_saida1, camada_saida2, camada_saida3])
    regressor.compile(optimizer='adam', loss='mse')
    # regressor.compile(optimizer='adam', loss='mean_absolute_error')
    regressor.fit(previsores[indice_treinamento], [venda_na[indice_treinamento], venda_eu[indice_treinamento], venda_jp[indice_treinamento]], epochs=10000, batch_size=100)

    previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores[indice_teste])

    matriz_previsoes_na.append(previsao_na)
    matriz_previsoes_eu.append(previsao_eu)
    matriz_previsoes_jp.append(previsao_jp)




print("Fim")
# utilizei essa forma de validação cruzada para conseguir fazer o código para esse tipo de regressão