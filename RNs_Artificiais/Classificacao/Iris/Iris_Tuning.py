import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import os
# Serve para ignorar o erro do AVX2 (por estar usando o tensorflow interno pelo anaconda)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base = pd.read_csv('iris.csv')

# iloc é um função que pode ser utilizada para fazer a divisão de um arquivo csv
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


# Importação da classe que será utilizada para a transformação em atributos numéricos
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
# Faz a transformação do atributo (classes) categórico para atributo numérico
classe = label_encoder.fit_transform(classe)
# Variável que vai receber a transformação em 3 dimensões
classe_dummy = np_utils.to_categorical(classe)


""" Confirguração da Rede Neural """
def criarRede(optimizer, kernel_initializer, activation, neurons, drops):

    classificador = Sequential()
    # Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 4 + 3 / 2 = 3.5
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=4))
    classificador.add(Dropout(drops))
    # Criação da segunda camada oculta
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(drops))
    # Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid
    classificador.add(Dense(units=3, activation='softmax'))

    classificador.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classificador


classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size': [10, 20],
              'epochs': [200, 400],
              'optimizer': ['adam', 'sgd'],
              'drops': [0.2, 0.3],
              # 'loss': ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
              'kernel_initializer': ['random_uniform', 'he_normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [4, 8]}


grid_search = GridSearchCV(estimator=classificador, param_grid=parametros, cv=5)
grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print(melhores_parametros)
print(melhor_precisao)