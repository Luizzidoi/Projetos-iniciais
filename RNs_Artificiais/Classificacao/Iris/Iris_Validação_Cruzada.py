import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
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


""" Criação da Rede Neural """
def criarRede():

    classificador = Sequential()
    # Criação da primeira camada oculta e definição da camada de entrada. Units = (entradas + saídas) / 2 = 4 + 3 / 2 = 3.5
    classificador.add(Dense(units=4, activation='relu', input_dim=4))
    # Criação da segunda camada oculta
    classificador.add(Dense(units=4, activation='relu'))
    # Camada de saída. Units = 1 (apenas uma resposta 0 ou 1) e função de saída sigmoid
    classificador.add(Dense(units=3, activation='softmax'))

    classificador.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return classificador


 # 'batch_size'=10 -> a cada 10 registros a rede faz a atualização dos pesos
classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)

# Função que faz a validação cruzada dividindo a base de dados em 10 (valor do cv) partes
resultados = cross_val_score(estimator=classificador, X=previsores, y=classe, cv=10, scoring='accuracy')


# Cálcula a média e o desvio. A média serve pq a validação cruzada traz 10 (valor do cv) resultados de probabilidade e para ter uma maior rpecisão da % da rede é feito uma média entre esses 10 valores
# O desvio é cálculado para ter uma idea se a rede teve overfiting
media = resultados.mean()   # Função que cálcula a média da variável 'resultados'
desvio = resultados.std()   # Função que cálcula o desvio padrão da variável 'resultados'


print(resultados)
print(media)
print(desvio)