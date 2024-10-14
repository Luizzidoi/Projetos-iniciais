""""
Redes Neurais - Boltzmann Machines
Aprendizagem não supervisionada
RBM - Restricted Boltzmann Machine
Sistemas de recomendação
Recomendação de filmes

"""


from rbm import RBM
import numpy as np

rbm = RBM(num_visible=6, num_hidden=2)


""" Criação de uma base de dados com 6 usuários """
base = np.array([[1,1,1,0,0,0], [1,0,1,0,0,0],
                 [1,1,1,0,0,0], [0,0,1,1,1,1],
                 [0,0,1,1,0,1], [0,0,1,1,0,1]])


""" Lista de filmes """
filmes = ["A bruxa", "Invocação do mal", "O chamado",
          "Se beber não case", "Gente grande", "American Pie"]


""" Treinamento da rede neural """
rbm.train(base, max_epochs=5000)
# Printa os pesos que a rede encontrou. Primeira linha e primeira coluna são unidades de Bias
pesos = rbm.weights


""" Criação de um novo usuário para a indicação"""
usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])
# Função que diz qual neuronio foi ativado
neuronio1 = rbm.run_visible(usuario1)
neuronio2 = rbm.run_visible(usuario2)


"""" Recomendação dos filmes aos novos usuários """
camada_escondida1 = neuronio1
recomendacao1 = rbm.run_hidden(camada_escondida1)
# len = 6 filmes
print('Recomendações para o usuário1:')
for i in range(len(usuario1[0])):
    if usuario1[0, i] == 0 and recomendacao1[0, i] == 1:
        print(filmes[i])

print('################################')
camada_escondida2 = neuronio2
recomendacao2 = rbm.run_hidden(camada_escondida2)
print('Recomendações para o usuário2:')
for i in range(len(usuario2[0])):
    if usuario2[0, i] == 0 and recomendacao2[0, i] == 1:
        print(filmes[i])


print("\nFim")