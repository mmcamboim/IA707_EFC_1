# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:45:26 2023

@author: mcamboim
"""
from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
mat = scipy.io.loadmat('C:\\Users\\mcamboim\\Documents\\UNICAMP\\IA707 - Computação Evolutiva\\Exercicios\\Exercício 1\\elshafei_QAP.mat')

plt.close('all')
plt.rcParams['axes.linewidth'] = 2.0
plt.rc('axes', axisbelow=True)
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

pop_size = 250
crossover_probability = 0.8
mutation_probability = 0.2
generations = 500
elitism_percentage = 0.95    
executions = 1

gen_alg = GeneticAlgorithm(mat['D'],mat['W'])
best_objective = np.zeros((generations,executions))
mean_objective = np.zeros((generations,executions))
for execution_idx in range(executions):
    print(f'\nExecução {execution_idx+1}/{executions}')
    gen_alg.runGa(pop_size=pop_size,crossover_probability=crossover_probability,mutation_probability=mutation_probability,elitism_percentage=elitism_percentage,generations=generations)
    best_objective[:,execution_idx] = gen_alg.best_objective_through_generations
    mean_objective[:,execution_idx] = gen_alg.mean_objective_through_generations

# Figures
best_objective_idx = np.argmax(best_objective[-1,:])
plt.figure(figsize=(12,6),dpi=150)
plt.plot(best_objective[:,best_objective_idx],lw=2,c='b')
plt.plot([17_212_548] * generations,ls='--',lw=2,c='r')
plt.ylabel('Função Objetivo []')
plt.xlabel('# de Gerações [n]')
plt.legend(['Evolução da Função Objetivo','Valor de Referência'])
plt.xlim([0,generations-1])
plt.grid(True,ls='dotted')
plt.tight_layout()

x_mean = np.array([np.mean(best_objective[generation,:]) for generation in range(generations)])
x_std = np.array([np.std(best_objective[generation,:],ddof=1) for generation in range(generations)])
plt.figure(figsize=(12,6),dpi=150)
plt.plot(x_mean,lw=2,c='b')
plt.plot([17_212_548] * generations,ls='--',lw=2,c='r')
plt.fill_between(np.arange(generations),x_mean-1*x_std,x_mean+x_std,alpha=0.5)
plt.ylabel('Função Objetivo []')
plt.xlabel('# de Gerações [n]')
plt.legend(['Valor Médio da Evolução da Função Objetivo','Valor de Referência','Desvio Padrão da Evolução da Função Objetivo'])
plt.xlim([0,generations-1])
plt.grid(True,ls='dotted')


plt.figure(figsize=(12,6),dpi=150)
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.plot(best_objective[:,i],lw=2,c='b')
    plt.plot(mean_objective[:,i],lw=2,c='g')
    plt.plot([17_212_548] * generations,ls='--',lw=2,c='r')
    plt.xlim([0,generations-1])
    #
    #plt.xlabel('# de Gerações [n]')
    if(i==0 or i == 5):
        plt.ylabel('Função Objetivo []')
    if(i>=5):
        plt.xlabel('# de Gerações [n]')
    if(i==0):
        plt.legend(['Melhor','Média'])
plt.tight_layout()



