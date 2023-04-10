# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 14:07:37 2023

@author: mcamboim
"""

from genetic_algorithm import GeneticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

mat = scipy.io.loadmat('C:\\Users\\mcamboim\\Documents\\UNICAMP\\IA707 - Computação Evolutiva\\Exercicios\\Exercício 1\\elshafei_QAP.mat')

gen_alg = GeneticAlgorithm(mat['D'],mat['W'])
possible_solution = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
best_solution = 1e9

for execution in range(10):
    for i in range(300*100):
        np.random.shuffle(possible_solution)
        solution_i = gen_alg.getObjectiveFunction(possible_solution)
        if(solution_i < best_solution):
            best_solution = solution_i
    print(best_solution)