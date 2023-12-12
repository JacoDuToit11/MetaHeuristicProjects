# python implementation of Grey wolf optimization (GWO)
# minimizing rastrigin and sphere function

from stat import FILE_ATTRIBUTE_DIRECTORY
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math # cos() for Rastrigin
import copy # array-copying convenience
import sys	 # max float
from scipy.optimize import rosen
from sympy import source


#-------fitness functions---------

def fitness_schwefel(x):
    total = 0
    for i in range(num_dim):
        sum = 0
        for j in range (0, i):
            sum += x[j]
        total += sum**2
    return total

# rastrigin function
def fitness_rastrigin(x):
    fitness_value = 0.0
    for i in range(num_dim):
        xi = x[i]
        fitness_value += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return fitness_value

#sphere function
def fitness_sphere(x):
    fitness_value = 0.0
    for i in range(num_dim):
        xi = x[i]
        fitness_value += (xi*xi)
    return fitness_value

def fitness_rosenbrock(x):
    fitness = rosen(x)
    return fitness

def fitness_step3(x):
    fitness = 0
    for i in range(0, num_dim):
        try:
            fitness += math.floor(math.pow(x[i], 2))
        except OverflowError as err:
            fitness = sys.float_info.max
    return fitness

def fitness_qing(x):
    fitness = 0
    for i in range(0, num_dim):
        try:
            fitness += pow((pow(x[i], 2)-(i+1)), 2)
        except OverflowError as err:
            fitness = sys.float_info.max
    return fitness

#-------------------------


# wolf class
class wolf:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random()
        self.position = [0.0 for i in range(dim)]

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.fitness = fitness(self.position) # curr fitness

# grey wolf optimization (GWO)
def gwo(fitness, max_iter, n, dim, minx, maxx, run_num, fitness_matrix):
    rnd = random.Random()

    # create n random wolves
    population = [wolf(fitness, dim, minx, maxx, i) for i in range(n)]

    # On the basis of fitness values of wolves
    # sort the population in asc order
    population = sorted(population, key = lambda temp: temp.fitness)

    # best 3 solutions will be called as
    # alpha, beta and gaama
    alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])

    # main loop of gwo
    Iter = 0
    while Iter < max_iter:

        fitness_matrix[run_num][Iter] = alpha_wolf.fitness

        # linearly decreased from 2 to 0
        a = 2*(1 - Iter/max_iter)

        # updating each population member with the help of best three members
        for i in range(n):
            A1, A2, A3 = a * (2 * rnd.random() - 1), a * (
            2 * rnd.random() - 1), a * (2 * rnd.random() - 1)
            C1, C2, C3 = 2 * rnd.random(), 2*rnd.random(), 2*rnd.random()

            X1 = [0.0 for i in range(dim)]
            X2 = [0.0 for i in range(dim)]
            X3 = [0.0 for i in range(dim)]
            Xnew = [0.0 for i in range(dim)]
            for j in range(dim):
                X1[j] = alpha_wolf.position[j] - A1 * abs(
                C1 * alpha_wolf.position[j] - population[i].position[j])
                X2[j] = beta_wolf.position[j] - A2 * abs(
                C2 * beta_wolf.position[j] - population[i].position[j])
                X3[j] = gamma_wolf.position[j] - A3 * abs(
                C3 * gamma_wolf.position[j] - population[i].position[j])
                Xnew[j]+= X1[j] + X2[j] + X3[j]
            
            for j in range(dim):
                Xnew[j]/=3.0
            
            # fitness calculation of new solution
            fnew = fitness(Xnew)

            # greedy selection
            if fnew < population[i].fitness:
                population[i].position = Xnew
                population[i].fitness = fnew
                
        # On the basis of fitness values of wolves
        # sort the population in asc order
        population = sorted(population, key = lambda temp: temp.fitness)

        # best 3 solutions will be called as
        # alpha, beta and gaama
        alpha_wolf, beta_wolf, gamma_wolf = copy.copy(population[: 3])
        
        Iter+= 1
    # end-while

    # returning the best solution
    return alpha_wolf.position
        
#----------------------------

def plot_fitness(gbest_matrix, file_directory, sourceFile):
    np_arr = np.array(gbest_matrix)
    avg_gbest = np.average(np_arr, axis=0)
    x_values = [0]
    for i in range (1, num_iter):
        x_values.append(i)
    std_gbest = np.std(np_arr, axis=0)
    upper_std_line = np.add(avg_gbest, std_gbest)
    lower_std_line = np.subtract(avg_gbest, std_gbest)

    fig = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Best Position Value")
    plt.plot(x_values, upper_std_line, label="average + standard deviation")
    plt.plot(x_values, lower_std_line, label="average - standard deviation")
    plt.plot(x_values, avg_gbest, label="average")
    plt.legend()
    if save:
        fig.savefig(file_directory+"/fitness.png")
    else: 
        plt.show()
    print("gbest: ", avg_gbest[num_iter-1], "with std dev: ", std_gbest[num_iter-1], file = sourceFile)

def rosenbrock():
    xmin = -30
    xmax = 30
    fitness= fitness_rosenbrock
    file_directory = "data/wolf/rosenbrock"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        gwo(fitness, num_iter, num_particles, num_dim, xmin, xmax, run_num, fitness_matrix)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def sphere():
    xmin = -5.12
    xmax = 5.12
    fitness= fitness_sphere
    file_directory = "data/wolf/sphere"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        gwo(fitness, num_iter, num_particles, num_dim, xmin, xmax, run_num, fitness_matrix)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def rastigrin():
    xmin = -5.12
    xmax = 5.12
    fitness= fitness_rastrigin
    file_directory = "data/wolf/rastigrin"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        gwo(fitness, num_iter, num_particles, num_dim, xmin, xmax, run_num, fitness_matrix)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def step3():
    xmin = -100
    xmax = 100
    fitness= fitness_step3
    file_directory = "data/wolf/step3"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        gwo(fitness, num_iter, num_particles, num_dim, xmin, xmax, run_num, fitness_matrix)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def qing():
    xmin = -500
    xmax = 500
    fitness= fitness_qing
    file_directory = "data/wolf/qing"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        gwo(fitness, num_iter, num_particles, num_dim, xmin, xmax, run_num, fitness_matrix)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def schwefel():
    xmin = -100
    xmax = 100
    fitness= fitness_schwefel
    file_directory = "data/wolf/schwefel"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        gwo(fitness, num_iter, num_particles, num_dim, xmin, xmax, run_num, fitness_matrix)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def main():
    global num_dim, num_particles, num_iter, num_runs, save
    num_dim = 30
    num_particles = 30
    num_iter = 2000
    num_runs = 20
    save = True
    #rosenbrock()
    #sphere()
    #rastigrin()
    #step3()
    #qing()
    schwefel()

if __name__ == "__main__":
    main()



