# differential evolution search of the two-dimensional sphere objective function
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
from scipy.optimize import rosen
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math

def fitness_schwefel(x):
    total = 0
    for i in range(num_dim):
        sum = 0
        for j in range (0, i):
            sum += x[j]
        total += sum**2
    return total

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

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])

# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def differential_evolution(pop_size, bounds, iter, F, cr, fitness, run_num, fitness_matrix):
    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions
    obj_all = [fitness(ind) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # run iterations of the algorithmxfx
    for i in range(iter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = fitness(pop[j])
            # compute objective function value for trial vector
            obj_trial = fitness(trial)
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        fitness_matrix[run_num][i] = best_obj
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
    return [best_vector, best_obj]

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
    fig.savefig(file_directory+"/fitness.png")
    print("gbest: ", avg_gbest[num_iter-1], "with std dev: ", std_gbest[num_iter-1], file = sourceFile)

def rosenbrock():
    xmin = -30
    xmax = 30
    fitness= fitness_rosenbrock

    bounds =  asarray([(xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)])

    for i in range(1, 5):    
        com = i
        if com == 1:
            F = 0.5
            cr = 0.7
        elif com == 2:
            F = 0.9
            cr = 0.1
        elif com == 3:
            F = 0.9
            cr = 0.9
        elif com == 4:
            F = 0.5
            cr = 0.3

        file_directory = "data/de/rosenbrock/C" + str(com)
        sourceFile = open(file_directory + '/Results.txt', 'w')
        fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
        tic = time.perf_counter()
        for run_num in range(0, num_runs):
            differential_evolution(pop_size, bounds, num_iter, F, cr, fitness, run_num, fitness_matrix)
        toc = time.perf_counter()
        print("time taken: ", toc - tic , file=sourceFile)
        print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
        plot_fitness(fitness_matrix, file_directory, sourceFile)

def sphere():
    xmin = -5.12
    xmax = 5.12
    fitness= fitness_sphere

    bounds =  asarray([(xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)])

    for i in range(1, 5):    
        com = i
        if com == 1:
            F = 0.5
            cr = 0.7
        elif com == 2:
            F = 0.9
            cr = 0.1
        elif com == 3:
            F = 0.9
            cr = 0.9
        elif com == 4:
            F = 0.5
            cr = 0.3

        file_directory = "data/de/sphere/C" + str(com)
        sourceFile = open(file_directory + '/Results.txt', 'w')
        fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
        tic = time.perf_counter()
        for run_num in range(0, num_runs):
            differential_evolution(pop_size, bounds, num_iter, F, cr, fitness, run_num, fitness_matrix)
        toc = time.perf_counter()
        print("time taken: ", toc - tic , file=sourceFile)
        print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
        plot_fitness(fitness_matrix, file_directory, sourceFile)


def rastigrin():
    xmin = -5.12
    xmax = 5.12
    fitness= fitness_rastrigin

    bounds =  asarray([(xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)])

    for i in range(1, 5):    
        com = i
        if com == 1:
            F = 0.5
            cr = 0.7
        elif com == 2:
            F = 0.9
            cr = 0.1
        elif com == 3:
            F = 0.9
            cr = 0.9
        elif com == 4:
            F = 0.5
            cr = 0.3

        file_directory = "data/de/rastigrin/C" + str(com)
        sourceFile = open(file_directory + '/Results.txt', 'w')
        fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
        tic = time.perf_counter()
        for run_num in range(0, num_runs):
            differential_evolution(pop_size, bounds, num_iter, F, cr, fitness, run_num, fitness_matrix)
        toc = time.perf_counter()
        print("time taken: ", toc - tic , file=sourceFile)
        print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
        plot_fitness(fitness_matrix, file_directory, sourceFile)

def step3():
    xmin = -100
    xmax = 100
    fitness= fitness_step3

    bounds =  asarray([(xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)])

    for i in range(1, 5):    
        com = i
        if com == 1:
            F = 0.5
            cr = 0.7
        elif com == 2:
            F = 0.9
            cr = 0.1
        elif com == 3:
            F = 0.9
            cr = 0.9
        elif com == 4:
            F = 0.5
            cr = 0.3

        file_directory = "data/de/step3/C" + str(com)
        sourceFile = open(file_directory + '/Results.txt', 'w')
        fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
        tic = time.perf_counter()
        for run_num in range(0, num_runs):
            differential_evolution(pop_size, bounds, num_iter, F, cr, fitness, run_num, fitness_matrix)
        toc = time.perf_counter()
        print("time taken: ", toc - tic , file=sourceFile)
        print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
        plot_fitness(fitness_matrix, file_directory, sourceFile)


def qing():
    xmin = -500
    xmax = 500
    fitness= fitness_qing

    bounds =  asarray([(xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)])

    for i in range(1, 5):    
        com = i
        if com == 1:
            F = 0.5
            cr = 0.7
        elif com == 2:
            F = 0.9
            cr = 0.1
        elif com == 3:
            F = 0.9
            cr = 0.9
        elif com == 4:
            F = 0.5
            cr = 0.3

        file_directory = "data/de/qing/C" + str(com)
        sourceFile = open(file_directory + '/Results.txt', 'w')
        fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
        tic = time.perf_counter()
        for run_num in range(0, num_runs):
            differential_evolution(pop_size, bounds, num_iter, F, cr, fitness, run_num, fitness_matrix)
        toc = time.perf_counter()
        print("time taken: ", toc - tic , file=sourceFile)
        print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
        plot_fitness(fitness_matrix, file_directory, sourceFile)

def schwefel():
    xmin = -100
    xmax = 100
    fitness= fitness_schwefel

    bounds =  asarray([(xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax) , (xmin, xmax), (xmin, xmax)
    , (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax), (xmin, xmax)])

    for i in range(1, 5):    
        com = i
        if com == 1:
            F = 0.5
            cr = 0.7
        elif com == 2:
            F = 0.9
            cr = 0.1
        elif com == 3:
            F = 0.9
            cr = 0.9
        elif com == 4:
            F = 0.5
            cr = 0.3

        file_directory = "data/de/schwefel/C" + str(com)
        sourceFile = open(file_directory + '/Results.txt', 'w')
        fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
        tic = time.perf_counter()
        for run_num in range(0, num_runs):
            differential_evolution(pop_size, bounds, num_iter, F, cr, fitness, run_num, fitness_matrix)
        toc = time.perf_counter()
        print("time taken: ", toc - tic , file=sourceFile)
        print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
        plot_fitness(fitness_matrix, file_directory, sourceFile)

def main():
    global num_dim, pop_size, num_iter, num_runs
    num_dim = 30
    pop_size = 30
    num_iter = 2000
    num_runs = 20

    #rosenbrock()
    #sphere()
    #rastigrin()
    #step3()
    #qing()
    schwefel()

if __name__ == "__main__":
    main()