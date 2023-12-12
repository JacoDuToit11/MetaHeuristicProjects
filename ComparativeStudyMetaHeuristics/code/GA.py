# genetic algorithm search for continuous function optimization
from numpy.random import randint
from numpy.random import rand
from scipy.optimize import rosen
import math
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

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

# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		# extract the substring
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]
		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])
		# convert string to integer
		integer = int(chars, 2)
		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
		# store
		decoded.append(value)
	return decoded

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, bounds, run_num, fitness_matrix):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))
    # enumerate generations
    for gen in range(num_iter):
        # decode population
        decoded = [decode(bounds, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(d) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                #print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
        fitness_matrix[run_num][gen] = best_eval
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]

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
    plt.ylabel("Global Best Value")
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
    file_directory = "data/ga/rosenbrock"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    bounds = [[0]*(2) for i in range(num_dim)]
    for i in range(0, num_dim):
        bounds[i][0] = xmin
        bounds[i][1] = xmax

    # perform the genetic algorithm search
    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        best, score = genetic_algorithm(fitness, bounds, run_num, fitness_matrix)
        decoded = decode(bounds, n_bits, best)
        print('f(%s) = %f' % (decoded, score))
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def sphere():
    xmin = -5.12
    xmax = 5.12
    fitness= fitness_sphere
    file_directory = "data/ga/sphere"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    bounds = [[0]*(2) for i in range(num_dim)]
    for i in range(0, num_dim):
        bounds[i][0] = xmin
        bounds[i][1] = xmax

    # perform the genetic algorithm search
    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        best, score = genetic_algorithm(fitness, bounds, run_num, fitness_matrix)
        decoded = decode(bounds, n_bits, best)
        print('f(%s) = %f' % (decoded, score))
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)


def rastigrin():
    xmin = -5.12
    xmax = 5.12
    fitness= fitness_rastrigin
    file_directory = "data/ga/rastigrin"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    bounds = [[0]*(2) for i in range(num_dim)]
    for i in range(0, num_dim):
        bounds[i][0] = xmin
        bounds[i][1] = xmax

    # perform the genetic algorithm search
    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        best, score = genetic_algorithm(fitness, bounds, run_num, fitness_matrix)
        decoded = decode(bounds, n_bits, best)
        print('f(%s) = %f' % (decoded, score))
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def step3():
    xmin = -100
    xmax = 100
    fitness= fitness_step3
    file_directory = "data/ga/step3"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    bounds = [[0]*(2) for i in range(num_dim)]
    for i in range(0, num_dim):
        bounds[i][0] = xmin
        bounds[i][1] = xmax

    # perform the genetic algorithm search
    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        best, score = genetic_algorithm(fitness, bounds, run_num, fitness_matrix)
        decoded = decode(bounds, n_bits, best)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)


def qing():
    xmin = -500
    xmax = 500
    fitness= fitness_qing
    file_directory = "data/ga/qing"
    sourceFile = open(file_directory + '/Results.txt', 'w')

    bounds = [[0]*(2) for i in range(num_dim)]
    for i in range(0, num_dim):
        bounds[i][0] = xmin
        bounds[i][1] = xmax

    # perform the genetic algorithm search
    fitness_matrix = [[0]*(num_iter) for i in range(num_runs)]
    tic = time.perf_counter()
    for run_num in range(0, num_runs):
        best, score = genetic_algorithm(fitness, bounds, run_num, fitness_matrix)
        decoded = decode(bounds, n_bits, best)
        print('f(%s) = %f' % (decoded, score))
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_fitness(fitness_matrix, file_directory, sourceFile)

def main():
    global num_dim, num_particles, num_iter, num_runs, save, n_bits, n_pop, r_cross, r_mut
    num_dim = 30
    num_particles = 30
    num_iter = 2000
    num_runs = 20
    save = True
    # bits per variable
    n_bits = 16
    # define the population size
    n_pop = 30
    # crossover rate
    r_cross = 0.9
    # mutation rate
    r_mut = 1.0 / (float(n_bits) * num_dim)

    #rosenbrock()
    #sphere()
    #rastigrin()
    #step3()
    #qing()

if __name__ == "__main__":
    main()