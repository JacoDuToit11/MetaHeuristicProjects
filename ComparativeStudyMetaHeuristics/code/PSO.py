import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen
import sys

test = False
if test:   
    num_dim = 30
    num_part = 30
    num_iter = 3000
    num_runs = 5
    num_iter_measured = num_iter
else:
    num_dim = 30
    num_part = 30
    num_iter = 2000
    num_runs = 20
    num_iter_measured = num_iter

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

def rosenbrock():
    global w, c1, c2, com, file_directory, save, fitnessFunc, x_min, x_max
    x_min = [-30] * num_dim
    x_max = [30] * num_dim
    save = True
    fitnessFunc = fitness_rosenbrock
    for i in range(1, 5):
        com = i
        if com == 1:
            w = 0.7
            c1 = 1.4
            c2 = 1.4
        elif com == 2:
            w = 0.9
            c1 = 0.7
            c2 = 0.7
        elif com == 3:
            w = 0.8
            c1 = 1.2
            c2 = 1.2
        elif com == 4:
            w = 0.6
            c1 = 1.8
            c2 = 1.8
        file_directory = "data/pso/rosenbrock/C" + str(com)
        run()

def rastigrin():
    global w, c1, c2, com, file_directory, save, fitnessFunc, x_min, x_max
    x_min = [-5.12] * num_dim
    x_max = [5.12] * num_dim
    save = True
    fitnessFunc = fitness_rastrigin
    for i in range(1, 5):
        com = i
        if com == 1:
            w = 0.7
            c1 = 1.4
            c2 = 1.4
        elif com == 2:
            w = 0.9
            c1 = 0.7
            c2 = 0.7
        elif com == 3:
            w = 0.8
            c1 = 1.2
            c2 = 1.2
        elif com == 4:
            w = 0.6
            c1 = 1.8
            c2 = 1.8
        file_directory = "data/pso/rastigrin/C" + str(com)
        run()

def sphere():
    global w, c1, c2, com, file_directory, save, fitnessFunc, x_min, x_max
    x_min = [-5.12] * num_dim
    x_max = [5.12] * num_dim
    save = True
    fitnessFunc = fitness_sphere
    for i in range(1, 5):
        com = i
        if com == 1:
            w = 0.7
            c1 = 1.4
            c2 = 1.4
        elif com == 2:
            w = 0.9
            c1 = 0.7
            c2 = 0.7
        elif com == 3:
            w = 0.8
            c1 = 1.2
            c2 = 1.2
        elif com == 4:
            w = 0.6
            c1 = 1.8
            c2 = 1.8
        file_directory = "data/pso/sphere/C" + str(com)
        run()

def step3():
    global w, c1, c2, com, file_directory, save, fitnessFunc, x_min, x_max
    x_min = [-100] * num_dim
    x_max = [100] * num_dim
    save = True
    fitnessFunc = fitness_step3
    for i in range(1, 5):
        com = i
        if com == 1:
            w = 0.7
            c1 = 1.4
            c2 = 1.4
        elif com == 2:
            w = 0.9
            c1 = 0.7
            c2 = 0.7
        elif com == 3:
            w = 0.8
            c1 = 1.2
            c2 = 1.2
        elif com == 4:
            w = 0.6
            c1 = 1.8
            c2 = 1.8
        file_directory = "data/pso/step3/C" + str(com)
        run()

def qing():
    global w, c1, c2, com, file_directory, save, fitnessFunc, x_min, x_max
    x_min = [-500] * num_dim
    x_max = [500] * num_dim
    save = True
    fitnessFunc = fitness_qing
    for i in range(1, 5):
        com = i
        if com == 1:
            w = 0.7
            c1 = 1.4
            c2 = 1.4
        elif com == 2:
            w = 0.9
            c1 = 0.7
            c2 = 0.7
        elif com == 3:
            w = 0.8
            c1 = 1.2
            c2 = 1.2
        elif com == 4:
            w = 0.6
            c1 = 1.8
            c2 = 1.8
        file_directory = "data/pso/qing/C" + str(com)
        run()

def schwefel():
    global w, c1, c2, com, file_directory, save, fitnessFunc, x_min, x_max
    x_min = [-100] * num_dim
    x_max = [100] * num_dim
    save = True
    fitnessFunc = fitness_schwefel
    for i in range(1, 5):
        com = i
        if com == 1:
            w = 0.7
            c1 = 1.4
            c2 = 1.4
        elif com == 2:
            w = 0.9
            c1 = 0.7
            c2 = 0.7
        elif com == 3:
            w = 0.8
            c1 = 1.2
            c2 = 1.2
        elif com == 4:
            w = 0.6
            c1 = 1.8
            c2 = 1.8
        file_directory = "data/pso/schwefel/C" + str(com)
        run()

def run():
    gbest_matrix = [[0]*(num_iter_measured+1) for i in range(num_runs)]
    global sourceFile
    sourceFile = open(file_directory + '/Results.txt', 'w')

    tic = time.perf_counter()
    for i in range(1, num_runs + 1):
        PSO(i, gbest_matrix)
    toc = time.perf_counter()
    print("time taken: ", toc - tic , file=sourceFile)
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")
    plot_gbest(gbest_matrix)
    sourceFile.close()

class Particle:
    def __init__(self):
        self.pos = []
        self.velocity = [0]*num_dim

        for i in range(0, num_dim):
            init_pos = random.uniform(x_min[i], x_max[i])
            self.pos.append(init_pos)

        self.pbest_pos = self.pos
        self.pbest_val = fitnessFunc(self.pos)
        
    def move(self):
        for i in range(0, num_dim):
            self.pos[i] = self.pos[i] + self.velocity[i]

    def update_velocity(self, gbest_pos):
        for i in range (0, num_dim):
            r1 = random.random()
            r2 = random.random()
            cognitive = c1*r1*(self.pbest_pos[i] - self.pos[i])
            social = c2*r2*(gbest_pos[i] - self.pos[i])

            # No Clamping Strategy
            self.velocity[i] = w*self.velocity[i] + cognitive +  social

    def evaluate_local(self):
        curr_pos_val = fitnessFunc(self.pos)

        if curr_pos_val < self.pbest_val:
            self.pbest_val = curr_pos_val
            self.pbest_pos = self.pos

def PSO(run_num, gbest_matrix):
        gbest_pos = []
        gbest_val = float('inf')

        swarm = []
        for i in range(0, num_part):
            swarm.append(Particle())

        measure = 0
        j = 1
        while j < num_iter + 1:
            for i in range (0, num_part):
                swarm[i].evaluate_local()
                if  swarm[i].pbest_val < gbest_val:
                    gbest_val = swarm[i].pbest_val
                    gbest_pos = list(swarm[i].pos)

            if j == 1:
                gbest_matrix[run_num-1][measure] = gbest_val
                measure += 1

            for i in range (0, num_part):
                swarm[i].update_velocity(gbest_pos)
                swarm[i].move()

            gbest_matrix[run_num-1][measure] = gbest_val
            measure += 1
            j += 1

def plot_gbest(gbest_matrix):
    np_arr = np.array(gbest_matrix)
    avg_gbest = np.average(np_arr, axis=0)
    x_values = [1]
    for i in range (1, num_iter_measured + 1):
        x_values.append(num_iter/num_iter_measured*i)
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
        fig.savefig(file_directory+"/gbest.png", bbox_inches='tight')
    else: 
        plt.show()
    print("gbest: ", avg_gbest[num_iter_measured], "with std dev: ", std_gbest[num_iter_measured], file = sourceFile)

# Algorithm parameters
def main():
    #rosenbrock()
    #sphere()
    #rastigrin()
    #step3()
    #qing()
    schwefel()

if __name__ == "__main__":
    main()



