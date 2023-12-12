import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen
from PyBenchFCN import SingleObjectiveProblem as SOP
import sys

test = False
if test:   
    num_dim = 3
    num_part = 30
    num_iter = 5
    num_runs = 5
    num_iter_measured = 5
else:
    num_dim = 30
    num_part = 30
    num_iter = 5000
    num_runs = 20
    num_iter_measured = 5000

# Boundary constraints
x_min = [-30] * num_dim
x_max = [30] * num_dim

def fitnessFunc(x):
    return Rosenbrock(x)

# -1 <= x_i <= 1
def Exponential(x):
    sum = 0
    for i in range(0, num_dim):
        sum += pow(x[i], 2)
    fitness = -math.exp(-0.5*sum)
    return fitness

# -30 <= x_i <= 30
def Rosenbrock(x):
    fitness = rosen(x)
    return fitness

# -500 <= x_i <= 500
def Qing(x):
    fitness = 0
    for i in range(0, num_dim):
        try:
            fitness += pow((pow(x[i], 2)-(i+1)), 2)
        except OverflowError as err:
            fitness = sys.float_info.max
    return fitness

# -100 <= x_i <= 100
def Step3(x):
    fitness = 0
    for i in range(0, num_dim):
        try:
            fitness += math.floor(math.pow(x[i], 2))
        except OverflowError as err:
            fitness = sys.float_info.max
    return fitness

def CosMix(x):
    fitness = 0
    sum1 = 0
    sum2 = 0
    for i in range(0, num_dim):
        try:
            sum1 += math.cos(5*math.pi*x[i])
        except OverflowError as err:
            sum1 = sys.float_info.max
        try:
            sum2 += math.pow(x[i], 2)
        except OverflowError as err:
            sum2 = sys.float_info.max
    try:
        fitness += -0.1*sum1 - sum2
    except OverflowError as err:
        fitness = sys.float_info.max
    return fitness

# Algorithm parameters
def main():
    global apply_clamp, w, c1, c2, file_directory, save, k
    for i in range(0, 4):
        for j in range(1, 5):
            # Clamping strategy
            kcom = i
            if kcom != 0:
                apply_clamp = True
            else:
                apply_clamp = False
            if kcom == 1:
                k = 0.1
            elif kcom == 2:
                k = 0.3
            elif kcom == 3:
                k = 0.5

            # Control parameters
            com = j
            if com == 1:
                w = 1
                c1 = 2
                c2 = 2
            elif com == 2:
                w = 0.9
                c1 = 2
                c2 = 2
            elif com == 3:
                w = 0.7
                c1 = 1.4
                c2 = 1.4
            elif com == 4:
                w = 0.9
                c1 = 0.7
                c2 = 0.7
            # Graph saving stuff
            save = True
            if kcom == 0:
                file_directory = "final/Rosenbrock/C" + str(com)
            else:
                file_directory = "final/Rosenbrock/K" + str(kcom) + "/C" + str(com)

            run()

def run():

    gbest_matrix = [[0]*(num_iter_measured+1) for i in range(num_runs)]
    diversity_matrix = [[0]*(num_iter_measured+1) for i in range(num_runs)]
    outside_matrix = [[0]*(num_iter_measured+1) for i in range(num_runs)]
    velocity_matrix = [[0]*(num_iter_measured+1) for i in range(num_runs)]
    
    global sourceFile
    sourceFile = open(file_directory + '/Results.txt', 'w')
    tic = time.perf_counter()

    for i in range(1, num_runs + 1):
        PSO(i, gbest_matrix, diversity_matrix, outside_matrix, velocity_matrix)

    toc = time.perf_counter()
    print(f"Executed {num_runs} runs in {toc - tic:0.4f} seconds")

    plot_diversity(diversity_matrix)
    plot_gbest(gbest_matrix)
    plot_outside(outside_matrix)
    plot_velocity(velocity_matrix)

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

            # Clamping Strategy
            if apply_clamp:
                temp_velocity = w*self.velocity[i] + cognitive +  social
                if temp_velocity > max_velocity[i]:
                    self.velocity[i] = max_velocity[i]
                elif temp_velocity < -max_velocity[i]:
                    self.velocity[i] = -max_velocity[i]
                else: 
                    self.velocity[i] = temp_velocity

    def evaluate_local(self):
        curr_pos_val = fitnessFunc(self.pos)

        if curr_pos_val < self.pbest_val:
            self.pbest_val = curr_pos_val
            self.pbest_pos = self.pos

def PSO(run_num, gbest_matrix, diversity_matrix, outside_matrix, velocity_matrix):
        gbest_pos = []
        gbest_val = float('inf')

        # Setting max_velocity for clamping
        if apply_clamp:
            global max_velocity
            max_velocity = []
            max_velocity = obtain_max_velocity()

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
                diversity_matrix[run_num-1][measure] = obtain_diversity(swarm)
                outside_matrix[run_num-1][measure] = obtain_perc_left_part(swarm)
                velocity_matrix[run_num-1][measure] = obtain_avg_velocity_mag(swarm)
                measure += 1

            for i in range (0, num_part):
                swarm[i].update_velocity(gbest_pos)
                swarm[i].move()

            gbest_matrix[run_num-1][measure] = gbest_val
            diversity_matrix[run_num-1][measure] = obtain_diversity(swarm)
            outside_matrix[run_num-1][measure] = obtain_perc_left_part(swarm)
            velocity_matrix[run_num-1][measure] = obtain_avg_velocity_mag(swarm)
            measure += 1
            j += 1

def obtain_max_velocity():
        max_velocity = []
        for i in range(0, num_dim):
            max_velocity.append(k*(x_max[i]-x_min[i]))
        return max_velocity

def obtain_diversity(swarm):
    np_swarm = np.array(swarm[0].pos)
    for i in range(1, num_part):
        temp = np.array(swarm[i].pos)
        np_swarm = np.vstack((np_swarm, temp))
    average_part = np.average(np_swarm, axis=0)

    total_dist = 0
    for i in range(0, num_part):
        temp = np.array(swarm[i].pos)
        total_dist += np.linalg.norm(average_part - temp)
    return total_dist/num_part

def obtain_perc_left_part(swarm):
    counter = 0
    for i in range(0, num_part):
        for j in range(0, num_dim):
            if swarm[i].pos[j] > x_max[j] or swarm[i].pos[j] < x_min[j]:
                counter += 1
                break
    return (counter/num_part)*100

def obtain_avg_velocity_mag(swarm):
    total_velocity_mag = 0
    for i in range(0, num_part):
        temp = np.array(swarm[i].velocity)
        total_velocity_mag += np.linalg.norm(temp)
    return total_velocity_mag/num_part

def plot_diversity(diversity_matrix):
    np_arr = np.array(diversity_matrix)
    avg_div = np.average(np_arr, axis=0)
    x_values = [1]
    for i in range (1, num_iter_measured + 1):
        x_values.append(num_iter/num_iter_measured*i)

    std_div = np.std(np_arr, axis=0)
    upper_std_line = np.add(avg_div, std_div)
    lower_std_line = np.subtract(avg_div, std_div)
    
    fig = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Swarm Diversity")
    plt.plot(x_values, upper_std_line, label="average + standard deviation")
    plt.plot(x_values, lower_std_line, label="average - standard deviation")
    plt.plot(x_values, avg_div, label="average")
    plt.legend()
    if save:
        fig.savefig(file_directory+"/div.png")
    else: 
        plt.show()

    print("At iteration 5000:", file = sourceFile)
    print("diversity: ", avg_div[num_iter_measured], "with std dev: ", std_div[num_iter_measured], file = sourceFile)

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
        fig.savefig(file_directory+"/gbest.png")
    else: 
        plt.show()
    print("gbest: ", avg_gbest[num_iter_measured], "with std dev: ", std_gbest[num_iter_measured], file = sourceFile)

def plot_outside(outside_matrix):
    np_arr = np.array(outside_matrix)
    avg_outside = np.average(np_arr, axis=0)
    std_outside = np.std(np_arr, axis=0)
    upper_std_line = np.add(avg_outside, std_outside)
    lower_std_line = np.subtract(avg_outside, std_outside)
    x_values = [1]
    for i in range (1, num_iter_measured + 1):
        x_values.append(num_iter/num_iter_measured*i)

    fig = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Percentage of Particles Outside of Search Space")
    plt.plot(x_values, upper_std_line, label="average + standard deviation")
    plt.plot(x_values, lower_std_line, label="average - standard deviation")
    plt.plot(x_values, avg_outside, label="average")
    plt.legend()
    if save:
        fig.savefig(file_directory+"/outside.png")
    else: 
        plt.show()
    print("percentage of particles outside: ", avg_outside[num_iter_measured], "with std dev: ", std_outside[num_iter_measured], file = sourceFile)

def plot_velocity(velocity_matrix):
    np_arr = np.array(velocity_matrix)
    avg_velocity = np.average(np_arr, axis=0)
    std_vel = np.std(np_arr, axis=0)
    upper_std_line = np.add(avg_velocity, std_vel)
    lower_std_line = np.subtract(avg_velocity, std_vel)
    x_values = [1]
    for i in range (1, num_iter_measured + 1):
        x_values.append(num_iter/num_iter_measured*i)
    
    fig = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Average Velocity Magnitude")
    plt.plot(x_values, upper_std_line, label="average + standard deviation")
    plt.plot(x_values, lower_std_line, label="average - standard deviation")
    plt.plot(x_values, avg_velocity, label="average")
    plt.legend()
    if save:
        fig.savefig(file_directory+"/vel.png")
    else: 
        plt.show()
    print("average velocity magnitude: ", avg_velocity[num_iter_measured], "with std dev: ", std_vel[num_iter_measured], file = sourceFile)

if __name__ == "__main__":
    main()



