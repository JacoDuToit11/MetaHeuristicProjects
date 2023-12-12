from matplotlib import pyplot as plt
import numpy as np
import random
import math
from pymoo.factory import get_problem
from pymoo.factory import get_performance_indicator
from pymoo.visualization.scatter import Scatter
from regex import R
from pymoo.problems.many.wfg import WFG1, WFG3

# ZDT1
def ZDT1_f1(x):
    if not inBounds(x):
        return float('inf')
    return x[0]

def ZDT1_f2(x):
    if not inBounds(x):
        return float('inf')
    sum = 0
    for i in range(1, len(x)):
        sum += x[i]
    g = 1 + (9/(num_dim-1))*sum
    h = 1 - math.sqrt(x[0]/g)
    return g*h

# ZDT2
def ZDT2_f1(x):
    return x[0]

def ZDT2_f2(x):
    sum = 0
    for i in range(1, len(x)):
        sum += x[i]
    g = 1 + (9/(num_dim-1))*sum
    h = 1 - math.pow(x[0]/g, 2)
    return g*h

# ZDT3
def ZDT3_f1(x):
    return x[0]

def ZDT3_f2(x):
    sum = 0
    for i in range(1, len(x)):
        sum += x[i]
    g = 1 + (9/(num_dim-1))*sum
    h = 1 - math.sqrt(x[0]/g) - (x[0]/g)*math.sin(10*math.pi*x[0])
    return g*h

def WFG1_f1(x):
    wfg = WFG1(n_var = num_dim, n_obj=3)
    solutions = wfg.evaluate(x)
    return solutions[0][0]

def WFG1_f2(x):
    wfg = WFG1(n_var = num_dim, n_obj=3)
    solutions = wfg.evaluate(x)
    return solutions[0][1]

def WFG1_f3(x):
    wfg = WFG1(n_var = num_dim, n_obj=3)
    solutions = wfg.evaluate(x)
    return solutions[0][2]

def WFG3_f1(x):
    wfg = WFG3(n_var = num_dim, n_obj=3)
    solutions = wfg.evaluate(x)
    return solutions[0][0]

def WFG3_f2(x):
    wfg = WFG3(n_var = num_dim, n_obj=3)
    solutions = wfg.evaluate(x)
    return solutions[0][1]

def WFG3_f3(x):
    wfg = WFG3(n_var = num_dim, n_obj=3)
    solutions = wfg.evaluate(x)
    return solutions[0][2]

test = False
if test:   
    num_dim = 30
    num_part1 = 33
    num_part2 = 17
    num_iter = 1000
    num_runs = 3
    MAX_ARCHIVE_SIZE = 50
    save = True
else:
    num_dim = 30
    num_part1 = 29
    num_part2 = 10
    num_part3 = 11
    num_iter = 2000
    num_runs = 20
    MAX_ARCHIVE_SIZE = 50
    save = True
    num_parts = [num_part1, num_part2, num_part3]

num_obj = 3
pf = get_problem("wfg3", num_dim, num_obj)
x_min = pf.xl
x_max = pf.xu
#x_min = [0] * num_dim
#x_max = [1] * num_dim

ref_point = np.array([10, 10, 10])

def fitnessFunc1(x):
    if not inBounds(x):
        return float('inf')
    return WFG3_f1(x)

def fitnessFunc2(x):
    if not inBounds(x):
        return float('inf')
    return WFG3_f2(x)

def fitnessFunc3(x):
    if not inBounds(x):
        return float('inf')
    return WFG3_f3(x)

def main():
    global def_c1, def_c2, def_c3, def_w
    def_c1 = 1.65
    def_c2 = 1.75
    def_c3 = 0.75
    def_w = 0.525

    print("start MGPSO")
    MGPSO()
    print("end MGPSO")
    print("start MMMGPSO")
    MMMGPSO()
    print("end MMMGPSO")
    
def MGPSO():
    file_directory = "Data/WFG3/MGPSO/"
    pf = get_problem("wfg3", num_dim, num_obj).pareto_front()
    #minimize
    igd = get_performance_indicator("igd", pf)
    #maximize
    hv = get_performance_indicator("hv", ref_point=ref_point)

    igd_mat = [[0]*(num_iter) for i in range(num_runs)]
    hp_mat = [[0]*(num_iter) for i in range(num_runs)]

    for num_run in range(0, num_runs):
        archive = []
        swarms = []
        for i in range(0, num_obj):
            swarms.append(PSO(i+1, num_parts[i]))
        for iter in range(0, num_iter):
            for pso in swarms:
                for i in range (0, pso.num_part):
                    pso.swarm[i].evaluate_local()
                    for j in range(0, len(pso.swarm)):
                        if pso.swarm[i].pbest_val < pso.swarm[j].nbest_val:
                            pso.swarm[j].nbest_pos = list(pso.swarm[i].pbest_pos)
                            pso.swarm[j].nbest_val = pso.swarm[i].pbest_val
                    archive = update_archive(pso.swarm[i].pos, archive)
            for pso in swarms:
                for i in range (0, pso.num_part):
                    a = getArchiveItem(archive)
                    pso.swarm[i].update_velocity(a.pos)
                    pso.swarm[i].move()
            A = getArchiveValues(archive)
            igd_mat[num_run][iter] = igd.do(A)
            hp_mat[num_run][iter] = hv.do(A)
        A = getArchiveValues(archive)
        file_name = file_directory + "ParetoFront" + str(num_run + 1)
        Scatter(legend=True).add(pf, label="True POF").add(A, label="MGPSO POF").save(file_name)
    drawFigures(file_directory, igd_mat, hp_mat)

def MMMGPSO():
    file_directory = "Data/WFG3/MMMGPSO/"
    pf = get_problem("wfg3", num_dim, num_obj).pareto_front()
    #minimize
    igd = get_performance_indicator("igd", pf)
    #maximize
    hv = get_performance_indicator("hv", ref_point=ref_point)

    igd_mat = [[0]*(num_iter) for i in range(num_runs)]
    hp_mat = [[0]*(num_iter) for i in range(num_runs)]

    z = 0.2
    radius = z*eucDistance(x_min, x_max)

    file_directory += "radius" + str(z) + "/"

    for num_run in range(0, num_runs):
        archive = []
        swarms = []
        for i in range(0, num_obj):
            swarms.append(PSO(i+1, num_parts[i]))
        for iter in range(0, num_iter):
            for pso in swarms:
                for i in range (0, pso.num_part):
                    pso.swarm[i].evaluate_local()
                sortParticles(pso.swarm)
                speciesSeed = []
                for i in range(0, pso.num_part):
                    found = False
                    for j in range(0, len(speciesSeed)):
                        if eucDistance(speciesSeed[j].pbest_pos, pso.swarm[i].pbest_pos) <= radius:
                            found = True
                            if speciesSeed[j].pbest_val < pso.swarm[i].nbest_val:
                                pso.swarm[i].nbest_pos = list(speciesSeed[j].pbest_pos)
                                pso.swarm[i].nbest_val = speciesSeed[j].pbest_val
                            break
                    if not found:
                        speciesSeed.append(pso.swarm[i])
                    archive = update_archive(pso.swarm[i].pos, archive)
            for pso in swarms:
                for i in range(0, pso.num_part):
                    a = getArchiveItem(archive)
                    pso.swarm[i].update_velocity(a.pos)
                    pso.swarm[i].move()
            A = getArchiveValues(archive)
            igd_mat[num_run][iter] = igd.do(A)
            hp_mat[num_run][iter] = hv.do(A)
        A = getArchiveValues(archive)
        file_name = file_directory + "ParetoFront" + str(num_run + 1)
        Scatter(legend=True).add(pf, label="True POF").add(A, label="MMMGPSO POF").save(file_name)
    drawFigures(file_directory, igd_mat, hp_mat)

def drawFigures(file_directory, igd_mat, hp_mat):
    sourceFile = open(file_directory + '/Results.txt', 'w')
    drawIGD(file_directory, igd_mat, sourceFile)
    drawHV(file_directory, hp_mat, sourceFile)

def drawIGD(file_directory, igd_mat, sourceFile):
    igd_mat = np.array(igd_mat)
    avg_igd = np.average(igd_mat, axis=0)
    x_values = [0]
    for i in range (0, num_iter - 1):
        x_values.append(i)

    std_div = np.std(igd_mat, axis=0)
    upper_std_line = np.add(avg_igd, std_div)
    lower_std_line = np.subtract(avg_igd, std_div)
    
    fig = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("IGD")
    plt.plot(x_values, upper_std_line, label = "average + standard deviation")
    plt.plot(x_values, lower_std_line, label = "average - standard deviation")
    plt.plot(x_values, avg_igd, label = "average")
    plt.legend()
    if save:
        fig.savefig(file_directory + "igd.png")
    else: 
        plt.show()
    print("At iteration ", num_iter, ": ", file = sourceFile)
    print("igd: ", avg_igd[num_iter - 1], "with std dev: ", std_div[num_iter - 1], file = sourceFile)

def drawHV(file_directory, hv_mat, sourceFile):
    hv_mat = np.array(hv_mat)
    avg_hv = np.average(hv_mat, axis=0)
    x_values = [0]
    for i in range (0, num_iter - 1):
        x_values.append(i)

    std_div = np.std(hv_mat, axis=0)
    upper_std_line = np.add(avg_hv, std_div)
    lower_std_line = np.subtract(avg_hv, std_div)
    
    fig = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("HV")
    plt.plot(x_values, upper_std_line, label = "average + standard deviation")
    plt.plot(x_values, lower_std_line, label = "average - standard deviation")
    plt.plot(x_values, avg_hv, label = "average")
    plt.legend()
    if save:
        fig.savefig(file_directory + "hv.png")
    else: 
        plt.show()
    print("hv: ", avg_hv[num_iter - 1], "with std dev: ", std_div[num_iter - 1], file = sourceFile)

class archiveItem:
    def  __init__(self, pos):
        self.pos = pos
        self.value1 = fitnessFunc1(pos)
        self.value2 = fitnessFunc2(pos)
        self.value3 = fitnessFunc3(pos)
        self.dist = 0

def update_archive(x, archive):
    #Only allow solutions in bounds
    if (not inBounds(x)):
        return archive

    x_archiveItem = archiveItem(x)

    for i in range(0, len(archive)):
        if x_archiveItem.value1 == archive[i].value1 and x_archiveItem.value2 == archive[i].value2:
            return archive
        if Dominated(x_archiveItem, archive[i]):
            return archive

    archive.append(x_archiveItem)
    # Remove entries that are dominated by the newly added item
    archive = [item for item in archive if not Dominated(item, x_archiveItem)]

    if len(archive) > MAX_ARCHIVE_SIZE:
        removeArchiveItem(archive)
    return archive

def getArchiveItem(archive):
    index1 = random.randint(0, len(archive)-1)
    index2 = random.randint(0, len(archive)-1)
    index3 = random.randint(0, len(archive)-1)
    mini_archive = [archive[index1], archive[index2], archive[index3]]
    mini_archive = crowdingDistance(mini_archive)
    max_index = 0
    for i in range(1, len(mini_archive)):
        if (mini_archive[i].dist > mini_archive[max_index].dist):
            max_index = i
    return mini_archive[max_index]

def sortDist(archive):
    for i in range(1, len(archive)):
            temp = archive[i]
            j = i - 1
            while j >= 0 and temp.dist > archive[j].dist:
                archive[j+1] = archive[j]
                j -= 1
            archive[j+1] = temp
    return archive

#returns true if item1 is dominated by item2, else false
def Dominated(item1, item2):
    if item1.value3 == None:
        if item2.value1 <= item1.value1 and item2.value2 <= item1.value2:
            if item2.value1 < item1.value1 or item2.value2 < item1.value2:
                return True   
        return False
    else:
        if item2.value1 <= item1.value1 and item2.value2 <= item1.value2 and item2.value3 <= item1.value3:
            if item2.value1 < item1.value1 or item2.value2 < item1.value2 or item2.value3 < item1.value3:
                return True   
        return False

def getArchiveValues(archive):
    values = []
    if archive[0].value3 == None:
        for i in range(0, len(archive)):
            item_values = [archive[i].value1, archive[i].value2]
            values.append(item_values)
        return np.array(values)
    else:
        for i in range(0, len(archive)):
            item_values = [archive[i].value1, archive[i].value2, archive[i].value3]
            values.append(item_values)
        return np.array(values)

def removeArchiveItem(archive):
    archive = crowdingDistance(archive)
    min_index = 0
    for i in range(1, len(archive)):
        if (archive[i].dist < archive[min_index].dist):
            min_index = i
    del archive[min_index]

def crowdingDistance(archive):
    n = len(archive)
    for i in range(0, n):
        archive[i].dist = 0
    
    for i in range(0, num_obj):
        archive = sort(archive, i + 1)
        for j in range(1, n-1):
            if (i == 0):
                archive[j].dist += archive[j+1].value1 - archive[j-1].value1
            elif (i == 1):
                archive[j].dist += archive[j+1].value2 - archive[j-1].value2
            elif (i == 2):
                archive[j].dist += archive[j+1].value3 - archive[j-1].value3
        archive[0].dist += archive[1].value1 + archive[1].value2 + archive[1].value3
        archive[n-1].dist += archive[n-2].value1 + archive[n-2].value2 + archive[n-2].value3
    return archive
    
def sort(archive, obj_func_index):
    if (obj_func_index == 1):
        for i in range(1, len(archive)):
            temp = archive[i]
            j = i - 1
            while j >= 0 and temp.value1 < archive[j].value1:
                archive[j+1] = archive[j]
                j -= 1
            archive[j+1] = temp
    if (obj_func_index == 2):
        for i in range(1, len(archive)):
            temp = archive[i]
            j = i - 1
            while j >= 0 and temp.value2 < archive[j].value2:
                archive[j+1] = archive[j]
                j -= 1
            archive[j+1] = temp
    if (obj_func_index == 3):
        for i in range(1, len(archive)):
            temp = archive[i]
            j = i - 1
            while j >= 0 and temp.value3 < archive[j].value3:
                archive[j+1] = archive[j]
                j -= 1
            archive[j+1] = temp 
    return archive

class Particle():
    def __init__(self, func_index):
        self.func_index = func_index
        self.pos = []
        self.velocity = [0]*num_dim
        self.lambdat = random.uniform(0, 1)

        for i in range(0, num_dim):
            init_pos = random.uniform(x_min[i], x_max[i])
            self.pos.append(init_pos)

        self.nbest_pos = list(self.pos)
        self.pbest_pos = list(self.pos)
        if func_index == 1:
            self.pbest_val = fitnessFunc1(self.pos)
            self.nbest_val = fitnessFunc1(self.pos)
        elif func_index == 2:
            self.nbest_val = fitnessFunc2(self.pos)
            self.pbest_val = fitnessFunc2(self.pos)
        elif func_index == 3:
            self.nbest_val = fitnessFunc3(self.pos)
            self.pbest_val = fitnessFunc3(self.pos)
    
    def move(self):
        for i in range(0, num_dim):
            self.pos[i] = self.pos[i] + self.velocity[i]

    def update_velocity(self, a):
        c1, c2, c3, w = parameters(self.lambdat)

        for i in range (0, num_dim):
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            cognitive_comp = c1*r1*(self.pbest_pos[i] - self.pos[i])
            social_comp = self.lambdat*c2*r2*(self.nbest_pos[i] - self.pos[i])
            archive_comp = (1 - self.lambdat)*c3*r3*(a[i] - self.pos[i])
            self.velocity[i] = w*self.velocity[i] + cognitive_comp +  social_comp + archive_comp

    def evaluate_local(self):
        if self.func_index == 1:
            curr_pos_val = fitnessFunc1(self.pos)
        elif self.func_index == 2:
            curr_pos_val = fitnessFunc2(self.pos)
        elif self.func_index == 3:
            curr_pos_val = fitnessFunc3(self.pos)
        if curr_pos_val < self.pbest_val:
            if (inBounds(self.pos)):
                self.pbest_val = curr_pos_val
                self.pbest_pos = list(self.pos)

def inBounds(x):
    for i in range(0, num_dim):
        if math.isnan(x[i]):
            return False
        if (x[i] < x_min[i] or x[i] > x_max[i]):
            return False
    return True

def parameters(lambdat):
    for i in range(0, 10):
        c1, c2, c3, w = sampleParameters()
        left = c1 + lambdat*c2 + (1-lambdat)*c3
        num1 = 4*(1-w**2)
        num2 = (c1**2 + (lambdat**2)*(c2**2) + ((1 - lambdat)**2)*(c3**2))*(1+w)
        denom2 = 3*(c1 + lambdat*c2 + (1 - lambdat)*c3)**2
        denom1 = 1 - w + num2/denom2
        right = num1/denom1
        if left < right:
            return c1, c2, c3, w
    return def_c1, def_c2, def_c3, def_w

def sampleParameters():
    c1 = random.uniform(0,2)
    c2 = random.uniform(0,2)
    c3 = random.uniform(0,2)
    w = random.uniform(0, 1)
    return c1, c2, c3, w

class PSO():
    def __init__(self, func_index, num_part):
        self.func_index = func_index
        self.num_part = num_part

        self.swarm = []
        for i in range(0, num_part):
            self.swarm.append(Particle(func_index))


def MMPSO():
    z = 0.2
    radius = z*eucDistance(x_min, x_max)
    func_index = 1
    num_part = 30
    pso = PSO(func_index, num_part)
    for iter in range(0, num_iter):
        for i in range (0, pso.num_part):
            pso.swarm[i].evaluate_local_new()
        sortParticles(pso.swarm)
        speciesSeed = []
        for i in range(0, len(pso.swarm)):
            found = False
            for j in range(0, len(speciesSeed)):
                if eucDistance(speciesSeed[j].pbest_pos, pso.swarm[i].pbest_pos) <= radius:
                    found = True
                    if speciesSeed[j].pbest_val < pso.swarm[i].nbest_val:
                        pso.swarm[i].nbest_pos = list(speciesSeed[j].pbest_pos)
                        pso.swarm[i].nbest_val = speciesSeed[j].pbest_val
                    break
            if not found:
                speciesSeed.append(pso.swarm[i])
        for i in range(0, len(pso.swarm)):
            pso.swarm[i].update_velocity_new(pso.swarm[i].nbest_pos)
            pso.swarm[i].move()

def eucDistance(a, b):
    dist = np.linalg.norm(np.array(a) - np.array(b))
    return dist

def sortParticles(swarm):
    for i in range(1, len(swarm)):
        temp = swarm[i]
        j = i - 1
        while j >= 0 and temp.pbest_val < swarm[j].pbest_val:
            swarm[j+1] = swarm[j]
            j -= 1
        swarm[j+1] = temp 
        

if __name__ == "__main__":
    main()