import os
import random
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Util.generate_init_solution import generate_solution_nearest2, generate_solution_random
from Util.load_data import *
from Util.operators import *
from Util.ALNS_config import *
import Util.load_data as Uload
import copy
import Util.config
import numpy as np
class Bee:
    def __init__(self, id, type):
        self.id = id
        self.type = type
        

    def setType(self, type):
        self.type = type

    def getType(self):
        return self.type

    def getId(self):
        return self.id

class Nectar:
    def __init__(self, solution, fitness=None):
        self.solution = solution  
        self.search_num = 0  
        
        self.fitness = fitness if fitness is not None else solution.get_fitness()
        self.bee = None

    def setBee(self, bee):
        self.bee = bee

    def add_search_num(self):
        self.search_num += 1

    def getBee(self):
        return self.bee



def neighbor_insertion(solution):
    task = random.randint(1, solution.task_num)
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    remove_(sequence_map, path_init_task_map, task)

    parent_positions = get_all_position(sequence_map, path_init_task_map, task, 'parent')

    parent_position = random.sample(parent_positions, 1)[0]
    insert_(sequence_map, path_init_task_map, task, parent_position, 'parent')

    child_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, 'child')
    child_position = random.sample(child_positions, 1)[0]
    insert_(sequence_map, path_init_task_map, task, child_position, 'child')
    return Solution(solution.instance, sequence_map, path_init_task_map)


def neighbor_swap(solution):
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    task_list = list(range(1, solution.task_num + 1))
    task_1 = random.sample(task_list, 1)[0]
    task_list.remove(task_1)
    task_2 = random.sample(task_list, 1)[0]
    
    if sequence_map[task_1]['parent_pre_task'] == 0:
        path_init_task_map[sequence_map[task_1]['parent']] = task_2

    if sequence_map[task_1]['child_pre_task'] == 0:
        path_init_task_map[sequence_map[task_1]['child']] = task_2

    if sequence_map[task_2]['parent_pre_task'] == 0:
        path_init_task_map[sequence_map[task_2]['parent']] = task_1

    if sequence_map[task_2]['child_pre_task'] == 0:
        path_init_task_map[sequence_map[task_2]['child']] = task_1

    
    task_1_parent = sequence_map[task_1]['parent']
    task_1_parent_pre_task = sequence_map[task_1]['parent_pre_task']
    task_1_parent_next_task = sequence_map[task_1]['parent_next_task']
    task_1_child = sequence_map[task_1]['child']
    task_1_child_pre_task = sequence_map[task_1]['child_pre_task']
    task_1_child_next_task = sequence_map[task_1]['child_next_task']

    if sequence_map[task_1]['parent_pre_task'] != 0 and sequence_map[task_1]['parent_pre_task'] != task_2:
        sequence_map[sequence_map[task_1]['parent_pre_task']]['parent_next_task'] = task_2

    if sequence_map[task_1]['parent_next_task'] != 0 and sequence_map[task_1]['parent_next_task'] != task_2:
        sequence_map[sequence_map[task_1]['parent_next_task']]['parent_pre_task'] = task_2

    if sequence_map[task_1]['child_pre_task'] != 0 and sequence_map[task_1]['child_pre_task'] != task_2:
        sequence_map[sequence_map[task_1]['child_pre_task']]['child_next_task'] = task_2

    if sequence_map[task_1]['child_next_task'] != 0 and sequence_map[task_1]['child_next_task'] != task_2:
        sequence_map[sequence_map[task_1]['child_next_task']]['child_pre_task'] = task_2

    ''''''

    if sequence_map[task_2]['parent_pre_task'] != 0 and sequence_map[task_2]['parent_pre_task'] != task_1:
        sequence_map[sequence_map[task_2]['parent_pre_task']]['parent_next_task'] = task_1

    if sequence_map[task_2]['parent_next_task'] != 0 and sequence_map[task_2]['parent_next_task'] != task_1:
        sequence_map[sequence_map[task_2]['parent_next_task']]['parent_pre_task'] = task_1

    if sequence_map[task_2]['child_pre_task'] != 0 and sequence_map[task_2]['child_pre_task'] != task_1:
        sequence_map[sequence_map[task_2]['child_pre_task']]['child_next_task'] = task_1

    if sequence_map[task_2]['child_next_task'] != 0 and sequence_map[task_2]['child_next_task'] != task_1:
        sequence_map[sequence_map[task_2]['child_next_task']]['child_pre_task'] = task_1

    sequence_map[task_1]['parent'] = sequence_map[task_2]['parent']
    if sequence_map[task_2]['parent_pre_task'] == task_1:
        sequence_map[task_1]['parent_pre_task'] = task_2
    else:
        sequence_map[task_1]['parent_pre_task'] = sequence_map[task_2]['parent_pre_task']

    if sequence_map[task_2]['parent_next_task'] == task_1:
        sequence_map[task_1]['parent_next_task'] = task_2
    else:
        sequence_map[task_1]['parent_next_task'] = sequence_map[task_2]['parent_next_task']

    sequence_map[task_1]['child'] = sequence_map[task_2]['child']

    if sequence_map[task_2]['child_pre_task'] == task_1:
        sequence_map[task_1]['child_pre_task'] = task_2
    else:
        sequence_map[task_1]['child_pre_task'] = sequence_map[task_2]['child_pre_task']

    if sequence_map[task_2]['child_next_task'] == task_1:
        sequence_map[task_1]['child_next_task'] = task_2
    else:
        sequence_map[task_1]['child_next_task'] = sequence_map[task_2]['child_next_task']


    sequence_map[task_2]['parent'] = task_1_parent

    if task_1_parent_pre_task == task_2:
        sequence_map[task_2]['parent_pre_task'] = task_1
    else:
        sequence_map[task_2]['parent_pre_task'] = task_1_parent_pre_task

    if task_1_parent_next_task == task_2:
        sequence_map[task_2]['parent_next_task'] = task_1
    else:
        sequence_map[task_2]['parent_next_task'] = task_1_parent_next_task

    sequence_map[task_2]['child'] = task_1_child

    if task_1_child_pre_task == task_2:
        sequence_map[task_2]['child_pre_task'] = task_1
    else:
        sequence_map[task_2]['child_pre_task'] = task_1_child_pre_task

    if task_1_child_next_task == task_2:
        sequence_map[task_2]['child_next_task'] = task_1
    else:
        sequence_map[task_2]['child_next_task'] = task_1_child_next_task

    return Solution(solution.instance, sequence_map, path_init_task_map)

def neighbor_route_swap(solution):

    path_map = solution.get_path_map()

    chain_flag = random.randint(0, 1)

    if chain_flag == 0:
        M_list = Util.config.M_parent_list
    else:
        M_list = Util.config.M_child_list

    swap_path_index_list = random.sample(M_list, 2)
    path_1_index = swap_path_index_list[0]
    path_2_index = swap_path_index_list[1]

    path_map_temp = copy_dict_int_list(path_map)
    path_1 = path_map_temp[path_1_index]
    path_map_temp[path_1_index] = path_map_temp[path_2_index]
    path_map_temp[path_2_index] = path_1
    sequence_map, path_init_task_map = path_map2sequence_map(path_map_temp)
    return Solution(solution.instance, sequence_map, path_init_task_map)

def get_neighbor_solution(solution):
    neighbor_list = [neighbor_insertion, neighbor_swap]
    neighbor_index = random.randint(0,1)
    neighbor_solution = neighbor_list[neighbor_index](solution)
    return neighbor_solution


def get_index_roulette(nectar_list, num):
    cost = np.array([nc.fitness for nc in nectar_list])
    for j in range(len(cost)):
        cost[j] = - cost[j]
    fitness_list = (cost - np.min(cost)) + 1e-3
    idx = np.random.choice(np.arange(len(nectar_list)), size=num, replace=True,
                           p=(fitness_list) / (fitness_list.sum()))
    return idx


def DABC(instance_name, max_iterations=100, time_limit_s=3600, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    start_time = time.time()
    instance = read_excel(instance_name + ".xlsx")
    nectar_list = []  
    onlooker_list = []  
    scout_list = []  
    task_num = len(instance)
    solution = generate_solution_nearest2(instance)
    employed_bee = Bee(0, 1)
    nectar = Nectar(solution)  
    nectar.setBee(employed_bee)  
    nectar_list.append(nectar)
    for i in range(1, employed_size):
        employed_bee = Bee(i, 1)
        solution = generate_solution_nearest2(instance, random.random())  
        nectar = Nectar(solution)  
        nectar.setBee(employed_bee)  
        nectar_list.append(nectar)

    for i in range(employed_size, onLooker_size + employed_size):
        onlooker_bee = Bee(i, 2)
        onlooker_list.append(onlooker_bee)

    best_fitness = min([item.fitness for item in nectar_list])
    best_index = [item.fitness for item in nectar_list].index(best_fitness)
    best_nectar = nectar_list[best_index]
    best_solution = best_nectar.solution



    count = 0

    # duration = task_num ** 2 * (Util.config.M_child + Util.config.M_parent) * C / 1000

    while count < max_iterations and time.time() - start_time < time_limit_s:
        count += 1
        
        new_nectar_list = []
        for nc in nectar_list:
            if time.time() - start_time > time_limit_s:
                break
            nc_solution = nc.solution
            nc_fitness = nc.fitness
            new_solution = get_neighbor_solution(nc_solution)
            new_fitness = new_solution.get_fitness()
            if new_fitness < nc_fitness:
                
                employed_bee = nc.bee
                new_nectar = Nectar(new_solution, new_fitness)
                new_nectar.setBee(employed_bee)
                new_nectar_list.append(new_nectar)
                
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
                    best_nectar = new_nectar
            else:
                nc.add_search_num()
                
                if nc.search_num > LIMIT and nc != best_nectar:
                    
                    bee = nc.bee
                    bee.setType(3)  
                    scout_list.append(bee)
                else:
                    new_nectar_list.append(nc)
            # else:
            #     new_nectar_list.append(nc)
        nectar_list = copy.deepcopy(new_nectar_list)  



        for _ in range(r):
            if len(nectar_list) == 0:
                
                continue

            
            onlooker_bee_num = len(onlooker_list)
            onlooker_nectar_index = get_index_roulette(nectar_list, onlooker_bee_num)  
            new_onlooker_list = []

            for i, onlooker_bee in enumerate(onlooker_list):
                if time.time() - start_time > time_limit_s:
                    break
                
                onlooker_nectar = nectar_list[onlooker_nectar_index[i]]  
                nectar_solution = onlooker_nectar.solution
                nectar_fitness = onlooker_nectar.fitness
                
                new_solution = get_neighbor_solution(nectar_solution)
                new_fitness = new_solution.get_fitness()
                if new_fitness < nectar_fitness:
                    
                    new_nectar = Nectar(new_solution, new_fitness)
                    onlooker_bee.setType(1)  
                    new_nectar.setBee(onlooker_bee)
                    nectar_list[onlooker_nectar_index[i]] = new_nectar
                    
                    origin_nectar_bee = onlooker_nectar.bee
                    origin_nectar_bee.setType(2)  
                    new_onlooker_list.append(origin_nectar_bee)
                    
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness
                        best_nectar = new_nectar
                else:
                    
                    onlooker_nectar.add_search_num()
                    
                    
                    #     onlooker_nectar

                    new_onlooker_list.append(onlooker_bee)

            onlooker_list = copy.deepcopy(new_onlooker_list)

        
        for scout_bee in scout_list:
            # print("scout_bee")
            new_solution = generate_solution_nearest2(instance, random.random())  
            new_nectar = Nectar(new_solution)
            new_fitness = new_nectar.fitness
            scout_bee.setType(1)  
            new_nectar.setBee(scout_bee)
            nectar_list.append(new_nectar)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                best_nectar = new_nectar
        scout_list = []  
        print(f"Generation {count}, best fitness: {round(best_fitness, 3)}, time: {time.time() - start_time}")
    return best_solution, best_fitness, time.time() - start_time







POP_SIZE = 100  
employed_size = int(POP_SIZE / 2)  
onLooker_size = int(POP_SIZE / 2)  
LIMIT = 1000  
r = 20  

#
if __name__ == "__main__":
    instance_name = "uniform_100_per_scale_20260413/size_40_uniform/T40_I8_uniform"
    # instance = read_excel(instance_name + ".xlsx")
    DABC(instance_name)
