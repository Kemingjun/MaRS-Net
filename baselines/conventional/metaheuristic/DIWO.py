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
import Util.config







POP_INITIAL_SIZE = 50     
POP_MAX_SIZE = 100         
S_MAX = 100                
S_MIN = 1                 



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




def get_neighbor_solution(solution):
    neighbor_list = [neighbor_insertion, neighbor_swap]
    neighbor_index = random.randint(0,1)
    neighbor_solution = neighbor_list[neighbor_index](solution)
    return neighbor_solution


def DIWO(instance_name, max_iterations=100, time_limit_s=3600, seed=None):
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    instance = read_excel(instance_name + ".xlsx")

    weed_list = []

    weed_hash_list = []

    best_solution = generate_solution_nearest2(instance)
    best_fitness = best_solution.get_fitness()

    weed_list.append(best_solution)

    for _ in range(POP_INITIAL_SIZE - 1):
        init_solution = generate_solution_random(instance)
        weed_list.append(init_solution)

    for solution in weed_list:
        weed_hash_list.append(solution.hash_key)


    count = 0

    while count < max_iterations and time.time() - start_time < time_limit_s:
        count += 1
        fitness_list = [solution.get_fitness() for solution in weed_list]
        min_fitness = min(fitness_list)
        max_fitness = max(fitness_list)

        best_fitness = min_fitness

        seed_list = []

        for weed_solution in weed_list:
            weed_fitness = weed_solution.get_fitness()

            
            try:
                if abs(max_fitness - min_fitness) < 1e-5:
                    seed_num = random.randint(S_MIN, S_MAX)
                else:
                    seed_num = math.floor(S_MAX - (weed_fitness - min_fitness) / (max_fitness - min_fitness) *\
                                          (S_MAX - S_MIN))
            except:
                seed_num = random.randint(S_MIN, S_MAX)

            for seed_index in range(seed_num):
                new_solution = get_neighbor_solution(weed_solution)
                if new_solution.hash_key not in weed_hash_list:
                    
                    seed_list.append(new_solution)
        weed_seed_list = seed_list + weed_list
        weed_seed_list = sorted(weed_seed_list, key=lambda x: x.get_fitness())

        best_solution = weed_seed_list[0]

        weed_list = weed_seed_list[:POP_MAX_SIZE]
        # weed_hash_list = []
        # for solution in weed_list:
        #     weed_hash_list.append(solution.hash_key)

        # print(f"time: {time.time() - start_time} best_fitness: {best_fitness}")

    return best_solution, best_fitness, time.time() - start_time


#
# # # #
# # # #
# if __name__ == "__main__":
#     instance_name = "T20_I2"
#     DIWO(instance_name)
#     pass



