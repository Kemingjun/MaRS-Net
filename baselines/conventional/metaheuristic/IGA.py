import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Util.generate_init_solution import generate_solution_nearest2
from Util.load_data import *
# import Util.load_data as Uload
# import math
from Util.operators import *
from Util.ALNS_config import *
import Util.load_data as Uload
import time
import Util.config




"""
config
"""


def destruct_construct(current_solution, d_num):
    destroyed_info = destroy_couple_random(current_solution, d_num)
    new_solution = repair_couple_greedy(*destroyed_info, current_solution)
    return new_solution


def local_search(solution, start_t, time_limit_s=3600):
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    task_list = list(range(1, solution.task_num + 1))
    random.shuffle(task_list)
    for task in task_list:  
        if time.time() - start_t > time_limit_s:
            return Solution(solution.instance, sequence_map, path_init_task_map)

        greedy_sequence_map = copy_dict_int_dict(sequence_map)
        greedy_path_init_task_map = copy_dict_int_int(path_init_task_map)
        fitness_min, _, _ = cal_fitness(solution.instance, greedy_sequence_map, greedy_path_init_task_map)

        remove_(sequence_map, path_init_task_map, task)

        is_improved = False

        parent_position_set = get_all_position(sequence_map, path_init_task_map, task, 'parent')
        for parent_position in parent_position_set:
            destroyed_sequence_map_temp = copy_dict_int_dict(sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, task, parent_position, 'parent')
            child_position_set = get_feasible_insert_position(destroyed_sequence_map_temp, path_init_task_map_temp,
                                                              task, 'child')
            for child_position in child_position_set:
                destroyed_sequence_map_temp_temp = copy_dict_int_dict(destroyed_sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp, task, child_position,
                        'child')
                fitness, _, _ = cal_fitness(solution.instance, destroyed_sequence_map_temp_temp,
                                            path_init_task_map_temp_temp)
                if fitness < fitness_min:
                    greedy_sequence_map = destroyed_sequence_map_temp_temp
                    greedy_path_init_task_map = path_init_task_map_temp_temp
                    fitness_min = fitness
                    is_improved = True
                    break
            if is_improved:
                break

        sequence_map = greedy_sequence_map
        path_init_task_map = greedy_path_init_task_map
    new_solution = Solution(solution.instance, sequence_map, path_init_task_map)
    # if new_solution.get_fitness() < solution.get_fitness():
    #     print("success")
    return new_solution


def IGA(instance_name, max_iterations=100, time_limit_s=3600, seed=None):
    if seed is not None:
        random.seed(seed)
    start_t = time.time()
    instance = read_excel(instance_name + ".xlsx")
    task_num = len(instance)
    d_num = math.ceil(task_num * d_num_coefficient)

    solution = generate_solution_nearest2(instance)

    current_fitness = solution.get_fitness()

    best_solution = solution
    best_fitness = current_fitness
    count = 0
    fitness_time_list = [[current_fitness, 0]]


    CONSTANT_T = get_T(instance)

    while count < max_iterations and time.time() - start_t < time_limit_s:
        count += 1

        # _solution = destruct_construct(solution, d_num)
        # new_solution = local_search(_solution, start_t)

        _solution = local_search(solution, start_t, time_limit_s)
        new_solution = destruct_construct(_solution, d_num)


        new_fitness = new_solution.get_fitness()
        if new_fitness < current_fitness:
            solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                fitness_time_list.append([best_fitness, time.time() - start_t])
        elif new_fitness == current_fitness:
            pass
        else:
            p_a = math.exp((current_fitness - new_fitness) / CONSTANT_T)
            if random.random() < p_a:
                solution = new_solution
                current_fitness = new_fitness
    #
    #     print(
    
    # print(f"best fitness:{best_fitness}  best solution:{best_solution}")

    return best_solution, best_fitness, time.time() - start_t

#
# if __name__ == "__main__":
#     instance_name = "T20_I2"
#     # T = get_T(instance)
#     IGA(instance_name)
#     pass


