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
import numpy as np
from Util.ALNS_config import *
from Util.operators import *



destructOperatorList = [destroy_couple_random,
                        destroy_couple_worst_cost,
                        destroy_couple_worst_distance,
                        destroy_couple_worst_tardiness,
                        destroy_couple_shaw]
constructOperatorList = [repair_couple_greedy,
                         repair_couple_greedy_urgency_priority,
                         repair_couple_greedy_cost_priority]

d_operator_num = len(destructOperatorList)  
c_operator_num = len(constructOperatorList)  
wDestruct = [1 for _ in range(d_operator_num)]  
wConstruct = [1 for _ in range(c_operator_num)]  


def destruct_construct(current_solution, d_num):
    destruct_index = np.random.choice(np.arange(len(wDestruct)), p=np.array(wDestruct) / sum(wDestruct))
    construct_index = np.random.choice(np.arange(len(wConstruct)), p=np.array(wConstruct) / sum(wConstruct))
    destroyed_info = destructOperatorList[destruct_index](current_solution, d_num)
    new_solution = constructOperatorList[construct_index](*destroyed_info, current_solution)
    return new_solution, destruct_index, construct_index



def ALNS(instance_name, max_iterations=1000, time_limit_s=3600, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    start_t = time.time()
    instance = read_excel(instance_name + ".xlsx")
    totalScoreDestruct = [0 for _ in range(d_operator_num)]  
    totalScoreConstruct = [0 for _ in range(c_operator_num)]
    timesDestruct = [0 for _ in range(d_operator_num)]  
    timesConstruct = [0 for _ in range(c_operator_num)]
    wDestruct = [1 for _ in range(d_operator_num)]  
    wConstruct = [1 for _ in range(c_operator_num)]  

    solution_table = {}  

    task_num = len(instance)
    d_num = math.ceil(task_num * d_num_coefficient)

    solution = generate_solution_nearest2(instance)  #
    # solution = generate_solution_random(instance)

    solution_table[solution.hash_key] = solution

    current_fitness = solution.get_fitness()

    best_solution = solution
    best_fitness = current_fitness
    count = 0



    CONSTANT_T = get_T(instance)

    # print(f"constant T is {CONSTANT_T}")

    count = 0

    while count <= max_iterations and time.time() - start_t < time_limit_s:
        count += 1
        new_solution, destruct_index, construct_index = destruct_construct(solution, d_num)

        is_new = False  
        if new_solution.hash_key not in solution_table.keys():
            solution_table[new_solution.hash_key] = new_solution
            is_new = True


        new_fitness = new_solution.get_fitness()

        scoreDestruct = 0
        scoreConstruct = 0

        if new_fitness < current_fitness:
            solution = new_solution
            current_fitness = new_fitness
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                scoreDestruct = sigma_1
                scoreConstruct = sigma_1
            else:
                if is_new:
                    scoreDestruct = sigma_2
                    scoreConstruct = sigma_2
        elif new_fitness == current_fitness:
            if is_new:
                scoreDestruct = sigma_2
                scoreConstruct = sigma_2
        else:
            p_a = math.exp((current_fitness - new_fitness) / CONSTANT_T)
            # print(f"delta:{(current_fitness - new_fitness)}  probability: {p_a}")
            if random.random() < p_a:
                solution = new_solution
                current_fitness = new_fitness
                if is_new:
                    scoreDestruct = sigma_3
                    scoreConstruct = sigma_3

        timesDestruct[destruct_index] += 1
        timesConstruct[construct_index] += 1
        totalScoreDestruct[destruct_index] += scoreDestruct
        totalScoreConstruct[construct_index] += scoreConstruct

        if count % l_s == 0:
            for i in range(d_operator_num):
                if timesDestruct[i] != 0:
                    dTime = timesDestruct[i]
                else:
                    dTime = 1

                wDestruct[i] = wDestruct[i] * (1 - rho) + rho * totalScoreDestruct[i] / dTime
                totalScoreDestruct[i] = 0
                timesDestruct[i] = 0
            for i in range(c_operator_num):
                if timesConstruct[i] != 0:
                    cTime = timesConstruct[i]
                else:
                    cTime = 1
                wConstruct[i] = wConstruct[i] * (1 - rho) + rho * totalScoreConstruct[i] / cTime
                totalScoreConstruct[i] = 0
                timesConstruct[i] = 0

        print(
            f"Iteration {count}, best fitness: {round(best_fitness, 4)}, "
            f"current fitness: {round(current_fitness, 4)}, new fitness: {round(new_fitness, 4)}, "
            f"d_index: {destruct_index}, c_index: {construct_index}, time: {round(time.time() - start_t, 4)}"
        )

    return best_solution, best_fitness, time.time() - start_t


if __name__ == "__main__":
    instance_name = "Synthetic_Dataset/size_10_uniform/T10_I1_uniform"
    # instance_name = "T10_I1"
    # instance = read_excel(instance_name + ".xlsx")
    solution, _, _ = ALNS(instance_name)
    print(solution.get_path_map())
    print(f"total_distance:{solution.distance}")
    # print(f"fixed_distance:{solution.fixed_distance}")
    # print(f"unfixed_distance:{solution.unfixed_distance}")
    print(f"tardiness:{solution.tardiness}")
    pass
#
