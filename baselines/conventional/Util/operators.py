import random
from Util.util import *
import Util.config as config
# from Util.config import *
import math
import Util.ALNS_config as ALNS_config
from Util.solution import Solution


def destroy_couple_random(solution, d_num):
    
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_sequence_map = solution.get_sequence_map()  
    destroyed_task_list = random.sample([task for task in range(1, solution.task_num + 1)], d_num)

    for destroyed_task in destroyed_task_list:
        remove_(destroyed_sequence_map, path_init_task_map, destroyed_task)

    return destroyed_sequence_map, path_init_task_map, destroyed_task_list

def destroy_couple_worst_cost(solution, d_num):
    
    task_cost_list = []  
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_task_list = []
    task_info_map = solution.info_map
    task_list = list(sequence_map.keys())
    for task in task_list:
        cost = task_info_map[task]['distance'] * config.weight + task_info_map[task]['tardiness'] * (1 - config.weight)
        task_cost_list.append([task, cost])
    task_cost_list = sorted(task_cost_list, key=lambda x: x[1], reverse=True)
    for i in range(d_num):
        destroyed_task_list.append(task_cost_list[i][0])

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_couple_worst_distance(solution, d_num):
    
    task_distance_list = []  
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_task_list = []
    task_list = list(sequence_map.keys())
    task_info_map = solution.info_map
    if task_info_map is None:
        solution.get_fitness()
        task_info_map = solution.info_map

    for task in task_list:
        distance = task_info_map[task]['distance']
        task_distance_list.append([task, distance])

    task_distance_list = sorted(task_distance_list, key=lambda x: x[1], reverse=True)
    for i in range(d_num):
        destroyed_task_list.append(task_distance_list[i][0])

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_couple_worst_tardiness(solution, d_num):
    
    task_tardiness_list = []  
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_task_list = []
    task_list = list(sequence_map.keys())
    task_info_map = solution.info_map

    for task in task_list:
        distance = task_info_map[task]['tardiness']
        task_tardiness_list.append([task, distance])

    task_tardiness_list = sorted(task_tardiness_list, key=lambda x: x[1], reverse=True)
    for i in range(d_num):
        destroyed_task_list.append(task_tardiness_list[i][0])

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_couple_shaw(solution, d_num):
    instance = solution.instance
    destroyed_task_list = []
    path_init_task_map = solution.get_path_init_task_map()
    sequence_map = solution.get_sequence_map()
    task_list = list(sequence_map.keys())
    reference_task = random.choice(task_list)
    reference_source_position = [instance[reference_task - 1][1], instance[reference_task - 1][2]]
    reference_destination_position = [instance[reference_task - 1][3], instance[reference_task - 1][4]]
    task_similarity_list = []  
    for task in task_list:
        task_source_position = [instance[task - 1][1], instance[task - 1][2]]
        task_destination_position = [instance[task - 1][3], instance[task - 1][4]]
        s_d = cal_distance(reference_source_position, task_source_position) + cal_distance(reference_destination_position, task_destination_position)
        s_c = cal_distance(reference_destination_position, task_destination_position)
        # s_d = (config.distance_cache[tuple([tuple(reference_source_position), tuple(task_source_position)])] +
        #        config.distance_cache[tuple([tuple(reference_destination_position), tuple(task_destination_position)])])
        # s_c = math.fabs(instance[task - 1][5] - instance[reference_task - 1][5])
        similarity = ALNS_config.shaw_weight * s_d + (1 - ALNS_config.shaw_weight) * s_c
        task_similarity_list.append([task, similarity])

    task_similarity_list = sorted(task_similarity_list, key=lambda x: x[1], reverse=False)
    for i in range(d_num):
        destroyed_task_list.append(task_similarity_list[i][0])

    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list


def destroy_couple_worst_misalignment(solution, d_num):
    
    task_misalignment_list = []  
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_task_list = []
    task_list = list(sequence_map.keys())
    task_info_map = solution.info_map


    for task in task_list:
        misalignment = task_info_map[task]['misalignment']
        task_misalignment_list.append([task, misalignment])

    task_misalignment_list = sorted(task_misalignment_list, key=lambda x: x[1], reverse=True)
    for i in range(d_num):
        destroyed_task_list.append(task_misalignment_list[i][0])
    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)
    return sequence_map, path_init_task_map, destroyed_task_list


def repair_couple_greedy(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    
    instance = current_solution.instance
    random.shuffle(destroyed_task_list)
    for destroyed_task in destroyed_task_list:
        greedy_destroyed_sequence_map = None
        greedy_path_init_task_map = None
        fitness_min = 1e6

        parent_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, 'parent')
        for parent_position in parent_position_set:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, parent_position, 'parent')
            child_position_set = get_feasible_insert_position(destroyed_sequence_map_temp, path_init_task_map_temp,
                                                              destroyed_task, 'child')
            for child_position in child_position_set:
                destroyed_sequence_map_temp_temp = copy_dict_int_dict(destroyed_sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp, destroyed_task, child_position,
                        'child')
                fitness, _, _ = cal_fitness(instance, destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp)
                if fitness < fitness_min:
                    greedy_destroyed_sequence_map = destroyed_sequence_map_temp_temp
                    greedy_path_init_task_map = path_init_task_map_temp_temp
                    fitness_min = fitness
        if greedy_destroyed_sequence_map == None:
            print("!!!!!!!!!bug!!!!!!!!! only the original position is feasible")

        destroyed_sequence_map = greedy_destroyed_sequence_map
        path_init_task_map = greedy_path_init_task_map
    new_solution = Solution(instance, destroyed_sequence_map, path_init_task_map)
    return new_solution



def repair_couple_greedy_cost_priority(destroyed_sequence_map, path_init_task_map, destroyed_task_list,
                                       current_solution):
    
    current_info_map = current_solution.info_map
    instance = current_solution.instance
    task_cost_list = []
    for task in destroyed_task_list:
        task_cost_list.append([task, current_info_map[task]["cost"]])
    task_cost_list = sorted(task_cost_list, key=lambda x: x[1], reverse=True)
    for task_cost in task_cost_list:
        destroyed_task = task_cost[0]
        greedy_destroyed_sequence_map = None
        greedy_path_init_task_map = None
        fitness_min = 1e6

        parent_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, 'parent')
        for parent_position in parent_position_set:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, parent_position, 'parent')
            child_position_set = get_feasible_insert_position(destroyed_sequence_map_temp, path_init_task_map_temp,
                                                              destroyed_task, 'child')
            for child_position in child_position_set:

                destroyed_sequence_map_temp_temp = copy_dict_int_dict(destroyed_sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp, destroyed_task, child_position,
                        'child')
                fitness, _, _ = cal_fitness(instance, destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp)
                if fitness < fitness_min:
                    greedy_destroyed_sequence_map = destroyed_sequence_map_temp_temp
                    greedy_path_init_task_map = path_init_task_map_temp_temp
                    fitness_min = fitness
        if greedy_destroyed_sequence_map == None:
            print("!!!!!!!!!bug!!!!!!!!! only the original position is feasible")
        destroyed_sequence_map = greedy_destroyed_sequence_map
        path_init_task_map = greedy_path_init_task_map
    new_solution = Solution(instance, destroyed_sequence_map, path_init_task_map)
    return new_solution



def repair_couple_greedy_urgency_priority(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    
    instance = current_solution.instance
    destroyed_task_l_time = []
    for task in destroyed_task_list:
        destroyed_task_l_time.append([task, instance[task - 1][5]])
    destroyed_task_l_time = sorted(destroyed_task_l_time, key=lambda x: x[1], reverse=False)
    for task_l_time in destroyed_task_l_time:
        destroyed_task = task_l_time[0]
        greedy_destroyed_sequence_map = None
        greedy_path_init_task_map = None
        fitness_min = 1e6

        parent_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, 'parent')
        for parent_position in parent_position_set:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, parent_position, 'parent')
            child_position_set = get_feasible_insert_position(destroyed_sequence_map_temp, path_init_task_map_temp,
                                                              destroyed_task, 'child')
            for child_position in child_position_set:

                destroyed_sequence_map_temp_temp = copy_dict_int_dict(destroyed_sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp, destroyed_task, child_position,
                        'child')
                fitness, _, _ = cal_fitness(instance, destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp)
                if fitness < fitness_min:
                    greedy_destroyed_sequence_map = destroyed_sequence_map_temp_temp
                    greedy_path_init_task_map = path_init_task_map_temp_temp
                    fitness_min = fitness
        if greedy_destroyed_sequence_map == None:
            print("!!!!!!!!!bug!!!!!!!!! only the original position is feasible")
        destroyed_sequence_map = greedy_destroyed_sequence_map
        path_init_task_map = greedy_path_init_task_map
    new_solution = Solution(instance, destroyed_sequence_map, path_init_task_map)
    return new_solution


def repair_couple_random(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    instance = current_solution.instance
    for destroyed_task in destroyed_task_list:
        
        insert_chain = random.sample(['parent', 'child'], 2)  
        
        all_position = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, insert_chain[0])
        position_chain = random.sample(all_position, 1)[0]

        insert_(destroyed_sequence_map, path_init_task_map, destroyed_task, position_chain, insert_chain[0])
        
        feasible_insert_position_set = get_feasible_insert_position(destroyed_sequence_map, path_init_task_map,
                                                                    destroyed_task, insert_chain[1])

        position_opposite_chain = random.sample(feasible_insert_position_set, 1)[0]  
        insert_(destroyed_sequence_map, path_init_task_map, destroyed_task, position_opposite_chain, insert_chain[1])
    new_solution = Solution(instance, destroyed_sequence_map, path_init_task_map)
    return new_solution


def repair_couple_second_greedy(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    
    instance = current_solution.instance
    random.shuffle(destroyed_task_list)
    for destroyed_task in destroyed_task_list:
        greedy_destroyed_sequence_map = None
        greedy_path_init_task_map = None
        second_destroyed_sequence_map = None
        second_path_init_task_map = None
        fitness_min = 1e6
        fitness_second_min = 1e7


        parent_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, 'parent')
        for parent_position in parent_position_set:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, parent_position, 'parent')
            child_position_set = get_feasible_insert_position(destroyed_sequence_map_temp, path_init_task_map_temp,
                                                              destroyed_task, 'child')
            for child_position in child_position_set:
                destroyed_sequence_map_temp_temp = copy_dict_int_dict(destroyed_sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp, destroyed_task, child_position,
                        'child')
                fitness, _, _ = cal_fitness(instance, destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp)
                if fitness < fitness_min:
                    fitness_second_min = fitness_min  
                    fitness_min = fitness  
                    second_destroyed_sequence_map = greedy_destroyed_sequence_map
                    second_path_init_task_map = greedy_path_init_task_map
                    greedy_destroyed_sequence_map = destroyed_sequence_map_temp_temp
                    greedy_path_init_task_map = path_init_task_map_temp_temp
                else:
                    if fitness_second_min >= fitness >= fitness_min:
                        fitness_second_min = fitness
                        second_destroyed_sequence_map = destroyed_sequence_map_temp_temp
                        second_path_init_task_map = path_init_task_map_temp_temp

        if greedy_destroyed_sequence_map == None:
            print("!!!!!!!!!bug!!!!!!!!! only the original position is feasible")

        if second_destroyed_sequence_map != None:
            destroyed_sequence_map = second_destroyed_sequence_map
            path_init_task_map = second_path_init_task_map
        else:
            destroyed_sequence_map = greedy_destroyed_sequence_map
            path_init_task_map = greedy_path_init_task_map

    new_solution = Solution(instance, destroyed_sequence_map, path_init_task_map)
    return new_solution


def repair_couple_greedy_noise(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    
    gamma = 0.1  
    instance = current_solution.instance
    random.shuffle(destroyed_task_list)
    for destroyed_task in destroyed_task_list:
        greedy_destroyed_sequence_map = None
        greedy_path_init_task_map = None
        fitness_min = 1e6

        parent_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, 'parent')
        for parent_position in parent_position_set:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, parent_position, 'parent')
            child_position_set = get_feasible_insert_position(destroyed_sequence_map_temp, path_init_task_map_temp,
                                                              destroyed_task, 'child')
            for child_position in child_position_set:

                destroyed_sequence_map_temp_temp = copy_dict_int_dict(destroyed_sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp, destroyed_task, child_position,
                        'child')
                fitness, _, _ = cal_fitness(instance, destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp)
                
                fitness = (1 + (gamma * random.uniform(-1, 1))) * fitness
                if fitness < fitness_min:
                    greedy_destroyed_sequence_map = destroyed_sequence_map_temp_temp
                    greedy_path_init_task_map = path_init_task_map_temp_temp
                    fitness_min = fitness
        if greedy_destroyed_sequence_map == None:
            print("!!!!!!!!!bug!!!!!!!!! only the original position is feasible")
        destroyed_sequence_map = greedy_destroyed_sequence_map
        path_init_task_map = greedy_path_init_task_map
    new_solution = Solution(instance, destroyed_sequence_map, path_init_task_map)
    return new_solution


def repair_couple_partial_greedy(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    
    partial_proportion = 0.9
    instance = current_solution.instance
    random.shuffle(destroyed_task_list)
    for destroyed_task in destroyed_task_list:
        greedy_destroyed_sequence_map = None
        greedy_path_init_task_map = None
        fitness_min = 1e6

        parent_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, 'parent')
        for parent_position in parent_position_set:
            destroyed_sequence_map_temp = copy_dict_int_dict(destroyed_sequence_map)
            path_init_task_map_temp = copy_dict_int_int(path_init_task_map)
            insert_(destroyed_sequence_map_temp, path_init_task_map_temp, destroyed_task, parent_position, 'parent')
            child_position_set = get_feasible_insert_position(destroyed_sequence_map_temp, path_init_task_map_temp,
                                                              destroyed_task, 'child')
            for child_position in child_position_set:
                if random.random() > partial_proportion:
                    continue
                destroyed_sequence_map_temp_temp = copy_dict_int_dict(destroyed_sequence_map_temp)
                path_init_task_map_temp_temp = copy_dict_int_int(path_init_task_map_temp)
                insert_(destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp, destroyed_task, child_position,
                        'child')
                fitness, _, _ = cal_fitness(instance, destroyed_sequence_map_temp_temp, path_init_task_map_temp_temp)
                if fitness < fitness_min:
                    greedy_destroyed_sequence_map = destroyed_sequence_map_temp_temp
                    greedy_path_init_task_map = path_init_task_map_temp_temp
                    fitness_min = fitness
        if greedy_destroyed_sequence_map == None:
            print("!!!!!!!!!bug!!!!!!!!! only the original position is feasible")

        destroyed_sequence_map = greedy_destroyed_sequence_map
        path_init_task_map = greedy_path_init_task_map
    new_solution = Solution(instance, destroyed_sequence_map, path_init_task_map)
    return new_solution