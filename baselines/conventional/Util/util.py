import math

import Util.config as config
# from Util.config import *
import Util.ALNS_config as ALNS_config
import time
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename='info.log')

feasible_list = []



def copy_set_int(original_set):
    
    copied_set = set()
    for item in original_set:
        copied_set.add(item)

    return copied_set


def copy_list_int(original_list):
    
    copied_list = []
    for item in copied_list:
        copied_list.append(item)

    return copied_list


def copy_dict_int_int(original_dict):
    
    copied_dict = {}

    
    for key, value in original_dict.items():
        copied_dict[key] = value

    return copied_dict


def copy_dict_int_list(original_dict):
    
    copied_dict = {}

    
    for key, value in original_dict.items():
        copied_list = []
        for v in value:
            copied_list.append(v)
        copied_dict[key] = copied_list

    return copied_dict


def copy_dict_int_dict(original_dict):
    
    copied_dict = {}

    
    for key, inner_dict in original_dict.items():
        
        copied_inner_dict = {}

        
        for inner_key, value in inner_dict.items():
            copied_inner_dict[inner_key] = value

        
        copied_dict[key] = copied_inner_dict

    return copied_dict


# %%
def cal_distance(source_axis, destination_axis):
    source_x = source_axis[0]
    source_y = source_axis[1]
    destination_x = destination_axis[0]
    destination_y = destination_axis[1]

    distance = math.fabs(source_x - destination_x) + math.fabs(source_y - destination_y)

    return distance
    #
    # if source_x == destination_x:
    
    #     distance = abs(destination_y - source_y)
    #     return distance
    # else:
    #     if (source_y > 50 > destination_y) or (source_y < 50 < destination_y):
    #         distance = abs(destination_x - source_x) + abs(destination_y - source_y)
    #         return distance
    #     else:
    
    #         distance_1 = min(abs(source_y - 0) + abs(destination_y - 0),
    #                          abs(source_y - 50) + abs(destination_y - 50),
    #                          abs(source_y - 100) + abs(destination_y - 100))
    #         distance_2 = abs(destination_x - source_x)
    #         return distance_1 + distance_2


def cal_fitness_old(instance, sequence_map, path_init_task_map):
    # start_time = time.perf_counter()
    total_distance = 0  
    total_tardiness = 0  

    info_map = {task: dict() for task in sequence_map.keys()}
    task_num = len(sequence_map)

    task_parent_closed_list = []
    task_child_closed_list = []

    for path_index, first_task in path_init_task_map.items():
        if first_task != 0:
            if path_index in config.M_parent_list:
                info_map[first_task]['parent_pre_d_time'] = 0
                task_parent_closed_list.append(first_task)
            else:
                info_map[first_task]['child_pre_e_time'] = 0
                task_child_closed_list.append(first_task)

    task_closed = []
    while len(task_closed) < task_num:
        task_to_calculate = None
        for task in task_parent_closed_list:
            if task in task_child_closed_list:
                task_to_calculate = task
                break
        if task_to_calculate is None:
            print("No enabled transition; infeasible solution.")
            return 1e6

        parent_pre_d_time = info_map[task_to_calculate]['parent_pre_d_time']
        child_pre_e_time = info_map[task_to_calculate]['child_pre_e_time']

        if sequence_map[task_to_calculate]['parent_pre_task'] == 0:
            parent_pre_position = config.M_position_map[sequence_map[task_to_calculate]['parent']]
        else:
            parent_pre_position = [instance[sequence_map[task_to_calculate]['parent_pre_task'] - 1][3],
                                   instance[sequence_map[task_to_calculate]['parent_pre_task'] - 1][4]]
        if sequence_map[task_to_calculate]['child_pre_task'] == 0:
            child_pre_position = config.M_position_map[sequence_map[task_to_calculate]['child']]
        else:
            child_pre_position = [instance[sequence_map[task_to_calculate]['child_pre_task'] - 1][3],
                                  instance[sequence_map[task_to_calculate]['child_pre_task'] - 1][4]]

        source_position = [instance[task_to_calculate - 1][1], instance[task_to_calculate - 1][2]]
        destination_position = [instance[task_to_calculate - 1][3], instance[task_to_calculate - 1][4]]

        # t_i_plus2i_minus = cal_distance(parent_pre_position, child_pre_position) / V
        # t_i_minus2j_s = cal_distance(child_pre_position, source_position) / V
        # t_j_s2j_d = cal_distance(source_position, destination_position) / V
        t_i_plus2i_minus = cal_distance(parent_pre_position, child_pre_position) / config.V
        t_i_minus2j_s = cal_distance(child_pre_position, source_position) / config.V  
        t_j_s2j_d = cal_distance(source_position, destination_position) / config.V
        
        
        # t_j_s2j_d = config.distance_cache[tuple([tuple(source_position), tuple(destination_position)])] / config.V  #

        t_idle_parent = max(0, child_pre_e_time - parent_pre_d_time - t_i_plus2i_minus)

        t_d = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + config.T_couple + t_i_minus2j_s + config.T_load + t_j_s2j_d + config.T_decouple
        t_e = t_d + instance[task_to_calculate - 1][6]

        info_map[task_to_calculate]['parent_decouple_time'] = t_d
        info_map[task_to_calculate]['child_end_time'] = t_e

        parent_next_task = sequence_map[task_to_calculate]['parent_next_task']
        child_next_task = sequence_map[task_to_calculate]['child_next_task']

        task_distance = (t_i_plus2i_minus + t_i_minus2j_s + t_j_s2j_d) * config.V
        task_tardiness = max(0, t_e - instance[task_to_calculate - 1][5])

        total_distance += task_distance
        total_tardiness += task_tardiness

        if parent_next_task != 0:
            info_map[parent_next_task]['parent_pre_d_time'] = t_d
            task_parent_closed_list.append(parent_next_task)
        if child_next_task != 0:
            info_map[child_next_task]['child_pre_e_time'] = t_e
            task_child_closed_list.append(child_next_task)

        task_closed.append(task_to_calculate)
        task_parent_closed_list.remove(task_to_calculate)
        task_child_closed_list.remove(task_to_calculate)
    fitness = total_distance * config.weight + total_tardiness * (1 - config.weight)
    # end_time = time.perf_counter()
    # FDD_time_list.append((end_time - start_time) * 10 ** 6)
    # print(f"FDD average time: {np.mean(FDD_time_list)}")
    return fitness, total_distance, total_tardiness



def code2path_map(code):
    parent_code = code[0]
    child_code = code[1]
    path_map = {}
    parent_index = 1
    path = []
    for task_index, task in enumerate(parent_code):
        if task_index == 0:
            continue
        if task == 0:
            path_map[parent_index] = path
            path = []
            parent_index += 1
            continue
        path.append(task)
        if task_index == len(parent_code) - 1:
            path_map[parent_index] = path

    child_index = config.M_parent + 1
    path = []
    for task_index, task in enumerate(child_code):
        if task_index == 0:
            continue
        if task == 0:
            path_map[child_index] = path
            path = []
            child_index += 1
            continue
        path.append(task)
        if task_index == len(child_code) - 1:
            path_map[child_index] = path
    return path_map




def path_map2sequence_map(path_map):
    sequence_map = {}
    path_init_task_map = {}
    for agv_index in path_map.keys():
        path = path_map[agv_index]
        if len(path) == 0:
            path_init_task_map[agv_index] = 0
            continue
        for task_index, task in enumerate(path):
            if task in sequence_map.keys():
                task_info = sequence_map[task]
            else:
                task_info = {}
            if agv_index in config.M_parent_list:
                task_info['parent'] = agv_index
                if task_index == 0:
                    task_info['parent_pre_task'] = 0
                    path_init_task_map[agv_index] = task
                else:
                    pre_task = path[task_index - 1]
                    task_info['parent_pre_task'] = pre_task
                if task_index == len(path) - 1:
                    
                    task_info['parent_next_task'] = 0
                else:
                    task_info['parent_next_task'] = path[task_index + 1]
            else:
                task_info['child'] = agv_index
                if task_index == 0:
                    task_info['child_pre_task'] = 0
                    path_init_task_map[agv_index] = task
                else:
                    pre_task = path[task_index - 1]
                    task_info['child_pre_task'] = pre_task
                if task_index == len(path) - 1:
                    
                    task_info['child_next_task'] = 0
                else:
                    task_info['child_next_task'] = path[task_index + 1]
            sequence_map[task] = task_info
    return sequence_map, path_init_task_map


# cal_fitness_time_list = []
def cal_fitness(instance, sequence_map, path_init_task_map):
    # start_time = time.perf_counter()
    total_distance = 0  
    total_tardiness = 0  

    info_map = {task: dict() for task in sequence_map.keys()}
    task_num = len(sequence_map)

    task_parent_closed_set = set()
    task_child_closed_set = set()

    for path_index, first_task in path_init_task_map.items():
        if first_task != 0:
            if path_index in config.M_parent_list:
                info_map[first_task]['parent_pre_d_time'] = 0
                task_parent_closed_set.add(first_task)
            else:
                info_map[first_task]['child_pre_e_time'] = 0
                task_child_closed_set.add(first_task)

    # task_closed = []
    task_closed_num = 0
    while task_closed_num < task_num:
        task_to_calculate = None
        for task in task_parent_closed_set:
            if task in task_child_closed_set:
                task_to_calculate = task
                break
        if task_to_calculate is None:
            print("No enabled transition; infeasible solution.")
            return 1e6
        task_closed_num += 1
        parent_pre_d_time = info_map[task_to_calculate]['parent_pre_d_time']
        child_pre_e_time = info_map[task_to_calculate]['child_pre_e_time']

        task_data = sequence_map[task_to_calculate]

        if task_data['parent_pre_task'] == 0:
            parent_pre_position = config.M_position_map[task_data['parent']]
        else:
            parent_pre_position = [instance[task_data['parent_pre_task'] - 1][3],
                                   instance[task_data['parent_pre_task'] - 1][4]]
        if task_data['child_pre_task'] == 0:
            child_pre_position = config.M_position_map[task_data['child']]
        else:
            child_pre_position = [instance[task_data['child_pre_task'] - 1][3],
                                  instance[task_data['child_pre_task'] - 1][4]]
        instance_task_data = instance[task_to_calculate - 1]
        source_position = [instance_task_data[1], instance_task_data[2]]
        destination_position = [instance_task_data[3], instance_task_data[4]]

        # t_i_plus2i_minus = cal_distance(parent_pre_position, child_pre_position) / V
        # t_i_minus2j_s = cal_distance(child_pre_position, source_position) / V
        # t_j_s2j_d = cal_distance(source_position, destination_position) / V
        t_i_plus2i_minus = cal_distance(parent_pre_position, child_pre_position) / config.V
        t_i_minus2j_s = cal_distance(child_pre_position, source_position) / config.V  
        t_j_s2j_d = cal_distance(source_position, destination_position) / config.V
        
        
        # t_j_s2j_d = config.distance_cache[tuple([tuple(source_position), tuple(destination_position)])] / config.V  #

        t_idle_parent = max(0, child_pre_e_time - parent_pre_d_time - t_i_plus2i_minus)

        t_d = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + config.T_couple + t_i_minus2j_s + config.T_load + t_j_s2j_d + config.T_decouple
        t_e = t_d + instance_task_data[6]

        # info_map[task_to_calculate]['parent_decouple_time'] = t_d
        # info_map[task_to_calculate]['child_end_time'] = t_e

        parent_next_task = task_data['parent_next_task']
        child_next_task = task_data['child_next_task']

        task_distance = (t_i_plus2i_minus + t_i_minus2j_s + t_j_s2j_d) * config.V
        task_tardiness = max(0, t_e - instance_task_data[5])

        total_distance += task_distance
        total_tardiness += task_tardiness

        if parent_next_task != 0:
            info_map[parent_next_task]['parent_pre_d_time'] = t_d
            task_parent_closed_set.add(parent_next_task)
        if child_next_task != 0:
            info_map[child_next_task]['child_pre_e_time'] = t_e
            task_child_closed_set.add(child_next_task)

        # task_closed.append(task_to_calculate)
        task_parent_closed_set.remove(task_to_calculate)
        task_child_closed_set.remove(task_to_calculate)
    fitness = total_distance * config.weight + total_tardiness * (1 - config.weight)
    # end_time = time.perf_counter()
    # FDD_time_list.append((end_time - start_time) * 10 ** 6)
    # print(f"FDD average time: {round(np.mean(FDD_time_list),4)}")
    # end_time = time.perf_counter()
    # cal_fitness_time_list.append(end_time-start_time)
    # if len(cal_fitness_time_list) > 1000:
    #     print(sum(cal_fitness_time_list)/len(cal_fitness_time_list))
    return fitness, total_distance, total_tardiness



FDD_time_list = []


def get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain):
    

    
    feasible_position_set = {(task, 0) for task in destroyed_sequence_map.keys() if
                             task != destroyed_task and destroyed_sequence_map[task][chain] is not None}  
    feasible_position_set = feasible_position_set.union({(task, 1) for task in destroyed_sequence_map.keys() if
                                                         task != destroyed_task and destroyed_sequence_map[task][
                                                             chain] is not None})  
    
    to_delete_list = []
    for position in feasible_position_set:
        task = position[0]
        direction = position[1]
        if direction == 0:
            pre_task = destroyed_sequence_map[task][chain + '_pre_task']
            if pre_task != 0 and pre_task is not None:
                if (pre_task, 1) in feasible_position_set:
                    to_delete_list.append((pre_task, 1))
        else:
            next_task = destroyed_sequence_map[task][chain + '_next_task']
            if next_task != 0 and next_task is not None:
                if (next_task, 0) in feasible_position_set:
                    to_delete_list.append((task, 1))
    for position in to_delete_list:
        feasible_position_set.discard(position)

    
    for path_index, first_task in path_init_task_map.items():
        if chain == 'child' and path_index in config.M_child_list and first_task == 0:
            feasible_position_set.add((0, path_index))
        if chain == 'parent' and path_index in config.M_parent_list and first_task == 0:
            feasible_position_set.add((0, path_index))
    return feasible_position_set


def get_feasible_insert_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain):
    
    # start_time = time.perf_counter()
    
    feasible_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain)
    # all_position_num = len(feasible_position_set)

    pre_to_explore_set = {destroyed_task}
    pre_explored_list = []
    while len(pre_to_explore_set) != 0:
        temp_set = copy_set_int(pre_to_explore_set)
        for task in temp_set:
            
            child_pre_task = destroyed_sequence_map[task]['child_pre_task']
            parent_pre_task = destroyed_sequence_map[task]['parent_pre_task']
            if child_pre_task != 0 and child_pre_task is not None:
                feasible_position_set.discard((child_pre_task, 0))
                
                
                if chain == 'child':
                    couple_task = destroyed_sequence_map[child_pre_task]['child_pre_task']
                else:
                    couple_task = destroyed_sequence_map[child_pre_task]['parent_pre_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 1))
                if child_pre_task not in pre_explored_list:
                    pre_to_explore_set.add(child_pre_task)
            if parent_pre_task != 0 and parent_pre_task is not None:
                feasible_position_set.discard((parent_pre_task, 0))
                if chain == 'child':
                    couple_task = destroyed_sequence_map[parent_pre_task]['child_pre_task']
                else:
                    couple_task = destroyed_sequence_map[parent_pre_task]['parent_pre_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 1))
                if parent_pre_task not in pre_explored_list:
                    pre_to_explore_set.add(parent_pre_task)
            pre_to_explore_set.remove(task)
            pre_explored_list.append(task)
    # print(f"pre_explored_list:{pre_explored_list}")

    next_to_explore_set = {destroyed_task}
    next_explored_list = []
    while len(next_to_explore_set) != 0:
        temp_set = copy_set_int(next_to_explore_set)
        for task in temp_set:
            child_next_task = destroyed_sequence_map[task]['child_next_task']
            parent_next_task = destroyed_sequence_map[task]['parent_next_task']
            if child_next_task != 0 and child_next_task is not None:
                feasible_position_set.discard((child_next_task, 1))
                if chain == 'child':
                    couple_task = destroyed_sequence_map[child_next_task]['child_next_task']
                else:
                    couple_task = destroyed_sequence_map[child_next_task]['parent_next_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 0))
                if child_next_task not in next_explored_list:
                    next_to_explore_set.add(child_next_task)
            if parent_next_task != 0 and parent_next_task is not None:
                feasible_position_set.discard((parent_next_task, 1))
                if chain == 'child':
                    couple_task = destroyed_sequence_map[parent_next_task]['child_next_task']
                else:
                    couple_task = destroyed_sequence_map[parent_next_task]['parent_next_task']
                if couple_task != 0 and couple_task is not None:
                    feasible_position_set.discard((couple_task, 0))
                if parent_next_task not in next_explored_list:
                    next_to_explore_set.add(parent_next_task)
            next_to_explore_set.remove(task)
            next_explored_list.append(task)
    # end_time = time.perf_counter()
    # BBS_time_list.append((end_time - start_time) * 10 ** 6)
    # print(f"BBS average time: {round(np.mean(BBS_time_list), 4)}")
    # print(f"next_explored_list:{next_explored_list}")
    
    # feasible_position_num = len(feasible_position_set)
    # feasible_list.append(round((feasible_position_num / all_position_num) * 100, 4))
    # infeasible_num_list.append(all_position_num - feasible_position_num)
    
    
    
    
    return feasible_position_set


BBS_time_list = []
infeasible_num_list = []


def insert_(destroyed_sequence_map, path_init_task_map, destroyed_task, insert_position, chain):
    
    if chain == 'parent':
        chain_opposite = 'child'
    else:
        chain_opposite = 'parent'
    if insert_position[0] != 0:
        insert_task = insert_position[0]
        direction = ['pre', 'next'][insert_position[1]]
        direction_opposite = ['pre', 'next'][insert_position[1] - 1]
        path_index = destroyed_sequence_map[insert_task][chain]
        direction_task = destroyed_sequence_map[insert_task][chain + '_' + direction + '_task']
        destroyed_sequence_map[insert_task][chain + '_' + direction + '_task'] = destroyed_task
        if direction_task != 0:
            destroyed_sequence_map[direction_task][
                chain + '_' + direction_opposite + '_task'] = destroyed_task
        else:
            if direction == 'pre':
                path_init_task_map[path_index] = destroyed_task
        if destroyed_task not in destroyed_sequence_map.keys():
            destroyed_sequence_map[destroyed_task] = {}
            destroyed_sequence_map[destroyed_task][chain] = path_index
            destroyed_sequence_map[destroyed_task][chain_opposite] = None
            destroyed_sequence_map[destroyed_task][chain + '_' + direction + '_task'] = direction_task
            destroyed_sequence_map[destroyed_task][chain_opposite + '_' + direction + '_task'] = None
            destroyed_sequence_map[destroyed_task][chain + '_' + direction_opposite + '_task'] = insert_task
            destroyed_sequence_map[destroyed_task][chain_opposite + '_' + direction_opposite + '_task'] = None
        else:
            destroyed_sequence_map[destroyed_task][chain] = path_index
            destroyed_sequence_map[destroyed_task][chain + '_' + direction + '_task'] = direction_task
            destroyed_sequence_map[destroyed_task][chain + '_' + direction_opposite + '_task'] = insert_task
    else:
        insert_path_index = insert_position[1]
        path_init_task_map[insert_path_index] = destroyed_task
        if destroyed_task not in destroyed_sequence_map.keys():
            destroyed_sequence_map[destroyed_task] = {}
            destroyed_sequence_map[destroyed_task][chain] = insert_path_index
            destroyed_sequence_map[destroyed_task][chain_opposite] = None
            destroyed_sequence_map[destroyed_task][chain + '_pre_task'] = 0
            destroyed_sequence_map[destroyed_task][chain_opposite + '_pre_task'] = None
            destroyed_sequence_map[destroyed_task][chain + '_next_task'] = 0
            destroyed_sequence_map[destroyed_task][chain_opposite + '_next_task'] = None
        else:
            destroyed_sequence_map[destroyed_task][chain] = insert_path_index
            destroyed_sequence_map[destroyed_task][chain + '_pre_task'] = 0
            destroyed_sequence_map[destroyed_task][chain + '_next_task'] = 0
    return destroyed_sequence_map, path_init_task_map


def remove_(sequence_map, path_init_task_map, task):
    
    parent_pre_task = sequence_map[task]['parent_pre_task']
    child_pre_task = sequence_map[task]['child_pre_task']
    parent_next_task = sequence_map[task]['parent_next_task']
    child_next_task = sequence_map[task]['child_next_task']

    if parent_pre_task == 0:
        parent = sequence_map[task]['parent']
        path_init_task_map[parent] = parent_next_task
    else:
        sequence_map[parent_pre_task]['parent_next_task'] = parent_next_task

    if parent_next_task != 0:
        sequence_map[parent_next_task]['parent_pre_task'] = parent_pre_task

    if child_pre_task == 0:
        child = sequence_map[task]['child']
        path_init_task_map[child] = child_next_task
    else:
        sequence_map[child_pre_task]['child_next_task'] = child_next_task

    if child_next_task != 0:
        sequence_map[child_next_task]['child_pre_task'] = child_pre_task

    del sequence_map[task]

    return sequence_map, path_init_task_map


def get_path_map(sequence_map, path_init_task_map):
    path_map = {}
    for path_index, init_task in path_init_task_map.items():
        if init_task == 0:
            
            path_map[path_index] = []
            continue
        if path_index in config.M_parent_list:
            path = [init_task]
            parent_next_task = sequence_map[init_task]['parent_next_task']
            while parent_next_task != 0:
                path.append(parent_next_task)
                parent_next_task = sequence_map[parent_next_task]['parent_next_task']
            path_map[path_index] = path
        else:
            path = [init_task]
            parent_next_task = sequence_map[init_task]['child_next_task']
            while parent_next_task != 0:
                path.append(parent_next_task)
                parent_next_task = sequence_map[parent_next_task]['child_next_task']
            path_map[path_index] = path

    return path_map


def get_T(instance):
    sum_dis = 0
    for info in instance:
        source_position = [info[1], info[2]]
        destination_position = [info[3], info[4]]
        # sum_dis += config.distance_cache[tuple([tuple(source_position), tuple(destination_position)])] / config.V
        sum_dis += cal_distance(source_position, destination_position) / config.V
    T = sum_dis / (len(instance) * (config.M_parent + config.M_child)) * ALNS_config.T_coefficient
    return T




