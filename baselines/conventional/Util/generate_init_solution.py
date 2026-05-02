import copy
import random
import time

from Util.solution import Solution
from Util.util import *
import Util.config
from Util.load_data import read_excel

def generate_solution_average(instance):
    task_num = len(instance)
    path_map = {i: [] for i in range(1, Util.config.M + 1)}
    task_list = list(range(1, task_num + 1))
    random.shuffle(task_list)
    for task in task_list:
        shortest_parent_path_index = -1
        shortest_parent_path_task_num = 1000
        for path_index in range(1, Util.config.M_parent + 1):
            path_task_num = len(path_map[path_index])
            if path_task_num < shortest_parent_path_task_num:
                shortest_parent_path_index = path_index

        shortest_child_path_index = -1
        shortest_child_path_task_num = 1000
        for path_index in range(Util.config.M_parent + 1, Util.config.M_parent + Util.config.M_child + 1):
            path_task_num = len(path_map[path_index])
            if path_task_num < shortest_child_path_task_num:
                shortest_child_path_index = path_index

        path_map[shortest_parent_path_index].append(task)
        path_map[shortest_child_path_index].append(task)

    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution



def generate_solution_random(instance):
    task_num = len(instance)
    sequence_map = {}
    path_init_task_map = {}
    
    for path_index in range(1, Util.config.M + 1):
        path_init_task_map[path_index] = 0
    for task in range(1, task_num + 1):
        chain_list = random.sample(['parent', 'child'], 2)
        chain = chain_list[0]
        coupled_chain = chain_list[1]
        all_position_set = get_all_position(sequence_map, path_init_task_map, task, chain)
        position = random.sample(all_position_set, 1)[0]
        insert_(sequence_map, path_init_task_map, task, position, chain)
        feasible_position_set = get_feasible_insert_position(sequence_map, path_init_task_map, task, coupled_chain)
        coupled_position = random.sample(feasible_position_set, 1)[0]
        insert_(sequence_map, path_init_task_map, task, coupled_position, coupled_chain)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution

def generate_infeasible_solution_random(instance):
    task_num = len(instance)
    sequence_map = {}
    path_init_task_map = {}
    
    for path_index in range(1, Util.config.M + 1):
        path_init_task_map[path_index] = 0
    for task in range(1, task_num + 1):
        chain_list = random.sample(['parent', 'child'], 2)
        chain = chain_list[0]
        coupled_chain = chain_list[1]
        all_position_set = get_all_position(sequence_map, path_init_task_map, task, chain)
        position = random.sample(all_position_set, 1)[0]
        insert_(sequence_map, path_init_task_map, task, position, chain)
        # feasible_position_set = get_feasible_insert_position(sequence_map, path_init_task_map, task, coupled_chain)
        couple_all_position_set = get_all_position(sequence_map, path_init_task_map, task, coupled_chain)
        coupled_position = random.sample(couple_all_position_set, 1)[0]
        insert_(sequence_map, path_init_task_map, task, coupled_position, coupled_chain)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution

def generate_solution_greedy1(instance):
    
    task_num = len(instance)
    agv_info_map = {}   
    for agv in range(1, Util.config.M + 1):
        agv_info_map[agv] = [0, Util.config.M_position_map[agv]]
    task_ddl_list = [[task, instance[task - 1][5]] for task in range(1, task_num + 1)]

def generate_solution_greedy2(instance):
    
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    task_time_list = [[task, instance[task - 1][5]] for task in task_list]
    task_time_list = sorted(task_time_list, key=lambda x: x[1], reverse=False)

    path_init_task_map = {}
    for path_index in range(1, Util.config.M + 1):
        path_init_task_map[path_index] = 0
    sequence_map = {}
    for task_time in task_time_list:
        # best_position = None
        task = task_time[0]
        best_fitness = 1e5
        best_solution = None
        parent_all_feasible_position_set = get_all_position(sequence_map, path_init_task_map, task, 'parent')
        for parent_position in parent_all_feasible_position_set:
            sequence_map_temp = copy.deepcopy(sequence_map)
            path_init_task_map_temp = copy.deepcopy(path_init_task_map)
            insert_(sequence_map_temp, path_init_task_map_temp, task, parent_position, 'parent')
            child_all_feasible_position_set = get_feasible_insert_position(sequence_map_temp, path_init_task_map_temp, task, 'child')
            for child_position in child_all_feasible_position_set:
                sequence_map_temp_temp = copy.deepcopy(sequence_map_temp)
                path_init_task_map_temp_temp = copy.deepcopy(path_init_task_map_temp)
                insert_(sequence_map_temp_temp, path_init_task_map_temp_temp, task, child_position, 'child')
                solution_temp = Solution(instance, sequence_map_temp_temp, path_init_task_map_temp_temp)
                fitness = solution_temp.get_fitness()
                if fitness < best_fitness:
                    # best_position = [parent_position, child_position]
                    best_fitness = fitness
                    best_solution = solution_temp
        sequence_map = best_solution.get_sequence_map()
        path_init_task_map = best_solution.get_path_init_task_map()
    return Solution(instance, sequence_map, path_init_task_map)

def generate_solution_nearest(instance):
    
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    parent_map = {}  
    child_map = {}   

    path_map = {}
    for path_index in range(1, Util.config.M + 1):
        path_map[path_index] = []

    for parent in Util.config.M_parent_list:
        parent_map[parent] = [0, Util.config.M_position_map[parent]]
    for child in Util.config.M_child_list:
        child_map[child] = [0, Util.config.M_position_map[child]]

    while len(task_list) != 0:
        min_parent_index = min(parent_map, key=lambda k: parent_map[k][0])
        parent_info = parent_map[min_parent_index]
        parent_idle_time = parent_info[0]
        parent_position = parent_info[1]
        
        # temporal distance and spatial distance
        min_child_index = None
        distance_min = 1e5
        for child in Util.config.M_child_list:
            child_info = child_map[child]
            child_idle_time = child_info[0]
            child_position = child_info[1]
            spatial_distance = cal_distance(parent_position, child_position)
            parent_arrive_time = (parent_idle_time + spatial_distance / Util.config.V) * Util.config.V  
            temporal_distance = max(0, child_idle_time - parent_arrive_time)  
            
            distance = spatial_distance + temporal_distance
            if distance < distance_min:
                min_child_index = child
                distance_min = distance
        
        child_min_info = child_map[min_child_index]
        child_min_idle_time = child_min_info[0]
        child_min_position = child_min_info[1]
        couple_time = max(child_min_idle_time, parent_idle_time + cal_distance(parent_position, child_min_position) / Util.config.V)
        couple_position = child_min_position
        
        task_min = None
        cost_min = 1e5
        t_d_min = None
        t_e_min = None
        idle_position = None
        for task in task_list:
            task_info = instance[task - 1]
            source_position = [task_info[1], task_info[2]]
            destination_position = [task_info[3], task_info[4]]
            t_operate = task_info[6]
            t_ddl = task_info[5]
            couple2source_distance = cal_distance(couple_position, source_position)
            source2destination_distance = cal_distance(source_position, destination_position)
            t_d = couple_time + Util.config.T_couple + couple2source_distance / Util.config.V + Util.config.T_load + source2destination_distance / Util.config.V + Util.config.T_decouple
            t_e = t_d + t_operate
            distance_cost = couple2source_distance + source2destination_distance
            tardiness_cost = max(0, t_e - t_ddl)
            
            cost = distance_cost * Util.config.weight + tardiness_cost * (1 - Util.config.weight)
            if cost < cost_min:
                task_min = task
                cost_min = cost
                t_d_min = t_d
                t_e_min = t_e
                idle_position = destination_position
        path_map[min_parent_index].append(task_min)
        path_map[min_child_index].append(task_min)
        parent_map[min_parent_index] = [t_d_min, idle_position]
        child_map[min_child_index] = [t_e_min, idle_position]
        task_list.remove(task_min)
    solution = Solution(instance, None, path_map)
    return solution





def generate_solution_nearest2(instance, weight_construct=None):
    
    if weight_construct is None:
        weight_construct = 0.8
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    parent_map = {}  
    child_map = {}   

    path_map = {}
    for path_index in range(1, Util.config.M + 1):
        path_map[path_index] = []

    for parent in Util.config.M_parent_list:
        parent_map[parent] = [0, Util.config.M_position_map[parent]]
    for child in Util.config.M_child_list:
        child_map[child] = [0, Util.config.M_position_map[child]]

    while len(task_list) != 0:
        min_parent_index = min(parent_map, key=lambda k: parent_map[k][0])
        parent_info = parent_map[min_parent_index]
        parent_idle_time = parent_info[0]
        parent_position = parent_info[1]
        
        # temporal distance and spatial distance
        min_child_index = None
        distance_min = 1e5
        for child in Util.config.M_child_list:
            child_info = child_map[child]
            child_idle_time = child_info[0]
            child_position = child_info[1]
            spatial_distance = cal_distance(parent_position, child_position)
            parent_arrive_time = (parent_idle_time + spatial_distance / Util.config.V) * Util.config.V  
            temporal_distance = max(0, child_idle_time - parent_arrive_time)  
            
            
            distance = spatial_distance + temporal_distance
            if distance < distance_min:
                min_child_index = child
                distance_min = distance
        
        child_min_info = child_map[min_child_index]
        child_min_idle_time = child_min_info[0]
        child_min_position = child_min_info[1]
        couple_time = max(child_min_idle_time, parent_idle_time + cal_distance(parent_position, child_min_position) / Util.config.V)
        couple_position = child_min_position
        
        task_min = None
        cost_min = 1e5
        t_d_min = None
        t_e_min = None
        idle_position = None
        for task in task_list:
            task_info = instance[task - 1]
            source_position = [task_info[1], task_info[2]]
            destination_position = [task_info[3], task_info[4]]
            t_operate = task_info[6]
            t_ddl = task_info[5]
            couple2source_distance = cal_distance(couple_position, source_position)
            source2destination_distance = cal_distance(source_position, destination_position)
            t_d = couple_time + Util.config.T_couple + couple2source_distance / Util.config.V + Util.config.T_load + source2destination_distance / Util.config.V + Util.config.T_decouple
            t_e = t_d + t_operate
            distance_cost = couple2source_distance
            urgent_cost = t_ddl - parent_idle_time     
            
            
            cost = distance_cost * weight_construct + urgent_cost * (1 - weight_construct)
            if cost < cost_min:
                task_min = task
                cost_min = cost
                t_d_min = t_d
                t_e_min = t_e
                idle_position = destination_position
        path_map[min_parent_index].append(task_min)
        path_map[min_child_index].append(task_min)
        parent_map[min_parent_index] = [t_d_min, idle_position]
        child_map[min_child_index] = [t_e_min, idle_position]
        task_list.remove(task_min)

    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution



def generate_solution_nearest3(instance, weight_construct=None):
    
    if weight_construct is None:
        weight_construct = 1
    task_num = len(instance)
    task_list = [task for task in range(1, task_num + 1)]
    parent_map = {}  
    child_map = {}   

    path_map = {}
    for path_index in range(1, Util.config.M + 1):
        path_map[path_index] = []

    for parent in Util.config.M_parent_list:
        parent_map[parent] = [0, Util.config.M_position_map[parent]]
    for child in Util.config.M_child_list:
        child_map[child] = [0, Util.config.M_position_map[child]]

    while len(task_list) != 0:
        min_parent_index = min(parent_map, key=lambda k: parent_map[k][0])
        parent_info = parent_map[min_parent_index]
        parent_idle_time = parent_info[0]
        parent_position = parent_info[1]


        min_child_index = min(child_map, key=lambda k: child_map[k][0])
        child_info = child_map[min_child_index]
        child_idle_time = child_info[0]
        child_position = child_info[1]
        
        # # temporal distance and spatial distance
        # min_child_index = None
        # distance_min = 1e5
        # for child in M_child_list:
        #     child_info = child_map[child]
        #     child_idle_time = child_info[0]
        #     child_position = child_info[1]
        #     spatial_distance = cal_distance(parent_position, child_position)
        
        
        #     """
        
        
        
        
        #     """
        #     distance = spatial_distance + temporal_distance
        #     if distance < distance_min:
        #         min_child_index = child
        #         distance_min = distance
        
        # child_min_info = child_map[min_child_index]
        # child_min_idle_time = child_min_info[0]
        # child_min_position = child_min_info[1]
        couple_time = max(child_idle_time, parent_idle_time + cal_distance(parent_position, child_position) / Util.config.V)
        couple_position = child_position
        
        task_min = None
        cost_min = 1e5
        t_d_min = None
        t_e_min = None
        idle_position = None
        for task in task_list:
            task_info = instance[task - 1]
            source_position = [task_info[1], task_info[2]]
            destination_position = [task_info[3], task_info[4]]
            t_operate = task_info[6]
            t_ddl = task_info[5]
            couple2source_distance = cal_distance(couple_position, source_position)
            source2destination_distance = cal_distance(source_position, destination_position)
            t_d = couple_time + Util.config.T_couple + couple2source_distance / Util.config.V + Util.config.T_load + source2destination_distance / Util.config.V + Util.config.T_decouple
            t_e = t_d + t_operate
            distance_cost = couple2source_distance
            urgent_cost = t_ddl - child_idle_time     
            
            
            cost = distance_cost * weight_construct + urgent_cost * (1 - weight_construct)
            if cost < cost_min:
                task_min = task
                cost_min = cost
                t_d_min = t_d
                t_e_min = t_e
                idle_position = destination_position
        path_map[min_parent_index].append(task_min)
        path_map[min_child_index].append(task_min)
        parent_map[min_parent_index] = [t_d_min, idle_position]
        child_map[min_child_index] = [t_e_min, idle_position]
        task_list.remove(task_min)

    sequence_map, path_init_task_map = path_map2sequence_map(path_map)
    solution = Solution(instance, sequence_map, path_init_task_map)
    return solution



if __name__ == "__main__":
    instance = read_excel("T30_I7.xlsx")
    start_time = time.perf_counter()
    greedy_solution = generate_solution_greedy2(instance)
    end_time = time.perf_counter()
    nearest_solution = generate_solution_nearest2(instance)
    print(
        f"Initial solution time: {(end_time - start_time)}, "
        f"greedy fitness: {greedy_solution.get_fitness()}, "
        f"nearest fitness: {nearest_solution.get_fitness()}"
    )

    # # base = 0
    # # for _ in range(11):
    # #     weight_construct = base + _ * 0.1
    # #     solution = generate_solution_nearest2(instance, weight_construct)
    # #     print(f"weight:{weight_construct} fitness:{solution.get_fitness()}")
    # # pass
    # # weight_construct = 0.8
    # # start_time1 = time.perf_counter()
    #
    # solution = generate_solution_nearest2(instance, 0.0)
    # print(f"0.0 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.1)
    # print(f"0.1 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.2)
    # print(f"0.2 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.3)
    # print(f"0.3 neighborhood fitness:{solution.get_fitness()}")
    #
    #
    # solution = generate_solution_nearest2(instance, 0.4)
    # print(f"0.4 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.5)
    # print(f"0.5 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.6)
    # print(f"0.6 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.7)
    # print(f"0.7 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.8)
    # print(f"0.8 neighborhood fitness:{solution.get_fitness()}")
    #
    # solution = generate_solution_nearest2(instance, 0.9)
    # print(f"0.9 neighborhood fitness:{solution.get_fitness()}")
    #
    #
    # solution = generate_solution_nearest2(instance, 1)
    # print(f"1.0 neighborhood fitness:{solution.get_fitness()}")
    #
    # # start_time2 = time.perf_counter()
    # # solution_greedy = generate_solution_average(instance)
    # # print(f"average fitness:{solution_greedy.get_fitness()} time:{time.perf_counter() - start_time2}")
    # #
    # # start_time2 = time.perf_counter()
    # # solution_random = generate_solution_nearest3(instance, 0.7)
    # # print(f"3 fitness:{solution_random.get_fitness()} time:{time.perf_counter() - start_time2}")
    # # pass







