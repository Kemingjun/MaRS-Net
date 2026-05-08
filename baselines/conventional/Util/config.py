from pathlib import Path


M_parent = 4  
M_child = 8  
M = M_parent + M_child
M_parent_list = list(range(1, M_parent + 1))
M_child_list = list(range(M_parent + 1, M_parent + M_child + 1))
V = 1.2  
T_couple = 8  
T_decouple = 8  
T_load = 30  
weight = 0.4

M_position_map = {robot: [0, 0] for robot in range(1, 13)}

# M_position_map = {1: [10, 25], 2: [95, 65], 3: [15, 5], 4: [45, 50], 5: [60, 80], 6: [90, 20]}
# M_position_map = {1: [10, 25], 2: [95, 65], 3: [10, 80], 4: [95, 40], 5: [15, 5], 6: [45, 50], 7: [60, 80], 8: [90, 20], 9: [15, 65], 10: [45, 95], 11: [60, 50], 12: [90, 80]}

# M_position_map = {1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0],
#                   5: [0, 0], 6: [0, 0], 7: [0, 0], 8: [0, 0], 9: [0, 0], 10: [0, 0], 11: [0, 0],
#                   12: [0, 0], 13: [0, 0], 14: [0, 0], 15: [0, 0], 16: [0, 0], 17: [0, 0], 18: [0, 0],
#                   19: [0, 0], 20: [0, 0]}


iter_max = 500  



def load_distance():
    filename = Path(__file__).resolve().with_name("distance_cache.txt")
    cache = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                
                parts = line.strip().split('_')
                p1 = tuple(map(int, parts[0][1:-1].split(', ')))  
                p2 = tuple(map(int, parts[1][1:-1].split(', ')))  
                distance = float(parts[2])  
                cache[tuple([p1, p2])] = distance  
    except FileNotFoundError:
        pass  
    return cache



# def distance_cache(filename):
#     with open(filename, 'w') as f:
#         for p_1_x in range(0, 101, 50):
#             for p_1_y in range(0, 101, 5):
#                 for p_2_x in range(0, 101, 50):
#                     for p_2_y in range(0, 101, 5):
#                         # dis = cal_distance([p_1_x, p_1_y], [p_2_x, p_2_y])
#                         dis = math.fabs(p_1_x - p_2_x) + math.fabs(p_1_y - p_2_y)
#                         f.write(f"{(p_1_x, p_1_y)}_{p_2_x, p_2_y}_{dis}\n")

distance_cache = load_distance()
# distance_cache("distance_cache.txt")
pass


