import pandas as pd
# from Model_Util.util import cal_distance
import json
import os
from pathlib import Path
optimal_solution_dict = {}

def read_txt():
    data_set = []
    path = "C:/Users/13360/Desktop/CHAGV/CHAGV_python/Util/instance.txt"
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = lines[3:]  
    data = []
    for line in lines:
        data_line = line.strip("\n").split()
        if len(data_line) == 0:
            
            if len(data) != 0:
                data_set.append(data)
                data = []
        elif data_line[0].startswith('T') or data_line[0].startswith('C'):
            continue
        else:
            data_line = list(map(float, data_line))
            data.append(data_line)
    return data_set


def read_excel(file_name):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))

    
    parent_dir = os.path.dirname(current_dir)

    
    file_path = os.path.join(parent_dir, "Instance", file_name)
    df = pd.read_excel(file_path)
    instance = [list(row) for index, row in df.iterrows()]
    return instance


def _resolve_instance_path(file_name):
    path = Path(file_name)
    candidates = [path]
    if path.suffix == "":
        candidates.append(path.with_suffix(".xlsx"))

    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate

    for parent in Path(__file__).resolve().parents:
        for candidate in candidates:
            instance_path = parent / "Instance" / candidate
            if instance_path.exists():
                return instance_path

    return Path(__file__).resolve().parent.parent / "Instance" / file_name


def read_excel(file_name):
    df = pd.read_excel(_resolve_instance_path(file_name))
    return [list(row) for index, row in df.iterrows()]

# class optimalSolution:
#     def __init__(self, instance, code, fitness):
#         self.instance = instance
#         self.code = code
#         self.fitness = fitness
#

#     def to_dict(self):
#         return {"instance": self.instance, "code": self.code, "fitness": self.fitness}
#

#     @classmethod
#     def from_dict(cls, dict_data):
#         return cls(dict_data["instance"], dict_data["code"], dict_data["fitness"])
#
#
#
#
# class CustomEncoder(json.JSONEncoder):
#     def encode(self, obj):
#         if isinstance(obj, list):

#             return json.dumps(obj)
#         return super().encode(obj)
#
# def init_optimal_solution():
#     global optimal_solution_dict
#     with open("D:/Paper/CHAGV/AHASP_python/Util/optimal_solution_0910.json", "r") as file:
#         optimal_solution_dict = json.load(file)
#     pass
#
# def update_optimal_solution():
#     global optimal_solution_dict
#     with open("D:/Paper/CHAGV/AHASP_python/Util/optimal_solution_0910.json", "w") as file:
#         json.dump(optimal_solution_dict, file, indent=4, separators=(",", ": "))












# def save_cache(filename, cache):
#     with open(filename, 'w') as f:
#         for (p1, p2), distance in cache.items():
#             f.write(f"{p1}, {p2}, {distance}\n")









# if __name__ == "__main__":
#     filename = 'distance_cache.txt'
#     load_distance(filename)
#     # read_excel("5_20240906210644.xlsx")
#     pass
