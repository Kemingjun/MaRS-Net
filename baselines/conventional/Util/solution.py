from Util.util import *
import Util.config


# %% solution

class Solution:
    def __init__(self, instance, sequence_map, path_init_task_map):

        self.instance = instance
        self.sequence_map = sequence_map
        self.path_init_task_map = path_init_task_map

        self.distance = None
        self.tardiness = None
        self.fitness = None
        self.feasible = None
        self.path_map = None

        self.task_num = len([task for task in self.sequence_map.keys()])  

        self.info_map = {task: dict() for task in sequence_map.keys()}  
        self.code = self.get_code()

        self.hash_key = hash(tuple(self.code[0] + self.code[1]))

        self.fixed_distance = 0
        self.unfixed_distance = 0

        self.plot_info = {task: dict() for task in sequence_map.keys()}  





        # frozen_sequence = frozenset(sequence_map.items())
        # self.hash_key = hash(frozen_sequence)

    def get_path_map(self):
        if self.path_map is not None:
            return copy_dict_int_list(self.path_map)
        path_map = {}
        for path_index, init_task in self.path_init_task_map.items():
            if init_task == 0:
                
                path_map[path_index] = []
                continue
            if path_index in Util.config.M_parent_list:
                path = [init_task]
                parent_next_task = self.sequence_map[init_task]['parent_next_task']
                while parent_next_task != 0:
                    path.append(parent_next_task)
                    parent_next_task = self.sequence_map[parent_next_task]['parent_next_task']
                path_map[path_index] = path
            else:
                path = [init_task]
                parent_next_task = self.sequence_map[init_task]['child_next_task']
                while parent_next_task != 0:
                    path.append(parent_next_task)
                    parent_next_task = self.sequence_map[parent_next_task]['child_next_task']
                path_map[path_index] = path
        self.path_map = path_map
        return path_map

    def get_code(self):
        code_parent = [0]
        code_child = [0]
        for path_index, init_task in self.path_init_task_map.items():
            if path_index in Util.config.M_parent_list:
                chain = "parent"
                code_list = code_parent
            else:
                chain = "child"
                code_list = code_child
            if init_task == 0:
                if path_index != Util.config.M_parent and path_index != Util.config.M_child + Util.config.M_parent:
                    code_list.append(0)
            else:
                path = [init_task]
                next_task = self.sequence_map[init_task][chain + '_next_task']
                while next_task != 0:
                    path.append(next_task)
                    next_task = self.sequence_map[next_task][chain + '_next_task']
                code_list += path
                if path_index != Util.config.M_parent and path_index != Util.config.M_child + Util.config.M_parent:
                    code_list.append(0)
        return [code_parent, code_child]
    # def get_code_by_path(self):
    #     path_map = self.get_path_map()
    #     code_parent = [0]
    #     code_child = [0]
    #
    #     for agv_index in M_parent_list:
    #         path = path_map[agv_index]
    #         code_parent += path
    #         if agv_index != M_parent:
    #             code_parent.append(0)
    #
    #     for agv_index in M_child_list:
    #         path = path_map[agv_index]
    #         code_child += path
    #         if agv_index != M_parent + M_child:
    #             code_child.append(0)
    #     code_ = [code_parent, code_child]
    #     if code_ != self.code:
    #         print("bug")

    def get_fitness(self):
        if self.fitness is not None:
            return self.fitness
        total_distance = 0  
        total_tardiness = 0  

        task_parent_closed_set = set()
        task_child_closed_set = set()

        for path_index, first_task in self.path_init_task_map.items():
            if first_task != 0:
                if path_index in Util.config.M_parent_list:
                    self.info_map[first_task]['parent_pre_d_time'] = 0
                    task_parent_closed_set.add(first_task)
                else:
                    self.info_map[first_task]['child_pre_e_time'] = 0
                    task_child_closed_set.add(first_task)

        task_closed = []
        while len(task_closed) < self.task_num:
            task_to_calculate = None
            for task in task_parent_closed_set:
                if task in task_child_closed_set:
                    task_to_calculate = task
                    break
            if task_to_calculate is None:
                print("No enabled transition; infeasible solution.")
                self.feasible = False
                self.fitness = 1e6
                return self.fitness


            self.plot_info[task_to_calculate]['carrier'] = self.sequence_map[task_to_calculate]['parent']
            self.plot_info[task_to_calculate]['shuttle'] = self.sequence_map[task_to_calculate]['child']
            self.plot_info[task_to_calculate]["source"] = (self.instance[task_to_calculate - 1][1], self.instance[task_to_calculate - 1][2])
            self.plot_info[task_to_calculate]["destination"] = (self.instance[task_to_calculate - 1][3], self.instance[task_to_calculate - 1][4])
            self.plot_info[task_to_calculate]["deadline"] = self.instance[task_to_calculate - 1][5]
            self.plot_info[task_to_calculate]["operation_time"] = self.instance[task_to_calculate - 1][6]








            parent_pre_d_time = self.info_map[task_to_calculate]['parent_pre_d_time']
            child_pre_e_time = self.info_map[task_to_calculate]['child_pre_e_time']

            if self.sequence_map[task_to_calculate]['parent_pre_task'] == 0:
                parent_pre_position = Util.config.M_position_map[self.sequence_map[task_to_calculate]['parent']]
            else:
                parent_pre_position = [self.instance[self.sequence_map[task_to_calculate]['parent_pre_task'] - 1][3],
                                       self.instance[self.sequence_map[task_to_calculate]['parent_pre_task'] - 1][4]]

            if self.sequence_map[task_to_calculate]['child_pre_task'] == 0:
                child_pre_position = Util.config.M_position_map[self.sequence_map[task_to_calculate]['child']]
            else:
                child_pre_position = [self.instance[self.sequence_map[task_to_calculate]['child_pre_task'] - 1][3],
                                      self.instance[self.sequence_map[task_to_calculate]['child_pre_task'] - 1][4]]

            source_position = [self.instance[task_to_calculate - 1][1], self.instance[task_to_calculate - 1][2]]
            destination_position = [self.instance[task_to_calculate - 1][3], self.instance[task_to_calculate - 1][4]]

            t_i_plus2i_minus = cal_distance(parent_pre_position, child_pre_position) / Util.config.V
            t_i_minus2j_s = cal_distance(child_pre_position, source_position) / Util.config.V
            t_j_s2j_d = cal_distance(source_position, destination_position) / Util.config.V

            t_idle_parent = max(0, child_pre_e_time - parent_pre_d_time - t_i_plus2i_minus)

            t_d = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Util.config.T_couple + t_i_minus2j_s + Util.config.T_load + t_j_s2j_d + Util.config.T_decouple
            t_e = t_d + self.instance[task_to_calculate - 1][6]
            self.info_map[task_to_calculate]['parent_start_time'] = parent_pre_d_time + t_idle_parent
            self.info_map[task_to_calculate]['attach_time'] = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus

            self.plot_info[task_to_calculate]['carrier_departure_time'] = parent_pre_d_time + t_idle_parent
            self.plot_info[task_to_calculate]['attach_start_time'] = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus
            self.plot_info[task_to_calculate]['attach_end_time'] = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Util.config.T_couple
            self.plot_info[task_to_calculate]['arrive_source_time'] = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Util.config.T_couple + t_i_minus2j_s
            self.plot_info[task_to_calculate]['pick_end_time'] = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Util.config.T_couple + t_i_minus2j_s + Util.config.T_load
            self.plot_info[task_to_calculate]['arrive_destination_time'] = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Util.config.T_couple + t_i_minus2j_s + Util.config.T_load + t_j_s2j_d
            self.plot_info[task_to_calculate]['detach_end_time'] = parent_pre_d_time + t_idle_parent + t_i_plus2i_minus + Util.config.T_couple + t_i_minus2j_s + Util.config.T_load + t_j_s2j_d + Util.config.T_decouple
            self.plot_info[task_to_calculate]['shuttle_complete_time'] = t_e





            self.info_map[task_to_calculate]['parent_decouple_time'] = t_d
            self.info_map[task_to_calculate]['child_end_time'] = t_e

            parent_next_task = self.sequence_map[task_to_calculate]['parent_next_task']
            child_next_task = self.sequence_map[task_to_calculate]['child_next_task']

            task_distance = (t_i_plus2i_minus + t_i_minus2j_s + t_j_s2j_d) * Util.config.V
            task_tardiness = max(0, t_e - self.instance[task_to_calculate - 1][5])

            self.plot_info[task_to_calculate]['distance'] = task_distance
            self.plot_info[task_to_calculate]['tardiness'] = task_tardiness


            self.info_map[task_to_calculate]['distance'] = task_distance
            self.info_map[task_to_calculate]['tardiness'] = task_tardiness

            self.info_map[task_to_calculate]['cost'] = task_distance * Util.config.weight + task_tardiness * (1 - Util.config.weight)

            self.info_map[task_to_calculate]['misalignment'] = abs(child_pre_e_time - parent_pre_d_time - t_i_plus2i_minus)

            total_distance += task_distance
            total_tardiness += task_tardiness

            self.fixed_distance += (t_j_s2j_d * Util.config.V)
            self.unfixed_distance += ((t_i_plus2i_minus+t_i_minus2j_s) * Util.config.V)

            if parent_next_task != 0:
                self.info_map[parent_next_task]['parent_pre_d_time'] = t_d
                task_parent_closed_set.add(parent_next_task)

            if child_next_task != 0:
                self.info_map[child_next_task]['child_pre_e_time'] = t_e
                task_child_closed_set.add(child_next_task)

            task_closed.append(task_to_calculate)
            task_parent_closed_set.remove(task_to_calculate)
            task_child_closed_set.remove(task_to_calculate)

        self.fitness = total_distance * Util.config.weight + total_tardiness * (1 - Util.config.weight)
        self.distance = total_distance
        self.tardiness = total_tardiness
        self.feasible = True
        return self.fitness


    def get_path_init_task_map(self):
        
        return copy_dict_int_int(self.path_init_task_map)

    def get_sequence_map(self):
        
        return copy_dict_int_dict(self.sequence_map)





if __name__ == "__main__":
    from Util.load_data import read_excel
    instance = read_excel("real_world/size_10/T10_I1.xlsx")
    path_map = {1: [2, 1, 10, 3, 6, 9, 8],
                 2: [5],
                 3: [],
                 4: [4, 7],
                 5: [7],
                 6: [2],
                 7: [9],
                 8: [8],
                 9: [],
                 10: [5, 10],
                 11: [1],
                 12: [3, 6, 4]}
    seq, init = path_map2sequence_map(path_map)
    solution = Solution(instance, seq, init)
    print(solution.get_fitness())
