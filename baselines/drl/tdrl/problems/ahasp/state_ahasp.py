import torch
from typing import NamedTuple, Dict
from utils.boolmask import mask_long2bool, mask_long_scatter
from problems.ahasp.paramet_ahasp import paramet_ahasp


class StateAHASP(NamedTuple):
    # Fixed input
    source_coords: torch.Tensor            
    destination_coords: torch.Tensor       
    deadline: torch.Tensor          
    operation_time: torch.Tensor    

    ids: torch.Tensor  # Keeps track of original fixed data index of rows


    
    cur_time: torch.Tensor    
    cur_coord: torch.Tensor   
    visited_: torch.Tensor    
    length: torch.Tensor      
    tardiness: torch.Tensor   
    i: torch.Tensor           # Keeps track of step


    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.source_coords.size(1))


    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        source_coords = input['source']

        batch_size, n_loc, _ = source_coords.size()

        cur_time = torch.zeros(batch_size, paramet_ahasp.ROBOT_NUM, dtype=torch.float, device=source_coords.device)
        cur_coord = torch.zeros(batch_size, paramet_ahasp.ROBOT_NUM, 2, dtype=torch.float, device=source_coords.device)
        length = torch.zeros(batch_size, dtype=torch.float, device=source_coords.device)
        tardiness = torch.zeros(batch_size, 1, dtype=torch.float, device=source_coords.device)

        return StateAHASP(
            source_coords=source_coords,
            destination_coords=input['destination'],
            deadline=input['deadline'],
            operation_time=input['operation_time'],
            ids=torch.arange(batch_size, dtype=torch.int64, device=source_coords.device)[:, None],  # Add steps dimension
            # prev_task=prev_task,
            cur_time=cur_time,
            cur_coord=cur_coord,
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc,  
                    dtype=torch.uint8, device=source_coords.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=source_coords.device)  # Ceil
            ),
            length=length,
            tardiness=tardiness,
            i=torch.zeros(1, dtype=torch.int64, device=source_coords.device)  # Vector with length num_steps
        )

    def update(self, selected_task, selected_robot_one_hot):

        assert self.i.size(0) == 1, "Can only update if state represents single step"
        selected_robot_one_hot = selected_robot_one_hot.bool()

        updated_time = self.cur_time.clone()
        cur_coord = self.cur_coord.clone()  

        selected_task_source_coord = self.source_coords[self.ids.squeeze(), selected_task]
        selected_task_destination_coord = self.destination_coords[self.ids.squeeze(), selected_task]

        

        
        
        parent_robot_coord = cur_coord[:, :4, :]  

        
        child_robot_coord = cur_coord[:, 4:, :]  

        
        
        selected_parent_coord = parent_robot_coord[selected_robot_one_hot[:, :4]]  
        selected_child_coord = child_robot_coord[selected_robot_one_hot[:, 4:]]  

        
        distance_to_child_robot = (selected_parent_coord - selected_child_coord).norm(p=1, dim=-1)

        
        distance_to_task_source = (selected_child_coord - selected_task_source_coord).norm(p=1, dim=-1)

        
        distance_from_task_source_to_destination = (selected_task_source_coord - selected_task_destination_coord).norm(
            p=1, dim=-1)

        
        total_distance = distance_to_child_robot + distance_to_task_source + distance_from_task_source_to_destination
        length = self.length + total_distance

        
        
        duration_to_child_robot = distance_to_child_robot / paramet_ahasp.ROBOT_VELOCITY
        
        selected_parent_time = self.cur_time[:, :4][selected_robot_one_hot[:, :4]]  
        selected_child_time = self.cur_time[:, 4:][selected_robot_one_hot[:, 4:]]  

        
        
        coupling_time = torch.max(selected_child_time, selected_parent_time + duration_to_child_robot) + paramet_ahasp.T_couple

        
        duration_to_task_source = (selected_child_coord - selected_task_source_coord).norm(p=1, dim=-1) / paramet_ahasp.ROBOT_VELOCITY

        
        duration_source_to_destination = (selected_task_source_coord - selected_task_destination_coord).norm(p=1, dim=-1) / paramet_ahasp.ROBOT_VELOCITY
        decoupling_time = coupling_time + duration_to_task_source + paramet_ahasp.T_load + duration_source_to_destination + paramet_ahasp.T_decouple

        
        duration_operation_for_selected_task = self.operation_time[self.ids.squeeze(), selected_task]
        child_complete_time = decoupling_time + duration_operation_for_selected_task

        selected_task_deadline = self.deadline[self.ids.squeeze(), selected_task]
        selected_task_tardiness = torch.clamp_min(child_complete_time - selected_task_deadline, 0)

        # mask = selected_robot_one_hot.unsqueeze(-1)  # (B, R, 1)
        
        cur_coord[selected_robot_one_hot] = selected_task_destination_coord.unsqueeze(1).expand(-1, cur_coord.size(1), -1)[selected_robot_one_hot]
        
        updated_time[:, :4][selected_robot_one_hot[:, :4]] = decoupling_time
        
        updated_time[:, 4:][selected_robot_one_hot[:, 4:]] = child_complete_time



        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, selected_task[:, None, None], 1)  
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, selected_task[:, None])

        return self._replace(
            cur_time=updated_time, cur_coord=cur_coord, visited_=visited_,
            length=length, i=self.i + 1, tardiness=self.tardiness + selected_task_tardiness.unsqueeze(-1),
        )

    def all_finished(self):
        return self.i.item() >= self.source_coords.size(1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_task

    def get_mask(self):
        visited = self.visited_.to(torch.bool)
        return visited

