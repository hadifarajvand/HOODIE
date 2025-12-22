from .decision_maker_base import DescisionMakerBase
import numpy as np


class RuleBased(DescisionMakerBase):
    def __init__(self, number_of_actions,local_cpu,foreign_cpus, possible_actions, task_feature_len, size_max, timeout_max, density_max,*args, **kwargs):
        self.number_of_actions = number_of_actions
        self.local_cpu = local_cpu
        self.foreign_cpus = foreign_cpus
        self.possible_actions = possible_actions
        self.task_feature_len = task_feature_len
        self.size_max = size_max
        self.timeout_max = timeout_max
        self.density_max = density_max


    def choose_action(self, observation,public_queues ,*args, **kwargs):
        actions = np.zeros(self.number_of_actions, dtype=float)
        # Recover unnormalized task attributes and waiting times
        task_size_norm = observation[0]
        priority_norm = observation[1]
        timeout_norm = observation[2]
        density_norm = observation[3]
        task_size = task_size_norm * self.size_max
        task_timeout = timeout_norm * self.timeout_max
        task_density = max(density_norm * self.density_max, 1e-6)
        local_waiting_time = observation[self.task_feature_len] * self.timeout_max
        offloading_waiting_time = observation[self.task_feature_len + 1] * self.timeout_max
        local_processing_time = (task_size * task_density) / max(self.local_cpu, 1e-6)
        actions[0] = local_waiting_time + local_processing_time
        
        
        for i in range(1, self.number_of_actions):
            target_id = self.possible_actions[i]
            queue_load = max(public_queues[target_id], 1)
            cpu_capacity = self.foreign_cpus[i-1] / queue_load
            foreign_processing_time = (task_size * task_density) / max(cpu_capacity, 1e-6)
            actions[i]  = offloading_waiting_time + foreign_processing_time
        return np.argmin(actions)
