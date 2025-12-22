import numpy as np
from .queues import PublicQueueManager
 
 
class Cloud:
    def __init__(self,
                 number_of_servers ,
                 computational_capacity,
                 slot_duration=1.0):
        self.number_of_servers=  number_of_servers
        self.computational_capacity = computational_capacity
        
        self.current_time=0
        self.supporting_servers = np.arange(self.number_of_servers)
        self.public_queue_manager = PublicQueueManager(id=self.number_of_servers,
                                                       computational_capacity=  self.computational_capacity,
                                                       supporting_servers= self.supporting_servers,
                                                       slot_duration=slot_duration)
    def reset(self):
        self.current_time=0
        self.public_queue_manager.reset()

    
    def step(self,current_time:int):
        rewards = self.public_queue_manager.step(current_time)
        return rewards
    
    def add_offloaded_tasks(self,offloaded_tasks,current_time:int=0):
        self.public_queue_manager.add_tasks(offloaded_tasks,current_time=current_time)

    def get_features(self):
        return self.public_queue_manager.get_queue_lengths()
    
 
    def get_active_queues(self,current_time:int):
        active_queues =self.public_queue_manager.get_active_queues(current_time)
        return active_queues
    
    def get_supporting_servers(self):
        return self.supporting_servers
