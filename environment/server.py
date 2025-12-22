import numpy as np
from .queues import ProcessingQueue,OffloadingQueue,PublicQueueManager


class Server():
    def __init__(self, 
                 id :int, 
                 private_queue_computational_capacity :float,
                 public_queues_computational_capacity :float,
                 outbound_connections,
                 inbound_connections,
                 horizontal_rate:float,
                 vertical_rate:float,
                 slot_duration:float):
        self.id=id
        self.private_queue_computational_capacity = private_queue_computational_capacity
        self.public_queues_computational_capacity = public_queues_computational_capacity
       
        
        self.processing_queue = ProcessingQueue(self.private_queue_computational_capacity,
                                                slot_duration=slot_duration)
        
        outbound_connections = np.array(outbound_connections)
        self.offloading_servers = np.where(outbound_connections!=0)[0]
        self.offloading_capacities = {s:outbound_connections[s] for s in self.offloading_servers}
        self.offloading_queue = OffloadingQueue(offloading_capacities = self.offloading_capacities,
                                                horizontal_rate=horizontal_rate,
                                                vertical_rate=vertical_rate,
                                                slot_duration=slot_duration)

        inbound_connections = np.array(inbound_connections)
        self.supporting_servers =  np.where(inbound_connections!=0)[0]
        self.public_queue_manager = PublicQueueManager(id=self.id,
                                                       computational_capacity=  self.public_queues_computational_capacity,
                                                       supporting_servers= self.supporting_servers,
                                                       slot_duration=slot_duration)
        self.current_time=0

    def reset(self):
            self.current_time=0
            self.processing_queue.reset()
            self.public_queue_manager.reset()
            self.offloading_queue.reset()   
    
    def get_waiting_times(self,current_time:int=0):
        return  self.processing_queue.get_waiting_time(current_time),self.offloading_queue.get_waiting_time(current_time)
    
    def add_offloaded_tasks(self,offloaded_tasks,current_time:int):
        self.public_queue_manager.add_tasks(offloaded_tasks,current_time=current_time)

    def step(self,action=None,local_task=None, current_time:int=0):
        transmited_task = None
        local_reward = 0.0
        events = []
        self.current_time = current_time
        if local_task:
            local_task.set_origin_server_id(self.id)
            if action ==self.id:  
                _, add_reward = self.processing_queue.add_task(local_task, current_time)
                local_reward += add_reward
            else:
                target_server_id = action
                local_task.set_target_server_id(target_server_id)
                transmited_task, add_reward = self.offloading_queue.add_task(local_task, current_time)
                local_reward += add_reward
        # process one slot for local processing and offloading queues
        step_reward, proc_events = self.processing_queue.step(current_time)
        local_reward += step_reward
        events.extend(proc_events)
        tx_task, off_reward, off_events = self.offloading_queue.step(current_time)
        if tx_task:
            transmited_task = tx_task
        local_reward += off_reward
        events.extend(off_events)

        foreign_rewards =  {}  # public processing handled in add_tasks
        total_rewards=  foreign_rewards
        if local_reward != 0:
            total_rewards[self.id] = local_reward
        return transmited_task,total_rewards,events
    
    def get_features(self,current_time:int=0):
        private_waiting_time,public_waiting_time = self.get_waiting_times(current_time)
        public_queues = self.public_queue_manager.get_queue_lengths()
        return np.array([private_waiting_time,
                         public_waiting_time]),public_queues
    
    def get_number_of_features(self):
        features,_  = self.get_features()
        return len(features)
    def get_number_of_actions(self):
        return 1+len(self.offloading_servers)
    
    def get_offliading_servers(self):
        return self.offloading_servers
    
    
    def get_active_queues(self,current_time:int):
        active_queues  =self.public_queue_manager.get_active_queues(current_time)
        return active_queues
    
    
    def get_supporting_servers(self):
        return self.supporting_servers
