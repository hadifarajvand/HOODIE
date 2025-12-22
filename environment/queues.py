from collections import deque
from .task import Task
import math

TIME_SLOT_DURATION = 1.0  # A in the paper (seconds)


class ProcessingQueue:
    def __init__(self,computational_capacity, slot_duration=TIME_SLOT_DURATION):
        self.computational_capacity = computational_capacity
        self.slot_duration = slot_duration
        self.queue = deque()
        self.queue_length = 0.0
        self.reset()
    def reset(self):
        self.queue.clear()
        self.queue_length = 0.0
        self.last_completion_time = -1
    def add_task(self,task:Task,current_time:int):
        t = task.copy()
        capacity_per_slot = max(self.computational_capacity * self.slot_duration, 1e-6)
        service_slots = max(1, math.ceil((t.get_remaining_size() * t.get_density()) / capacity_per_slot))
        start_time = max(self.last_completion_time + 1, current_time)
        completion_time = start_time + service_slots - 1
        finish_time = min(completion_time, t.get_timeout())
        self.last_completion_time = finish_time
        self.queue.append({
            "task": t,
            "planned_completion": completion_time,
            "finish_time": finish_time
        })
        self.queue_length += t.get_remaining_size()
        return None, 0.0
    def step(self,current_time:int):
        reward = 0.0
        events = []
        while self.queue and current_time >= self.queue[0]["finish_time"]:
            entry = self.queue.popleft()
            t = entry["task"]
            self.queue_length -= t.get_remaining_size()
            if entry["planned_completion"] <= t.get_timeout():
                delay = entry["planned_completion"] - t.get_arrival_time()
                reward += -delay
                events.append({"origin": t.get_origin_server_id(), "delay": delay, "dropped": False, "reward": -delay})
            else:
                reward += -t.drop_penalty
                events.append({"origin": t.get_origin_server_id(), "delay": None, "dropped": True, "reward": -t.drop_penalty})
        return reward, events
    def get_waiting_time(self,current_time:int):
        if not self.queue:
            return 0.0
        return max(0.0, self.last_completion_time - current_time + 1)


class OffloadingQueue:
    def __init__(self,offloading_capacities, horizontal_rate, vertical_rate, slot_duration):   
        self.offloading_capacities = offloading_capacities
        self.horizontal_rate = horizontal_rate
        self.vertical_rate = vertical_rate
        self.slot_duration = slot_duration
        self.queue = deque()
        self.queue_length = 0.0
        self.reset()
    def reset(self):
        self.queue.clear()
        self.queue_length = 0.0
        self.last_completion_time = -1
    def get_effective_rate(self, target_server_id):
        if target_server_id == max(self.offloading_capacities.keys()):
            return self.vertical_rate
        weight = self.offloading_capacities.get(target_server_id, 0.0)
        return weight * self.horizontal_rate
    def add_task(self,task:Task,current_time:int):
        t = task.copy()
        rate = max(self.get_effective_rate(t.get_target_server_id()) * self.slot_duration, 1e-6)
        service_slots = max(1, math.ceil(t.get_remaining_size() / rate))
        start_time = max(self.last_completion_time + 1, current_time)
        completion_time = start_time + service_slots - 1
        finish_time = min(completion_time, t.get_timeout())
        self.last_completion_time = finish_time
        self.queue.append({
            "task": t,
            "planned_completion": completion_time,
            "finish_time": finish_time
        })
        self.queue_length += t.get_remaining_size()
        return None, 0.0
    def step(self,current_time:int):
        transmitted_task = None
        reward = 0.0
        events = []
        while self.queue and current_time >= self.queue[0]["finish_time"]:
            entry = self.queue.popleft()
            t = entry["task"]
            self.queue_length -= t.get_remaining_size()
            if entry["planned_completion"] <= t.get_timeout():
                transmitted_task = t.copy()
                transmitted_task.arrival_time = entry["finish_time"] + 1
                elapsed = entry["finish_time"] - t.get_arrival_time()
                transmitted_task.timeout_delay = max(0, t.timeout_delay - elapsed)
            else:
                reward += -t.drop_penalty
                events.append({"origin": t.get_origin_server_id(), "delay": None, "dropped": True, "reward": -t.drop_penalty})
        return transmitted_task, reward, events
    def get_waiting_time(self,current_time:int):
        if not self.queue:
            return 0.0
        return max(0.0, self.last_completion_time - current_time + 1)


class PublicQueue:
    def __init__(self, slot_duration=TIME_SLOT_DURATION):
        self.queue_length = 0.0
        self.queue = deque()
        self.slot_duration = slot_duration
        self.last_completion_time = -1
    def reset(self):
        self.queue_length = 0.0
        self.queue.clear()
        self.last_completion_time = -1
    def add_task(self,task:Task,current_time:int,computational_capacity:float):
        t = task.copy()
        self.queue.append(t)
        self.queue_length += t.get_remaining_size()
    def step(self,computational_capacity,current_time:int,active_queues:int):
        reward = 0.0
        events = []
        # drop expired tasks
        while self.queue and current_time >= self.queue[0].get_timeout():
            expired = self.queue.popleft()
            self.queue_length -= expired.get_remaining_size()
            reward += -expired.drop_penalty
            events.append({"origin": expired.get_origin_server_id(), "delay": None, "dropped": True, "reward": -expired.drop_penalty})
        if not self.queue or computational_capacity<=0 or active_queues<=0:
            return reward, events
        # dynamic completion recomputation: process head with current share until either head finishes or share exhausted
        share = self.slot_duration * computational_capacity / max(active_queues,1)
        remaining_share = share
        while self.queue and remaining_share > 0:
            t = self.queue[0]
            work = remaining_share / max(t.get_density(),1e-6)
            consume = min(work, t.get_remaining_size())
            self.queue_length -= consume
            t.remain -= consume
            remaining_share -= consume * max(t.get_density(),1e-6)
            if t.get_remaining_size() <= 0:
                delay = current_time - t.get_arrival_time()
                reward += -delay
                events.append({"origin": t.get_origin_server_id(), "delay": delay, "dropped": False, "reward": -delay})
                self.queue.popleft()
            else:
                break
        return reward, events

    def get_queue_length(self):
        return self.queue_length


class PublicQueueManager():
    def __init__(self,
                 id,
                 computational_capacity,
                 supporting_servers,
                 slot_duration=TIME_SLOT_DURATION):
        self.id = id
        self.computational_capacity = computational_capacity
        self.supporting_servers = supporting_servers
        self.public_queues ={}
        for server_id in self.supporting_servers:
            self.public_queues[server_id] = PublicQueue(slot_duration=slot_duration)
       
    
    def reset(self):
        for _,q in self.public_queues.items():
            q.reset()
    
    def get_public_queue_server_length(self,server_id):
        return self.public_queues[server_id].get_queue_length()
    
    def get_active_queues(self,current_time:int):
        active_queues =0
        for _,q in self.public_queues.items():
            if len(q.queue) > 0:
                active_queues +=1
        return active_queues
        
    def add_tasks(self,recieved_tasks=[],current_time:int=0):
        rewards = {}
        if not recieved_tasks:
            return rewards, []
        events = []
        for task in recieved_tasks:
            assert task.get_target_server_id() == self.id
            origin_server_id = task.get_origin_server_id()
            self.public_queues[origin_server_id].add_task(task,current_time,0.0)
        active = self.get_active_queues(current_time)
        for origin_id,q in self.public_queues.items():
            reward, ev = q.step(self.computational_capacity,current_time,active)
            if reward !=0:
                rewards[origin_id] = rewards.get(origin_id,0) + reward
            events.extend(ev)
        return rewards, events
    
    def step(self,current_time:int):
        rewards = {}
        events = []
        active = self.get_active_queues(current_time)
        for origin_id,q in self.public_queues.items():
            reward, ev = q.step(self.computational_capacity,current_time,active)
            if reward !=0:
                rewards[origin_id] = rewards.get(origin_id,0) + reward
            events.extend(ev)
        return rewards, events
    
    
    def get_queue_lengths(self):
        queue_lengths = {}
        for server_id in self.supporting_servers:
            queue_lengths[server_id] = self.public_queues[server_id].get_queue_length()
        return queue_lengths
    
