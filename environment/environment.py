from collections import deque
from .server import Server
from .cloud import Cloud
from .task_generator import TaskGenerator
from .matchmaker import Matchmaker
from utils import merge_dicts,dict_to_array,remove_diagonal_and_reshape
import numpy as np
import math
import torch
class Environment():
    def _compute_clusters(self):
        adjacency = np.array(self.connection_matrix)[:self.number_of_servers, :self.number_of_servers]
        cluster_ids = [-1] * self.number_of_servers
        cid = 0
        for i in range(self.number_of_servers):
            if cluster_ids[i] != -1:
                continue
            stack = [i]
            cluster_ids[i] = cid
            while stack:
                node = stack.pop()
                neighbors = np.where(adjacency[node] > 0)[0]
                for nb in neighbors:
                    if cluster_ids[nb] == -1:
                        cluster_ids[nb] = cid
                        stack.append(nb)
            cid += 1
        # extra cluster for Cloud (even though origins remain servers)
        num_clusters = cid + 1
        return cluster_ids, num_clusters
    def __init__(self, 
                 static_frequency,
                 number_of_servers,
                 private_cpu_capacities,
                 public_cpu_capacities,
                 connection_matrix,
                 cloud_computational_capacity,
                 episode_time,
                 task_arrive_probabilities,
                 task_size_mins,
                 task_size_maxs,
                 task_size_distributions,
                timeout_delay_mins,
                timeout_delay_maxs,
                timeout_delay_distributions,
                priotiry_mins,
                priotiry_maxs,
                priotiry_distributions,
                computational_density_mins,
                computational_density_maxs,
                computational_density_distributions,
                 drop_penalty_mins,
                 drop_penalty_maxs,
                 drop_penalty_distributions,  
                 horizontal_rate=10.0,
                 vertical_rate=30.0,
                 slot_duration=1.0,
                 delay_prob=0.1,
                 max_delay=1,
                 ec_link_delay=1,
                 lookback_window=10,
                 recovery_period=None,
                 number_of_clouds=1) -> None:
        self.number_of_servers = number_of_servers
        self.number_of_clouds = number_of_clouds
        self.horizontal_rate = horizontal_rate
        self.vertical_rate = vertical_rate
        self.slot_duration = slot_duration
        self.delay_prob = delay_prob
        self.max_delay = max_delay
        self.ec_link_delay = ec_link_delay
        self.recovery_period = recovery_period if recovery_period is not None else max(1, max_delay * 2)
        self.current_time = 0
        self.episode_time_end = episode_time +max(timeout_delay_maxs)
        self.connection_matrix=  connection_matrix
        self.lookback_window = lookback_window
        self.rho_length = (self.number_of_servers + self.number_of_clouds) * self.number_of_servers
        self.cluster_ids, self.num_clusters = self._compute_clusters()
        get_column = lambda m, i: [row[i] for row in m]
        self.task_generators = [TaskGenerator(id=i,
                                              episode_time=episode_time,
                                              task_arrive_probability=task_arrive_probabilities[i],
                                              size_min=task_size_mins[i],
                                              size_max = task_size_maxs[i],
                                                size_distribution = task_size_distributions[i],
                                                timeout_delay_min = timeout_delay_mins[i],
                                                timeout_delay_max = timeout_delay_maxs[i],
                                                timeout_delay_distribution = timeout_delay_distributions[i],
                                                priotiry_min = priotiry_mins[i],
                                                priotiry_max = priotiry_maxs[i],
                                                priotiry_distribution = priotiry_distributions[i],
                                                computational_density_min = computational_density_mins[i],
                                                computational_density_max = computational_density_maxs[i],
                                                computational_density_distribution = computational_density_distributions[i],
                                                drop_penalty_min = drop_penalty_mins[i],
                                                drop_penalty_max = drop_penalty_maxs[i],
                                                drop_penalty_distribution = drop_penalty_distributions[i])
                                for i in range(number_of_servers)]         
        self.servers = [Server( id=i,
                                private_queue_computational_capacity=  private_cpu_capacities[i],
                                public_queues_computational_capacity= public_cpu_capacities[i],
                                outbound_connections=  self.connection_matrix[i],
                                inbound_connections=get_column(self.connection_matrix,i),
                                horizontal_rate=self.horizontal_rate,
                                vertical_rate=self.vertical_rate,
                                slot_duration=self.slot_duration) 
                        for i in range(number_of_servers)]
       
        self.matchmakers = [Matchmaker(id=s.id,
                                       offloading_servers=s.get_offliading_servers())
                            for s in self.servers]
        self.cloud = Cloud(number_of_servers=number_of_servers,
                           computational_capacity=cloud_computational_capacity,
                           slot_duration=self.slot_duration)
        
        
        self.number_of_task_features=  self.task_generators[0].generate().get_number_of_features()
        self.number_of_server_features = self.servers[0].get_number_of_features()
        self.number_of_features = self.number_of_task_features + self.number_of_server_features
        self.static_frequency = static_frequency
        self.static_counter = 0
        
        self.max_waiting_time = max(timeout_delay_maxs)
        self.get_task_features_maxs()
        self.load_history = None
        self.load_cache = None
        self.delay_counters = None
        self.ec_cache = None
        self.ec_delay_counters = None
        self.ec_pending = None
        self.metrics = None
        self.reset()
    def reset(self):
        
        
        
        if self.static_frequency:
            if self.static_counter % self.static_frequency ==0:
                np.random.seed(42)
                torch.manual_seed(42)
                self.static_counter+=1
        self.current_time = 0
        self.load_history = deque([np.zeros(self.rho_length, dtype=np.float32) for _ in range(self.lookback_window)],
                                  maxlen=self.lookback_window)
        self.load_cache = np.zeros(self.rho_length, dtype=np.float32)
        self.delay_counters = np.zeros(self.rho_length, dtype=int)
        self.ec_cache = np.zeros(self.rho_length, dtype=np.float32)
        self.ec_delay_counters = np.zeros(self.rho_length, dtype=int)
        self.ec_last_broadcast = np.zeros(self.num_clusters, dtype=int)
        self.ec_pending = []
        self.ec_pending = []  # list of tuples (deliver_time, source_cluster, payload)
        self.metrics = {"processed": 0, "dropped": 0, "delay_sum": 0.0}
        for task_generator in self.task_generators:
            task_generator.reset()
        for server in self.servers:
            server.reset()
        self.cloud.reset()
        self.reset_transmitted_tasks()
        self.tasks= [t.step() for t in self.task_generators]
        
        observations = self.pack_observation()
        done = False
        info = {}
        self.actions  =[{
                'local':0,
                'horisontal':0,
                'cloud':0
            }
            for _ in range(self.number_of_servers)]
        return observations,done, info
    def reset_transmitted_tasks(self):
        self.horisontal_transmitted_tasks = [[] for _ in range(self.number_of_servers+self.number_of_clouds)]
    
    def scale_rewards(self,reward):
        return reward/self.max_reward
    
    def get_task_features_maxs(self):
        self.feature_maxes = self.task_generators[0].get_maxs()
        for g in self.task_generators:
            np.maximum(self.feature_maxes, g.get_maxs(), out=self.feature_maxes)  # Compare and store the max values

    def scale_task_features(self,task_features):
        return task_features/self.feature_maxes
    def scale_waiting_times(self,waiting_times):
        return waiting_times/self.max_waiting_time
    def pack_observation(self):
        server_observations = np.zeros((self.number_of_servers,self.number_of_features))
        public_queues_legth  = [np.array([]) for key in range(self.number_of_servers+self.number_of_clouds)]
        assert len(self.tasks) == self.number_of_servers
        for s in range(self.number_of_servers):
            if self.tasks[s]:
                task_features = self.tasks[s].get_features()
            else:
                task_features = np.zeros(self.number_of_task_features)
            task_features = self.scale_task_features(task_features)
            waiting_times,server_public_queues = self.servers[s].get_features(self.current_time)     
            waiting_times = self.scale_waiting_times(waiting_times)       
            server_features = np.concatenate([task_features,waiting_times])
            server_observations[s] = server_features
        
            for q in server_public_queues:
                public_queues_legth[q] = np.append(public_queues_legth[q], server_public_queues[q])

        cloud_public_queues = self.cloud.get_features()
        for q in cloud_public_queues:
            public_queues_legth[q] = np.append(public_queues_legth[q], cloud_public_queues[q])      
        
        # Per-node public queue lengths (rho) summed in bits
        queue_length_vector = []
        for server in self.servers:
            queue_length_vector.append(sum(server.public_queue_manager.get_queue_lengths().values()))
        queue_length_vector.append(sum(self.cloud.public_queue_manager.get_queue_lengths().values()))
        queue_length_vector = np.array(queue_length_vector, dtype=np.float32)

        # Flatten per-queue lengths (rho_{origin,target}) for richer state
        rho_matrix = np.zeros((self.number_of_servers + self.number_of_clouds, self.number_of_servers), dtype=np.float32)
        for idx, server in enumerate(self.servers):
            lengths = server.public_queue_manager.get_queue_lengths()
            for origin, length in lengths.items():
                rho_matrix[idx, origin] = length
        cloud_lengths = self.cloud.public_queue_manager.get_queue_lengths()
        for origin, length in cloud_lengths.items():
            rho_matrix[-1, origin] = length
        rho_flat = rho_matrix.flatten()

        # Pub-Sub delay with recovery on per-queue load vector (cluster-correlated)
        observed_rho = self._simulate_ec_messages(rho_flat)
        # Track true per-queue lengths in history (shape: W x (N+1)*N) for the LSTM input
        self.load_history.append(rho_flat.copy())
        load_history_matrix = np.stack(self.load_history, axis=0)

        local_observations = []
        for i in range(len(server_observations)):
            augmented = np.concatenate([
                server_observations[i],
                public_queues_legth[i],
                queue_length_vector,
                rho_flat,
                observed_rho
            ])
            local_observations.append(augmented)
            
        return local_observations,load_history_matrix

    def _simulate_ec_messages(self, rho_flat: np.ndarray) -> np.ndarray:
        """
        Simulate EC-based pub-sub: each cluster EC refreshes its view unless delayed; periodic recovery forces refresh.
        """
        observed = np.zeros_like(rho_flat)
        # schedule new EC broadcasts with link delay
        for cluster in range(self.num_clusters):
            payload = []
            for row_idx in range(self.number_of_servers + self.number_of_clouds):
                if self.cluster_ids[row_idx % self.number_of_servers] != cluster:
                    continue
                start = row_idx * self.number_of_servers
                end = start + self.number_of_servers
                payload.extend(rho_flat[start:end].tolist())
            deliver_time = self.current_time + self.ec_link_delay
            self.ec_pending.append((deliver_time, cluster, payload))
        # deliver pending messages
        remaining = []
        for deliver_time, cluster, payload in self.ec_pending:
            if deliver_time <= self.current_time:
                # apply payload to all rows owned by cluster
                idx_payload = 0
                for row_idx in range(self.number_of_servers + self.number_of_clouds):
                    if self.cluster_ids[row_idx % self.number_of_servers] != cluster:
                        continue
                    start = row_idx * self.number_of_servers
                    end = start + self.number_of_servers
                    self.ec_cache[start:end] = payload[idx_payload:idx_payload + self.number_of_servers]
                    self.ec_delay_counters[start:end] = 0
                    idx_payload += self.number_of_servers
                self.ec_last_broadcast[cluster] = self.current_time
            else:
                remaining.append((deliver_time, cluster, payload))
        self.ec_pending = remaining
        # max_delay recovery
        for cluster in range(self.num_clusters):
            if (self.current_time - self.ec_last_broadcast[cluster]) > self.max_delay:
                for row_idx in range(self.number_of_servers + self.number_of_clouds):
                    if self.cluster_ids[row_idx % self.number_of_servers] != cluster:
                        continue
                    start = row_idx * self.number_of_servers
                    end = start + self.number_of_servers
                    self.ec_cache[start:end] = rho_flat[start:end]
                    self.ec_delay_counters[start:end] = 0
                self.ec_last_broadcast[cluster] = self.current_time
        observed[:] = self.ec_cache
        return observed
    
    def add_action_info(self,action,server_id,task):
        if task:
            if action ==server_id:
                self.actions[server_id]['local'] +=1
            elif action == self.number_of_servers:
                self.actions[server_id]['cloud'] +=1
            else:
                self.actions[server_id]['horisontal'] +=1
    def step(self,actions):


        tasks_arrived = [0 if t is None else 1 for t in self.tasks]
        rewards = {}
        task_events = []
        
        assert len(actions) == self.number_of_servers
        if self.current_time >=self.episode_time_end:
            done = True
        else:
            done = False
        self.current_time +=1
        
        for s in self.servers:
            s.add_offloaded_tasks(self.horisontal_transmitted_tasks[s.id],current_time=self.current_time)
        self.cloud.add_offloaded_tasks(self.horisontal_transmitted_tasks[-1],current_time=self.current_time)
        self.reset_transmitted_tasks()

        for server_id in range(self.number_of_servers):
            action = self.matchmakers[server_id].match_action(server_id,actions[server_id])
            self.add_action_info(action,server_id,self.tasks[server_id])
            transmited_task, server_reward, ev = self.servers[server_id].step(action,self.tasks[server_id], current_time=self.current_time)
            rewards = merge_dicts(rewards,server_reward)
            task_events.extend(ev)
            if transmited_task:
                origin_server_id = transmited_task.get_origin_server_id()
                assert origin_server_id == server_id
                target_server_id = transmited_task.get_target_server_id()
                
                self.horisontal_transmitted_tasks[target_server_id].append(transmited_task) 
        # process offloaded tasks deterministically at destinations and advance public queues
        for target_id in range(self.number_of_servers):
            incoming = self.horisontal_transmitted_tasks[target_id]
            if incoming:
                reward_updates, ev = self.servers[target_id].public_queue_manager.add_tasks(incoming,current_time=self.current_time)
                rewards = merge_dicts(rewards,reward_updates)
                task_events.extend(ev)
            # process any completions/drops even without new arrivals
            step_rewards, ev = self.servers[target_id].public_queue_manager.step(self.current_time)
            rewards = merge_dicts(rewards, step_rewards)
            task_events.extend(ev)
        cloud_incoming = self.horisontal_transmitted_tasks[-1]
        if cloud_incoming:
            reward_updates, ev = self.cloud.public_queue_manager.add_tasks(cloud_incoming,current_time=self.current_time)
            rewards = merge_dicts(rewards,reward_updates)
            task_events.extend(ev)
        cloud_step_rewards, ev = self.cloud.public_queue_manager.step(self.current_time)
        rewards = merge_dicts(rewards, cloud_step_rewards)
        task_events.extend(ev)

        self.tasks= [t.step() for t in self.task_generators]     
        # immediate drop if timeout already expired upon arrival
        for idx, t in enumerate(self.tasks):
            if t is None:
                continue
            if t.get_timeout() <= self.current_time:
                drop_reward = -t.drop_penalty
                rewards[idx] = rewards.get(idx, 0) + drop_reward
                task_events.append({"origin": idx, "delay": None, "dropped": True, "reward": drop_reward})
                self.tasks[idx] = None
       
        observations = self.pack_observation()
        rewards_array  = np.full(self.number_of_servers, np.nan, dtype=np.float32)
        for idx,val in rewards.items():
            rewards_array[idx] = val
        for ev in task_events:
            if ev["dropped"]:
                self.metrics["dropped"] += 1
            else:
                self.metrics["processed"] += 1
                if ev["delay"] is not None:
                    self.metrics["delay_sum"] += ev["delay"]
        
        info  ={}
        info['rewards'] = rewards_array
        info['tasks_arrived'] = np.array(tasks_arrived)
        info['tasks_dropped'] = np.zeros_like(rewards_array)
        info['task_events'] = task_events
        proc = self.metrics["processed"]
        avg_delay = self.metrics["delay_sum"]/proc if proc>0 else np.nan
        info['kpi'] = {
            "processed": proc,
            "dropped": self.metrics["dropped"],
            "avg_delay": avg_delay
        }
        
        return observations,rewards_array, done, info
        
    
    def get_server_dimensions(self,id):
        
        local_observations, load_history_matrix = self.pack_observation()
        local_observation = local_observations[id]
        return (
            len(local_observation),
            load_history_matrix.shape[1],
            self.servers[id].get_number_of_actions()
        )
    def get_task_features(self):
        return self.task_generators[0].get_number_of_features()
    
    def get_task_feature_count(self):
        return self.number_of_task_features
    
    def get_episode_actions(self):
        return self.actions
    
    def get_foreign_cpus(self,id):
        available_servers = np.array(self.matchmakers[id].get_rows())
        available_servers = available_servers[available_servers!=id]
        available_servers = available_servers[available_servers!=self.number_of_servers]
        public_cpus = np.array([s.public_queues_computational_capacity for s in self.servers])
        
        available_public_cpus = public_cpus[available_servers]
        
        available_public_cpus = np.append(available_public_cpus, self.cloud.computational_capacity)
        return available_public_cpus
        
        
        
