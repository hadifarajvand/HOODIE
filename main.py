from environment import Environment
from decision_makers import Agent, AllHorizontal, AllLocal, AllVertical,Random,SingleAgent,RoundRobin,RuleBased
from lr_schedulers import constant,Linear
import numpy as np
import argparse
import torch
import os
import json
import pickle
import matplotlib.pyplot as plt
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_folder', type=str, default='log_folder', help='Path to the log folder')
    parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='Path to the hyperparameters file')
    parser.add_argument('--epochs', type=int, default=2, help='Device to use')
    parser.add_argument('--validate', type=bool, default=False, help='Device to use')
    parser.add_argument('--plot', action='store_true', help='Save KPI plots to log folder')
    args  = parser.parse_args()

    os.makedirs(args.log_folder, exist_ok=True)
    with open(args.hyperparameters_file) as f:
        hyperparameters = json.load(f)
            
    number_of_servers=  hyperparameters['number_of_servers']
    env = Environment(
        number_of_servers=hyperparameters['number_of_servers'],
        private_cpu_capacities=hyperparameters['private_cpu_capacities'],
        public_cpu_capacities=hyperparameters['public_cpu_capacities'],
        connection_matrix=hyperparameters['connection_matrix'],
        cloud_computational_capacity=hyperparameters['cloud_computational_capacity'],
        episode_time=hyperparameters['episode_time'],
        static_frequency=hyperparameters['static_frequency'],
        task_arrive_probabilities=hyperparameters['task_arrive_probabilities'],
        task_size_mins=hyperparameters['task_size_mins'],
        task_size_maxs=hyperparameters['task_size_maxs'],
        task_size_distributions=hyperparameters['task_size_distributions'],
        timeout_delay_mins=hyperparameters['timeout_delay_mins'],
        timeout_delay_maxs=hyperparameters['timeout_delay_maxs'],
        timeout_delay_distributions=hyperparameters['timeout_delay_distributions'],
        priotiry_mins=hyperparameters['priotiry_mins'],
        priotiry_maxs=hyperparameters['priotiry_maxs'],
        priotiry_distributions=hyperparameters['priotiry_distributions'],
        computational_density_mins=hyperparameters['computational_density_mins'],
        computational_density_maxs=hyperparameters['computational_density_maxs'],
        computational_density_distributions=hyperparameters['computational_density_distributions'],
        drop_penalty_mins=hyperparameters['drop_penalty_mins'],
        drop_penalty_maxs=hyperparameters['drop_penalty_maxs'],
        drop_penalty_distributions=hyperparameters['drop_penalty_distributions'],
        horizontal_rate=hyperparameters['horizontal_rate'],
        vertical_rate=hyperparameters['vertical_rate'],
        slot_duration=hyperparameters['slot_duration'],
        delay_prob=hyperparameters['delay_prob'],
        max_delay=hyperparameters['max_delay'],
        lookback_window=hyperparameters['lstm_time_step']
    )
    
    
    scheduler_file= args.log_folder+'/scheduler.pth'
    scheduler_choices ={
        'constant': constant,
        'Linear': Linear(start=hyperparameters['learning_rate'],
                        end=hyperparameters['learning_rate_end'],
                            number_of_epochs=hyperparameters['lr_scheduler_epochs'])
        }
    
    scheduler =scheduler_choices[hyperparameters['scheduler_choice']]
    
    
    with open(scheduler_file, 'wb') as f:
        pickle.dump(scheduler, f)
    
    decision_makers = []
    
    decision_makers_choice ={
        'drl': Agent,
        'all_horizontal': AllHorizontal,
        'all_local': AllLocal,
        'all_vertical': AllVertical,
        'random': Random,
        'round_robin':RoundRobin,
        'rule_based':RuleBased,
        "single":SingleAgent
    }
    chosen_descision_maker = decision_makers_choice[hyperparameters['decision_makers']]

    decision_makers = []
    
    
    for i in range(number_of_servers):
        state_dimensions,foreign_queues,number_of_actions = env.get_server_dimensions(i)      
        lstm_shape = foreign_queues 

        decision_maker_params ={'number_of_actions': number_of_actions} 
        
        if hyperparameters['decision_makers'] == 'drl':
            decision_maker_params = {
                'id': i,
                'state_dimensions': state_dimensions,
                'lstm_shape': lstm_shape,
                'number_of_actions': number_of_actions,
                'hidden_layers': hyperparameters['hidden_layers'],
                'lstm_layers': hyperparameters['lstm_layers'],
                'lstm_time_step': hyperparameters['lstm_time_step'],
                'dropout_rate': hyperparameters['dropout_rate'],
                'dueling': hyperparameters['dueling'],
                'epsilon': hyperparameters['epsilon'],
                'epsilon_decrement': hyperparameters['epsilon_decrement'],
                'epsilon_end': hyperparameters['epsilon_end'],
                'gamma': hyperparameters['gamma'],
                'learning_rate': hyperparameters['learning_rate'],
                'scheduler_file':scheduler_file,
                'loss_function': getattr(torch.nn, hyperparameters['loss_function']),
                'optimizer': getattr(torch.optim, hyperparameters['optimizer']),
                'checkpoint_folder': args.log_folder+ '/agent_'+str(i)+'.pth',
                'save_model_frequency': hyperparameters['save_model_frequency'],
                'update_weight_percentage': hyperparameters['update_weight_percentage'],
                'memory_size': hyperparameters['memory_size'],
                'batch_size': hyperparameters['batch_size'],
                'replace_target_iter': hyperparameters['replace_target_iter'],
                'device': device
            }
        
        if hyperparameters['decision_makers'] == 'rule_based':
            foreign_cpus = env.get_foreign_cpus(i)
            possible_actions = env.matchmakers[i].get_rows()
            decision_maker_params = {
                'number_of_actions': number_of_actions,
                'local_cpu': hyperparameters['private_cpu_capacities'][i],
                'foreign_cpus':foreign_cpus,
                'possible_actions': possible_actions,
                'task_feature_len': env.get_task_feature_count(),
                'size_max': hyperparameters['task_size_maxs'][i],
                'timeout_max': hyperparameters['timeout_delay_maxs'][i],
                'density_max': hyperparameters['computational_density_maxs'][i]
            }
        
    
        decision_maker = chosen_descision_maker(**decision_maker_params)
        decision_makers.append(decision_maker)
        
        
        
    for key in hyperparameters:
        if key != 'connection_matrix':
            print(key ," : ",hyperparameters[key])
    total_epochs = args.epochs
    kpi_history = []
    for epoch in range(total_epochs):
        accumulated_rewards = []
        observations,done, info = env.reset()
        local_observations,load_history_matrix =observations
        last_kpi = None
        while not done:
            actions = np.zeros(number_of_servers, dtype=int)
            for i in range(number_of_servers):
                actions[i] = decision_makers[i].choose_action(local_observations[i],load_history_matrix)
            observations,rewards,done,info = env.step(actions)
            local_observations_,load_history_matrix_ =observations
            if not args.validate:
                for i in range(number_of_servers):
                        if np.isnan(rewards[i]):
                            continue
                        decision_makers[i].store_transitions(state = local_observations[i],
                                                        lstm_state=load_history_matrix,
                                                        action = actions[i],
                                                        reward= rewards[i],
                                                        new_state=local_observations_[i],
                                                        new_lstm_state=load_history_matrix_,
                                                        done=done)
                        decision_makers[i].learn()
                        
            local_observations,load_history_matrix  = local_observations_,load_history_matrix_
            accumulated_rewards.append(np.nansum(rewards))
            last_kpi = info.get("kpi")
        
        avg_reward = sum(accumulated_rewards)/len(accumulated_rewards)
        print(f'Epoch {epoch} Accumulated rewards: {avg_reward}')
        if last_kpi:
            print("KPI: processed={}, dropped={}, avg_delay={}".format(
                last_kpi.get('processed'),
                last_kpi.get('dropped'),
                last_kpi.get('avg_delay')))
            kpi_history.append(last_kpi)
        for decision_maker in decision_makers:
            decision_maker.reset_lstm_history()
    if args.plot and kpi_history:
        processed = [k['processed'] for k in kpi_history]
        dropped = [k['dropped'] for k in kpi_history]
        delays = [k['avg_delay'] for k in kpi_history]
        epochs = list(range(len(kpi_history)))
        fig, axs = plt.subplots(3,1, figsize=(8,9))
        axs[0].plot(epochs, processed, label='Processed')
        axs[0].set_ylabel('Processed')
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[1].plot(epochs, dropped, label='Dropped', color='r')
        axs[1].set_ylabel('Dropped')
        axs[1].grid(True, linestyle='--', alpha=0.5)
        axs[2].plot(epochs, delays, label='Avg Delay', color='g')
        axs[2].set_ylabel('Avg Delay')
        axs[2].set_xlabel('Epoch')
        axs[2].grid(True, linestyle='--', alpha=0.5)
        for ax in axs:
            ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(args.log_folder, 'kpi.png')
        plt.savefig(plot_path)
        print(f'KPI plot saved to {plot_path}')

                                
                    
if __name__ == "__main__":
    main()
