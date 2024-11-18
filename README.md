Paper Title : "﻿HOODIE: Hybrid Computation Offloading via Distributed Deep Reinforcement Learning in Delay-aware Cloud-Edge Continuum"

Cite as: Anastasios Giannopoulos, Ilias Paralikas, Sotirios Spantideas, et al. HOODIE: Hybrid Computation Offloading via Distributed Deep Reinforcement Learning in Delay-aware Cloud-Edge Continuum. TechRxiv. November 14, 2024.

DOI: 10.36227/techrxiv.173161211.11891475/v1

**ABSTRACT** Cloud-Edge Computing Continuum (CEC) system, where edge and cloud nodes are seamlessly connected, is dedicated to handle substantial computational loads offloaded by end-users. These tasks can suffer from delays or be dropped entirely when deadlines are missed, particularly under fluctuating network conditions and resource limitations. The CEC is coupled with the need for hybrid task offloading, where the task placement decisions concern whether the tasks are processed locally, offloaded vertically to the cloud, or horizontally to interconnected edge servers. In this paper, we present a distributed hybrid task offloading scheme (HOODIE) designed to jointly optimize the tasks latency and drop rate, under dynamic CEC traffic. HOODIE employs a model-free deep reinforcement learning (DRL) framework, where distributed DRL agents at each edge server autonomously determine offloading decisions without global task distribution awareness. To further enhance the system pro-activity and learning stability, we incorporate techniques such as Long Short-term Memory (LSTM), Dueling deep Q-networks (DQN), and double-DQN. Extensive simulation results demonstrate that HOODIE effectively reduces task drop rates and average task processing delays, outperforming several baseline methods under changing CEC settings and dynamic conditions.



How to use :

In order to run the default version one just needs to run the command 
```
python main.py
```

This will create a new run and will store it in a folder called *log_folder*. If ones wants to change the default values for the simulations, they need to tun the hyperparameters script, with the appropriated command line arguments, as shown below

```
python hyperparameters --number_of_servers 10 
```
After that they need to rerun the main script.

Note that one must rename the folder, or else the weights of the previous run will be loaded.
