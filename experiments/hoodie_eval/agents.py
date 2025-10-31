from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pickle
import torch

from decision_makers.agent import Agent as BaseAgent
from lr_schedulers import Linear, constant

from .adapters import _deep_update  # reuse utility


class TelemetryAgent(BaseAgent):
    """
    Extension of the original Agent that records the training loss after each
    learning step so that convergence curves can be reproduced.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loss_history: List[float] = []

    def learn(self) -> None:  # type: ignore[override]
        def weighted_add_state_dicts():
            eval_state_dict = self.Q_eval_network.state_dict()
            target_state_dict = self.Q_target_network.state_dict()
            new_state_dict = {}
            for key in target_state_dict.keys():
                new_state_dict[key] = (
                    self.update_weight_percentage * eval_state_dict[key]
                    + target_state_dict[key] * (1 - self.update_weight_percentage)
                )
            return new_state_dict

        def get_lstm_sequence(index):
            start_index = max(0, index - self.lstm_time_step)
            return torch.tensor(self.lstm_memory[start_index:index])

        if self.memory_counter <= self.batch_size + self.lstm_time_step:
            return

        self.Q_eval_network.train()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_iter == 0:
            new_target_weights = weighted_add_state_dicts()
            self.Q_target_network.load_state_dict(new_target_weights)

        self.optimizer.zero_grad()

        max_memory = min(self.memory_counter, self.memory_size)
        batch_indices = np.random.choice(
            range(self.lstm_time_step, max_memory), self.batch_size, replace=False
        )

        state_batch = torch.tensor(self.state_memory[batch_indices]).to(self.device)
        lstm_sequence_batch = [get_lstm_sequence(index) for index in batch_indices]
        lstm_sequence_batch = torch.stack(lstm_sequence_batch).to(self.device)
        action_batch = torch.tensor(self.action_memory[batch_indices]).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[batch_indices]).to(self.device)
        next_state_batch = torch.tensor(self.new_state_memory[batch_indices]).to(self.device)
        next_lstm_sequence_batch = [get_lstm_sequence(index + 1) for index in batch_indices]
        next_lstm_sequence_batch = torch.stack(next_lstm_sequence_batch).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch_indices]).to(self.device)

        q_eval = self.Q_eval_network(state_batch, lstm_sequence_batch).gather(
            1, action_batch.unsqueeze(1)
        ).squeeze(1)
        q_next_eval = self.Q_eval_network(next_state_batch, next_lstm_sequence_batch)
        next_actions = torch.argmax(q_next_eval, dim=1)
        q_next_target = self.Q_target_network(next_state_batch, next_lstm_sequence_batch)
        q_target_next = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        mask = terminal_batch == 0
        q_target_next = q_target_next * mask
        q_target = reward_batch + self.gamma * q_target_next

        loss = self.loss_function(q_eval, q_target)
        loss.backward()
        self.optimizer.step()

        self.loss_history.append(float(loss.detach().cpu().item()))

        self.epsilon = max(self.epsilon - self.epsilon_decrement, self.epsilon_end)

        self._last_loss = float(loss.detach().cpu().item())

        with open(self.scheduler_file, "wb") as f:
            pickle.dump(self.schedueler_fun, f)

        if self.learn_step_counter % self.save_model_frequency == 0:
            self.store_model()

        self.scheduler.step()


@dataclass
class AgentBundle:
    agents: List[TelemetryAgent]
    log_dir: Path


def build_telemetry_agents(
    env,
    agent_config,
    hyperparameters: Mapping[str, Any],
    log_dir: Path,
) -> AgentBundle:
    """
    Construct one telemetry-enabled agent per edge server using the provided
    hyperparameter set and agent configuration.
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    hp = copy.deepcopy(dict(hyperparameters))
    overrides = copy.deepcopy(agent_config.overrides or {})

    if agent_config.lookback_window:
        overrides.setdefault("lstm_time_step", agent_config.lookback_window)

    _deep_update(hp, overrides)

    hp["dueling"] = agent_config.dueling
    hp["lstm_layers"] = hp.get("lstm_layers", 1 if agent_config.lstm else 0)
    if not agent_config.lstm:
        hp["lstm_layers"] = 0
        hp["lstm_time_step"] = 1
    if not agent_config.double:
        hp["replace_target_iter"] = hp.get("replace_target_iter", 2000) * 100

    scheduler_file = log_dir / "scheduler.pth"

    scheduler_choice = hp.get("scheduler_choice", "constant")
    scheduler_map = {
        "constant": constant,
        "Linear": Linear(
            start=hp.get("learning_rate", 1e-6),
            end=hp.get("learning_rate_end", hp.get("learning_rate", 1e-6)),
            number_of_epochs=hp.get("lr_scheduler_epochs", agent_config.episodes),
        ),
    }
    scheduler = scheduler_map.get(scheduler_choice, constant)

    with open(scheduler_file, "wb") as f:
        pickle.dump(scheduler, f)

    agents: List[TelemetryAgent] = []

    for i in range(env.number_of_servers):
        state_dimensions, foreign_queues, number_of_actions = env.get_server_dimensions(i)
        lstm_shape = foreign_queues

        agent = TelemetryAgent(
            id=i,
            state_dimensions=state_dimensions,
            lstm_shape=lstm_shape,
            number_of_actions=number_of_actions,
            hidden_layers=hp["hidden_layers"],
            lstm_layers=hp.get("lstm_layers", 0),
            lstm_time_step=hp.get("lstm_time_step", 1),
            dropout_rate=hp["dropout_rate"],
            dueling=hp.get("dueling", True),
            epsilon=hp["epsilon"],
            epsilon_decrement=hp["epsilon_decrement"],
            epsilon_end=hp["epsilon_end"],
            gamma=hp["gamma"],
            learning_rate=hp["learning_rate"],
            scheduler_file=str(scheduler_file),
            loss_function=getattr(torch.nn, hp["loss_function"]),
            optimizer=getattr(torch.optim, hp["optimizer"]),
            checkpoint_folder=str(log_dir / f"agent_{i}.pth"),
            save_model_frequency=hp.get("save_model_frequency", 100),
            update_weight_percentage=hp.get("update_weight_percentage", 1.0),
            memory_size=hp["memory_size"],
            batch_size=hp["batch_size"],
            replace_target_iter=hp["replace_target_iter"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        agents.append(agent)

    return AgentBundle(agents=agents, log_dir=log_dir)
