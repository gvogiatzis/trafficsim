"""
Multi-Agent PPO Ensemble for Decentralized Traffic Control

This module implements a coordinator for multiple independent PPO agents, where
each agent controls a subset of traffic lights in the network. The agents learn
simultaneously in a shared environment but make decisions independently.

Key Features:
- Independent Learning: Each agent has its own actor-critic networks
- Decentralized Control: Agents act independently without explicit communication
- Coordinated Training: All agents train when their buffers are full
- Flexible Agent Assignment: Traffic lights can be partitioned across agents

Architecture:
    Multi-Agent System (Decentralized)
    ├── Agent 0 controls Traffic Lights [TL1, TL2]
    ├── Agent 1 controls Traffic Lights [TL3, TL4]
    └── Agent 2 controls Traffic Lights [TL5]

Each agent:
- Observes only its controlled lanes (local observation)
- Takes actions only for its traffic lights
- Receives reward based on its controlled area
- Learns independently but affects others through environment
"""

from .ppo_agent import PPOAgent
import torch
import numpy as np
from typing import Dict, Tuple, List


class PPOEnsemble:
    """
    Ensemble of independent PPO agents for multi-agent traffic control.

    This class manages multiple PPO agents, each controlling a subset of traffic
    lights. The agents learn simultaneously but independently (no parameter sharing
    or direct communication). They interact through the shared environment.
    """

    def __init__(
        self,
        schema: Dict[int, Tuple[int, int]],
        network_layers: List[int],
        learning_rate: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        buffer_capacity: int = 2048
    ):
        """
        Initialize an ensemble of PPO agents based on the provided schema.

        Parameters
        ----------
        schema : Dict[int, Tuple[int, int]]
            Maps agent_id to (state_size, num_actions) tuple.
            Example: {0: (12, 4), 1: (8, 4)} means:
                - Agent 0: observes 12 features, has 4 possible actions
                - Agent 1: observes 8 features, has 4 possible actions
        network_layers : list of int
            Hidden layer sizes shared by all agents, e.g., [1024, 1024]
        learning_rate : float, optional
            Learning rate for all agents (default: 0.0003)
        discount_factor : float, optional
            Discount factor (gamma) for all agents (default: 0.99)
        gae_lambda : float, optional
            GAE lambda parameter for all agents (default: 0.95)
        clip_epsilon : float, optional
            PPO clipping parameter for all agents (default: 0.2)
        value_coef : float, optional
            Value loss coefficient for all agents (default: 0.5)
        entropy_coef : float, optional
            Entropy bonus coefficient for all agents (default: 0.01)
        ppo_epochs : int, optional
            Number of training epochs per update for all agents (default: 10)
        batch_size : int, optional
            Training batch size for all agents (default: 64)
        buffer_capacity : int, optional
            Trajectory buffer size for all agents (default: 2048)

        Notes
        -----
        All agents share the same hyperparameters but have independent:
        - Neural networks (different input/output dimensions based on their assignments)
        - Trajectory buffers (store different experiences)
        - Policy distributions (sample actions independently)
        """
        self.network_layers = network_layers

        # Create a dictionary of independent PPO agents
        self.agents: Dict[int, PPOAgent] = dict()

        # Instantiate one PPO agent for each entry in the schema
        for id, (state_size, num_actions) in schema.items():
            self.agents[id] = PPOAgent(
                state_size=state_size,
                num_actions=num_actions,
                network_layers=network_layers,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                ppo_epochs=ppo_epochs,
                batch_size=batch_size,
                buffer_capacity=buffer_capacity
            )

    def choose_action(
        self,
        multi_state: Dict[int, np.ndarray],
        deterministic: bool = False
    ) -> Tuple[Dict[int, int], Dict[int, float], Dict[int, float]]:
        """
        Select actions for all agents based on their local observations.

        Each agent independently chooses an action using its own policy.
        For PPO, we also need to store the log probabilities and values
        for later training.

        Parameters
        ----------
        multi_state : Dict[int, np.ndarray]
            Maps agent_id to its local state observation.
            Example: {0: array([1,2,3]), 1: array([4,5])}
        deterministic : bool, optional
            If True, all agents exploit (choose most probable action) for testing
            If False, sample from policy distribution for training (default: False)

        Returns
        -------
        actions : Dict[int, int]
            Maps agent_id to the action it chose
        log_probs : Dict[int, float]
            Maps agent_id to log probability of its action
        values : Dict[int, float]
            Maps agent_id to value estimate of its state
        """
        actions: Dict[int, int] = dict()
        log_probs: Dict[int, float] = dict()
        values: Dict[int, float] = dict()

        # Each agent independently selects an action based on its local state
        for id, agent in self.agents.items():
            if deterministic:
                # For evaluation: just get the action
                actions[id] = agent.choose_action(multi_state[id], deterministic=True)
                log_probs[id] = 0.0  # Not used during evaluation
                values[id] = 0.0  # Not used during evaluation
            else:
                # For training: get action, log prob, and value
                action, log_prob, value = agent.get_action_and_value(multi_state[id])
                actions[id] = action
                log_probs[id] = log_prob
                values[id] = value

        return actions, log_probs, values

    def remember(
        self,
        multi_state: Dict[int, np.ndarray],
        multi_action: Dict[int, int],
        multi_reward: Dict[int, float],
        multi_next_state: Dict[int, np.ndarray],
        done: bool,
        multi_log_prob: Dict[int, float],
        multi_value: Dict[int, float]
    ):
        """
        Store experiences for all agents in their respective trajectory buffers.

        Each agent stores its own (s, a, r, s', done, log_prob, value) tuple.
        Note that the 'done' flag is shared (episode termination affects all agents).

        Parameters
        ----------
        multi_state : Dict[int, np.ndarray]
            Current states for each agent
        multi_action : Dict[int, int]
            Actions taken by each agent
        multi_reward : Dict[int, float]
            Rewards received by each agent (can be different)
        multi_next_state : Dict[int, np.ndarray]
            Next states for each agent
        done : bool
            Whether the episode terminated (shared across all agents)
        multi_log_prob : Dict[int, float]
            Log probabilities of actions under collection policy
        multi_value : Dict[int, float]
            Value estimates for current states
        """
        # Store experience in each agent's independent trajectory buffer
        for id, agent in self.agents.items():
            agent.remember(
                multi_state[id],
                multi_action[id],
                multi_reward[id],
                multi_next_state[id],
                done,
                multi_log_prob[id],
                multi_value[id]
            )

    def train(self):
        """
        Train all agents that have collected enough experience.

        Each agent independently trains on its own trajectory buffer when full.
        This implements parallel, decentralized learning where agents learn from
        their own experiences without sharing parameters or gradients.
        """
        # Each agent checks if it should train and performs training if ready
        for id, agent in self.agents.items():
            if agent.should_train():
                agent.train()

    def should_train(self) -> bool:
        """
        Check if any agent is ready to train.

        Returns
        -------
        bool
            True if at least one agent has a full buffer, False otherwise
        """
        return any(agent.should_train() for agent in self.agents.values())

    def load_from_file(self, fname: str):
        """
        Load trained models for all agents from files.

        The files should follow the naming convention:
        {fname}_{agent_id}_actor.pt and {fname}_{agent_id}_critic.pt

        Parameters
        ----------
        fname : str
            Base path to the saved multi-agent model files
        """
        for id, agent in self.agents.items():
            agent.load_from_file(f"{fname}_{id}")

    def save_to_file(self, fname: str):
        """
        Save trained models for all agents to files.

        Saves actor and critic for each agent with naming convention:
        {fname}_{agent_id}_actor.pt and {fname}_{agent_id}_critic.pt

        Parameters
        ----------
        fname : str
            Base path where models should be saved
        """
        for id, agent in self.agents.items():
            agent.save_to_file(f"{fname}_{id}")
