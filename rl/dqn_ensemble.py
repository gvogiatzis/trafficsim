"""
Multi-Agent DQN Ensemble for Decentralized Traffic Control

This module implements a coordinator for multiple independent DQN agents, where
each agent controls a subset of traffic lights in the network. The agents learn
simultaneously in a shared environment but make decisions independently.

Key Features:
- Independent Learning: Each agent has its own Q-network and replay buffer
- Decentralized Control: Agents act independently without explicit communication
- Coordinated Training: All agents train synchronously from their experiences
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

from .dqn_agent import DQNAgent
import torch
import numpy as np

from typing import Dict

class DQNEnsemble:
    """
    Ensemble of independent DQN agents for multi-agent traffic control.

    This class manages multiple DQN agents, each controlling a subset of traffic
    lights. The agents learn simultaneously but independently (no parameter sharing
    or direct communication). They interact through the shared environment.
    """

    def __init__(self, schema, network_layers, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=32, memory_capacity=10000):
        """
        Initialize an ensemble of DQN agents based on the provided schema.

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
            Learning rate for all agents (default: 0.001)
        discount_factor : float, optional
            Discount factor (gamma) for all agents (default: 0.99)
        epsilon : float, optional
            Initial exploration rate for all agents (default: 1.0)
        epsilon_decay : float, optional
            Decay factor applied to epsilon after each episode (default: 0.999)
        epsilon_min : float, optional
            Minimum epsilon value (not currently enforced) (default: 0.01)
        batch_size : int, optional
            Training batch size for all agents (default: 32)
        memory_capacity : int, optional
            Replay buffer size for all agents (default: 10000)

        Notes
        -----
        All agents share the same hyperparameters but have independent:
        - Neural networks (different input/output dimensions based on their assignments)
        - Replay buffers (store different experiences)
        - Exploration schedules (epsilon decays together but actions chosen independently)
        """
        self.network_layers = network_layers

        # Create a dictionary of independent DQN agents
        self.agents: Dict[int, DQNAgent] = dict()

        # Instantiate one DQN agent for each entry in the schema
        for id, (state_size, num_actions) in schema.items():
            self.agents[id] = DQNAgent(
                state_size=state_size,
                num_actions=num_actions,
                network_layers=network_layers,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                batch_size=batch_size,
                memory_capacity=memory_capacity
            )        

    def choose_action(self, multi_state: Dict[str, np.ndarray], deterministic=False):
        """
        Select actions for all agents based on their local observations.

        Each agent independently chooses an action using its own policy (epsilon-greedy).
        There is no explicit coordination or communication between agents during action
        selection, though they implicitly coordinate through the shared environment.

        Parameters
        ----------
        multi_state : Dict[int, np.ndarray]
            Maps agent_id to its local state observation.
            Example: {0: array([1,2,3]), 1: array([4,5])}
        deterministic : bool, optional
            If True, all agents exploit (ignore epsilon) for testing (default: False)

        Returns
        -------
        Dict[int, int]
            Maps agent_id to the action it chose.
            Example: {0: 2, 1: 0} means agent 0 chose action 2, agent 1 chose action 0
        """
        a: Dict[str, int] = dict()

        # Each agent independently selects an action based on its local state
        for id, agent in self.agents.items():
            a[id] = agent.choose_action(multi_state[id], deterministic)

        return a

    def remember(self, multi_state: Dict[str, np.ndarray], multi_action: Dict[str, int], multi_reward: Dict[str, float], multi_next_state: Dict[str, np.ndarray], done: bool):
        """
        Store experiences for all agents in their respective replay buffers.

        Each agent stores its own (s, a, r, s', done) tuple. Note that the 'done'
        flag is shared (episode termination affects all agents simultaneously).

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
        """
        # Store experience in each agent's independent replay buffer
        for id, agent in self.agents.items():
            agent.remember(
                multi_state[id],
                multi_action[id],
                multi_reward[id],
                multi_next_state[id],
                done
            )

    def replay(self):
        """
        Train all agents simultaneously using their replay buffers.

        Each agent independently samples from its own buffer and updates its network.
        This implements parallel, decentralized learning where agents learn from
        their own experiences without sharing parameters or gradients.
        """
        # Each agent performs one training step independently
        for id, agent in self.agents.items():
            agent.replay()

    def update_target_model(self):
        """
        Update target networks for all agents.

        Synchronizes each agent's target network with its online network.
        Typically called periodically (e.g., once per episode or every N steps).
        """
        # Update target network for each agent
        for id, agent in self.agents.items():
            agent.update_target_model()

    def decay_epsilon(self):
        """
        Decay exploration rate for all agents.

        All agents decay epsilon together, maintaining synchronized exploration schedules.
        This ensures all agents transition from exploration to exploitation at similar rates.
        """
        # Decay epsilon for each agent (synchronized decay)
        for id, agent in self.agents.items():
            agent.decay_epsilon()

    def load_from_file(self, fname):
        """
        Load trained models for all agents from a single file.

        The file should contain a dictionary mapping agent_id to state_dict.
        This allows resuming training or deploying a trained multi-agent system.

        Parameters
        ----------
        fname : str
            Path to the saved multi-agent model file (.pt or .pth)

        Expected file structure:
        {
            0: <state_dict for agent 0>,
            1: <state_dict for agent 1>,
            ...
        }
        """
        # Load the multi-agent model dictionary
        multi_model = torch.load(fname)

        # Load each agent's weights from the corresponding entry
        for id, agent in self.agents.items():
            agent.load_from_dict(multi_model[id])

    def save_to_file(self, fname):
        """
        Save trained models for all agents to a single file.

        Saves a dictionary mapping agent_id to state_dict, allowing all agents
        to be loaded together for deployment or resuming training.

        Parameters
        ----------
        fname : str
            Path where the multi-agent model should be saved (.pt or .pth)

        Saved file structure:
        {
            0: <state_dict for agent 0>,
            1: <state_dict for agent 1>,
            ...
        }
        """
        # Create dictionary of all agent models
        multi_model = dict()
        for id, agent in self.agents.items():
            multi_model[id] = agent.model.to_dict()

        # Save as a single file
        torch.save(multi_model, fname)