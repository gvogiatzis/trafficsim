"""
Proximal Policy Optimization (PPO) Agent Implementation

This module implements a single PPO agent using the clipped surrogate objective
with an actor-critic architecture. The agent learns a policy (actor) and value
function (critic) through interaction with an environment.

Key Features:
- Actor-Critic: Separate networks for policy and value function
- Clipped Objective: Prevents destructively large policy updates
- GAE: Generalized Advantage Estimation for variance reduction
- Multiple Epochs: Reuses collected experience for sample efficiency

References:
- PPO: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- GAE: Schulman et al. (2016) "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

from .models import MLPnet


class PPOAgent:
    """
    A single PPO agent that learns to control traffic lights using policy gradients.

    This class implements the PPO algorithm with clipped surrogate objective.
    It maintains two neural networks: an actor (policy) and a critic (value function).
    """

    def __init__(
        self,
        state_size: int,
        num_actions: int,
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
        Initialize the PPO agent with actor-critic networks and training parameters.

        Parameters
        ----------
        state_size : int
            Dimensionality of the state space (input to the networks)
        num_actions : int
            Number of possible discrete actions
        network_layers : list of int
            Hidden layer sizes, e.g., [512, 512] creates two hidden layers
        learning_rate : float, optional
            Learning rate for both actor and critic (default: 0.0003)
        discount_factor : float, optional
            Gamma parameter for future reward discounting, in [0,1] (default: 0.99)
        gae_lambda : float, optional
            Lambda parameter for GAE, controls bias-variance tradeoff (default: 0.95)
        clip_epsilon : float, optional
            Clipping parameter for PPO objective (default: 0.2)
        value_coef : float, optional
            Coefficient for value loss in total loss (default: 0.5)
        entropy_coef : float, optional
            Coefficient for entropy bonus to encourage exploration (default: 0.01)
        ppo_epochs : int, optional
            Number of epochs to train on each batch of collected experience (default: 10)
        batch_size : int, optional
            Mini-batch size for SGD updates (default: 64)
        buffer_capacity : int, optional
            Maximum number of transitions to collect before training (default: 2048)
        """
        self.state_size = state_size
        self.num_actions = num_actions
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity

        # Actor network: outputs action probabilities (policy)
        self.actor = MLPnet(state_size, *network_layers, num_actions)

        # Critic network: outputs state value estimate
        self.critic = MLPnet(state_size, *network_layers, 1)

        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experience buffer: stores trajectories for on-policy learning
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, float, float]] = []

    def choose_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select an action using the current policy.

        Parameters
        ----------
        state : np.ndarray
            Current state observation from the environment
        deterministic : bool, optional
            If True, select the most probable action (for evaluation)
            If False, sample from the policy distribution (for training) (default: False)

        Returns
        -------
        int
            Selected action index in range [0, num_actions-1]
        """
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            # Get action logits from actor network
            logits = self.actor(state_tensor)
            probs = torch.softmax(logits, dim=-1)

            if deterministic:
                # Exploitation: choose most probable action
                action = torch.argmax(probs, dim=-1).item()
            else:
                # Exploration: sample from probability distribution
                action = torch.multinomial(probs, num_samples=1).item()

        return action

    def get_action_and_value(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        Select an action and compute log probability and value estimate.

        This method is used during experience collection to store the log
        probability of the taken action for later PPO updates.

        Parameters
        ----------
        state : np.ndarray
            Current state observation

        Returns
        -------
        action : int
            Selected action
        log_prob : float
            Log probability of the selected action under current policy
        value : float
            Estimated value of the current state
        """
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            # Get action probabilities
            logits = self.actor(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1)
            log_prob = torch.log(probs.gather(1, action)).item()

            # Get value estimate
            value = self.critic(state_tensor).item()

        return action.item(), log_prob, value

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ):
        """
        Store an experience tuple in the trajectory buffer.

        PPO is an on-policy algorithm that requires storing the log probability
        of actions under the policy that collected them.

        Parameters
        ----------
        state : np.ndarray
            State observation at time t
        action : int
            Action taken at time t
        reward : float
            Reward received after taking action
        next_state : np.ndarray
            State observation at time t+1
        done : bool
            Whether the episode terminated after this transition
        log_prob : float
            Log probability of the action under the collection policy
        value : float
            Value estimate V(s) from the critic
        """
        self.buffer.append((state, action, reward, next_state, done, log_prob, value))

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        next_values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE provides a way to estimate advantages that balances bias and variance
        through the lambda parameter. It computes a weighted average of n-step
        advantages.

        Parameters
        ----------
        rewards : List[float]
            Rewards at each timestep
        values : List[float]
            Value estimates V(s_t) at each timestep
        next_values : List[float]
            Value estimates V(s_{t+1}) at each timestep
        dones : List[bool]
            Done flags at each timestep

        Returns
        -------
        advantages : np.ndarray
            Advantage estimates A(s,a) for each timestep
        returns : np.ndarray
            Target values (returns) for value function training
        """
        advantages = []
        gae = 0

        # Compute GAE backwards through time
        for t in reversed(range(len(rewards))):
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.discount_factor * next_values[t] * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages, dtype=np.float32)
        # Returns = Advantages + Values (for value function training target)
        returns = advantages + np.array(values, dtype=np.float32)

        return advantages, returns

    def train(self):
        """
        Train the agent using collected experience with PPO algorithm.

        This method performs multiple epochs of mini-batch updates on the
        collected trajectory data. It updates both actor (policy) and critic
        (value function) using the PPO clipped objective.
        """
        if len(self.buffer) < self.batch_size:
            return

        # Unpack buffer
        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*self.buffer)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = list(rewards)
        dones = list(dones)
        old_log_probs = np.array(old_log_probs, dtype=np.float32)
        values = list(values)

        # Compute next state values for GAE
        next_values = []
        for next_state, done in zip(next_states, dones):
            if done:
                next_values.append(0.0)
            else:
                with torch.no_grad():
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                    next_values.append(self.critic(next_state_tensor).item())

        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(rewards, values, next_values, dones)

        # Normalize advantages for more stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float)
        returns_tensor = torch.tensor(returns, dtype=torch.float)

        # Perform multiple epochs of training on the collected data
        for _ in range(self.ppo_epochs):
            # Create random mini-batches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]

                # ===== Actor (Policy) Update =====
                # Get current action log probabilities
                logits = self.actor(batch_states)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1))

                # Compute probability ratio: π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # PPO clipped surrogate objective
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Entropy bonus to encourage exploration
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # ===== Critic (Value Function) Update =====
                # Get current value estimates
                values_pred = self.critic(batch_states).squeeze(1)

                # Value loss (MSE between predicted and target values)
                critic_loss = self.value_coef * nn.MSELoss()(values_pred, batch_returns)

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        # Clear buffer after training (on-policy learning)
        self.buffer.clear()

    def should_train(self) -> bool:
        """
        Check if enough experience has been collected to trigger training.

        Returns
        -------
        bool
            True if buffer has reached capacity, False otherwise
        """
        return len(self.buffer) >= self.buffer_capacity

    def load_from_file(self, fname: str):
        """
        Load trained actor and critic models from disk.

        Parameters
        ----------
        fname : str
            Base path to the saved model files (without extension)
            Will load from {fname}_actor.pt and {fname}_critic.pt
        """
        self.actor.load_state_dict(torch.load(f"{fname}_actor.pt"))
        self.critic.load_state_dict(torch.load(f"{fname}_critic.pt"))

    def load_from_dict(self, state_dict: dict):
        """
        Load model weights from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing 'actor' and 'critic' state dicts
        """
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])

    def save_to_file(self, fname: str):
        """
        Save the trained actor and critic models to disk.

        Parameters
        ----------
        fname : str
            Base path where models should be saved (without extension)
            Will save to {fname}_actor.pt and {fname}_critic.pt
        """
        torch.save(self.actor.state_dict(), f"{fname}_actor.pt")
        torch.save(self.critic.state_dict(), f"{fname}_critic.pt")
