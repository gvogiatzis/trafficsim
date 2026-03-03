"""
Deep Q-Network (DQN) Agent Implementation

This module implements a single DQN agent using the Double DQN algorithm with
experience replay. The agent learns to map states to action values (Q-values)
through interaction with an environment.

Key Features:
- Double DQN: Uses separate online and target networks to reduce overestimation
- Experience Replay: Stores and samples past experiences for more stable learning
- Epsilon-Greedy: Balances exploration and exploitation
- Flexible Architecture: Configurable MLP network layers

References:
- DQN: Mnih et al. (2015) "Human-level control through deep reinforcement learning"
- Double DQN: van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from typing import Callable

from .models import MLPnet, loadModel, saveModel, loadModel_from_dict

class DQNAgent:
    """
    A single Deep Q-Network agent that learns to control traffic lights.

    This class implements the Double DQN algorithm with experience replay buffer.
    It maintains two neural networks: an online model for action selection and
    a target model for stable Q-value estimation during training.
    """

    def __init__(self, state_size, num_actions, network_layers, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=32, memory_capacity=10000, debug=False):
        """
        Initialize the DQN agent with neural networks and training parameters.

        Parameters
        ----------
        state_size : int
            Dimensionality of the state space (input to the network)
        num_actions : int
            Number of possible actions (output dimension)
        network_layers : list of int
            Hidden layer sizes, e.g., [512, 512] creates two hidden layers
        learning_rate : float, optional
            Learning rate for the Adam optimizer (default: 0.001)
        discount_factor : float, optional
            Gamma parameter for future reward discounting, in [0,1] (default: 0.99)
        epsilon : float, optional
            Initial probability of choosing random action for exploration (default: 1.0)
        epsilon_decay : float, optional
            Multiplicative decay factor applied to epsilon after each episode (default: 0.999)
        epsilon_min : float, optional
            Minimum value for epsilon (not enforced in current implementation) (default: 0.01)
        batch_size : int, optional
            Number of experiences to sample from replay buffer for each training step (default: 32)
        memory_capacity : int, optional
            Maximum number of experiences to store in replay buffer (default: 10000)
        debug : bool, optional
            If True, loads a flow matrix for debugging action selection (default: False)
        """
        # Store hyperparameters
        self.state_size = state_size
        self.num_actions = num_actions
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor  # Gamma: importance of future rewards

        # Exploration parameters
        self.epsilon = epsilon  # Current exploration rate
        self.epsilon_decay = epsilon_decay  # Decay factor applied each episode
        self.epsilon_min = epsilon_min  # Floor for epsilon (currently not enforced)

        # Training parameters
        self.batch_size = batch_size  # Mini-batch size for SGD
        self.memory_capacity = memory_capacity

        # Experience replay buffer: stores (state, action, reward, next_state, done) tuples
        self.memory = []

        # Online model: used for action selection and updated every training step
        self.model = MLPnet(state_size, *network_layers, num_actions)

        # Target model: used for computing target Q-values, updated periodically
        # This stabilizes training by reducing the moving target problem
        self.target_model = MLPnet(state_size, *network_layers, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize with same weights

        self.debug = debug

        # Adam optimizer for gradient descent
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Mean Squared Error loss for Q-value regression
        self.criterion = nn.MSELoss()

        # Debug mode: Load external flow matrix for oracle action selection
        if self.debug:
            self.W = torch.tensor(np.loadtxt('flowmat.txt'), dtype=torch.float32)

        

    def choose_action(self, state, deterministic=False):
        """
        Select an action using epsilon-greedy policy.

        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the action with highest Q-value (exploitation).

        Parameters
        ----------
        state : array-like
            Current state observation from the environment
        deterministic : bool, optional
            If True, always exploit (ignore epsilon), used during testing (default: False)

        Returns
        -------
        int
            Selected action index in range [0, num_actions-1]
        """
        # Exploitation: use the learned policy to select best action
        if deterministic or np.random.uniform(0, 1) >= self.epsilon:
            # Convert state to tensor and add batch dimension
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)

            # Disable gradient computation for inference (saves memory and computation)
            with torch.no_grad():
                # Forward pass through the online model to get Q-values for all actions
                q_values = self.model(state_tensor)
                # Select action with maximum Q-value
                action = torch.argmax(q_values).item()

            # Debug mode: Override with oracle action based on flow matrix
            if self.debug:
                # Compute action scores using external flow matrix (for testing/debugging)
                action = np.argsort((state_tensor.squeeze() @ self.W).numpy())[-1]
        else:
            # Exploration: choose a random action uniformly
            action = random.randint(0, self.num_actions - 1)

        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store an experience tuple in the replay buffer.

        The replay buffer implements experience replay, a key component of DQN that
        breaks temporal correlations in the training data by sampling random batches
        from past experiences. This leads to more stable and efficient learning.

        Parameters
        ----------
        state : array-like
            State observation at time t
        action : int
            Action taken at time t
        reward : float
            Reward received after taking action
        next_state : array-like
            State observation at time t+1
        done : bool
            Whether the episode terminated after this transition
        """
        # Add new experience to the buffer
        self.memory.append((state, action, reward, next_state, done))

        # Keep buffer size at capacity by removing oldest experience (FIFO)
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)



    def replay_supervised(self, target_fun: Callable[[np.ndarray], np.ndarray]):
        """
        Train the model using supervised learning with an oracle target function.

        This method supports imitation learning or knowledge distillation by training
        the Q-network to match the output of an expert policy or heuristic function.
        Useful for pre-training or curriculum learning.

        Parameters
        ----------
        target_fun : Callable[[np.ndarray], np.ndarray]
            Oracle function that maps states to action scores/values.
            Should return a tensor of shape (batch_size, num_actions)

        Notes
        -----
        This uses CrossEntropyLoss rather than MSE, treating it as a classification
        problem where the network learns to predict the expert's action choices.
        """
        # Wait until we have enough experiences
        if len(self.memory) < self.batch_size:
            return

        # Sample random mini-batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert states to tensor
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float)

        # Forward pass: get predicted Q-values
        q_values = self.model(state_batch)

        # Get target actions from oracle function (supervised signal)
        # Takes argmax to get the action indices
        _, targets = torch.max(target_fun(state_batch), dim=1)

        # Use cross-entropy loss for classification (predict expert's action)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(q_values, targets)

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def replay(self):
        """
        Train the model using experience replay with Double DQN algorithm.

        This is the main training method that samples a batch of experiences from
        the replay buffer and updates the Q-network parameters using the Bellman
        equation. It implements Double DQN to reduce overestimation bias.

        Double DQN Algorithm:
        1. Use online model to SELECT the best next action
        2. Use target model to EVALUATE that action's Q-value
        3. Update online model to match: Q(s,a) ≈ r + γ * Q_target(s', argmax_a' Q_online(s', a'))

        This decoupling of action selection and evaluation reduces the tendency
        of standard DQN to overestimate Q-values.
        """
        # Wait until we have enough experiences for a full batch
        if len(self.memory) < self.batch_size:
            return

        # Sample random mini-batch from replay buffer
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # Convert all components to tensors
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float)
        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float)
        done_batch = torch.tensor(done_batch, dtype=torch.float)

        # ===== Double DQN Target Computation =====
        # Step 1: Use ONLINE model to select best action for next state
        next_q_values = self.model(next_state_batch)  # Q_online(s', a') for all a'
        _, best_next_actions = torch.max(next_q_values, dim=1)  # argmax_a' Q_online(s', a')

        # Step 2: Use TARGET model to evaluate the selected action
        next_q_values_target = self.target_model(next_state_batch)  # Q_target(s', a') for all a'
        max_next_q_values_target = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

        # Step 3: Compute target using Bellman equation
        # Target = r + γ * Q_target(s', argmax_a' Q_online(s', a'))
        # Multiply by (1 - done) to zero out terminal states
        targets = reward_batch + self.discount_factor * max_next_q_values_target * (1.0 - done_batch)

        # Note: Standard DQN would use the same network for both selection and evaluation:
        # next_q_values = self.target_model(next_state_batch)
        # max_next_q_values, _ = torch.max(next_q_values, dim=1)
        # targets = reward_batch + self.discount_factor * max_next_q_values * (1.0 - done_batch)

        # ===== Compute Current Q-values =====
        # Get Q-values for all actions in current state
        q_values = self.model(state_batch)  # Shape: (batch_size, num_actions)
        # Extract Q-values for the actions that were actually taken
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)

        # ===== Compute Loss and Update =====
        # MSE between predicted Q(s,a) and target Q-value
        loss = self.criterion(q_values, targets)

        # Standard PyTorch optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """
        Synchronize the target network with the online network.

        The target network is updated periodically (not every step) to provide
        stable Q-value targets during training. This prevents the "moving target"
        problem where the target values change too rapidly.

        Typically called every N training steps where N is a hyperparameter
        (e.g., every 100-1000 steps, or once per episode).
        """
        # Copy all weights from online model to target model
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon) after each episode.

        Implements geometric decay: epsilon *= epsilon_decay
        This gradually shifts from exploration to exploitation as training progresses.

        Note: The current implementation does NOT enforce epsilon_min. To add a floor:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        """
        # Multiplicative decay (geometric decay schedule)
        self.epsilon *= self.epsilon_decay

    def load_from_file(self, fname):
        """
        Load a trained model from disk.

        Both the online and target networks are set to the loaded weights.
        This is used for resuming training or for testing/deployment.

        Parameters
        ----------
        fname : str
            Path to the saved model file (.pt or .pth)
        """
        # Load the saved model into both networks
        self.model = loadModel(fname)
        self.target_model = loadModel(fname)

    def load_from_dict(self, state_dict):
        """
        Load model weights from a state dictionary.

        Similar to load_from_file but takes a dictionary instead of a filename.
        Used when loading from a multi-agent ensemble file.

        Parameters
        ----------
        state_dict : dict
            PyTorch state dictionary containing model weights
        """
        # Reconstruct models from state dictionary
        self.model = loadModel_from_dict(state_dict)
        self.target_model = loadModel_from_dict(state_dict)

    def save_to_file(self, fname):
        """
        Save the trained model to disk.

        Only saves the online model (not the target model) since the target
        can be reconstructed by copying the online model's weights.

        Parameters
        ----------
        fname : str
            Path where the model should be saved (.pt or .pth extension)
        """
        # Save only the online model
        saveModel(self.model, fname)
