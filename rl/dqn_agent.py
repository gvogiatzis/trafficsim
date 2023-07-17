import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from typing import Callable

from .models import MLPnet, loadModel, saveModel, loadModel_from_dict

class DQNAgent:
    def __init__(self, state_size, num_actions, network_layers, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=32, memory_capacity=10000, debug=False):
        self.state_size = state_size
        self.num_actions = num_actions
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.memory = []
        self.model = MLPnet(state_size, *network_layers, num_actions)
        self.target_model = MLPnet(state_size, *network_layers, num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.debug=debug
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=10*self.learning_rate)
        self.criterion = nn.MSELoss()

        # <DEBUG>
        if self.debug:
            self.W = torch.tensor(np.loadtxt('flowmat.txt'), dtype=torch.float32)
        # </DEBUG>

    def choose_action(self, state, deterministic=False):
        if deterministic or np.random.uniform(0, 1)>=self.epsilon:
            # Exploit: choose the action with maximum Q-value for the current state
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()
            
            # <DEBUG>
            if self.debug:
                action = np.argsort((state_tensor.squeeze() @ self.W).numpy())[-1]
            # </DEBUG>
        else:
            # Explore: choose a random action
            action = random.randint(0, self.num_actions - 1)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)



    def replay_supervised(self, target_fun: Callable[[np.ndarray], np.ndarray]):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float)

        q_values = self.model(state_batch)

        # This comes from the target function (supervised learning)
        _, targets = torch.max(target_fun(state_batch), dim=1)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def replay(self):
        # print("in replay")
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        # state_batch=torch.stack(state_batch)
        # next_state_batch=torch.stack(next_state_batch)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float)
        action_batch = torch.tensor(action_batch, dtype=torch.long)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float)
        done_batch = torch.tensor(done_batch, dtype=torch.float)

        # <test>
        next_q_values = self.model(next_state_batch)
        next_q_values_target = self.target_model(next_state_batch)
        _, best_next_actions = torch.max(next_q_values, dim=1)
        max_next_q_values_target = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
        targets = reward_batch + self.discount_factor * max_next_q_values_target * (1.0 - done_batch)
        # </test>

        # # This comes from the predictive model itself
        # next_q_values = self.target_model(next_state_batch)
        # max_next_q_values, _ = torch.max(next_q_values, dim=1)
        # targets = reward_batch + self.discount_factor * max_next_q_values * (1.0 - done_batch)

        # targets = reward_batch + self.discount_factor * max_next_q_values
        q_values = self.model(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def load_from_file(self, fname):
        self.model = loadModel(fname)
        self.target_model = loadModel(fname)
        # self.target_model.load_state_dict(self.model.state_dict())

    def load_from_dict(self, state_dict):
        self.model = loadModel_from_dict(state_dict)
        self.target_model = loadModel_from_dict(state_dict)
        # self.target_model.load_state_dict(self.model.state_dict())
    
    def save_to_file(self, fname):
        saveModel(self.model, fname)
