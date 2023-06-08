from .models import MLPnet, loadModel, saveModel, loadModel_from_dict
from .dqn_agent import DQNAgent
from sumoenv import TrafficControlEnv
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from typing import Dict

class DQNEnsemble:
    def __init__(self, env:TrafficControlEnv, network_layers, learning_rate=0.001, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01, batch_size=32, memory_capacity=10000):
        self.network_layers = network_layers
        self.agents: Dict[int,DQNAgent] = dict()
        schema = env.get_action_breakdown()
        for id, (state_size, num_actions) in schema.items():
            self.agents[id] = DQNAgent(state_size=state_size, num_actions=num_actions, network_layers=network_layers,learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, batch_size=batch_size, memory_capacity=memory_capacity)        

    def choose_action(self, multi_state: Dict[str, np.ndarray]):
        a: Dict[str, int] = dict()
        for id, agent in self.agents.items():
            a[id] = agent.choose_action(multi_state[id])            
        return a

    def remember(self, multi_state: Dict[str, np.ndarray], multi_action: Dict[str, int], multi_reward: Dict[str, float], multi_next_state: Dict[str, np.ndarray], done:bool):
        for id, agent in self.agents.items():
            agent.remember(multi_state[id],multi_action[id], multi_reward[id], multi_next_state[id], done)

    def replay(self):
        for id, agent in self.agents.items():
            agent.replay()

    def update_target_model(self):
        for id, agent in self.agents.items():
            agent.update_target_model()

    def decay_epsilon(self):
        for id, agent in self.agents.items():
            agent.decay_epsilon()
 
    def load_from_file(self, fname):
        multi_model = torch.load(fname)
        for id, agent in self.agents.items():
            agent.load_from_dict(multi_model[id])
    
    def save_to_file(self, fname):
        multi_model=dict()
        for id, agent in self.agents.items():
            multi_model[id] = agent.model.to_dict()
        torch.save(multi_model,fname)