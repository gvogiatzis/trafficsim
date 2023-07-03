"""Simple wrapper of sumo for training RL traffic control agents

This script contains the SimTrafficControlEnv class that can be used to interface
with the SUMO simulator. There is also a small demo __main__ scrip that will
setup a simple SUMO env and step it through for a few iterations. A simple usage
example usage is as follows:


from import sumoenv import SimTrafficControlEnv
from random import randint

env = SimTrafficControlEnv()

env.reset()
A = env.get_num_actions()
for t in range(1000):
    obs, r = env.step(randint(0,A-1))
"""

import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sumolib
import traci
from typing import Dict, Tuple

import random, pickle

class SimTrafficControlEnv:
    """ The main class that interfaces to sumo.
    
    It exposes its basic functionality for initializing a network, setting up
    basic routes and spawning of vehicles. The simulation is stepped through
    with traffic light actions passed through and observations from lane
    occupancy received back from sumo.
    """
    def __init__(self, net_fname = 'sumo_data/RussianJunction/RussianJunction.net.xml', vehicle_spawn_rate=0.015, state_wrapper=None, episode_length=500, sumo_timestep=20, use_gui=False, seed=None,step_length=1, output_path="output", record_tracks=False, car_length=5, record_screenshots = False, gui_config_file = None, real_routes_file = None, greedy_action=False,random_action=False):
        """ A basic constructor. We read the network file with sumolib and we
        start the sumo (or sumo-gui) program. We then initialize routes and save
        the state for quick reloading whenever we reset.
        """
        random.seed(seed)
        self.record_tracks = record_tracks
        self.total_steps_run=0
        self.current_episode = 0
        self.output_path = output_path
        self.episode_length = episode_length
        self._net_fname = net_fname
        self.vehicle_spawn_rate = vehicle_spawn_rate
        self.action_to_multiaction_dict = None
        self.multiaction_to_action_dict = None
        self.route_dict = None
        self._net = sumolib.net.readNet(self._net_fname, withPrograms=True)
        self._sumo = None
        self._vehcnt = 0
        self.state_wrapper = state_wrapper
        self.episode_step_countdown=episode_length
        self.sumo_timestep = sumo_timestep
        self.use_gui = use_gui
        self.car_length = car_length
        self.record_screenshots = record_screenshots
        self.gui_config_file = gui_config_file
        self.real_routes_file = real_routes_file
        self.greedy_action = greedy_action
        self.random_action=random_action
        self.sim_observation=None

        sanes = None

        sumo_command=['sumo-gui'] if self.use_gui else ['sumo']
        # sumo_command.extend(['-n',self._net_fname,'--start','--quit-on-end','--no-warnings','--no-step-log'])
        sumo_command.extend(['-n',self._net_fname,'--start','--quit-on-end','--no-warnings','--no-step-log', '--step-length', str(step_length)])

        if self.use_gui and self.gui_config_file is not None:
            sumo_command.extend(['-g',self.gui_config_file])

        if self.real_routes_file is not None:
            with open(self.real_routes_file, 'rb') as f:
                # self.real_routes = list(pickle.load(f)["trajectories"].keys())
                self.real_routes = pickle.load(f)["active_routes"]
        else:
            self.real_routes = None

        if seed is not None:
            sumo_command.extend(['--seed',str(seed)])
        else:
            sumo_command.extend(['--random'])
        
        traci.start(sumo_command,verbose=True, label="default")

        self._sumo = traci.getConnection(label="default")


        self.action_to_multiaction_dict, self.multiaction_to_action_dict = self.initialize_actions()


        self.sim_observation = np.zeros(shape=(self.get_obs_dim()))
        self.green_lanes_per_action = self.get_green_lanes_per_action()
        print(f"green_lanes_per_action={self.green_lanes_per_action}")

    def reset(self, seed = None, decentralized=False):
        """
        Initalizes a new environment. 
        
        Must be called when we need to start a new episode. In the traffic
        scenarios of sumo this is not actually necessary as we have a continuous
        loop. You can use in case of deadlock.

        Parameters
        ----------
        seed : int, optional
            A random seed passed to sumo for repeatability (not implemented yet)
        
        Returns
        -------
        observation: numpy.array or self.state_wrapper(observation) if
        self.state_wrapper is defined.
        """

        self.episode_step_countdown = self.episode_length
        

        self.sim_observation = np.random.randint(low=0,high=100,size=self.sim_observation.shape)
        return self.sim_observation
    

    def create_dataset(self, dataset_size=10000):
        dataset=[]
        for i in range(dataset_size):
            datapoint = np.random.randint(low=0,high=100,size=(self.get_obs_dim()))
            target = (self.sim_observation @ self.green_lanes_per_action).argmax()
            dataset.append((datapoint, target))
        return dataset


    def step(self, action=None):
        # print(f"action={action}")
        """ Steps the simulation through one timestep
        
        Executes a single simulation step after passing an action to the
        environment. Returns the observation of the new state and the reward.

        Parameters
        ----------
        action : int (or None)
            the action to send to sumo. Must be between 0 and
            self.get_num_actions()-1. Each integer encodes a combination of
            traffic lights across all junctions. 

        multi_action : int (or None)
            This is a dict mapping between traffic light ID and phase to set that traffic light to. 
        
        Returns
        -------
        observation: numpy array
            The new state of the simulation after the action was implemented

        reward: float
            The reward obtained by the agent for that timestep

        done: boolean
            is true if the episode is finished
        """

        self.episode_step_countdown -= 1

        done = self.episode_step_countdown==0



        total_greens = self.sim_observation @ self.green_lanes_per_action 
        
        if action == total_greens.argmax():
        # if action == 10:
            R = 100
        else:
            R = 0
        # print(f"R={R}")
        self.sim_observation = np.random.randint(low=0,high=100,size=self.sim_observation.shape)
        return self.sim_observation, R, done


    def close(self):
        """ Closes the simulation

        Makes sure the socket connection to the sumo environment is closed and the environment cleans after itself. Always when you don't need it anymore.

        Parameters
        ----------
        none
        
        Returns
        -------
        none
        """
        if traci.isLoaded():
            self._sumo.close()
            self._sumo=None



    def get_num_trafficlights(self):
        """ 
        Returns the number of traffic lights in the network
        """
        tls=self._net.getTrafficLights()
        return len(tls) 

    def get_num_actions(self):
        """ Returns the number of possible actions corresponding to all traffic
        light combinations """
        tls=self._net.getTrafficLights() 
        if len(tls)>0:     
            dim = 1
            for tl in tls:
                logic = tl.getPrograms()['0']
                dim *= len(logic.getPhases())
            return dim
        else:
            return 0
    
    def sample_action(self):
        """ Picks a random action
        
        The action picked is an integer between 0 and self.get_num_actions()-1
        inclusive 
        """
        return random.randint(0, self.get_num_actions()-1)    
    

    def get_obs_dim(self):
        """ Returns the dimensionality of the observation vector
        """
        return sum(len(e.getLanes()) for e in self._net.getEdges())


    def action_to_multiaction(self, a:int):
        if type(a) is int:
            return self.action_to_multiaction_dict[a]

    def get_green_lanes_per_action(self):
        lanes = sum([e.getLanes() for e in self._net.getEdges()],[])
        lanes = [lane.getID() for lane in lanes]
        
        lane_to_ind = {lane:i for i, lane in enumerate(lanes)}

        result = np.zeros((len(lanes),self.get_num_actions()))

        for act in range(self.get_num_actions()):
            ma = self.action_to_multiaction(act)
            for tls_id, a in ma.items():
                logic = self._sumo.trafficlight.getAllProgramLogics(tls_id)[0]
                controlled_lanes = self._sumo.trafficlight.getControlledLanes(tls_id)
                phases = logic.getPhases()
                for lane, s in zip(controlled_lanes,phases[a].state):
                    if s in ['g','G']:
                        row = lane_to_ind[lane]
                        result[row,act] = 1.0
        return result


    def initialize_actions(self):
        """
        This initializes dictionary mappings from a centralized action to a multi-action dict and vice versa.
        """
        action_list = []
        ma_to_a = dict()
        for tl in self._sumo.trafficlight.getIDList():
            logic = self._sumo.trafficlight.getAllProgramLogics(tl)[0]
            nphases = len(logic.getPhases())
            if len(action_list)==0:
                action_list = [[(tl, n)] for n in range(nphases)]
            else:
                action_list = [[(tl,n)] + d for n in range(nphases) for d in action_list]
        a_to_ma = {i:{x:y for x,y in r}    for i,r in enumerate(action_list)}
        ma_to_a = {tuple(y for x,y in sorted(r,key=lambda x:x[0])):i for i,r in enumerate(action_list)}
        return a_to_ma, ma_to_a
