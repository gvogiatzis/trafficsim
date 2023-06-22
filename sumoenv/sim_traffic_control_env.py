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

        # if self.record_tracks and not os.path.exists(f"{self.output_path}/sumo_tracks"):
        #     os.makedirs(f"{self.output_path}/sumo_tracks")
        #     print(f"created dir: {self.output_path}/sumo_tracks")
        # if self.record_screenshots and not os.path.exists(f"{self.output_path}/sumo_screenshots"):
        #     os.makedirs(f"{self.output_path}/sumo_screenshots")
        #     print(f"created dir: {self.output_path}/sumo_screenshots")

        if seed is not None:
            sumo_command.extend(['--seed',str(seed)])
        else:
            sumo_command.extend(['--random'])
        
        traci.start(sumo_command,verbose=True, label="default")

        self._sumo = traci.getConnection(label="default")
        self._initialize_routes()
        self._sumo.simulation.saveState('state.sumo')

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

        # self._sumo.simulation.loadState('state.sumo')

        # self._spawnVehicles()
        # self._sumo.simulationStep()
        self.episode_step_countdown = self.episode_length


        
    
        # if decentralized:
        #     r: Dict[str, Tuple[np.ndarray, float]] = dict()
        #     for tlID in self._sumo.trafficlight.getIDList():
        #         r[tlID] = self.get_local_state(tlID)
        #     return r
        # else:
        #     return self.get_total_state()

        

        self.sim_observation = np.random.randint(low=0,high=100,size=self.sim_observation.shape)
        return self.sim_observation
    

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
        # decentralized = (type(action) is dict)

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
        if os.path.exists('state.sumo'):
            os.remove('state.sumo')
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

    def action_to_multiaction(self, a:int):
        if type(a) is int:
            return self.action_to_multiaction_dict[a]

    def multiaction_to_action(self, multi_action:dict):
        key = tuple(a for id, a in sorted(multi_action.items(), key = lambda x:x[0]))
        return self.multiaction_to_action_dict[key]
    
    def get_action_breakdown(self):
        """ Returns the schema of all independent decisions. It is a dictionary who's keys are the traffic light IDs and the items are (obs_dim, num_actions) pairs. This can be used to create independent (discrete) RL agents solving each light. Obs_dim is the dimension of the observation vector while num_actions is the number of phases for that particular traffic light.        
        """
        schema = dict()
        for tlID in self._sumo.trafficlight.getIDList():
            logic = self._sumo.trafficlight.getAllProgramLogics(tlID)[0]
            lanes = self._sumo.trafficlight.getControlledLanes(tlID)
            phases = logic.getPhases()
            schema[tlID] = (len(lanes), len(phases))
        return schema

    def get_obs_dim(self):
        """ Returns the dimensionality of the observation vector
        """
        return sum(len(e.getLanes()) for e in self._net.getEdges())


    def set_all_lights(self, state):
        """
        Sets all lights in all traffic junction to the same state. The input state is one of the chars of 'rugGyYuoO'
        """
        # print(f"setting all lights to {state}")
        for tlID in self._sumo.trafficlight.getIDList():
            n = len(self._sumo.trafficlight.getControlledLanes(tlID))
            self._sumo.trafficlight.setRedYellowGreenState(tlID, state * n)

    def get_total_state(self):
        """
        This gets the lane observations for the entire traffic network
        """        
        # lanes = self._sumo.lane.getIDList() # This inludes :xyz internals
        lanes = sum([e.getLanes() for e in self._net.getEdges()],[])
        result = []
        for lane in lanes:
            # result.append(self._sumo.lane.getLastStepVehicleNumber(lane.getID()))
            result.append(self._sumo.lane.getLastStepHaltingNumber(lane.getID()))
        if self.state_wrapper is not None:
            return self.state_wrapper(np.array(result))
        else:
            return np.array(result)
        
    def get_local_state(self, tlID):
        """
        This gets the lane observations for all the lanes controlled by this traffic light
        """
        lanes = self._sumo.trafficlight.getControlledLanes(tlID)
        result=[]
        for lane in lanes:
            result.append(self._sumo.lane.getLastStepHaltingNumber(lane))
        if self.state_wrapper is not None:
            return self.state_wrapper(np.array(result))
        else:
            return np.array(result)

    def get_total_hallting_number(self):
        """
        This is the total number of cars stopping in the entire network.
        """
        lanes = sum([e.getLanes() for e in self._net.getEdges()],[])
        r = sum(self._sumo.lane.getLastStepHaltingNumber(lane.getID()) for lane in lanes)
        return r

    def get_local_hallting_number(self, tlID):
        """
        This is the total number of cars stopping in this traffic light
        """
        lanes = self._sumo.trafficlight.getControlledLanes(tlID)
        r = sum(self._sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        return r

    def _getCurrentTotalTimeLoss(self):
        dt = self._sumo.simulation.getDeltaT()
        timeloss = 0

        vehIDs = self._sumo.vehicle.getIDList()
        for vehID in vehIDs:
            Vmax = self._sumo.vehicle.getAllowedSpeed(vehID)
            V = self._sumo.vehicle.getSpeed(vehID)
            timeloss += (1 - V/Vmax) * dt
        # return timeloss / len(vehIDs)
        return timeloss

    def get_route_trajectories(self, save_file = None, plot_traj=False):   
        """
        Helper method for generating all possible vehicle routes (with detailed x,y trajectories for each). This is used in the getallroutes script which must be called before the routegui.
        """     
        route_traj = dict()
        route_waypoints=dict()
        print("_getAllRouteIDs")
        print(self._getAllRouteIDs())
        for routeID in self._getAllRouteIDs():
            route=[]
            waypoints=[]
            vehID = "veh"+routeID
            self._sumo.vehicle.add(vehID, routeID, departLane="best", )
            self._sumo.simulationStep()
            self._sumo.simulationStep()
            self.set_all_lights('r')
            found_red_light=False
            while self._sumo.vehicle.getIDCount()>0:
                if self._sumo.vehicle.getSpeed(vehID)==0.0 and not found_red_light:
                    found_red_light = True
                    waypoints.append(self._sumo.vehicle.getPosition(vehID))
                    self.set_all_lights('G')
                route.append(self._sumo.vehicle.getPosition(vehID))
                self._sumo.simulationStep()
            
            route_traj[routeID] = route
            route_waypoints[routeID] = waypoints
        if save_file is not None:
            with open(save_file, 'wb') as f:
                pickle.dump({"trajectories":route_traj, "waypoints":route_waypoints},f)
        if plot_traj:
            plt.figure()
            for r,xy in route_traj.items():
                X = [x for (x,y) in xy]
                Y = [y for (x,y) in xy]
                plt.plot(X,Y,'-')
                if r in route_waypoints:
                    wps = route_waypoints[r]
                    X = [x for (x,y) in wps]
                    Y = [y for (x,y) in wps]
                    plt.plot(X,Y,marker='o',markersize=5)
            plt.show()
        print(f"found {len(route_traj)} routes")
        return route_traj

    def _saveVehicles(self, output_path, use_total_time=False):
        if use_total_time:
            fname = f"{output_path}/{self.total_steps_run:009}.txt"
        else:
            step = self.episode_length-self.episode_step_countdown
            fname = f"{output_path}/{self.current_episode:03}_{step:04d}.txt"
        with open(fname, "w") as f:
            for v in self._sumo.vehicle.getIDList():
                route_edges = self._sumo.vehicle.getRoute(v)
                x,y = self._sumo.vehicle.getPosition(v)
                e0 = route_edges[0]
                e1 = route_edges[-1]
                routeID = self.route_dict[e0][e1]
                f.write(f"{routeID}, {x}, {y}\n")


    def _spawnVehicles(self):
        if self.real_routes is not None:
            all_routeIDs = self.real_routes 
        else:
            all_routeIDs = self._getAllRouteIDs()
        for routeID in all_routeIDs:
            if random.random() < self.vehicle_spawn_rate:
                vehID = f"veh{self._vehcnt:08d}"
                self._sumo.vehicle.add(vehID, routeID)
                self._sumo.vehicle.setLength(vehID, self.car_length)
                self._sumo.vehicle.setWidth(vehID, self.car_length/3.5)
                self._vehcnt +=1

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

    def _get_TLS_demand_breakdown(self, tls_id):
        logic = self._sumo.trafficlight.getAllProgramLogics(tls_id)[0]
        lanes = self._sumo.trafficlight.getControlledLanes(tls_id)
        phases = logic.getPhases()
        demand = []
        for phase in phases:
            phasedemand=0
            for lane, s in zip(lanes,phase.state):
                if s in ['g','G']:
                    phasedemand += self._sumo.lane.getLastStepVehicleNumber(lane)
                    # phasedemand += self._sumo.lane.getLastStepOccupancy(lane)
            demand.append(phasedemand) 
        return demand


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

    def _applyAction(self, action: int):
        """ Applies a traffic control action to the traffic lights
        """
        if len(self.action_to_multiaction_dict)>0:
            for (tl, a) in self.action_to_multiaction_dict[action]:
                self._sumo.trafficlight.setPhase(tl,a)

    def _applyMultiaction(self, multi_action: dict):
        """ Applies a traffic control multi_action dict to the traffic lights
        """
        for tlID, action in multi_action.items():
            self._sumo.trafficlight.setPhase(tlID,action)


    def _getAllRouteIDs(self):
        return [s for s in self._sumo.route.getIDList() if s[0]!='!']

    
    def _initialize_routes(self):
        self.route_dict = defaultdict(dict)
        edges = self._net.getEdges()
        for e1 in edges:
            if e1.is_fringe():
                for e2 in self._net.getReachable(e1):
                    if e2.is_fringe():
                        if e1!=e2 and e1.getID().replace('-','') != e2.getID().replace('-',''):
                            routeID = f"{e1.getID()}->{e2.getID()}"
                            self._sumo.route.add(routeID, [e1.getID(), e2.getID()])
                            self.route_dict[e1.getID()][e2.getID()] = routeID
