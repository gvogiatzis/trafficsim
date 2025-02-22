"""Simple wrapper of sumo for training RL traffic control agents

This script contains the TrafficControlEnv class that can be used to interface
with the SUMO simulator. There is also a small demo __main__ scrip that will
setup a simple SUMO env and step it through for a few iterations. A simple usage
example usage is as follows:


from import sumoenv import TrafficControlEnv
from random import randint

env = TrafficControlEnv()

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
from itertools import product, chain

import random, pickle

class TrafficControlEnv:
    """ The main class that interfaces to sumo.
    
    It exposes its basic functionality for initializing a network, setting up
    basic routes and spawning of vehicles. The simulation is stepped through
    with traffic light actions passed through and observations from lane
    occupancy received back from sumo.
    """
    def __init__(self, net_fname = 'sumo_data/RussianJunction/RussianJunction.net.xml', vehicle_spawn_rate=0.015, state_wrapper=None, episode_length=500, sumo_timestep=20, use_gui=False, seed=None,step_length=1, output_path="output", record_tracks=False, car_length=5, record_screenshots = False, gui_config_file = None, real_routes_file = None,agent_lights_file=None):
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


        # set self.simple_state to True if you want a simplified state that already computes 
        # a nice feature (sum of all cars halting that will be getting a green)
        # for each action. 
        self.simple_state = True


        # prepare sumo command
        sumo_command=['sumo-gui'] if self.use_gui else ['sumo']
        sumo_command.extend(['-n',self._net_fname,'--start','--quit-on-end','--no-warnings','--no-step-log', '--step-length', str(step_length)])

        # if in gui mode, use gui config if available (used to set viewpoints car sizes etc)
        if self.use_gui and self.gui_config_file is not None:
            sumo_command.extend(['-g',self.gui_config_file])

        # used if we need to restrict to specific routes as obtained by routegui
        if self.real_routes_file is not None:
            with open(self.real_routes_file, 'rb') as f:
                self.real_routes = pickle.load(f)["active_routes"]
        else:
            self.real_routes = None

        # set book keeping information for logging tracks and screenshots (used for image generation)
        if self.record_tracks and not os.path.exists(f"{self.output_path}/sumo_tracks"):
            os.makedirs(f"{self.output_path}/sumo_tracks")
            print(f"created dir: {self.output_path}/sumo_tracks")
        if self.record_screenshots and not os.path.exists(f"{self.output_path}/sumo_screenshots"):
            os.makedirs(f"{self.output_path}/sumo_screenshots")
            print(f"created dir: {self.output_path}/sumo_screenshots")

        # sort out random seeds for repeatable experiments
        if seed is not None:
            sumo_command.extend(['--seed',str(seed)])
        else:
            sumo_command.extend(['--random'])
        
        traci.start(sumo_command,verbose=True, label="default")

        self._sumo = traci.getConnection(label="default")
        self._initialize_routes()
        self._sumo.simulation.saveState('state.sumo')

        # compute schema that maps combinations of actions into actuall traffic lights for
        # the multiple agents
        self.schema = self._compute_schema(agent_lights_file)

    def _compute_schema(self, agent_lights_file):
        # agent_lights_file is a text file containing one line per agent. In each line
        # are the traffic light ids that are to be controlled by each agent.
        # if no file is present we assume a single agent that simultaneously controls all
        # traffic lights 
        agent_tls = dict()
        ag_id = 0
        if agent_lights_file is not None:
            with open(agent_lights_file) as file:
                for line in file:
                    agent_tls[ag_id] = [str.strip() for str in line.split(",")]
                    ag_id += 1
        else:
            agent_tls[ag_id]=self._sumo.trafficlight.getIDList()

        # 'schema' is a dictionary that for maps an agentID to a set of objects:
        # controlledTLs: is a list of all the traffic lights that are controlled by that agent
        # controlledlanes: is a list of all the lanes that are controlled by that agent
        # a_to_tl_to_pha: is a dictionary that maps an actionID to a trafficlightID->phase dictionary
        # a_to_green_lanes: is a dictionary that maps an actionID to a list of lanes that will get a green under that action
        # dims: a (num_lanes, num_phases) tuple. The number of lanes is the size of the observation space and num_phases
        #       is the number of possible actions that this agent can take. It is the input and output dimension of
        #       the Neural Network that represents policy for that agent.
        schema = dict()

        for aID, tlIDs in agent_tls.items():
            schema[aID]={"controlledTLs":tlIDs}
            schema[aID]["controlledlanes"]=[]
          
            # computing the mapping between traffic light and phase for each of the 
            # available actions.
            num_lanes = 0
            num_phases = 1
            tl_pha_lists=[]
            # tl_pha_lists is a list of lists as follows:
            # [[(tl1,1),...,(tl1,nphases_for_tl1)],
            # [(tl2,1),...,(tl2,nphases_for_tl2)],...]
            # In the end we will compute a cartesian product of all those lists
            # and the result will be stored in tl_pha_combinations
            # Then it's a simple case of converting that into an action->tl->phaseID mapping
            for tlID in tlIDs:
                logic = self._sumo.trafficlight.getAllProgramLogics(tlID)[0]
                lanes = self._sumo.trafficlight.getControlledLanes(tlID)
                nphases = len(logic.getPhases())
                tl_pha_lists.append([(tlID, n) for n in range(nphases)])           
                num_lanes += len(lanes)
                schema[aID]["controlledlanes"] += lanes
            tl_pha_combinations = list(product(*tl_pha_lists))
            num_actions = len(tl_pha_combinations)
            a_to_tl_to_pha = {a:{tl:pha for tl,pha in tl_pha}    for a,tl_pha in enumerate(tl_pha_combinations)}

            # computing green lanes corresponding to each action
            a_to_green_lanes=dict()
            for a in a_to_tl_to_pha.keys():
                a_to_green_lanes[a]=[]
                for tl,pha in a_to_tl_to_pha[a].items():
                    pha_to_greenlanes = self._get_green_lanes_per_phase(tl)
                    a_to_green_lanes[a].extend(pha_to_greenlanes[pha])

            schema[aID]["a_to_tl_to_pha"] = a_to_tl_to_pha
            schema[aID]["a_to_green_lanes"]=a_to_green_lanes

            # set self.simple_state to True if you want a simplified state that already computes 
            # a nice feature (sum of all cars halting that will be getting a green)
            # for each action. 
            if self.simple_state:
                schema[aID]["dims"] = (num_actions, num_actions)
            else:
                schema[aID]["dims"] = (num_lanes, num_actions)

            # print(schema[aID]["dims"])
            # for a, lanes in schema[aID]["a_to_green_lanes"].items():
            #     print(f"action {a}, # green lanes={len(lanes)}")

        return schema        


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

        self._sumo.simulation.loadState('state.sumo')

        self._spawnVehicles()
        self._sumo.simulationStep()
        self.episode_step_countdown = self.episode_length

        multi_state, _ = self._get_states_and_rewards()
        return multi_state
    
    def step(self, action=None):
        """ Steps the simulation through one timestep
        
        Executes a single simulation step after passing an action to the
        environment. Returns the observation of the new state and the reward.

        Parameters
        ----------
        action : dict[int,int]
            This is a dict mapping between agentID and action to be taken by that agent
        
        Returns
        -------
        observation: dict[int,ndarray]
            This is a dict mapping between agentID and the new state of the simulation after the action was implemented

        reward: dict[int,float]
            This is a dict mapping between agentID and the reward obtained by the agent for that timestep

        done: boolean
            is true if the episode is finished
        """
        if action is None:
            action = self.choose_random_action()

        self._applyMultiaction(action)

        for _ in range(self.sumo_timestep):
            self._spawnVehicles()
            if self.use_gui and self.record_screenshots:
                self._sumo.gui.screenshot("View #0", f"{self.output_path}/sumo_screenshots/{self.total_steps_run:009}.png")
            self._sumo.simulationStep()
        
            if self.record_tracks:
                self._saveVehicles(f"{self.output_path}/sumo_tracks", use_total_time=True)
            self.total_steps_run+=1

        self.episode_step_countdown -= 1

        done = self.episode_step_countdown==0
        multi_state, multi_reward = self._get_states_and_rewards()

        return multi_state, multi_reward, done

    def choose_random_action(self):
        multi_action = dict()
        for agID in self.schema.keys():
            num_actions = self.schema[agID]["dims"][1]
            multi_action[agID]=random.randint(0, num_actions-1)
        return multi_action

    def choose_greedy_action(self):
        multi_action = dict()
        for agID in self.schema.keys():
            tlIDs = self.schema[agID]["controlledTLs"]            
            # this computes the current demand that will be let through for each TL phase
            tl_demand = {tl:self._get_TLS_demand_breakdown(tl) for tl in tlIDs}

            # finding the best action for that agend
            a_to_tl_to_pha = self.schema[agID]["a_to_tl_to_pha"] # mapping from agent action to individual TL phases
            best_action = -1
            highest_demand=-1
            for a, pha in a_to_tl_to_pha.items():
                demand = sum(tl_demand[tl][pha] for tl,pha in pha.items())
                if demand>highest_demand:
                    highest_demand = demand
                    best_action = a                    
            multi_action[agID]=best_action
        return multi_action


    def _get_green_lanes_per_phase(self, tls_id):
        ''' 
        Returns a dictionary that maps each phase to the list of lanes that turn green in a traffic light

        Parameters
        ----------
        tls_id:   str
            a traffic light ID
        
        Returns
        -------
        pha_to_green_lanes: int->[str]

        an array D such that if p is a phase of the traffic light, D[p] = the total demand waiting to be released if we enable phase p. 
        '''
        logic = self._sumo.trafficlight.getAllProgramLogics(tls_id)[0]
        lanes = self._sumo.trafficlight.getControlledLanes(tls_id)
        phases = logic.getPhases()
        pha_to_green_lanes=dict()
        for phase_id,phase in enumerate(phases):# phases are whole objects- we just need their index
            pha_to_green_lanes[phase_id]=[]
            for lane, s in zip(lanes,phase.state):
                if s in ['g','G']:
                    pha_to_green_lanes[phase_id].append(lane)
        return pha_to_green_lanes


    def _get_TLS_demand_breakdown(self, tls_id):
        ''' 
        Returns a breakdown of current demand for each phase.

        Parameters
        ----------
        tls_id:   str
            a traffic light ID
        
        Returns
        -------
        demand: [int]

        an array D such that if p is a phase of the traffic light, D[p] = the total demand waiting to be released if we enable phase p. 
        '''
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

    def _applyMultiaction(self, multi_action: dict):
        """ Applies a traffic control multi_action dict to the traffic lights
        """
        for agID, action in multi_action.items():
            tl_to_phase = self.schema[agID]["a_to_tl_to_pha"][action]
            for tlID,pha in tl_to_phase.items():
                self._sumo.trafficlight.setPhase(tlID,pha)

    def _get_states_and_rewards(self):
        multi_state:Dict[int, np.ndarray] = dict()
        multi_reward:Dict[int, float] = dict()

        for agID in self.schema.keys():
            multi_state[agID] = []
            lanes = self.schema[agID]["controlledlanes"]
            
            # set self.simple_state to True if you want a simplified state that already computes 
            # a nice feature (sum of all cars halting that will be getting a green)
            # for each action. 
            if self.simple_state:
                a_to_green_lanes = self.schema[agID]["a_to_green_lanes"]
                for a, green_lanes in a_to_green_lanes.items():
                    multi_state[agID].append(sum(self._sumo.lane.getLastStepHaltingNumber(lane) for lane in green_lanes))  
            else:
                for lane in lanes:
                    multi_state[agID].append(self._sumo.lane.getLastStepHaltingNumber(lane))
                    # multi_state[agID].append(self._sumo.lane.getLastStepOccupancy(lane))
 
            if self.state_wrapper is not None:
                multi_state[agID] = self.state_wrapper(multi_state[agID])

            multi_reward[agID] = -sum(self._sumo.lane.getLastStepHaltingNumber(lane) for lane in lanes)
            # multi_reward[agID] = sum(1.0-self._sumo.lane.getLastStepOccupancy(lane) for lane in lanes)
            # multi_reward[agID] = sum(self._sumo.lane.getLastStepMeanSpeed(lane) for lane in lanes)

        return multi_state, multi_reward

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

    def get_action_breakdown(self):
        """ Returns the schema of all independent decisions for all agents. It is a dict of dicts that maps agentID -> (obs_dim, num_actions) Obs_dim is the dimension of the observation vector while num_actions is the number of phases for that particular agent.        
        """
        minischema = dict()
        for agID in self.schema.keys():
            minischema[agID] = self.schema[agID]["dims"]
        return minischema
    
 

    def set_all_lights(self, state):
        """
        Sets all lights in all traffic junction to the same state. The input state is one of the chars of 'rugGyYuoO'
        """
        # print(f"setting all lights to {state}")
        for tlID in self._sumo.trafficlight.getIDList():
            n = len(self._sumo.trafficlight.getControlledLanes(tlID))
            self._sumo.trafficlight.setRedYellowGreenState(tlID, state * n)


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
        # print("_getAllRouteIDs")
        # print(self._getAllRouteIDs())
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
