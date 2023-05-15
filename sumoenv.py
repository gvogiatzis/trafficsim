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
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sumolib
import traci

import random, pickle

class TrafficControlEnv:
    """ The main class that interfaces to sumo.
    
    It exposes its basic functionality for initializing a network, setting up
    basic routes and spawning of vehicles. The simulation is stepped through
    with traffic light actions passed through and observations from lane
    occupancy received back from sumo.
    """
    def __init__(self, net_fname = 'sumo_data/test/Test.net.xml', vehicle_spawn_rate=0.015, state_wrapper=None, episode_length=500, sumo_timestep=20, use_gui=False, seed=None,step_length=1, output_path=None):
        """ A basic constructor. We read the network file with sumolib and we
        start the sumo (or sumo-gui) program. We then initialize routes and save
        the state for quick reloading whenever we reset.
        """
        random.seed(seed)
        self.total_steps_run=0
        self.current_episode = 0
        self.output_path = output_path
        self.episode_length = episode_length
        self._net_fname = net_fname
        self.vehicle_spawn_rate = vehicle_spawn_rate
        self._net = sumolib.net.readNet(self._net_fname, withPrograms=True)
        self._sumo = None
        self._vehcnt = 0
        self.state_wrapper = state_wrapper
        self.episode_step_countdown=episode_length
        self.sumo_timestep = sumo_timestep
        self.use_gui = use_gui
        sumo_command=['sumo-gui'] if self.use_gui else ['sumo']
        # sumo_command.extend(['-n',self._net_fname,'--start','--quit-on-end','--no-warnings','--no-step-log'])
        sumo_command.extend(['-n',self._net_fname,'--start','--quit-on-end','--no-warnings','--no-step-log', '--step-length', str(step_length)])

        if seed is not None:
            sumo_command.extend(['--seed',str(seed)])
        else:
            sumo_command.extend(['--random'])
        traci.start(sumo_command,verbose=True, label="sim1")

        self._sumo = traci.getConnection(label="sim1")
        self._initialize_routes()
        self._sumo.simulation.saveState('state.sumo')
        
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


    def reset(self, seed = None):
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

        self.actionCombinations = self._getActionCombinations()
        return self._getObservation()
    
    def step(self, action=None):
        """ Steps the simulation through one timestep
        
        Executes a single simulation step after passing an action to the
        environment. Returns the observation of the new state and the reward.

        Parameters
        ----------
        action : int
            the action to send to sumo. Must be between 0 and
            self.get_num_actions()-1. Each integer encodes a combination of
            traffic lights across all junctions. 
        
        Returns
        -------
        observation: numpy array
            The new state of the simulation after the action was implemented

        reward: float
            The reward obtained by the agent for that timestep

        done: boolean
            is true if the episode is finished
        """
        if action is not None:
            self._applyAction(action)
        for _ in range(self.sumo_timestep):            
            self._spawnVehicles()
            self._sumo.simulationStep()
            self.total_steps_run+=1
        
            if self.output_path is not None:
                self._saveVehicles(self.output_path, use_total_time=True)

        self.episode_step_countdown -= 1

        done = self.episode_step_countdown==0
        return self._getObservation(), -self._getCurrentTotalTimeLoss(), done
    
    def _saveVehicles(self, output_path, use_total_time=False):
        if use_total_time:
            fname = f"{output_path}/{self.total_steps_run:009}.pk"
        else:
            step = self.episode_length-self.episode_step_countdown
            fname = f"{output_path}/{self.current_episode:03}_{step:04d}.pk"
        with open(fname, "w") as f:
            for v in self._sumo.vehicle.getIDList():
                route_edges = self._sumo.vehicle.getRoute(v)
                x,y = self._sumo.vehicle.getPosition(v)
                e0 = route_edges[0]
                e1 = route_edges[-1]
                routeID = self.route_dict[e0][e1]
                f.write(f"{routeID}, {x}, {y}\n")

    def _applyAction(self, action: int):
        """ Applies a traffic control action to the traffic lights
        """
        actionCombinations = self._getActionCombinations()
        if len(actionCombinations)>0:
            for (tl, a) in actionCombinations[action]:
                self._sumo.trafficlight.setPhase(tl,a)



    def close(self):
        """
        Closes the simulation
        """
        if os.path.exists('state.sumo'):
            os.remove('state.sumo')
        if traci.isLoaded():
            self._sumo.close()
            self._sumo=None
        

    def _spawnVehicles(self):
        for routeID in self._getAllRouteIDs():
            if random.random() < self.vehicle_spawn_rate:
                vehID = f"veh{self._vehcnt:08d}"
                self._sumo.vehicle.add(vehID, routeID)
                self._vehcnt +=1

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

    def _getActionCombinations(self):
        result = []
        for tl in self._sumo.trafficlight.getIDList():
            logic = self._sumo.trafficlight.getAllProgramLogics(tl)[0]
            nphases = len(logic.getPhases())
            if len(result)==0:
                result = [[(tl, n)] for n in range(nphases)]
            else:
                result = [[(tl,n)] + d for n in range(nphases) for d in result]                    
        return result

    def _getAllRouteIDs(self):
        return [s for s in self._sumo.route.getIDList() if s[0]!='!']
    
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
    
    def _getObservation(self):
        # lanes = self._sumo.lane.getIDList() # This includes :xyz internals
        lanes = sum([e.getLanes() for e in self._net.getEdges()],[])
        result = []
        for lane in lanes:
            # result.append(self._sumo.lane.getLastStepVehicleNumber(lane.getID()))
            result.append(self._sumo.lane.getLastStepHaltingNumber(lane.getID()))
        if self.state_wrapper is not None:
            return self.state_wrapper(np.array(result))
        else:
            return np.array(result)

    def _initialize_routes(self):
        self.route_dict = defaultdict(dict)
        routecnt = 0
        edges = self._net.getEdges()
        for e1 in edges:
            if e1.is_fringe():
                for e2 in self._net.getReachable(e1):
                    if e2.is_fringe():
                        if e1!=e2 and e1.getID().replace('-','') != e2.getID().replace('-',''):
                            routeID = f"route{routecnt:04d}:{e1.getID()}->{e2.getID()}"
                            self._sumo.route.add(routeID, [e1.getID(), e2.getID()])
                            self.route_dict[e1.getID()][e2.getID()] = routeID
                            routecnt += 1

    def get_route_trajectories(self, save_file = None, plot_traj=False):
        route_traj = dict()
        for routeID in self._getAllRouteIDs():
            route=[]
            vehID = "veh"+routeID
            self._sumo.vehicle.add(vehID, routeID)
            self._sumo.simulationStep()
            while self._sumo.vehicle.getIDCount()>0:
                route.append(self._sumo.vehicle.getPosition(vehID))
                self._sumo.simulationStep()
            route_traj[routeID] = route
        if save_file is not None:
            with open(save_file, 'wb') as f:
                pickle.dump(route_traj,f)
        if plot_traj:
            plt.figure()
            for r,xy in route_traj.items():
                X = [x for (x,y) in xy]
                Y = [y for (x,y) in xy]
                plt.plot(X,Y,'-')
            plt.show()

        return route_traj


if __name__ == "__main__":
    # env = TrafficControlEnv(net_fname = 'RussianJunction/RussianJunctionNoLights.net.xml', step_length=0.01, use_gui=True)

    # route_traj = env.get_route_trajectories(save_file="sumo_routes.pts", plot_traj=True)
    # env.close()



    # R=[]
    # env.reset()
    # A = env.get_num_actions()
    # for t in range(1000):
    #     obs, r, done = env.step(randint(0,A-1))
    #     R.append(r)
    # env.close()
    # plt.plot(R,'b-', label="negative total time loss")
    # plt.legend()
    # plt.show()

    env = TrafficControlEnv(net_fname = 'RussianJunction/RussianJunctionNoLights.net.xml', use_gui=False, output_path="sumo_outputs", step_length=0.1)
    print(env._sumo.route.getIDList())
    env.close()

    # from random import randint
    # env = TrafficControlEnv(net_fname = 'RussianJunction/RussianJunctionNoLights.net.xml', use_gui=True, output_path="sumo_outputs", step_length=0.1)
    # env.reset()
    # done = False
    # for i in range(50):
    #     obs, r, done = env.step()
    # A = env.get_num_actions()
    # if A == 0:
    #     while not done:
    #         obs, r, done = env.step()
    # else:
    #     while not done:
    #         obs, r, done = env.step(randint(0,A-1))
