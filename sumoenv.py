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

import numpy as np
import matplotlib.pyplot as plt
import sumolib
import traci

from random import random, randint


class TrafficControlEnv:
    """ The main class that interfaces to sumo.
    
    It exposes its basic functionality for initializing a network, setting up
    basic routes and spawning of vehicles. The simulation is stepped through
    with traffic light actions passed through and observations from lane
    occupancy received back from sumo.
    """
    def __init__(self, net_fname = 'sumo_data/test/Test.net.xml', vehicle_spawn_rate=0.015):
        """ A basic constructor
        """

        self._net_fname = net_fname
        self.vehicle_spawn_rate = vehicle_spawn_rate
        self._net = sumolib.net.readNet(self._net_fname)
        self._sumo = None
        self._vehcnt = 0
        

    def get_num_actions(self):
        """ Returns the number of possible actions corresponding to all traffic
        light combinations """

        tls = self._sumo.trafficlight.getIDList()
        dim = 1
        for tl in tls:
            logic = self._sumo.trafficlight.getAllProgramLogics(tl)[0]
            dim *= len(logic.getPhases())
        return dim
    
    def sample_action(self):
        """ Picks a random action
        
        The action picked is an integer between 0 and self.get_num_actions()-1
        inclusive 
        """
        return randint(0, self.get_num_actions()-1)    
    
    
    def get_obs_dim(self):
        """ Returns the dimensionality of the observation vector
        """

        return self._sumo.lane.getIDCount()


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

        """

        if self._sumo is not None and self._sumo.isLoaded():
            self._sumo.close()
            
        traci.start(['sumo-gui','-n',self._net_fname,'--start','--quit-on-end'])
        self._sumo = traci.getConnection()
        self._initialize_routes()
        self._spawnVehicles()
        self._sumo.simulationStep()

        self.actionCombinations = self._getActionCombinations()
        return self._getObservation(), 0.0
    
    def step(self, action):
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
        """

        for (tl, a) in self._getActionCombinations()[action]:
            self._sumo.trafficlight.setPhase(tl,a)
        self._spawnVehicles()
        self._sumo.simulationStep()
        return self._getObservation(), -self._getCurrentTotalTimeLoss()
    
    def close(self):
        """
        Closes the simulation
        """
        if traci.isLoaded():
            self._sumo.close()
            self._sumo=None

    def _spawnVehicles(self):
        for routeID in self._getAllRouteIDs():
            if random() < self.vehicle_spawn_rate:
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
        lanes = self._sumo.lane.getIDList()
        result = []
        for lane in lanes:
            result.append(self._sumo.lane.getLastStepVehicleNumber(lane))
        return np.array(result)

    def _initialize_routes(self):
        routecnt = 0
        edge_counts=dict()
        edges = self._net.getEdges()
        for e1 in edges:
            edge_counts[e1.getID()]=[]
            if e1.is_fringe():
                for e2 in self._net.getReachable(e1):
                    if e2.is_fringe():
                        if e1!=e2 and e1.getID().replace('-','') != e2.getID().replace('-',''):
                            routeID = f"route{routecnt:04d}:{e1.getID()}->{e2.getID()}"
                            self._sumo.route.add(routeID, [e1.getID(), e2.getID()])
                            routecnt += 1
    

if __name__ == "__main__":
    env = TrafficControlEnv()

    R=[]

    env.reset()
    A = env.get_num_actions()
    for t in range(1000):
        obs, r = env.step(randint(0,A-1))
        R.append(r)

    env.close()

    plt.plot(R,'b-', label="negative total time loss")
    plt.legend()
    plt.show()
