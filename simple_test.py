import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import matplotlib.pyplot as plt

import sumolib
import traci


from random import random
net_fname = 'sumo_data/test/Test.net.xml'
# net_fname = 'sumo_data/ThreeLaneJunction.net.xml'

def getActionCombinations():
    result = []
    for tl in traci.trafficlight.getIDList():
        logic = traci.trafficlight.getAllProgramLogics(tl)[0]
        nphases = len(logic.getPhases())
        if len(result)==0:
            result = [[(tl, n)] for n in range(nphases)]
        else:
            result = [[(tl,n)] + d for n in range(nphases) for d in result]                    
    return result

def get_TLS_demand_breakdown(tls_id):
    logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    lanes = traci.trafficlight.getControlledLanes(tls_id)
    phases = logic.getPhases()
    demand = []
    for phase in phases:
        phasedemand=0
        for lane, s in zip(lanes,phase.state):
            if s in ['g','G']:
                phasedemand += traci.lane.getLastStepVehicleNumber(lane)
                # phasedemand += traci.lane.getLastStepOccupancy(lane)
        demand.append(phasedemand) 
    return demand

def getCurrentTotalTimeLoss():
    dt = traci.simulation.getDeltaT()
    timeloss = 0

    vehIDs = traci.vehicle.getIDList()
    for vehID in vehIDs:
        Vmax = traci.vehicle.getAllowedSpeed(vehID)
        V = traci.vehicle.getSpeed(vehID)
        timeloss += (1 - V/Vmax) * dt
    return timeloss

def init_routes():
    edges = net.getEdges()
    route_ids=[]
    routecnt = 0
    edge_counts=dict()
    for e1 in edges:
        edge_counts[e1.getID()]=[]
        if e1.is_fringe():
            for e2 in net.getReachable(e1):
                if e2.is_fringe():
                    if e1!=e2 and e1.getID().replace('-','') != e2.getID().replace('-',''):
                        routeID = f"route{routecnt:04d}:{e1.getID()}->{e2.getID()}"
                        traci.route.add(routeID, [e1.getID(), e2.getID()])
                        routecnt += 1
                        route_ids.append(routeID)


vehcnt=0


p = 0.015


timeloss=[]
net = sumolib.net.readNet(net_fname)
# traci.start(['sumo-gui','-n',net_fname,'--start', '-d 20'])
traci.start(['sumo-gui','-n',net_fname,'--start','--quit-on-end'])
# traci.start(['sumo','-n',net_fname,'--start'])

init_routes()
edge_counts = dict()

# while traci.simulation.getMinExpectedNumber() > 0:
for i in range(1000):
    if i % 1 ==0:
        for tl_id in traci.trafficlight.getIDList():
            demand = get_TLS_demand_breakdown(tl_id)
            best_phase = np.argmax(demand)
            traci.trafficlight.setPhase(tl_id,best_phase)

    if i%200 == 0 and i>0:       
        traci.close()
        traci.start(['sumo-gui','-n',net_fname,'--start','--quit-on-end'])
        init_routes()
        
    for edgeID, edgecount in edge_counts.items():
        edgecount.append(traci.edge.getLastStepVehicleNumber(edgeID))        
    timeloss.append(getCurrentTotalTimeLoss())
    route_ids = [s for s in traci.route.getIDList() if s[0]!='!']
    print(len(route_ids))
    for routeID in route_ids:
        if random() < p:
            vehID = f"veh{vehcnt:08d}"
            traci.vehicle.add(vehID, routeID)
            vehcnt +=1
    traci.simulationStep()

# traci.close()

# for edgeID, edgecount in edge_counts.items():
#     plt.plot(edgecount, '-', label=edgeID)
# plt.legend()
# plt.show()

plt.plot(timeloss, '-', label='time loss')
plt.legend()
plt.show()
