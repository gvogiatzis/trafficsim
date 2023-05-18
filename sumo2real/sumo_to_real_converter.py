from sklearn.neighbors import NearestNeighbors
from  scipy.interpolate import interp1d
import numpy as np
import glob
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle

import typer


class SumoToRealConverter:
    def __init__(self, real_route_fname, sumo_routes_fname):
        with open(real_route_fname, 'rb') as f:
            self.real_route_verts = pickle.load(f)
        with open(sumo_routes_fname,'rb') as f:
            self.sumo_routes = pickle.load(f)

        self.sumo_route_t=dict()
        self.real_route_t=dict()
        self.nbrs = dict()
        self.xfun = dict()
        self.yfun = dict()

        for routeID, verts in self.sumo_routes.items():
            verts = np.array(verts)
            d = np.insert(np.sqrt(np.sum((verts[1:]-verts[:-1]) **2, axis=1)).cumsum(),0,0)
            t = d/d[-1]
            self.sumo_route_t[routeID] = t
            self.nbrs[routeID] = NearestNeighbors(n_neighbors=2).fit(verts)

        for routeID, verts in self.real_route_verts.items():
            verts = np.array(verts)
            d = np.insert(np.sqrt(np.sum((verts[1:]-verts[:-1]) **2, axis=1)).cumsum(),0,0)
            t = d/d[-1]
            self.real_route_t[routeID] = t

            xs,ys=list(zip(*verts))
            self.xfun[routeID] = interp1d(t, xs, kind='cubic')
            self.yfun[routeID] = interp1d(t, ys, kind='cubic')


    def _convert_vehicle(self,routeID, x,y):
        dist,ind = self.nbrs[routeID].kneighbors([(x,y)])
        dist = dist.squeeze()
        ind = ind.squeeze()
        ts = self.sumo_route_t[routeID]

        # the closer we are to each point, the further the t value should be 
        # from that point.
        t = (ts[ind[0]]*dist[1] + ts[ind[1]]*dist[0])/(dist[0]+dist[1])

        xi = self.xfun[routeID](t)
        yi = self.yfun[routeID](t)

        return xi,yi



    def convert(self, vehiclelist_fname):
        """
        vehiclelist_fname is the file name that contains a comma separated vehiclelist. 
        
        vehiclelist is a list of (x,y,routeID) triplets
        
        output is list of 
        """
        vehiclelist=[]
        with open(vehiclelist_fname, "r") as f:
            for line in f.readlines():
                words=[s.strip() for s in line.split(',')]
                vehiclelist.append([words[0], float(words[1]),float(words[2])])
        result=[]
        
        for routeID, x, y in vehiclelist:
            if routeID in self.real_route_verts:
                # print(f"route {routeID} has been defined ")
                xi,yi = self._convert_vehicle(routeID, x,y)
                result.append([xi,yi])
        return result