from sklearn.neighbors import NearestNeighbors
from  scipy.interpolate import interp1d
import numpy as np
import pickle



class SumoToRealConverter:
    def __init__(self, real_route_fname, sumo_routes_fname):
        with open(real_route_fname, 'rb') as f:
            r = pickle.load(f)
            self.real_route_verts = r['trajectories']
            self.real_route_waypts = r['waypoints']
        with open(sumo_routes_fname,'rb') as f:
            r = pickle.load(f)
            self.sumo_routes = r['trajectories']
            self.sumo_waypoints = r['waypoints']

        self.sumo_route_t=dict()
        self.real_route_t=dict()
        self.nbrs = dict()
        self.realt_to_xfun = dict()
        self.realt_to_yfun = dict()
        self.sumot_to_realt_fun = dict()

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
            self.realt_to_xfun[routeID] = interp1d(t, xs, kind='cubic')
            self.realt_to_yfun[routeID] = interp1d(t, ys, kind='cubic')

        for routeID, sumo_wps in self.sumo_waypoints.items():
            if routeID in self.real_route_waypts:
                real_wps = self.real_route_waypts[routeID]
                ts_sumo = [0.0]
                ts_real = [0.0]
                for i in range(min(len(real_wps), len(sumo_wps))):
                    ts_sumo.append(self._sumo_wpoint_tval(routeID,*sumo_wps[i]))
                    ts_real.append(self._real_wpoint_tval(routeID,*real_wps[i]))
                ts_sumo.append(1.0)
                ts_real.append(1.0)
                print(f"{routeID}: {list(zip(ts_sumo,ts_real))}")
                self.sumot_to_realt_fun[routeID] = interp1d(ts_sumo, ts_real, kind='linear')

    def _real_wpoint_tval(self, routeID, wp_x,wp_y):
        ts = np.linspace(0,1,100)
        xfun = self.realt_to_xfun[routeID]
        yfun = self.realt_to_yfun[routeID]
        xs = xfun(ts)
        ys = yfun(ts)
        d = np.sqrt((xs-wp_x)**2+(ys-wp_y)**2)
        return ts[d.argmin()]


    def _sumo_wpoint_tval(self, routeID, wp_x,wp_y):
        dist,ind = self.nbrs[routeID].kneighbors([(wp_x,wp_y)])
        dist = dist.squeeze()
        ind = ind.squeeze()
        ts = self.sumo_route_t[routeID]

        # the closer we are to each point, the further the t value should be 
        # from that point.
        if np.abs(dist[0]+dist[1])<1e-6:
            return 0.5*(ts[ind[0]]+ts[ind[1]])
        else:
            t = (ts[ind[0]]*dist[1] + ts[ind[1]]*dist[0])/(dist[0]+dist[1])
            return t


    def _convert_vehicle(self,routeID, x,y):
        dist,ind = self.nbrs[routeID].kneighbors([(x,y)])
        dist = dist.squeeze()
        ind = ind.squeeze()
        ts = self.sumo_route_t[routeID]

        # the closer we are to each point, the further the t value should be 
        # from that point.
        sumo_t = (ts[ind[0]]*dist[1] + ts[ind[1]]*dist[0])/(dist[0]+dist[1])

        real_t = self.sumot_to_realt_fun[routeID](sumo_t)

        xi = self.realt_to_xfun[routeID](real_t)
        yi = self.realt_to_yfun[routeID](real_t)

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