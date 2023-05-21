from sklearn.neighbors import NearestNeighbors
from  scipy.interpolate import interp1d, make_interp_spline
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob

class BboxHistogram:
    def __init__(self, real_route_fname, bbox_data_path):
        self.real_route_fname = real_route_fname

        with open(real_route_fname, 'rb') as f:
            r = pickle.load(f)
            self.real_route_verts = r['trajectories']
            self.real_route_waypts = r['waypoints']
        self.route_id_pts=dict()

        box_fnames = sorted(glob.glob(bbox_data_path))
        box_data = np.vstack([np.loadtxt(f) for f in box_fnames])
        self.box_data = box_data
        self.box_data = self.box_data[(self.box_data[:,0]==2),:]
        self.box_data_nrnb = \
            NearestNeighbors(n_neighbors=10).fit(self.box_data[:,1:3]+0.5*self.box_data[:,3:5])

    def find_nearest(self, x, y):
        d, i = self.box_data_nrnb.kneighbors([[x,y]])
        return np.median(self.box_data[i,:],axis=1).squeeze()