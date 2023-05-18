from sklearn.neighbors import NearestNeighbors
from  scipy.interpolate import interp1d, make_interp_spline
import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

class BboxHistogram:
    def __init__(self, real_route_fname, bbox_data_path, background_img_fname):
        self.real_route_fname = real_route_fname
        with open(real_route_fname, 'rb') as f:
            self.real_route_verts = pickle.load(f)
        self.route_id_pts=dict()
                
        self.route_pts = np.vstack([self._interpolate(verts) for routeID, verts in self.real_route_verts.items()])
        self.nrnb = NearestNeighbors(n_neighbors=1).fit(self.route_pts)
        self.background_img = plt.imread(background_img_fname)
        self.pixel_dist = 10
        self.N_bins = 100
        self.hist=[]

        h,w,c=self.background_img.shape

        box_fnames = sorted(glob.glob(bbox_data_path))
        box_data = np.vstack([np.loadtxt(f) for f in box_fnames])
        box_data[:,1] *= w
        box_data[:,3] *= w
        box_data[:,2] *= h
        box_data[:,4] *= h
        self.box_data = box_data
        dist,ind = self.nrnb.kneighbors(box_data[:,1:3])
        dist = dist.squeeze()
        ind = ind.squeeze()

        non_orphan_route_pts=[]
        for i in tqdm(range(len(self.route_pts))):
            # print(f"{i+1}/{len(self.route_pts)}")
            c = (box_data[:,0]==2) & (dist<self.pixel_dist) & (ind==i)
            histsampler = self._HistSampler(box_data[c,3:5])
            if histsampler.cdf is not None:
                non_orphan_route_pts.append(self.route_pts[i,:])
                self.hist.append(histsampler)
        
        self.route_pts = np.array(non_orphan_route_pts)
        self.nrnb = NearestNeighbors(n_neighbors=1).fit(self.route_pts)

    def sample(self, xy):
        d, i = self.nrnb.kneighbors(xy)
        xy_sample = self.hist[i.item()].sample()
        return xy_sample

    class _HistSampler:
        def __init__(self, data, nbins=25):
            self.hist, x_bins,y_bins = np.histogram2d(data[:,0],data[:,1],nbins)
            self.x_bins = (x_bins[:-1] + x_bins[1:])/2
            self.y_bins = (y_bins[:-1] + y_bins[1:])/2
            cdf = np.cumsum(self.hist.flatten())
            if cdf[-1]==0:
                self.cdf=None
            else:
                self.cdf = cdf / cdf[-1]

        def sample(self, n=1):
            if self.cdf is None:
                return None
            else:
                values = np.random.rand(n)
                value_bins = np.searchsorted(self.cdf, values)
                x_idx, y_idx = np.unravel_index(value_bins,
                                                (len(self.x_bins),
                                                len(self.y_bins)))
                random_from_cdf = np.column_stack((self.x_bins[x_idx],
                                                self.y_bins[y_idx]))
                new_x, new_y = random_from_cdf.T
                return np.array(list(zip(new_x,new_y)))
        

    
    def _interpolate(self, verts):
        verts = np.array(verts)
        d = np.insert(np.sqrt(np.sum((verts[1:]-verts[:-1]) **2, axis=1)).cumsum(),0,0)
        t = d/d[-1]
        ts = np.linspace(0.0,1.0,int(d[-1]/5)) # interpolate approx. every 5 pixels along the spline

        return make_interp_spline(t, verts)(ts)




