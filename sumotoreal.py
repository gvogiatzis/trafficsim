import numpy as np
from  scipy.interpolate import interp1d, make_interp_spline
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import pickle


class RouteEditor:
    def __init__(self, real_img_fname, sumo_routes_fname, save_file_name=None, load_file_name=None):
        self.selected_line_width = 5
        self.unselected_line_width = 1

        self.showverts=True
        self.epsilon = 10
        self._ind = None #index of selected vertex of selected route
        self.save_file_name = save_file_name
        self.load_file_name = load_file_name

        with open(sumo_routes_fname,'rb') as f:
            self.routes = pickle.load(f)

        self.fig_real, self.ax_real = plt.subplots()
        self.fig_sumo, self.ax_sumo = plt.subplots()
        self.fig_real.canvas.manager.set_window_title("Route editor")
        self.fig_sumo.canvas.manager.set_window_title("SUMO route selector")

        self.image = plt.imread(real_img_fname)
        self.ax_real.imshow(self.image)


        self.sumo_route_lines=dict()
        for r,xy in self.routes.items():
            X = [x for (x,y) in xy]
            Y = [y for (x,y) in xy]
            self.sumo_route_lines[r] = self.ax_sumo.plot(X,Y,'-', linewidth=1)[0]
        self.selected_route = list(self.sumo_route_lines.keys())[0]

        
        self.real_route_splines=dict()
        self.real_route_lines=dict()
        self.real_route_verts=dict()

        if self.load_file_name is not None:
            self._load_routes(self.load_file_name)
        # self._update_all()

        self.fig_sumo.canvas.mpl_connect('key_press_event', self._on_key_press_sumo)
        self.fig_real.canvas.mpl_connect('key_press_event', self._on_key_press_real)
        self.fig_real.canvas.mpl_connect('button_press_event', self._on_button_press_real)
        self.fig_real.canvas.mpl_connect('button_release_event', self._on_button_release_real)
        self.fig_real.canvas.mpl_connect('motion_notify_event', self._on_mouse_move_real)

    def start_gui(self):
        plt.show()

    def stop_gui(self):
        plt.close()


    def _on_mouse_move_real(self, event):
        """Callback for mouse movements."""
        if (self._ind is None
                or event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return

        srID = self.selected_route
        if srID not in self.real_route_verts:
            return
        
        verts = self.real_route_verts[srID]
        verts[self._ind] = event.xdata, event.ydata
        # line = self.real_route_lines[srID]
        # line.set_data(zip(*verts))
        # self.fig_real.canvas.draw_idle()
        # self.fig_real.canvas.flush_events()
        self._update_selected()


    def _update_all(self):
        for routeID, verts in self.real_route_verts.items():            
            if routeID not in self.real_route_lines:
                self.real_route_lines[routeID] = Line2D([],[],linewidth=self.unselected_line_width,marker='o',markerfacecolor='w',markersize=self.unselected_line_width, ls='--', color=self.sumo_route_lines[routeID].get_color())
                self.real_route_splines[routeID] = Line2D([],[],linewidth=self.unselected_line_width,ls='-',  color=self.sumo_route_lines[routeID].get_color())
                self.ax_real.add_line(self.real_route_splines[routeID])  
                self.ax_real.add_line(self.real_route_lines[routeID])        
            line = self.real_route_lines[routeID]
            spline = self.real_route_splines[routeID]
            line.set_data(zip(*verts))
            if len(verts)>=4:
                x, y = self._interpolate(*zip(*verts))
                spline.set_data(x, y)
            else:   
                spline.set_data([], [])

        self.sumo_route_lines[self.selected_route].set_linewidth(self.selected_line_width)
        if self.selected_route in self.real_route_lines:
            self.real_route_lines[self.selected_route].set_linewidth(self.selected_line_width)
            self.real_route_lines[self.selected_route].set_markersize(self.selected_line_width)
            self.real_route_splines[self.selected_route].set_linewidth(self.selected_line_width)
            
        self.fig_sumo.canvas.draw_idle()
        self.fig_sumo.canvas.flush_events()
        self.fig_real.canvas.draw_idle()
        self.fig_real.canvas.flush_events()

    def _update_selected(self):
        if self.selected_route in self.real_route_verts:
            verts = self.real_route_verts[self.selected_route]                
            line = self.real_route_lines[self.selected_route]
            line.set_data(zip(*verts))
            spline = self.real_route_splines[self.selected_route]
            if len(verts)>=4:
                x, y = self._interpolate(*zip(*verts))
                spline.set_data(x, y)
            else:
                spline.set_data([], [])
        
            background = self.fig_real.canvas.copy_from_bbox(self.ax_real.bbox)
            self.fig_real.canvas.restore_region(background)
            self.ax_real.draw_artist(spline)
            self.ax_real.draw_artist(line)
            self.fig_real.canvas.blit(self.ax_real.bbox)
        else:
            self.fig_sumo.canvas.draw_idle()
            self.fig_sumo.canvas.flush_events()
            self.fig_real.canvas.draw_idle()
            self.fig_real.canvas.flush_events()

    def _on_key_press_real(self, event):
        if event.key in ["up", "down"]: # change selected route
            delta = 1 if event.key=="up" else -1
            self._modify_selected_route(delta, inreal=True)
        if event.key == 'a': #add point
            srID = self.selected_route
            if srID not in self.real_route_verts:
                self.real_route_verts[srID] = []
                self.real_route_lines[srID] = Line2D([],[],linewidth=self.selected_line_width,marker='o',ls='--', markerfacecolor='w', markersize=self.selected_line_width, color=self.sumo_route_lines[srID].get_color())
                
                self.real_route_splines[srID] = Line2D([],[],linewidth=self.selected_line_width,ls='-', color=self.sumo_route_lines[srID].get_color())
                self.ax_real.add_line(self.real_route_splines[srID])        
                self.ax_real.add_line(self.real_route_lines[srID])        
            verts = self.real_route_verts[srID]
            verts.append((event.xdata, event.ydata))
        if event.key == 'd': #delete point
            if self.selected_route in self.real_route_verts:
                verts = self.real_route_verts[self.selected_route]
                t = self._get_ind_under_point(event)
                if t is not None:
                    del verts[t]
        elif event.key == 'i':  #insert point
            t = self._get_lineseg_under_point(event)
            if t is not None:
                verts = self.real_route_verts[self.selected_route]
                verts.insert(t+1, (event.xdata, event.ydata))
        elif event.key == 'w': #save route data
            if self.save_file_name is not None:
                self._save_routes(self.save_file_name)
                print(f"Wrote routes in {self.save_file_name}")
                
        self._update_all()


            # self.line.set_data(zip(*self.verts))
            # x, y = interpolate(*zip(*self.verts))
            # self.spline.set_data(x, y)

    def _on_key_press_sumo(self,event):
        print(event.key)
        if event.key in ["up", "down"]:
            delta = 1 if event.key=="up" else -1
            self._modify_selected_route(delta)
        self._update_all()

    def _on_button_press_real(self, event):
        """Callback for mouse button presses."""
        if (event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts):
            return
        self._ind = self._get_ind_under_point(event)

    def _on_button_release_real(self, event):
        """Callback for mouse button releases."""
        if (event.button != MouseButton.LEFT
                or not self.showverts):
            return
        self._ind = None

    def _save_routes(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.real_route_verts, f)

    def _load_routes(self, filename):
        with open(filename, 'rb') as f:
            self.real_route_verts = pickle.load(f)
        self._update_all()


    def _modify_selected_route(self, delta, inreal=False):
        route_keys = self.real_route_lines.keys() if inreal else self.sumo_route_lines.keys()
        routeIDs = list(route_keys)
        old_routeID = self.selected_route
        if self.selected_route in routeIDs:
            i = routeIDs.index(self.selected_route)
            i_new = (i+delta) % len(routeIDs)
            new_routeID = routeIDs[i_new]
        else:
            new_routeID=routeIDs[0]
        
        ulw = self.unselected_line_width
        slw = self.selected_line_width

        self.sumo_route_lines[old_routeID].set_linewidth(ulw)
        self.sumo_route_lines[new_routeID].set_linewidth(slw)
        
        if old_routeID in self.real_route_lines:
            self.real_route_lines[old_routeID].set_linewidth(ulw)
            self.real_route_lines[old_routeID].set_markersize(ulw)
            self.real_route_splines[old_routeID].set_linewidth(ulw)
        if new_routeID in self.real_route_lines:
            self.real_route_lines[new_routeID].set_linewidth(slw)
            self.real_route_lines[new_routeID].set_markersize(slw)
            self.real_route_splines[new_routeID].set_linewidth(slw)
        self.selected_route = new_routeID

    def _get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """        
        srID = self.selected_route
        if srID not in self.real_route_verts:
            return None
        verts = self.real_route_verts[srID]
        line = self.real_route_lines[srID]
        xyt = line.get_transform().transform(verts)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        ind = d.argmin()
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def _get_lineseg_under_point(self, event):
        srID = self.selected_route
        verts = self.real_route_verts[srID]
        line = self.real_route_lines[srID]
        xyt = line.get_transform().transform(verts)
        p = (event.x, event.y)
        d_min = np.inf
        t_min = None
        for t in range(len(verts)-1):
            s0 = xyt[t]
            s1 = xyt[t + 1]
            d = self._dist_point_to_segment(p, s0, s1)  
            if d<d_min:
                d_min = d
                t_min = t
        if d_min > self.epsilon:
            t_min = None
        return t_min


    def _dist(self, x, y):
        """
        Return the distance between two points.
        """
        d = x - y
        return np.sqrt(np.dot(d, d))

    def _dist_point_to_segment(self, p, s0, s1):
        """
        Get the distance of a point to a segment.
        *p*, *s0*, *s1* are *xy* sequences
        This algorithm from
        http://geomalgorithms.com/a02-_lines.html
        """
        v = s1 - s0
        w = p - s0
        c1 = np.dot(w, v)
        if c1 <= 0:
            return self._dist(p, s0)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return self._dist(p, s1)
        b = c1 / c2
        pb = s0 + b * v
        return self._dist(p, pb)

    def _interpolate(self, x, y):
        i = np.arange(len(x))

        interp_i = np.linspace(0, i.max(), 100 * i.max())

        xi = interp1d(i, x, kind='cubic')(interp_i)
        yi = interp1d(i, y, kind='cubic')(interp_i)

        return xi,yi



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
            print(routeID)
            if routeID in self.real_route_verts:
                # print(f"route {routeID} has been defined ")
                xi,yi = self._convert_vehicle(routeID, x,y)
                result.append([xi,yi])
        return result

from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob
import time

if __name__ == "__main__":
    route_editor = RouteEditor("RussianJunction/image.png", "sumo_routes.pk", save_file_name="routes.pk", load_file_name="routes.pk")
    # route_editor = RouteEditor("RussianJunction/image.png", "sumo_routes.pts", save_file_name="routes.pk")
    route_editor.start_gui()

    # converter = SumoToRealConverter("routes.pk", "sumo_routes.pts")
    # image = plt.imread("RussianJunction/image.png")
    
    # fig, ax = plt.subplots()
    # # plt.ion()
    # ax.imshow(image)
    # line = Line2D([],[],marker='o',markersize=5,color='r',ls='')
    # ax.add_line(line)
   
    # tracks = sorted(glob.glob("sumo_outputs/*.pk"))

    # for f in sorted(glob.glob("sumo_outputs/*.pk")):
    #     print(f)
    #     r=converter.convert(f)
    #     if len(r)==0:
    #         line.set_data([],[])
    #     else:      
    #         line.set_data(zip(*r))
    #     plt.pause(0.001)
    #     # time.sleep(0.1)

    # r=converter.convert("sumo_outputs/000_0332.pk")

