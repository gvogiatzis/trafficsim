import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumo2real import SumoToRealConverter, BboxHistogram
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import glob
import typer
from tqdm import tqdm
from typing import Tuple
from typing_extensions import Annotated




app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command("fitbbox")
def fitbbox(background_img_fname,
            real_route_fname="real_routes.pk",
            bbox_data_path="boxes/*.txt", 
            save_fname="bboxhist.pk",
            output_path="output/converter"):
    """
    Fit bbox distribution to real data from yolo_v7
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    real_route_fname =    os.path.join(output_path,real_route_fname)
    save_fname =    os.path.join(output_path,save_fname)

    bbox_hist = BboxHistogram(real_route_fname=real_route_fname,bbox_data_path=bbox_data_path, background_img_fname=background_img_fname)
    with open(save_fname, "wb") as f:
        pickle.dump(bbox_hist, f)

@app.command("convertseq")
def convertseq(img_fname,
               real_route_fname = os.path.join("converter","real_routes.pk"),
               sumo_routes_fname = os.path.join("converter","sumo_routes.pk"),
               bbox_hist_fname = os.path.join("converter","bboxhist.pk"),
               sumo_track_path = "sumo_tracks",
               output_figs_path= "sumo_to_real_figs",
               output_txt_path = "sumo_to_real_txt",
               output_path = "output",
               image_wh: Annotated[Tuple[int,int], typer.Option()] = (1920,1080),
               save_fig: bool = True,
               draw_fig: bool = True):
    """
    Convert a sumo track sequence to real image (x,y,w,h) tracks
    """ 

    output_figs_path = os.path.join(output_path,output_figs_path)
    output_txt_path = os.path.join(output_path,output_txt_path)
    sumo_track_path = os.path.join(output_path,sumo_track_path)
    real_route_fname = os.path.join(output_path,real_route_fname)
    sumo_routes_fname = os.path.join(output_path,sumo_routes_fname)
    bbox_hist_fname = os.path.join(output_path,bbox_hist_fname)
    
    w,h = image_wh
    if (w is None or h is None) and not draw_fig and not save_fig:
        print("Unknown image width/height. Please enter the save_fig or draw_fig option or input the size with the image_wh option.")
        raise typer.Abort()
    if save_fig and not os.path.exists(output_figs_path):
        os.makedirs(output_figs_path)
    if not os.path.exists(output_txt_path):
        os.makedirs(output_txt_path)
    
    converter = SumoToRealConverter(real_route_fname=real_route_fname, sumo_routes_fname=sumo_routes_fname)

    if save_fig or draw_fig:
        fig, ax = plt.subplots()
        ax.imshow(plt.imread(img_fname))
        line = Line2D([],[],marker='o',ls='',color='r',markersize=2)    
        line2 = Line2D([],[],marker='o',ls='',color='b',markersize=2)    
        ax.add_line(line)
        ax.add_line(line2)

    with open(bbox_hist_fname, "rb") as f:
        bbox_hist = pickle.load(f)

    for i, f in enumerate(tqdm(sorted(glob.glob(f"{sumo_track_path}/*.txt")))):
        r=converter.convert(f)
        with open(f"{output_txt_path}/frame{i:06d}.txt", "w") as f:
            xs,ys=[],[]
            for xy in r:
                det = bbox_hist.find_nearest(*xy)
                f.write(", ".join(str(n) for n in det)+"\n")
                bbox_x = det[1]*w
                bbox_y = det[2]*h
                bbox_w = det[3]*w
                bbox_h = det[4]*h
                xs.append(bbox_x)
                ys.append(bbox_y)
                
                # wh = bbox_hist.sample([xy])
                # if wh is not None:
                #     wh = wh.squeeze()
                #     xy -= wh/2
                #     f.write(f"2, {xy[0]/w}, {xy[1]/h}, {wh[0]/w}, {wh[1]/h}, 1.0\n")
                if draw_fig or save_fig:
                    ax.add_patch(Rectangle([bbox_x,bbox_y],bbox_w,bbox_h,fill=None, alpha=1))
            if len(r)==0:
                line.set_data([],[])
                line2.set_data([],[])
            else:      
                line.set_data(xs, ys)
                line2.set_data(*zip(*r))
        plt.pause(0.001)
        if save_fig:
            fig.savefig(f"{output_figs_path}/img{i:06d}.png")
        if draw_fig or save_fig:
             ax.patches.clear()
        
        # time.sleep(0.1)


if __name__ == "__main__":
    app()

