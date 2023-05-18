from sumo2real import SumoToRealConverter, BboxHistogram
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import glob
import typer
import os
from tqdm import tqdm
from typing import Tuple
from typing_extensions import Annotated


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command("fitbbox")
def fitbbox(real_route_fname="real_routes.pk",
            bbox_data_path="boxes/*.txt", 
            background_img_fname="sumo_data/RussianJunction/image.png",
            save_fname="bboxhist.pk"):
    """
    Fit bbox distribution to real data from yolo_v7
    """
    bbox_hist = BboxHistogram(real_route_fname=real_route_fname,bbox_data_path=bbox_data_path, background_img_fname=background_img_fname)
    with open(save_fname, "wb") as f:
        pickle.dump(bbox_hist, f)

@app.command("convertseq")
def convertseq(real_route_fname = "real_routes.pk",
               sumo_routes_fname = "sumo_routes.pk",
               img_fname = "sumo_data/RussianJunction/image.png",
               sumo_track_path="sumo_tracks",
               bbox_hist_fname="bboxhist.pk",
               output_figs_path="sumo_to_real_figs",
               output_txt_path="sumo_to_real_txt",
               image_wh:Annotated[Tuple[int,int], typer.Option()] =(1920,1080),
               save_fig=True,
               draw_fig=True):
    """
    Convert a sumo track sequence to real image (x,y,w,h) tracks
    """ 
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
        ax.add_line(line)

    with open(bbox_hist_fname, "rb") as f:
        bbox_hist = pickle.load(f)

    for i, f in enumerate(tqdm(sorted(glob.glob(f"{sumo_track_path}/*.txt")))):
        r=converter.convert(f)
        if len(r)==0:
            line.set_data([],[])
        else:      
            line.set_data(zip(*r))
        with open(f"{output_txt_path}/frame{i:06d}.txt", "w") as f:
            for xy in r:
                wh = bbox_hist.sample([xy])
                if wh is not None:
                    wh = wh.squeeze()
                    xy -= wh/2
                    f.write(f"2, {xy[0]/w}, {xy[1]/h}, {wh[0]/w}, {wh[1]/h}, 1.0\n")
                    if draw_fig or save_fig:
                        ax.add_patch(Rectangle(xy,wh[0],wh[1],fill=None, alpha=1))
        plt.pause(0.001)
        if save_fig:
            fig.savefig(f"{output_figs_path}/img{i:06d}.png")
        if draw_fig or save_fig:
             ax.patches.clear()
        
        # time.sleep(0.1)


if __name__ == "__main__":
    app()

