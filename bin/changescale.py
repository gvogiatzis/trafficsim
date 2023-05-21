# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from typing import Tuple, Optional as Opt
# from typing_extensions import Annotated as Ann
# import typer

# app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

# @app.command()
# def change_scale(file_path: Ann[str,typer.Argument(help="The path of the csv files to be rescaled")],
#     step_length: float = 0.01, 
#     use_gui: bool = True,
#     save_file: str = "sumo_routes.pk",
#     output_path: str = os.path.join("output","converter"),
#     plot_traj: bool = True):
#     """
#     Save the trajectories of all available routes in the network.
#     """
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     save_file = os.path.join(output_path, save_file)
#     env = TrafficControlEnv(net_fname = net_fname, step_length=step_length, use_gui=use_gui)
#     env.get_route_trajectories(save_file=save_file, plot_traj=plot_traj)
#     env.close()

# if __name__ == "__main__":
#     app()


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer
from typing import Tuple, Optional as Opt, List
from typing_extensions import Annotated as Ann
import tqdm

app = typer.Typer(no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command()
def find(file_list: Ann[List[str], typer.Argument(help="One or more csv files to be rescaled")],
         multiply: Ann[List[str], typer.Option("--multiply", "-m", help="A list of 'int:float' pairs that defines the collumn index and scale factor that it will be multiplied by. E.g. 2:123.0 means row[2]*=123.0")] = [],
         divide: Ann[List[str], typer.Option("--divide", "-d", help="A list of 'int:float' pairs that defines the collumn index and scale factor that it will be divided by. E.g. 2:123.0 means row[2]/=123.0")] = []):
    for fpath in tqdm.tqdm(file_list):
        with open(fpath, "r") as f1:
            path, fname = os.path.split(fpath)
            with open(os.path.join(path, "res_"+fname), "w") as f2:
                for line in f1.readlines():
                    ws = line.split(" ")
                    for i,s in list(map(int, q.split(":")) for q in multiply):
                        ws[i] = str(float(ws[i])*s)
                    for i,s in list(map(int, q.split(":")) for q in divide):
                        ws[i] = str(float(ws[i])/s)
                    f2.write(" ".join(ws))
if __name__ == "__main__":
    app()