import sys
import os.path
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if os.path.basename(sys.argv[0]) != "netinfo.py":
    print("running from jupyter")
    sys.argv=["netinfo.py", "sumo_data/TwoJunction.net.xml"]


import typer
from typing import Optional as Opt, List, Tuple
from typing_extensions import Annotated as Ann
from types import SimpleNamespace


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command()
def main(net_fname: Ann[str, typer.Argument(help="the filename of the sumo network to use")],
         flowmat_fname: Ann[Opt[str], typer.Option(help="If present, provides the filename where the flowmatrix for the network will be saved. The flow matrix is a structure that links centralized actions with particular lanes that are shown the green light. It is a matrix of imensions num_actions x num_lanes and contains a 1 if the particular action green-lights a particular lane, and 0 otherwise.")] = None):
    # print(locals())

    from sumoenv import TrafficControlEnv

    env = TrafficControlEnv(net_fname=net_fname)

    print(f"Network filename: {net_fname}")
    print(f"Number of traffic lights: {env.get_num_trafficlights()}")
    print(f"Number of actions: {env.get_num_actions()}")
    print(f"Dimension of observation space: {env.get_obs_dim()}")
    print(f"All action combinations:")
    env.reset()

    W = env.get_green_lanes_per_action()
    print(f"flow matrix shape: {W.shape}")
    # a = env.action_to_multiaction_dict
    # for a in a:
    #     print(k)

    # print(f"Number of routes: {len(env.get_route_trajectories())}")

    if flowmat_fname is not None:
        print(f"Saving flow matrix in {flowmat_fname}")
        np.savetxt(flowmat_fname, env.get_green_lanes_per_action())
    env.close()


if __name__ == "__main__":
    app()

    