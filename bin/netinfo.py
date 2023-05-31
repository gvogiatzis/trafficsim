import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if os.path.basename(sys.argv[0]) != "netinfo.py":
    print("running from jupyter")
    sys.argv=["netinfo.py", "sumo_data/TwoJunction.net.xml"]


import typer
from typing import Optional as Opt, List, Tuple
from typing_extensions import Annotated as Ann
from types import SimpleNamespace


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)
state = SimpleNamespace() # state variable that will hold common set of options




@app.command()
def main(net_fname: Ann[str, typer.Argument(help="the filename of the sumo network to use")]):
    print(locals())

    from sumoenv import TrafficControlEnv

    env = TrafficControlEnv(net_fname=net_fname)

    print(f"Network filename: {net_fname}")
    print(f"Number of traffic lights: {env.get_num_trafficlights()}")
    print(f"Number of actions: {env.get_num_actions()}")
    print(f"Dimension of observation space: {env.get_obs_dim()}")
    # print(f"Number of routes: {len(env.get_route_trajectories())}")
    env.close()


if __name__ == "__main__":
    app()

    