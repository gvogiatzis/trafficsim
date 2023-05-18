import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumoenv import TrafficControlEnv

import typer

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command("allroutes")
def save_route_trajectories(
    net_fname: str = typer.Argument(help="The file name of the sumo network"), 
    step_length: float = 0.01, 
    use_gui: bool = True,
    save_file: str = "sumo_routes.pk",
    output_path: str = os.path.join("output","converter"),
    plot_traj: bool = True):
    """
    Save the trajectories of all available routes in the network.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_file = os.path.join(output_path, save_file)
    env = TrafficControlEnv(net_fname = net_fname, step_length=step_length, use_gui=use_gui)
    env.get_route_trajectories(save_file=save_file, plot_traj=plot_traj)
    env.close()

if __name__ == "__main__":
    app()