#!/usr/bin/env python
"""
Network Information Utility

This script displays comprehensive information about a SUMO network file,
including traffic light configuration, action space dimensions, and agent
breakdowns for multi-agent setups.

Usage Examples:
    # Display basic network info
    python netinfo.py sumo_data/RussianJunction/RussianJunction.net.xml

    # Display info and save flow matrix
    python netinfo.py sumo_data/Grid2by2.net.xml --flowmat-fname output/flowmat.txt

    # Multi-agent configuration
    python netinfo.py sumo_data/Grid2by2.net.xml --agent-lights-file agents.txt
"""

import sys
import os.path
import numpy as np
from typing import Optional, Dict

# Add parent directory to path so we can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Support for running from Jupyter
if os.path.basename(sys.argv[0]) != "netinfo.py":
    print("Running from Jupyter notebook")
    sys.argv = ["netinfo.py", "sumo_data/TwoJunction.net.xml"]

import typer
from typing_extensions import Annotated as Ann

# Initialize Typer CLI application
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)


def build_flow_matrix(env) -> np.ndarray:
    """Build flow matrix showing which lanes get green for each action.

    The flow matrix is a 2D array of shape (num_actions, num_lanes) where
    element [i,j] is 1 if action i gives green light to lane j, else 0.

    Parameters
    ----------
    env : TrafficControlEnv
        Initialized traffic control environment

    Returns
    -------
    np.ndarray
        Binary flow matrix (num_actions x num_lanes)
    """
    # Get action breakdown for all agents
    action_breakdown = env.get_action_breakdown()

    # For single-agent case, build flow matrix directly
    if len(action_breakdown) == 1:
        agent_id = list(action_breakdown.keys())[0]
        action_mapping = env.schema[agent_id]

        # Get all unique lanes
        all_lanes = action_mapping.controlled_lanes
        lane_to_idx = {lane: idx for idx, lane in enumerate(all_lanes)}

        # Build binary matrix
        num_actions = action_mapping.num_actions
        num_lanes = len(all_lanes)
        flow_matrix = np.zeros((num_actions, num_lanes), dtype=int)

        for action_id, green_lanes in enumerate(action_mapping.green_lanes_per_action):
            for lane in green_lanes:
                if lane in lane_to_idx:
                    flow_matrix[action_id, lane_to_idx[lane]] = 1

        return flow_matrix
    else:
        print("Warning: Flow matrix generation for multi-agent setups not yet implemented")
        print("         Flow matrix will be empty")
        return np.array([])


@app.command()
def main(
    net_fname: Ann[str, typer.Argument(help="Path to SUMO network file (.net.xml)")],
    flowmat_fname: Ann[Optional[str], typer.Option(
        help="If provided, saves the flow matrix to this file. The flow matrix is a "
             "binary matrix of shape (num_actions, num_lanes) where element [i,j] is 1 "
             "if action i gives green light to lane j, else 0."
    )] = None,
    agent_lights_file: Ann[Optional[str], typer.Option(
        help="Path to file defining agent-to-traffic-light assignments. "
             "Format: one line per agent with comma-separated traffic light IDs."
    )] = None
) -> None:
    """
    Display comprehensive information about a SUMO network configuration.

    This utility creates a traffic control environment and reports:
    - Number of traffic lights in the network
    - Total action space size (for single-agent control)
    - Per-agent observation and action dimensions
    - Flow matrix structure (which lanes get green for each action)
    """
    from sumoenv import TrafficControlEnv

    # Initialize environment
    print(f"\nInitializing environment for: {net_fname}")
    env = TrafficControlEnv(
        net_fname=net_fname,
        agent_lights_file=agent_lights_file
    )

    # Reset to ensure environment is ready
    env.reset()

    # Display basic network information
    print("\n" + "="*60)
    print("NETWORK INFORMATION")
    print("="*60)
    print(f"Network file:           {net_fname}")
    print(f"Traffic lights:         {env.get_num_trafficlights()}")
    print(f"Total action space:     {env.get_num_actions():,} (single-agent)")

    # Display per-agent breakdown
    action_breakdown = env.get_action_breakdown()
    print(f"\nNumber of agents:       {len(action_breakdown)}")
    print("\nPer-Agent Configuration:")
    print("-" * 60)

    for agent_id, (obs_dim, num_actions) in action_breakdown.items():
        action_mapping = env.schema[agent_id]
        print(f"  Agent {agent_id}:")
        print(f"    Traffic lights:     {action_mapping.num_traffic_lights} "
              f"{action_mapping.tl_ids}")
        print(f"    Controlled lanes:   {len(action_mapping.controlled_lanes)}")
        print(f"    Observation dim:    {obs_dim}")
        print(f"    Action space:       {num_actions}")

    # Build and optionally save flow matrix
    if flowmat_fname is not None:
        print("\n" + "="*60)
        print("FLOW MATRIX")
        print("="*60)

        flow_matrix = build_flow_matrix(env)

        if flow_matrix.size > 0:
            print(f"Flow matrix shape:      {flow_matrix.shape}")
            print(f"                        (actions: {flow_matrix.shape[0]}, "
                  f"lanes: {flow_matrix.shape[1]})")
            print(f"Saving to:              {flowmat_fname}")
            np.savetxt(flowmat_fname, flow_matrix, fmt='%d')
        else:
            print("Flow matrix not generated (multi-agent mode)")

    print("\n" + "="*60)

    # Clean up
    env.close()


if __name__ == "__main__":
    app()