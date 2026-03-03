"""SUMO Traffic Control Environment for Reinforcement Learning

This module provides a Python interface to the SUMO (Simulation of Urban MObility)
traffic simulator for training reinforcement learning agents to control traffic lights.

Key Features:
- Multi-agent support: Multiple independent agents can control different subsets of traffic lights
- Flexible state representations: Per-lane occupancy or per-action demand
- Action mapping: Clean abstraction using ActionMapping dataclass
- Configurable rewards: Based on halting vehicles, occupancy, or speed
- Data collection: Support for recording trajectories and screenshots

Architecture:
- ActionMapping: Encapsulates the relationship between abstract action IDs and traffic light phases
- TrafficControlEnv: Main environment class implementing standard RL interface (reset, step)

Usage Example:
    from sumoenv import TrafficControlEnv
    from random import randint

    # Initialize environment
    env = TrafficControlEnv(
        net_fname='sumo_data/network.net.xml',
        vehicle_spawn_rate=0.05,
        episode_length=500
    )

    # Training loop
    state = env.reset()
    for t in range(1000):
        action = {agent_id: randint(0, num_actions-1)
                  for agent_id, (obs_dim, num_actions) in env.get_action_breakdown().items()}
        state, reward, done = env.step(action)
        if done:
            state = env.reset()

    env.close()
"""

import os
import logging
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sumolib
import traci
from typing import Dict, Tuple, List, Callable, Optional
from itertools import product, chain
from dataclasses import dataclass

import random, pickle

# Set up logger for this module
logger = logging.getLogger(__name__)


class EpisodeRecorder:
    """Records data during an episode for analysis and visualization.

    This class collects episode statistics including states, actions, rewards,
    and optional detailed trajectory data. It provides methods to save and
    export the collected data.

    Attributes
    ----------
    episode_id : int
        Episode number
    collect_trajectories : bool
        Whether to record detailed vehicle trajectories
    states : List[Dict[int, List[float]]]
        State observations at each timestep
    actions : List[Dict[int, int]]
        Actions taken at each timestep
    rewards : List[Dict[int, float]]
        Rewards received at each timestep
    """

    def __init__(self, episode_id: int, collect_trajectories: bool = False):
        """Initialize episode recorder.

        Parameters
        ----------
        episode_id : int
            Episode number for identification
        collect_trajectories : bool, optional
            If True, record detailed trajectory data (default: False)
        """
        self.episode_id = episode_id
        self.collect_trajectories = collect_trajectories
        self.states: List[Dict[int, List[float]]] = []
        self.actions: List[Dict[int, int]] = []
        self.rewards: List[Dict[int, float]] = []
        self.trajectories: List[Dict[str, Tuple[float, float]]] = []

    def record_step(
        self,
        state: Dict[int, List[float]],
        action: Dict[int, int],
        reward: Dict[int, float],
        trajectories: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> None:
        """Record data from a single timestep.

        Parameters
        ----------
        state : Dict[int, List[float]]
            State observations for each agent
        action : Dict[int, int]
            Actions taken by each agent
        reward : Dict[int, float]
            Rewards received by each agent
        trajectories : Dict[str, Tuple[float, float]], optional
            Vehicle positions (if collect_trajectories is True)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if self.collect_trajectories and trajectories is not None:
            self.trajectories.append(trajectories)

    def get_episode_summary(self) -> Dict:
        """Compute summary statistics for the episode.

        Returns
        -------
        Dict
            Summary containing episode_id, total_reward, length, etc.
        """
        # Compute total rewards per agent
        total_rewards = {}
        if len(self.rewards) > 0:
            for agent_id in self.rewards[0].keys():
                total_rewards[agent_id] = sum(r[agent_id] for r in self.rewards)

        return {
            'episode_id': self.episode_id,
            'length': len(self.states),
            'total_rewards': total_rewards,
            'mean_rewards': {aid: total_rewards[aid] / len(self.rewards)
                           for aid in total_rewards.keys()} if len(self.rewards) > 0 else {}
        }

    def save_to_file(self, filepath: str) -> None:
        """Save episode data to a pickle file.

        Parameters
        ----------
        filepath : str
            Path where data should be saved
        """
        data = {
            'episode_id': self.episode_id,
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'trajectories': self.trajectories if self.collect_trajectories else None,
            'summary': self.get_episode_summary()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved episode {self.episode_id} data to {filepath}")

    def clear(self) -> None:
        """Clear all recorded data."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.trajectories.clear()


@dataclass
class ActionMapping:
    """Maps action IDs to traffic light phase combinations for a single agent.

    This class encapsulates all traffic light control logic for a single agent,
    including action execution, state observation, and reward computation. It provides
    a clean abstraction between abstract action IDs and the underlying SUMO traffic
    light phases.

    Attributes
    ----------
    tl_ids : List[str]
        Ordered list of traffic light IDs controlled by this agent
    phase_combinations : List[Tuple[int, ...]]
        For each action ID, a tuple of phases (one per traffic light)
        Example: phase_combinations[5] = (2, 1) means action 5 sets
        tl_ids[0] to phase 2 and tl_ids[1] to phase 1
    controlled_lanes : List[str]
        All lane IDs controlled by this agent's traffic lights
    green_lanes_per_action : List[List[str]]
        For each action, list of lanes that will receive green light
    green_lanes_per_phase_cache : Dict[str, Dict[int, List[str]]]
        Cached mapping from traffic_light_id -> phase_id -> list of green lanes
        This avoids repeated SUMO API calls to parse phase states
    reward_fn : Optional[Callable], optional
        Custom reward function with signature: fn(sumo_connection, controlled_lanes) -> float
        If None, uses default reward (negative sum of halting vehicles)
    dims : Tuple[int, int]
        (observation_dim, num_actions) - input/output dimensions for RL agent

    Methods
    -------
    get_phases(action)
        Get the phase settings for a given action
    apply_action(action, sumo_connection)
        Execute an action by setting traffic light phases in SUMO
    get_state(sumo_connection, simple_state)
        Compute state observation from current traffic conditions
    get_reward(sumo_connection)
        Compute reward based on current traffic conditions
    """
    tl_ids: List[str]
    phase_combinations: List[Tuple[int, ...]]
    controlled_lanes: List[str]
    green_lanes_per_action: List[List[str]]
    green_lanes_per_phase_cache: Dict[str, Dict[int, List[str]]]
    reward_fn: Optional[Callable] = None
    dims: Tuple[int, int] = (0, 0)

    def get_phases(self, action: int) -> Tuple[int, ...]:
        """Get phase setting for each traffic light given an action ID.

        Parameters
        ----------
        action : int
            Action ID in range [0, num_actions-1]

        Returns
        -------
        Tuple[int, ...]
            Phase ID for each traffic light in tl_ids order
        """
        return self.phase_combinations[action]

    def apply_action(self, action: int, sumo_connection):
        """Apply action directly to SUMO traffic lights.

        Parameters
        ----------
        action : int
            Action ID to execute
        sumo_connection : traci.Connection
            Active SUMO/TraCI connection
        """
        phases = self.phase_combinations[action]
        for tl_id, phase in zip(self.tl_ids, phases):
            sumo_connection.trafficlight.setPhase(tl_id, phase)

    def get_state(self, sumo_connection, simple_state: bool) -> List[float]:
        """Compute state observation for this agent.

        Parameters
        ----------
        sumo_connection : traci.Connection
            Active SUMO/TraCI connection
        simple_state : bool
            If True, return per-action demand (sum of halting vehicles per action)
            If False, return per-lane halting numbers

        Returns
        -------
        List[float]
            State vector representing current traffic conditions
        """
        if simple_state:
            # State: per-action demand (sum of halting vehicles that would get green)
            state = []
            for green_lanes in self.green_lanes_per_action:
                demand = sum(sumo_connection.lane.getLastStepHaltingNumber(lane)
                           for lane in green_lanes)
                state.append(demand)
        else:
            # State: per-lane halting vehicle count
            state = [sumo_connection.lane.getLastStepHaltingNumber(lane)
                    for lane in self.controlled_lanes]

        return state

    def get_reward(self, sumo_connection) -> float:
        """Compute reward for this agent based on current traffic conditions.

        Uses custom reward function if provided, otherwise defaults to negative
        sum of halting vehicles (which encourages minimizing congestion).

        Parameters
        ----------
        sumo_connection : traci.Connection
            Active SUMO/TraCI connection

        Returns
        -------
        float
            Reward value (higher is better)
        """
        if self.reward_fn is not None:
            # Use custom reward function
            return self.reward_fn(sumo_connection, self.controlled_lanes)
        else:
            # Default reward: negative total halting vehicles (minimize congestion)
            return -sum(sumo_connection.lane.getLastStepHaltingNumber(lane)
                       for lane in self.controlled_lanes)

    def get_best_greedy_action(self, sumo_connection) -> int:
        """Select the greedy action that maximizes immediate demand satisfaction.

        The greedy policy chooses the action (phase combination) that gives green
        light to the most waiting vehicles. This serves as a simple baseline policy.

        Parameters
        ----------
        sumo_connection : traci.Connection
            Active SUMO/TraCI connection

        Returns
        -------
        int
            Action ID with highest current demand
        """
        # Compute demand for each phase of each traffic light using cached data
        tl_demand = {}
        for tl_id in self.tl_ids:
            phase_demands = []
            for phase_id in sorted(self.green_lanes_per_phase_cache[tl_id].keys()):
                green_lanes = self.green_lanes_per_phase_cache[tl_id][phase_id]
                demand = sum(sumo_connection.lane.getLastStepVehicleNumber(lane)
                           for lane in green_lanes)
                phase_demands.append(demand)
            tl_demand[tl_id] = phase_demands

        # Find action with highest total demand across all traffic lights
        best_action = 0
        highest_demand = -1
        for action_id, phases in enumerate(self.phase_combinations):
            # Sum demand for this phase combination
            total_demand = sum(tl_demand[tl_id][phase]
                             for tl_id, phase in zip(self.tl_ids, phases))
            if total_demand > highest_demand:
                highest_demand = total_demand
                best_action = action_id

        return best_action

    @property
    def num_actions(self) -> int:
        """Number of possible actions for this agent."""
        return len(self.phase_combinations)

    @property
    def num_traffic_lights(self) -> int:
        """Number of traffic lights controlled by this agent."""
        return len(self.tl_ids)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"ActionMapping(tls={self.num_traffic_lights}, "
                f"actions={self.num_actions}, "
                f"lanes={len(self.controlled_lanes)}, "
                f"dims={self.dims})")

class TrafficControlEnv:
    """SUMO-based traffic control environment for reinforcement learning.

    This class provides a standard RL interface (reset, step, close) for training
    traffic light control agents in SUMO simulations. It supports both single-agent
    and multi-agent configurations, with flexible state representations and reward
    functions.

    The environment handles:
    - SUMO process management (start, stop, state saving/loading)
    - Route initialization and vehicle spawning
    - Action-to-traffic-light mapping via ActionMapping dataclass
    - State observation and reward computation
    - Data collection (trajectories, screenshots)

    Attributes
    ----------
    schema : Dict[int, ActionMapping]
        Maps agent_id to its ActionMapping instance (action space definition)
    simple_state : bool
        If True, use per-action demand state; if False, use per-lane halting numbers
    episode_length : int
        Number of timesteps per episode
    sumo_timestep : int
        Number of SUMO simulation steps between RL timesteps
    """
    def __init__(self, net_fname = 'sumo_data/RussianJunction/RussianJunction.net.xml', vehicle_spawn_rate=0.015, episode_length=500, sumo_timestep=20, use_gui=False, seed=None,step_length=1, output_path="output", record_tracks=False, car_length=5, record_screenshots = False, gui_config_file = None, real_routes_file = None,agent_lights_file=None, reward_fn=None, simple_state=False, spawn_strategy='uniform', record_episodes=False):
        """Initialize the SUMO traffic control environment.

        This constructor:
        1. Reads the SUMO network file
        2. Starts the SUMO/SUMO-GUI process
        3. Initializes routes between fringe edges
        4. Saves initial state for fast episode resets
        5. Computes the action schema for all agents

        Parameters
        ----------
        net_fname : str, optional
            Path to SUMO network file (.net.xml)
        vehicle_spawn_rate : float, optional
            Probability of spawning a vehicle per route per timestep (default: 0.015)
        episode_length : int, optional
            Number of RL timesteps per episode (default: 500)
        sumo_timestep : int, optional
            Number of SUMO steps between RL actions (default: 20)
        use_gui : bool, optional
            If True, use SUMO GUI instead of headless mode (default: False)
        seed : int, optional
            Random seed for reproducibility (default: None, random seed)
        step_length : float, optional
            SUMO simulation step length in seconds (default: 1.0)
        output_path : str, optional
            Directory for saving outputs (default: "output")
        record_tracks : bool, optional
            If True, save vehicle positions to files (default: False)
        car_length : float, optional
            Vehicle length in SUMO units (default: 5.0)
        record_screenshots : bool, optional
            If True, save GUI screenshots each timestep (default: False)
        gui_config_file : str, optional
            Path to SUMO GUI configuration file (default: None)
        real_routes_file : str, optional
            Path to pickle file with allowed routes (default: None, all routes)
        agent_lights_file : str, optional
            Path to file defining agent-to-traffic-light assignments (default: None, single agent)
            Format: one line per agent with comma-separated traffic light IDs
        reward_fn : Callable, optional
            Custom reward function with signature: fn(sumo_connection, controlled_lanes) -> float
            If None, uses default reward of negative sum of halting vehicles (default: None)
        simple_state : bool, optional
            If False, state is per-lane halting vehicle count (default)
            If True, state is per-action demand (sum of halting vehicles that would get green)
        spawn_strategy : str or Callable, optional
            Vehicle spawning strategy. Options:
            - 'uniform': Constant spawn rate for all routes (default)
            - 'time_varying': Spawn rate varies with rush hour pattern
            - 'route_dependent': Different spawn rates per route
            - Callable: Custom function(episode_step, route_id) -> spawn_probability
        record_episodes : bool, optional
            If True, record detailed episode data (states, actions, rewards) using
            EpisodeRecorder for analysis and visualization (default: False)
        """
        # Set random seed for reproducibility
        random.seed(seed)

        # Store reward function for passing to ActionMapping
        self.reward_fn = reward_fn

        # Store configuration parameters
        self.record_tracks = record_tracks
        self.record_episodes = record_episodes
        self.episode_recorder: Optional[EpisodeRecorder] = None
        self.total_steps_run = 0  # Global step counter across all episodes
        self.current_episode = 0
        self.episode_step = 0  # Current step within episode (0 to episode_length-1)
        self.output_path = output_path
        self.episode_length = episode_length
        self._net_fname = net_fname
        self.vehicle_spawn_rate = vehicle_spawn_rate
        self.spawn_strategy = spawn_strategy
        self.route_dict = None  # Maps (start_edge, end_edge) -> route_id
        self._vehcnt = 0  # Vehicle ID counter
        self.sumo_timestep = sumo_timestep
        self.use_gui = use_gui
        self.car_length = car_length
        self.record_screenshots = record_screenshots
        self.gui_config_file = gui_config_file
        self.real_routes_file = real_routes_file

        # Read SUMO network file
        self._net = sumolib.net.readNet(self._net_fname, withPrograms=True)
        self._sumo = None  # Will hold TraCI connection

        # State representation mode:
        # - simple_state=False: state is per-lane halting vehicle count (default)
        # - simple_state=True: state is per-action demand (sum of halting vehicles that would get green)
        self.simple_state = simple_state


        # Build SUMO command line
        sumo_command = ['sumo-gui'] if self.use_gui else ['sumo']
        sumo_command.extend([
            '-n', self._net_fname,
            '--start',              # Start simulation immediately
            '--quit-on-end',        # Exit when simulation ends
            '--no-warnings',        # Suppress warning messages
            '--no-step-log',        # Suppress step logging
            '--step-length', str(step_length)
        ])

        # Add GUI configuration if provided (sets viewpoints, zoom, car sizes, etc.)
        if self.use_gui and self.gui_config_file is not None:
            sumo_command.extend(['-g', self.gui_config_file])

        # Load restricted route set if provided (useful for avoiding difficult routes)
        if self.real_routes_file is not None:
            with open(self.real_routes_file, 'rb') as f:
                self.real_routes = pickle.load(f)["active_routes"]
        else:
            self.real_routes = None  # Use all routes

        # Create output directories for data collection
        if self.record_tracks and not os.path.exists(f"{self.output_path}/sumo_tracks"):
            os.makedirs(f"{self.output_path}/sumo_tracks")
            logger.info(f"Created directory: {self.output_path}/sumo_tracks")
        if self.record_screenshots and not os.path.exists(f"{self.output_path}/sumo_screenshots"):
            os.makedirs(f"{self.output_path}/sumo_screenshots")
            logger.info(f"Created directory: {self.output_path}/sumo_screenshots")

        # Add random seed for reproducibility
        if seed is not None:
            sumo_command.extend(['--seed', str(seed)])
        else:
            sumo_command.extend(['--random'])

        # Start SUMO process and get TraCI connection
        traci.start(sumo_command, verbose=True, label="default")
        self._sumo = traci.getConnection(label="default")

        # Initialize routes between fringe edges
        self._initialize_routes()

        # Save initial state for fast episode resets
        self._sumo.simulation.saveState('state.sumo')

        # Compute action schema mapping agent IDs to ActionMapping instances
        # This defines which traffic lights each agent controls and their action spaces
        self.schema = self._compute_schema(agent_lights_file)

    def _parse_agent_assignments(self, agent_lights_file: Optional[str]) -> Dict[int, List[str]]:
        """Parse agent-to-traffic-light assignments from file.

        Parameters
        ----------
        agent_lights_file : str or None
            Path to file containing TL assignments (one line per agent, comma-separated TL IDs)
            If None, creates a single agent controlling all traffic lights

        Returns
        -------
        Dict[int, List[str]]
            Maps agent_id to list of traffic light IDs
        """
        agent_tls = {}
        if agent_lights_file is not None:
            with open(agent_lights_file) as file:
                for ag_id, line in enumerate(file):
                    agent_tls[ag_id] = [s.strip() for s in line.split(",")]
        else:
            # Default: single agent controls all traffic lights
            agent_tls[0] = list(self._sumo.trafficlight.getIDList())
        return agent_tls

    def _build_green_lanes_cache(self, tl_ids: List[str]) -> Dict[str, Dict[int, List[str]]]:
        """Build cache of green lanes per phase for each traffic light.

        Parameters
        ----------
        tl_ids : List[str]
            List of traffic light IDs to cache

        Returns
        -------
        Dict[str, Dict[int, List[str]]]
            Maps tl_id -> phase_id -> list of green lanes
        """
        cache = {}
        for tl_id in tl_ids:
            cache[tl_id] = self._get_green_lanes_per_phase(tl_id)
        return cache

    def _build_phase_combinations(self, tl_ids: List[str]) -> Tuple[List[Tuple[int, ...]], List[str], List[Tuple[Tuple[str, int], ...]]]:
        """Build all possible phase combinations for the given traffic lights.

        Parameters
        ----------
        tl_ids : List[str]
            List of traffic light IDs

        Returns
        -------
        Tuple containing:
            - phase_combinations: List of phase tuples (one per action)
            - controlled_lanes: All lanes controlled by these traffic lights
            - tl_pha_combinations: Full (tl, phase) combinations for computing green lanes
        """
        controlled_lanes = []
        tl_pha_lists = []

        # Build list of (traffic_light, phase) combinations for each TL
        for tl_id in tl_ids:
            logic = self._sumo.trafficlight.getAllProgramLogics(tl_id)[0]
            lanes = self._sumo.trafficlight.getControlledLanes(tl_id)
            nphases = len(logic.getPhases())
            tl_pha_lists.append([(tl_id, phase) for phase in range(nphases)])
            controlled_lanes.extend(lanes)

        # Compute Cartesian product to get all action combinations
        tl_pha_combinations = list(product(*tl_pha_lists))

        # Convert to simple tuple format (just phase IDs)
        phase_combinations = [
            tuple(phase for tl, phase in tl_pha)
            for tl_pha in tl_pha_combinations
        ]

        return phase_combinations, controlled_lanes, tl_pha_combinations

    def _compute_green_lanes_per_action(
        self,
        tl_pha_combinations: List[Tuple[Tuple[str, int], ...]],
        green_lanes_cache: Dict[str, Dict[int, List[str]]]
    ) -> List[List[str]]:
        """Compute which lanes get green light for each action.

        Parameters
        ----------
        tl_pha_combinations : List[Tuple[Tuple[str, int], ...]]
            List of (tl_id, phase) combinations for each action
        green_lanes_cache : Dict[str, Dict[int, List[str]]]
            Cached green lanes per phase for each traffic light

        Returns
        -------
        List[List[str]]
            For each action, list of lanes that receive green light
        """
        green_lanes_per_action = []
        for tl_pha in tl_pha_combinations:
            green_lanes = []
            for tl_id, phase in tl_pha:
                green_lanes.extend(green_lanes_cache[tl_id][phase])
            green_lanes_per_action.append(green_lanes)
        return green_lanes_per_action

    def _compute_schema(self, agent_lights_file: Optional[str]) -> Dict[int, ActionMapping]:
        """Compute the action schema for all agents.

        Reads agent-to-traffic-light assignments and computes the complete
        mapping between abstract action IDs and traffic light phase combinations.

        Parameters
        ----------
        agent_lights_file : str or None
            Path to file containing TL assignments (one line per agent, comma-separated TL IDs)
            If None, creates a single agent controlling all traffic lights

        Returns
        -------
        Dict[int, ActionMapping]
            Maps agent_id to its ActionMapping instance
        """
        # Parse agent-to-traffic-light assignments
        agent_tls = self._parse_agent_assignments(agent_lights_file)

        schema = {}

        for agent_id, tl_ids in agent_tls.items():
            # Build cache of green lanes per phase for this agent's traffic lights
            green_lanes_per_phase_cache = self._build_green_lanes_cache(tl_ids)

            # Build all possible phase combinations
            phase_combinations, controlled_lanes, tl_pha_combinations = \
                self._build_phase_combinations(tl_ids)

            # Compute green lanes for each action
            green_lanes_per_action = self._compute_green_lanes_per_action(
                tl_pha_combinations, green_lanes_per_phase_cache
            )

            # Determine observation and action dimensions
            num_actions = len(phase_combinations)
            if self.simple_state:
                # State is per-action demand (one value per possible action)
                obs_dim = num_actions
            else:
                # State is per-lane occupancy
                obs_dim = len(controlled_lanes)

            # Create ActionMapping dataclass instance
            schema[agent_id] = ActionMapping(
                tl_ids=tl_ids,
                phase_combinations=phase_combinations,
                controlled_lanes=controlled_lanes,
                green_lanes_per_action=green_lanes_per_action,
                green_lanes_per_phase_cache=green_lanes_per_phase_cache,
                reward_fn=self.reward_fn,
                dims=(obs_dim, num_actions)
            )

        return schema        


    def reset(self, seed: Optional[int] = None, decentralized: bool = False) -> Dict[int, List[float]]:
        """
        Initalizes a new environment. 
        
        Must be called when we need to start a new episode. In the traffic
        scenarios of sumo this is not actually necessary as we have a continuous
        loop. You can use in case of deadlock.

        Parameters
        ----------
        seed : int, optional
            A random seed passed to sumo for repeatability (not implemented yet)
        
        Returns
        -------
        observation: numpy.array
            Initial state observation for each agent
        """

        self._sumo.simulation.loadState('state.sumo')

        self._spawnVehicles()
        self._sumo.simulationStep()
        self.episode_step = 0  # Reset episode step counter

        # Create new episode recorder if recording is enabled
        if self.record_episodes:
            self.episode_recorder = EpisodeRecorder(
                episode_id=self.current_episode,
                collect_trajectories=self.record_tracks
            )

        multi_state, _ = self._get_states_and_rewards()
        return multi_state
    
    def step(self, action: Optional[Dict[int, int]] = None) -> Tuple[Dict[int, List[float]], Dict[int, float], bool]:
        """ Steps the simulation through one timestep
        
        Executes a single simulation step after passing an action to the
        environment. Returns the observation of the new state and the reward.

        Parameters
        ----------
        action : dict[int,int]
            This is a dict mapping between agentID and action to be taken by that agent
        
        Returns
        -------
        observation: dict[int,ndarray]
            This is a dict mapping between agentID and the new state of the simulation after the action was implemented

        reward: dict[int,float]
            This is a dict mapping between agentID and the reward obtained by the agent for that timestep

        done: boolean
            is true if the episode is finished
        """
        if action is None:
            action = self.choose_random_action()

        self._applyMultiaction(action)

        for _ in range(self.sumo_timestep):
            self._spawnVehicles()
            if self.use_gui and self.record_screenshots:
                self._sumo.gui.screenshot("View #0", f"{self.output_path}/sumo_screenshots/{self.total_steps_run:009}.png")
            self._sumo.simulationStep()

            if self.record_tracks:
                self._saveVehicles(f"{self.output_path}/sumo_tracks", use_total_time=True)
            self.total_steps_run += 1

        self.episode_step += 1

        done = self.episode_step >= self.episode_length
        multi_state, multi_reward = self._get_states_and_rewards()

        # Record episode data if recording is enabled
        if self.record_episodes and self.episode_recorder is not None:
            self.episode_recorder.record_step(multi_state, action, multi_reward)

        return multi_state, multi_reward, done

    def choose_random_action(self) -> Dict[int, int]:
        """Choose random action for each agent.

        Returns
        -------
        Dict[int, int]
            Maps agent_id to randomly chosen action
        """
        multi_action = dict()
        for agID, action_mapping in self.schema.items():
            multi_action[agID] = random.randint(0, action_mapping.num_actions - 1)
        return multi_action

    def choose_greedy_action(self) -> Dict[int, int]:
        """Choose greedy action for each agent.

        Greedy policy selects the action that gives green light to the most
        waiting vehicles (maximum immediate demand satisfaction). This delegates
        to ActionMapping.get_best_greedy_action() for clean encapsulation.

        Returns
        -------
        Dict[int, int]
            Maps agent_id to greedily chosen action
        """
        multi_action = {}
        for agID, action_mapping in self.schema.items():
            multi_action[agID] = action_mapping.get_best_greedy_action(self._sumo)
        return multi_action


    def _get_green_lanes_per_phase(self, tls_id: str) -> Dict[int, List[str]]:
        '''
        Returns a dictionary that maps each phase to the list of lanes that turn green in a traffic light

        Parameters
        ----------
        tls_id:   str
            a traffic light ID

        Returns
        -------
        pha_to_green_lanes: Dict[int, List[str]]

        an array D such that if p is a phase of the traffic light, D[p] = the total demand waiting to be released if we enable phase p.
        '''
        logic = self._sumo.trafficlight.getAllProgramLogics(tls_id)[0]
        lanes = self._sumo.trafficlight.getControlledLanes(tls_id)
        phases = logic.getPhases()
        pha_to_green_lanes=dict()
        for phase_id,phase in enumerate(phases):# phases are whole objects- we just need their index
            pha_to_green_lanes[phase_id]=[]
            for lane, s in zip(lanes,phase.state):
                if s in ['g','G']:
                    pha_to_green_lanes[phase_id].append(lane)
        return pha_to_green_lanes


    def _get_TLS_demand_breakdown(self, tls_id: str, green_lanes_per_phase_cache: Optional[Dict[int, List[str]]] = None) -> List[int]:
        """Returns a breakdown of current demand for each phase.

        Uses cached green lanes per phase data (if provided) or computes it fresh
        to determine which lanes get green light in each phase, then sums the
        vehicle count on those lanes.

        Parameters
        ----------
        tls_id : str
            Traffic light ID
        green_lanes_per_phase_cache : Dict[int, List[str]], optional
            Cached mapping of phase_id -> green lanes for this traffic light.
            If None, will compute fresh via _get_green_lanes_per_phase.

        Returns
        -------
        List[int]
            Demand for each phase, where demand[p] = total vehicles waiting
            on lanes that would receive green in phase p
        """
        # Get mapping of phase_id -> green lanes (use cache if available)
        if green_lanes_per_phase_cache is not None:
            pha_to_green_lanes = green_lanes_per_phase_cache
        else:
            pha_to_green_lanes = self._get_green_lanes_per_phase(tls_id)

        # Compute demand for each phase by summing vehicles on green lanes
        demand = []
        for phase_id in sorted(pha_to_green_lanes.keys()):
            phasedemand = sum(
                self._sumo.lane.getLastStepVehicleNumber(lane)
                for lane in pha_to_green_lanes[phase_id]
            )
            demand.append(phasedemand)

        return demand

    def _applyMultiaction(self, multi_action: Dict[int, int]) -> None:
        """Apply multi-agent action to traffic lights.

        Parameters
        ----------
        multi_action : Dict[int, int]
            Maps agent_id to action_id to execute
        """
        for agID, action in multi_action.items():
            self.schema[agID].apply_action(action, self._sumo)

    def _get_states_and_rewards(self) -> Tuple[Dict[int, List[float]], Dict[int, float]]:
        """Compute state observations and rewards for all agents.

        This method delegates to the ActionMapping dataclass methods for clean
        encapsulation. Each agent's state and reward are computed independently
        based on its controlled traffic lights and lanes.

        Returns
        -------
        multi_state : Dict[int, np.ndarray]
            Maps agent_id to its state observation vector
        multi_reward : Dict[int, float]
            Maps agent_id to its scalar reward value
        """
        multi_state: Dict[int, np.ndarray] = dict()
        multi_reward: Dict[int, float] = dict()

        # Compute state and reward for each agent
        for agID, action_mapping in self.schema.items():
            # Get state observation from ActionMapping
            # State format depends on self.simple_state flag:
            # - False: per-lane halting vehicle counts
            # - True: per-action demand (halting vehicles that would get green)
            multi_state[agID] = action_mapping.get_state(self._sumo, self.simple_state)

            # Get reward from ActionMapping
            # Reward is negative sum of halting vehicles (minimize congestion)
            multi_reward[agID] = action_mapping.get_reward(self._sumo)

        return multi_state, multi_reward

    def get_episode_recorder(self) -> Optional[EpisodeRecorder]:
        """Get the current episode recorder.

        Returns
        -------
        Optional[EpisodeRecorder]
            Current episode recorder, or None if recording is disabled
        """
        return self.episode_recorder

    def save_episode_data(self, filepath: Optional[str] = None) -> None:
        """Save current episode data to file.

        Parameters
        ----------
        filepath : str, optional
            Path to save file. If None, uses default naming in output_path
        """
        if self.episode_recorder is None:
            logger.warning("No episode recorder available. Enable record_episodes=True")
            return

        if filepath is None:
            filepath = f"{self.output_path}/episode_{self.current_episode:04d}.pkl"

        self.episode_recorder.save_to_file(filepath)

    def get_episode_summary(self) -> Optional[Dict]:
        """Get summary statistics for current episode.

        Returns
        -------
        Optional[Dict]
            Episode summary, or None if recording is disabled
        """
        if self.episode_recorder is None:
            return None
        return self.episode_recorder.get_episode_summary()

    def close(self) -> None:
        """Close the SUMO simulation and clean up resources.

        This method:
        1. Removes the saved state file
        2. Closes the TraCI connection to SUMO
        3. Terminates the SUMO process

        Should be called when done with the environment to avoid zombie processes.
        """
        # Remove saved state file
        if os.path.exists('state.sumo'):
            os.remove('state.sumo')

        # Close TraCI connection and terminate SUMO
        if traci.isLoaded():
            self._sumo.close()
            self._sumo = None



    def get_num_trafficlights(self) -> int:
        """Get the total number of traffic lights in the network.

        Returns
        -------
        int
            Number of traffic lights
        """
        tls = self._net.getTrafficLights()
        return len(tls)

    def get_num_actions(self) -> int:
        """Get the total number of possible actions for single-agent control.

        This computes the Cartesian product of all traffic light phases,
        representing the total action space if a single agent controlled all
        traffic lights simultaneously.

        Note: For multi-agent setups, use get_action_breakdown() instead.

        Returns
        -------
        int
            Total number of action combinations (product of phase counts)
        """
        tls = self._net.getTrafficLights()
        if len(tls) > 0:
            # Compute product of phase counts across all traffic lights
            dim = 1
            for tl in tls:
                logic = tl.getPrograms()['0']
                dim *= len(logic.getPhases())
            return dim
        else:
            return 0

    def get_action_breakdown(self) -> Dict[int, Tuple[int, int]]:
        """Get observation and action dimensions for all agents.

        Returns
        -------
        Dict[int, Tuple[int, int]]
            Maps agent_id to (obs_dim, num_actions) tuple
            - obs_dim: dimension of observation vector
            - num_actions: number of possible actions for that agent
        """
        return {agID: action_mapping.dims for agID, action_mapping in self.schema.items()}
    
 

    def set_all_lights(self, state: str) -> None:
        """Set all traffic lights to the same state.

        Useful for debugging or creating specific traffic scenarios.

        Parameters
        ----------
        state : str
            Traffic light state character: 'r' (red), 'g' (green), 'G' (green priority),
            'y' (yellow), 'Y' (yellow priority), 'u' (red/yellow), 'o' (off blinking), 'O' (off)
        """
        for tlID in self._sumo.trafficlight.getIDList():
            # Set all lanes of this traffic light to the same state
            n = len(self._sumo.trafficlight.getControlledLanes(tlID))
            self._sumo.trafficlight.setRedYellowGreenState(tlID, state * n)

    def _getCurrentTotalTimeLoss(self) -> float:
        """Compute total time loss for all vehicles in the simulation.

        Time loss is the difference between ideal travel time (at max speed) and
        actual travel time. This is an alternative reward metric.

        Returns
        -------
        float
            Total time loss across all vehicles (in seconds)
        """
        dt = self._sumo.simulation.getDeltaT()
        timeloss = 0

        vehIDs = self._sumo.vehicle.getIDList()
        for vehID in vehIDs:
            Vmax = self._sumo.vehicle.getAllowedSpeed(vehID)  # Maximum allowed speed
            V = self._sumo.vehicle.getSpeed(vehID)             # Current speed
            timeloss += (1 - V / Vmax) * dt                    # Time loss = (1 - v/vmax) * dt

        return timeloss

    def get_route_trajectories(self, save_file: Optional[str] = None, plot_traj: bool = False) -> Dict[str, List[Tuple[float, float]]]:
        """Generate trajectory data for all possible routes in the network.

        This method simulates a vehicle along each route and records its (x,y)
        positions at each timestep. It also records waypoint locations where
        vehicles stop at red lights. Used for visualization and route analysis.

        Parameters
        ----------
        save_file : str, optional
            Path to save trajectories as pickle file (default: None)
        plot_traj : bool, optional
            If True, plot all trajectories using matplotlib (default: False)

        Returns
        -------
        Dict[str, List[Tuple[float, float]]]
            Maps route_id to list of (x, y) positions along the route
        """
        route_traj = dict()
        route_waypoints = dict()
        # Simulate each route to collect trajectory data
        for routeID in self._getAllRouteIDs():
            route = []
            waypoints = []
            vehID = "veh" + routeID

            # Add vehicle to simulation
            self._sumo.vehicle.add(vehID, routeID, departLane="best")
            self._sumo.simulationStep()
            self._sumo.simulationStep()

            # Set all lights red to create a stopping point (waypoint)
            self.set_all_lights('r')
            found_red_light = False

            # Record trajectory until vehicle exits network
            while self._sumo.vehicle.getIDCount() > 0:
                # Record waypoint when vehicle first stops at red light
                if self._sumo.vehicle.getSpeed(vehID) == 0.0 and not found_red_light:
                    found_red_light = True
                    waypoints.append(self._sumo.vehicle.getPosition(vehID))
                    self.set_all_lights('G')  # Turn lights green to let vehicle proceed

                # Record current position
                route.append(self._sumo.vehicle.getPosition(vehID))
                self._sumo.simulationStep()

            route_traj[routeID] = route
            route_waypoints[routeID] = waypoints
        # Save trajectories to file if requested
        if save_file is not None:
            with open(save_file, 'wb') as f:
                pickle.dump({"trajectories": route_traj, "waypoints": route_waypoints}, f)

        # Plot trajectories if requested
        if plot_traj:
            plt.figure()
            for r, xy in route_traj.items():
                X = [x for (x, y) in xy]
                Y = [y for (x, y) in xy]
                plt.plot(X, Y, '-')

                # Plot waypoints (red light stopping points)
                if r in route_waypoints:
                    wps = route_waypoints[r]
                    X = [x for (x, y) in wps]
                    Y = [y for (x, y) in wps]
                    plt.plot(X, Y, marker='o', markersize=5)
            plt.show()

        logger.info(f"Found {len(route_traj)} routes in network")
        return route_traj

    def _saveVehicles(self, output_path: str, use_total_time: bool = False) -> None:
        """Save current vehicle positions and routes to a text file.

        Used for data collection and visualization. Each line contains:
        route_id, x_position, y_position

        Parameters
        ----------
        output_path : str
            Directory where vehicle data will be saved
        use_total_time : bool, optional
            If True, use global timestep for filename; if False, use episode number
        """
        # Generate filename based on timestep
        if use_total_time:
            fname = f"{output_path}/{self.total_steps_run:009}.txt"
        else:
            fname = f"{output_path}/{self.current_episode:03}_{self.episode_step:04d}.txt"

        # Write vehicle data to file
        with open(fname, "w") as f:
            for v in self._sumo.vehicle.getIDList():
                route_edges = self._sumo.vehicle.getRoute(v)
                x, y = self._sumo.vehicle.getPosition(v)
                e0 = route_edges[0]   # Start edge
                e1 = route_edges[-1]  # End edge
                routeID = self.route_dict[e0][e1]
                f.write(f"{routeID}, {x}, {y}\n")

    def _get_spawn_probability(self, route_id: str) -> float:
        """Compute spawn probability for a route based on the spawn strategy.

        Parameters
        ----------
        route_id : str
            Route identifier

        Returns
        -------
        float
            Spawn probability for this route at current timestep
        """
        if callable(self.spawn_strategy):
            # Custom spawn function
            return self.spawn_strategy(self.episode_step, route_id)
        elif self.spawn_strategy == 'uniform':
            # Constant spawn rate (default)
            return self.vehicle_spawn_rate
        elif self.spawn_strategy == 'time_varying':
            # Simulate rush hour pattern: higher at beginning and end, lower in middle
            progress = self.episode_step / self.episode_length
            # Peak at 0.25 and 0.75 of episode (morning/evening rush)
            rush_hour_factor = 1.0 + 0.5 * (np.sin(4 * np.pi * progress - np.pi/2) + 1)
            return self.vehicle_spawn_rate * rush_hour_factor
        elif self.spawn_strategy == 'route_dependent':
            # Different rates for different routes (example: use route length as proxy)
            # This is a simple heuristic; could be customized based on route characteristics
            base_rate = self.vehicle_spawn_rate
            # Vary spawn rate by route hash (deterministic but varied)
            route_factor = 0.5 + (hash(route_id) % 100) / 100.0
            return base_rate * route_factor
        else:
            # Default to uniform if unknown strategy
            logger.warning(f"Unknown spawn_strategy '{self.spawn_strategy}', using uniform")
            return self.vehicle_spawn_rate

    def _spawnVehicles(self) -> None:
        """Spawn new vehicles probabilistically according to spawn strategy.

        For each route, a vehicle is spawned with probability determined by the
        spawn strategy. This creates stochastic traffic patterns. If real_routes
        is set, only those routes are used; otherwise all routes are available.
        """
        # Use restricted route set if provided, otherwise use all routes
        if self.real_routes is not None:
            all_routeIDs = self.real_routes
        else:
            all_routeIDs = self._getAllRouteIDs()

        # Probabilistically spawn vehicles on each route
        for routeID in all_routeIDs:
            spawn_prob = self._get_spawn_probability(routeID)
            if random.random() < spawn_prob:
                vehID = f"veh{self._vehcnt:08d}"
                self._sumo.vehicle.add(vehID, routeID)
                self._sumo.vehicle.setLength(vehID, self.car_length)
                self._sumo.vehicle.setWidth(vehID, self.car_length / 3.5)
                self._vehcnt += 1

    def _getAllRouteIDs(self) -> List[str]:
        """Get all valid route IDs from SUMO.

        Filters out internal routes (those starting with '!').

        Returns
        -------
        List[str]
            List of route IDs
        """
        return [s for s in self._sumo.route.getIDList() if s[0] != '!']

    def _initialize_routes(self) -> None:
        """Initialize all possible routes between fringe edges in the network.

        Creates a route for each pair of fringe (boundary) edges that are
        reachable from each other. Routes are named "start_edge->end_edge"
        and stored in both SUMO and the route_dict for lookup.

        Notes
        -----
        Excludes routes where start equals end, or where edges are mirror
        directions of the same road (e.g., "edge1" and "-edge1").
        """
        self.route_dict = defaultdict(dict)
        edges = self._net.getEdges()

        # Create routes between all reachable fringe edge pairs
        for e1 in edges:
            if e1.is_fringe():  # Only consider boundary edges
                for e2 in self._net.getReachable(e1):
                    if e2.is_fringe():
                        # Exclude self-loops and mirror directions
                        if e1 != e2 and e1.getID().replace('-', '') != e2.getID().replace('-', ''):
                            routeID = f"{e1.getID()}->{e2.getID()}"
                            self._sumo.route.add(routeID, [e1.getID(), e2.getID()])
                            self.route_dict[e1.getID()][e2.getID()] = routeID
