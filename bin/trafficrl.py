#!/usr/bin/env python
"""
Traffic RL Training Script

This is the main command-line interface for training and testing reinforcement learning
agents (DQN and PPO) to control traffic lights in SUMO simulations. It provides
comprehensive options for configuring the environment, training hyperparameters, and
data collection.

Usage Examples:
    # Basic DQN training
    python trafficrl.py sumo_data/RussianJunction/RussianJunction.net.xml

    # PPO training with GUI and custom parameters
    python trafficrl.py sumo_data/Grid2by2.net.xml --algorithm ppo --use-gui --num-episodes 100 --lr 0.0003

    # Testing a trained DQN model
    python trafficrl.py sumo_data/RussianJunction/RussianJunction.net.xml --test --in-model-fname output/models/model.pt

    # Multi-agent PPO training
    python trafficrl.py sumo_data/Grid2by2.net.xml --algorithm ppo --agent-lights-file agents.txt --num-episodes 200

Features:
- Single and multi-agent RL training
- Support for DQN and PPO algorithms
- Configurable hyperparameters for each algorithm
- Baseline policies (random, greedy)
- Data collection for imitation learning
- Model saving and loading
- SUMO GUI integration
"""

import sys
import os.path
# Add parent directory to path so we can import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typer
from typing import Optional as Opt, List, Tuple
from typing_extensions import Annotated as Ann
from random import random

# Initialize Typer CLI application with help options
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=True)


@app.command()
def main(net_fname: Ann[str, typer.Argument(help="the filename of the sumo network to use")],
         # ===== Environment Configuration =====
         vehicle_spawn_rate: Ann[Opt[float], typer.Option(help="The average rate at which new vehicles are being spawned")] = 0.05,

         episode_length: Ann[Opt[int], typer.Option(help='the number of timesteps for each episode')] = 20,

         use_gui: Ann[Opt[bool], typer.Option(help="If set, performs the simulation using the sumo-gui command, i.e. with a graphical interface")] = False,

         sumo_timestep: Ann[Opt[int], typer.Option(help='the number of sumo timesteps between RL timesteps (i.e. when actions are taken)')] = 20,

         seed: Ann[Opt[int], typer.Option(help='Random seed to be passed to sumo. This guarantees reproducible results. If not given, a different seed is chosen each time.')] = None,

         step_length: Ann[Opt[float], typer.Option(help='The length of a single timestep in the simulation in seconds. Set to <1.0 for finer granularity and >1.0 for speed (and less accuracy)')] = 1.0,

         car_length: Ann[Opt[float], typer.Option(help='The length of a car in sumo units. Increase to ensure cars stay away from each other when converted into the real world')] = 5.0,

         output_path: Ann[Opt[str], typer.Option(help='The output path for saving all outputs')] = "output",

         # ===== Algorithm Selection =====
         algorithm: Ann[Opt[str], typer.Option(help='The RL algorithm to use. Options: "dqn" (Deep Q-Network) or "ppo" (Proximal Policy Optimization)')] = "dqn",

         # ===== Training Configuration =====
         num_episodes: Ann[Opt[int], typer.Option(help='The number of episodes to train the agent')] = 50,

         in_model_fname: Ann[Opt[str], typer.Option(help='filename of a previously saved agent model which will be used as a starting point for further training. If not set, a new network is initialised according to the network-layers option.')] = None,

         network_layers: Ann[Opt[str], typer.Option(help="A string of integers separated by 'x' chars, denoting the size and number of hidden layers of the network architecture. E.g. '512x512x256' would create three hidden layers of dims 512,512 and 256. Ignored if 'in_model_fname' option is set.")] = "1024x1024",

         plot_reward: Ann[Opt[bool], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] = False,

         cuda: Ann[Opt[bool], typer.Option(help="If set (and if CUDA is available), will use GPU acceleration.")] = True,

         # ===== Visualization and Data Collection =====
         gui_config_file: Ann[Opt[str], typer.Option(help="A filename of a viewsettings configuration file.")] = None,

         real_routes_file: Ann[Opt[str], typer.Option(help="The real routes file saved by routegui. If set, will restrict vehicle generation in sumo to the routes that appear in that file. Use if you want to avoid certain difficult routes in your junction.")] = None,

         record_screenshots: Ann[Opt[bool], typer.Option(help="If set, will record a screenshot per timestep in [OUTPUT_PATH]/sumo_screenshots.")] = False,

         record_tracks: Ann[Opt[bool], typer.Option(help="If set, will save sumo vehicle tracks during each simulation step in [OUTPUT_PATH]/sumo_tracks.")] = False,

         # ===== Action Policy Options =====
         greedy_prob: Ann[Opt[float], typer.Option(help="A number between 0.0 and 1.0.  The probability of choosing the greedy action in each timestep.")] = 0.0,

         random_action: Ann[Opt[bool], typer.Option(help="If set, will apply a random action. This is a useful benchmark. If used in conjunction with training, will act as imitation RL where the agent is shown only the random actions being applied. Effectively equivalent to lambda = 0.")] = False,

         # ===== DQN Hyperparameters =====
         gamma: Ann[Opt[float], typer.Option(help='the discount factor for training models')]
          = 0.99,

         epsilon: Ann[Opt[float], typer.Option(help="The initial probability of choosing a random action in each timestep. Increase to help with exploration at the expense of worse performance. This will decay geometrically until it reaches epsilon_final.")]
          = 0.1,

         epsilon_final: Ann[Opt[float], typer.Option(help="The final probability of choosing a random action in each timestep. Epsilon keeps decaying geometrically until it reaches this value at the final episode.")]
          = 0.01,

         batch_size: Ann[Opt[int], typer.Option(help='the sample batch size for optimizing the models')]
          = 32,

         replay_buffer_size: Ann[Opt[int], typer.Option(help="The size of the replay buffer used by each DQNAgent.")]
          = 500000,

         update_freq: Ann[Opt[int], typer.Option(help="This is the number of timesteps between model updates. ")] = 2,

         lr: Ann[Opt[float], typer.Option(help="The learning rate of the networks.")]
          = 0.0001,

         # ===== Model Management =====
         out_model_fname: Ann[Opt[str], typer.Option(help="If set, gives the filename to use when saving the trained model. If not set, the name of the network is used with a .pt extension")] = None,

         save_intermediate: Ann[Opt[bool], typer.Option(help="If set, saves the trained model after every epoch at {output_path}/model/model{epoch:04d}.pt")]
          = False,

         test: Ann[Opt[bool], typer.Option(help="If set, performs only testing of a pre-trained agent model.")] = False,

         # ===== Multi-Agent Configuration =====
         agent_lights_file: Ann[Opt[str], typer.Option(help='filename consisting of the TLs that are assigned to each RL agent. Each line corresponds to a different agent consists of a comma-separated list of TL ids to be controlled by that agent. A file with N lines corresponds to N agents. If not given then a single agent controlling all TLs is created.')] = None,

         record_input_output: Ann[Opt[bool], typer.Option(help="If set, records detailed input (state) output (action) pairs for each agent as a set of csv files.")] = False):
    """
    Main entry point for training and testing RL traffic control agents (DQN or PPO).

    This function handles command-line argument parsing, environment initialization,
    and orchestrates the training/testing loop. It supports both single-agent and
    multi-agent configurations with extensive customization options.

    The function performs the following steps:
    1. Parse and validate command-line arguments
    2. Initialize the SUMO traffic control environment
    3. Call the RL training/testing loop
    4. Display results and save the trained model
    5. Clean up resources

    Parameters are organized into groups:
    - Algorithm Selection: Choose between DQN and PPO
    - Environment: SUMO simulation configuration
    - Training: Episode count, model architecture
    - Visualization: GUI, screenshots, trajectory recording
    - Action Policy: Baseline policies (random, greedy)
    - RL Hyperparameters: Learning rate, epsilon (DQN), gamma, etc.
    - Model Management: Loading/saving models
    - Multi-Agent: Agent-to-traffic-light assignments
    """

    # ===== Setup and Initialization =====

    # If no output model filename provided, derive it from the network filename
    if out_model_fname is None:
        out_model_fname = f"{os.path.splitext(os.path.basename(net_fname))[0]}.pt"

    # Import required modules
    from sumoenv import TrafficControlEnv
    import matplotlib.pyplot as plt
    import numpy as np

    # Parse network architecture string (e.g., "1024x1024" -> [1024, 1024])
    network_layers = [int(s) for s in network_layers.split("x") if s.isnumeric()]

    # ===== Environment Initialization =====
    # Create the SUMO traffic control environment with all specified parameters
    env = TrafficControlEnv(
        net_fname=net_fname,
        vehicle_spawn_rate=vehicle_spawn_rate,
        episode_length=episode_length,
        use_gui=use_gui,
        sumo_timestep=sumo_timestep,
        seed=seed,
        step_length=step_length,
        output_path=output_path,
        record_tracks=record_tracks,
        car_length=car_length,
        record_screenshots=record_screenshots,
        gui_config_file=gui_config_file,
        real_routes_file=real_routes_file,
        agent_lights_file=agent_lights_file
    )

    # ===== Validate Algorithm Choice =====
    if algorithm not in ['dqn', 'ppo']:
        print(f"Error: Unknown algorithm '{algorithm}'. Must be 'dqn' or 'ppo'.")
        return

    # ===== Training/Testing Loop =====
    # Execute the main RL loop and collect rewards for each episode
    rewards = rl_loop(
        env=env,
        algorithm=algorithm,
        cuda=cuda,
        network_layers=network_layers,
        output_path=output_path,
        gamma=gamma,
        replay_buffer_size=replay_buffer_size,
        num_episodes=num_episodes,
        test=test,
        lr=lr,
        epsilon=epsilon,
        epsilon_final=epsilon_final,
        batch_size=batch_size,
        save_intermediate=save_intermediate,
        in_model_fname=in_model_fname,
        out_model_fname=out_model_fname,
        update_freq=update_freq,
        random_action=random_action,
        greedy_prob=greedy_prob,
        record_input_output=record_input_output
    )

    # ===== Results Display =====
    # In test mode, print statistics of the collected rewards
    if test:
        print(f"Average reward is: {np.mean(rewards):0.1f} \u00B1 {np.std(rewards):0.1f}")

    # Plot reward curve if requested
    if plot_reward:
        print('plotting reward')
        plt.plot(rewards, '-')
        plt.show()

    # ===== Cleanup =====
    print("closing env")
    env.close()



def rl_loop(env, algorithm, cuda, network_layers, output_path, gamma, replay_buffer_size, num_episodes, test, lr, epsilon, epsilon_final, batch_size, save_intermediate, in_model_fname, out_model_fname, update_freq, random_action, greedy_prob, record_input_output):
    """
    Main reinforcement learning training/testing loop.

    This function implements the core training algorithm:
    1. Initialize RL agents (DQN or PPO) based on environment schema
    2. For each episode:
        a. Reset environment
        b. While not done:
            - Select action (learned, greedy, or random)
            - Step environment
            - Store experience in buffer
            - Train networks (DQN: every step, PPO: when buffer full)
            - Update target networks periodically (DQN only)
        c. Decay exploration rate (DQN only)
        d. Save model checkpoints if requested
    3. Return collected rewards for analysis

    Parameters
    ----------
    env : TrafficControlEnv
        The SUMO traffic control environment
    algorithm : str
        The RL algorithm to use ('dqn' or 'ppo')
    cuda : bool
        Whether to use GPU acceleration (currently not implemented)
    network_layers : list of int
        Hidden layer sizes for agent networks
    output_path : str
        Directory for saving models and data
    gamma : float
        Discount factor for future rewards
    replay_buffer_size : int
        Maximum size of experience replay buffer (DQN) or trajectory buffer (PPO)
    num_episodes : int
        Number of training/testing episodes
    test : bool
        If True, run in test mode (no training, deterministic actions)
    lr : float
        Learning rate for neural network optimization
    epsilon : float
        Initial exploration rate (DQN only, ignored for PPO)
    epsilon_final : float
        Final exploration rate after decay (DQN only, ignored for PPO)
    batch_size : int
        Number of experiences to sample per training step
    save_intermediate : bool
        If True, save model after each episode
    in_model_fname : str or None
        Path to pre-trained model to load
    out_model_fname : str
        Filename for saving final trained model
    update_freq : int
        Number of timesteps between target network updates (DQN only, ignored for PPO)
    random_action : bool
        If True, use random action policy (baseline)
    greedy_prob : float
        Probability of using greedy action instead of learned policy
    record_input_output : bool
        If True, save state-action pairs to CSV files

    Returns
    -------
    list of float
        Total reward for each episode
    """
    from rl import DQNEnsemble, PPOEnsemble

    # ===== Agent Initialization =====
    # Get the schema defining each agent's observation and action dimensions
    mini_schema = env.get_action_breakdown()
    # Format: {agent_id: (obs_dim, num_actions)}

    # ===== Data Recording Setup =====
    # Create CSV files to record state-action pairs for each agent (for imitation learning)
    if record_input_output:
        files = {}
        for agentID, (obs_dim, num_actions) in mini_schema.items():
            # Create one CSV file per agent
            files[agentID] = open(f"{output_path}/agent_io_{agentID}.csv", "w")
            # Write header: x0,x1,x2,...,xn,a (state features and action)
            files[agentID].write(",".join(map(lambda x: f"x{x}", range(obs_dim))))
            files[agentID].write(",a\n")

    # ===== Create RL Agent Ensemble =====
    if algorithm == 'dqn':
        # Calculate epsilon decay rate to reach epsilon_final by the last episode
        # Uses geometric decay: epsilon_final = epsilon * (decay_rate)^(num_episodes-1)
        epsilon_decay = (epsilon_final / epsilon) ** (1 / (num_episodes - 1))

        agent = DQNEnsemble(
            schema=mini_schema,
            network_layers=network_layers,
            learning_rate=lr,
            discount_factor=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            memory_capacity=replay_buffer_size
        )
        print(f"Using DQN with {len(agent.agents)} agents")
        print(f"DQN config: lr={lr}, gamma={gamma}, epsilon={epsilon:.3f}, buffer={replay_buffer_size}, batch={batch_size}")

    elif algorithm == 'ppo':
        # PPO-specific hyperparameters
        # Scale buffer size to episode length (train every 2-4 episodes)
        steps_per_episode = env.episode_length
        ppo_buffer_size = min(replay_buffer_size, max(256, steps_per_episode * 3))

        # Use larger batch size for stable policy updates
        ppo_batch_size = max(batch_size, min(64, ppo_buffer_size // 4))

        agent = PPOEnsemble(
            schema=mini_schema,
            network_layers=network_layers,
            learning_rate=lr,
            discount_factor=gamma,
            batch_size=ppo_batch_size,
            buffer_capacity=ppo_buffer_size
        )
        print(f"Using PPO with {len(agent.agents)} agents")
        print(f"PPO config: lr={lr}, gamma={gamma}, buffer={ppo_buffer_size}, batch={ppo_batch_size}, steps_per_episode={steps_per_episode}")
        print(f"PPO will train approximately every {ppo_buffer_size // steps_per_episode} episodes")

    # Load pre-trained model if provided
    if in_model_fname is not None:
        agent.load_from_file(in_model_fname)

    # Create output directory for models if it doesn't exist
    if not os.path.exists(f"{output_path}/models/"):
        os.makedirs(f"{output_path}/models/")

    # ===== Main Training/Testing Loop =====
    rewards = []  # Store total reward for each episode
    steps_to_update = update_freq  # Counter for target network updates (DQN only)

    for e in range(num_episodes):
        done = False
        S_new = env.reset()  # Initialize new episode
        tot_reward = 0  # Accumulator for episode reward

        # Episode loop: continue until episode terminates
        while not done:
            S = S_new  # Current state

            # ===== Action Selection =====
            # Track whether action came from learned policy (needed for PPO experience storage)
            used_learned_policy = False

            # Choose action based on policy mode
            if random_action:
                # Baseline: Random policy for comparison
                A = env.choose_random_action()
                log_probs = None  # Not needed for random actions
                values = None
            elif greedy_prob > 0.0 and random() <= greedy_prob:
                # Greedy heuristic: Choose action that maximizes immediate benefit
                # (gives green light to most waiting vehicles)
                A = env.choose_greedy_action()
                log_probs = None  # Not needed for greedy actions
                values = None
            else:
                # Learned policy: Use RL agent
                used_learned_policy = True
                # In test mode, always exploit (deterministic=True)
                if algorithm == 'dqn':
                    A = agent.choose_action(S, deterministic=test)
                    log_probs = None  # DQN doesn't use log probs
                    values = None
                elif algorithm == 'ppo':
                    A, log_probs, values = agent.choose_action(S, deterministic=test)

            # ===== Data Recording =====
            # Record state-action pairs to CSV files if requested
            if record_input_output:
                for agentID, state in S.items():
                    files[agentID].write(",".join(map(str, state)))
                    files[agentID].write(",")
                    files[agentID].write(str(A[agentID]) + "\n")

            # ===== Environment Step =====
            # Execute action and observe next state and reward
            S_new, R, done = env.step(action=A)

            # Calculate total reward across all agents
            tot_R = sum(r for agID, r in R.items())
            tot_reward += tot_R

            # ===== Training Step =====
            # Only train if not in test mode
            if not test:
                if algorithm == 'dqn':
                    # DQN: Store experience in replay buffer
                    # DQN is off-policy, so we can learn from any action source
                    agent.remember(S, A, R, S_new, done)

                    # Train networks via experience replay (one gradient step)
                    agent.replay()

                    # Periodically update target networks for stable training
                    steps_to_update -= 1
                    if steps_to_update == 0:
                        agent.update_target_model()
                        steps_to_update = update_freq

                    # Decay exploration rate
                    agent.decay_epsilon()

                elif algorithm == 'ppo':
                    # PPO: Only store experiences from the learned policy
                    # PPO is on-policy, so we can't learn from random/greedy actions
                    if used_learned_policy:
                        agent.remember(S, A, R, S_new, done, log_probs, values)

                    # Train when buffer is full (on-policy learning)
                    if agent.should_train():
                        print(f"  [PPO training at episode {e+1}]")
                        agent.train()

        # ===== Episode Complete =====
        rewards.append(tot_reward)

        # Print progress
        if test:
            print(f"Testing: {e+1}/{num_episodes} tot_reward={tot_reward}", end='\n')
        else:
            print(f"Training: {e+1}/{num_episodes} tot_reward={tot_reward}", end='\n')

        # Save intermediate model checkpoint if requested
        if save_intermediate:
            save_fname = os.path.join(output_path, 'models', f"{os.path.splitext(out_model_fname)[0]}{e:04d}.pt")
            agent.save_to_file(save_fname)

    # ===== Save Final Model =====
    final_save_name = os.path.join(output_path, 'models', out_model_fname)
    agent.save_to_file(final_save_name)

    # ===== Cleanup =====
    # Close all CSV files if recording was enabled
    if record_input_output:
        for agentID, f in files.items():
            f.close()

    return rewards

# ===== Script Entry Point =====
if __name__ == "__main__":
    # Run the Typer CLI application (parses arguments and calls main())
    app()
