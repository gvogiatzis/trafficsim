# SUMO Traffic Control RL Environment

A reinforcement learning environment for training autonomous traffic control agents using [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/). This system enables training Deep Q-Network (DQN) agents to optimize traffic light control policies for reducing congestion and improving traffic flow.

## Overview

This project provides a complete framework for:
- Interfacing with SUMO traffic simulations through Python
- Training single or multi-agent RL systems for traffic light control
- Evaluating different control policies (learned, greedy, random)
- Converting SUMO simulations to real-world coordinate systems
- Collecting trajectory data and generating synthetic training datasets

## Key Features

- **Multi-Agent Support**: Control multiple traffic lights independently or coordinate multiple agents across a network
- **Flexible State Representations**: Choose between raw lane occupancy or aggregated demand features
- **DQN Implementation**: Deep Q-Network with experience replay and target networks (double DQN)
- **Multiple Action Policies**: Train RL agents, use greedy heuristics, or random baselines
- **SUMO Integration**: Full TraCI API integration with both GUI and headless modes
- **Route Management**: Filter and restrict vehicle routes based on real-world data
- **Data Collection**: Record vehicle trajectories, screenshots, and state-action pairs for analysis
- **Sim-to-Real Tools**: Convert SUMO coordinates to real-world GPS coordinates

## Architecture

### Core Components

#### 1. Environment (`sumoenv/`)
- **`traffic_control_env.py`**: Main RL environment implementing OpenAI Gym-like interface
  - Manages SUMO simulation lifecycle
  - Handles multi-agent action/observation spaces
  - Computes rewards based on traffic metrics (halting vehicles)
  - Supports both simple (demand-based) and complex (lane-level) state representations

- **`sim_traffic_control_env.py`**: Simplified simulation environment for testing without SUMO

#### 2. RL Agents (`rl/`)
- **`dqn_agent.py`**: Single DQN agent implementation
  - Deep Q-Network with target network
  - Experience replay buffer
  - Epsilon-greedy exploration
  - Adam optimizer with MSE loss

- **`dqn_ensemble.py`**: Multi-agent coordinator
  - Manages multiple independent DQN agents
  - Synchronizes training and action selection
  - Handles multi-agent state/action/reward dictionaries

- **`models.py`**: Neural network architectures (MLP)

#### 3. Simulation Tools (`bin/`)
- **`trafficrl.py`**: Main training script with extensive CLI options
- **`getallroutes.py`**: Extract all possible routes from a SUMO network
- **`routegui.py`**: GUI tool for selecting/filtering routes
- **`sumotoreal.py`**: Convert SUMO coordinates to real-world coordinates

#### 4. Coordinate Conversion (`sumo2real/`)
- **`sumo_to_real_converter.py`**: Transform SUMO simulation data to GPS coordinates
- **`route_editor.py`**: Interactive route editing tools
- **`bbox_histogram.py`**: Spatial analysis utilities

## Installation

### Prerequisites
```bash
# Install SUMO
# On macOS:
brew install sumo

# On Ubuntu:
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

# Verify installation
sumo --version
```

### Python Dependencies
```bash
pip install numpy matplotlib torch sumolib traci typer
```

### Environment Setup
```bash
# Set SUMO_HOME (add to ~/.bashrc or ~/.zshrc)
export SUMO_HOME=/usr/share/sumo  # Adjust path as needed
export PATH=$PATH:$SUMO_HOME/bin
```

## Usage Guide

### Quick Start: Training a Traffic Control Agent

#### 1. Basic Training Command
```bash
python bin/trafficrl.py sumo_data/RussianJunction/RussianJunction.net.xml \
  --vehicle-spawn-rate 0.05 \
  --episode-length 500 \
  --num-episodes 100 \
  --output-path ./output
```

#### 2. Training with GUI Visualization
```bash
python bin/trafficrl.py sumo_data/RussianJunction/RussianJunction.net.xml \
  --use-gui \
  --vehicle-spawn-rate 0.05 \
  --episode-length 200 \
  --num-episodes 50
```

#### 3. Multi-Agent Training
```bash
# First, create a file 'agent_lights.txt' with traffic light assignments:
# tl1,tl2
# tl3,tl4

python bin/trafficrl.py sumo_data/Grid2by2.net.xml \
  --agent-lights-file agent_lights.txt \
  --num-episodes 100
```

#### 4. Testing a Trained Model
```bash
python bin/trafficrl.py sumo_data/RussianJunction/RussianJunction.net.xml \
  --test \
  --in-model-fname ./output/models/RussianJunction.pt \
  --num-episodes 20 \
  --use-gui
```

### Command Line Options

#### Environment Options
- `--vehicle-spawn-rate`: Rate at which vehicles spawn (default: 0.05)
- `--episode-length`: Number of timesteps per episode (default: 20)
- `--sumo-timestep`: SUMO steps between RL actions (default: 20)
- `--step-length`: Simulation timestep length in seconds (default: 1.0)
- `--use-gui`: Enable SUMO GUI visualization
- `--seed`: Random seed for reproducibility
- `--car-length`: Vehicle length in SUMO units (default: 5.0)

#### Training Options
- `--num-episodes`: Total training episodes (default: 50)
- `--network-layers`: Network architecture, e.g., "1024x1024" (default: "1024x1024")
- `--lr`: Learning rate (default: 0.0001)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon`: Initial exploration rate (default: 0.1)
- `--epsilon-final`: Final exploration rate (default: 0.01)
- `--batch-size`: Training batch size (default: 32)
- `--replay-buffer-size`: Experience replay buffer size (default: 500000)
- `--update-freq`: Target network update frequency (default: 2)

#### Multi-Agent Options
- `--agent-lights-file`: File specifying traffic light assignments per agent

#### Baseline Policies
- `--random-action`: Use random action policy
- `--greedy-prob`: Probability of using greedy action (0.0-1.0)

#### Data Collection
- `--record-tracks`: Save vehicle trajectories
- `--record-screenshots`: Save screenshots (requires --use-gui)
- `--record-input-output`: Save state-action pairs as CSV

#### Model Management
- `--in-model-fname`: Load pre-trained model
- `--out-model-fname`: Filename for saving trained model
- `--save-intermediate`: Save model after each episode

### Python API Usage

#### Basic Training Loop
```python
from sumoenv import TrafficControlEnv
from rl import DQNEnsemble

# Initialize environment
env = TrafficControlEnv(
    net_fname='sumo_data/RussianJunction/RussianJunction.net.xml',
    vehicle_spawn_rate=0.05,
    episode_length=500,
    use_gui=False
)

# Get action space dimensions
schema = env.get_action_breakdown()
# schema: {agent_id: (obs_dim, num_actions)}

# Create DQN ensemble
agent = DQNEnsemble(
    schema=schema,
    network_layers=[1024, 1024],
    learning_rate=0.0001,
    discount_factor=0.99,
    epsilon=0.1
)

# Training loop
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Choose action
        action = agent.choose_action(state)

        # Step environment
        next_state, reward, done = env.step(action)

        # Store experience and train
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += sum(reward.values())

    print(f"Episode {episode}: Reward = {total_reward}")
    agent.decay_epsilon()

# Save model
agent.save_to_file('trained_model.pt')
env.close()
```

#### Using Different Action Policies
```python
# Random action
random_action = env.choose_random_action()

# Greedy action (maximize cars getting green light)
greedy_action = env.choose_greedy_action()

# Learned action
learned_action = agent.choose_action(state, deterministic=True)

# Step with chosen action
next_state, reward, done = env.step(action)
```

### Working with Routes

#### Extract All Routes
```bash
python bin/getallroutes.py sumo_data/RussianJunction/RussianJunction.net.xml \
  --output-file routes.pkl
```

#### Filter Routes with GUI
```bash
python bin/routegui.py --route-file routes.pkl
# This creates an interactive GUI to select active routes
# Saves filtered routes for use with --real-routes-file
```

#### Use Filtered Routes in Training
```bash
python bin/trafficrl.py sumo_data/RussianJunction/RussianJunction.net.xml \
  --real-routes-file filtered_routes.pkl
```

## State and Reward Design

### State Representation

The environment supports two state representation modes (controlled by `simple_state` flag):

#### Simple State (default)
- **Dimension**: `num_actions`
- **Content**: For each action, sum of halting vehicles that would receive green light
- **Advantage**: Directly encodes action utility, faster learning

#### Complex State
- **Dimension**: `num_lanes` (all controlled lanes)
- **Content**: Halting vehicle count per lane
- **Advantage**: More detailed, allows learning complex policies

### Reward Function
```python
reward = -sum(halting_vehicles_per_lane)
```
The agent is rewarded for minimizing total halting vehicles across all controlled lanes.

### Actions

#### Single Agent
- **Action Space**: Cartesian product of all traffic light phases
- **Example**: 2 traffic lights with 4 phases each = 16 possible actions

#### Multi-Agent
- **Action Space**: Each agent controls a subset of traffic lights
- **Action**: Dictionary `{agent_id: phase_combination}`
- **Coordination**: Agents act independently but affect shared environment

## File Structure

```
trafficsim/
├── bin/                          # Executable scripts
│   ├── trafficrl.py             # Main training CLI
│   ├── getallroutes.py          # Route extraction
│   ├── routegui.py              # Route filtering GUI
│   └── sumotoreal.py            # Coordinate conversion
├── sumoenv/                      # SUMO environment wrappers
│   ├── traffic_control_env.py   # Main RL environment
│   └── sim_traffic_control_env.py # Simplified sim environment
├── rl/                          # RL algorithms
│   ├── dqn_agent.py            # Single DQN agent
│   ├── dqn_ensemble.py         # Multi-agent coordinator
│   ├── models.py               # Neural network models
│   └── supervised_learning_pretrainer.py
├── sumo2real/                   # Coordinate transformation
│   ├── sumo_to_real_converter.py
│   ├── route_editor.py
│   └── bbox_histogram.py
├── sumo_data/                   # SUMO network files
│   ├── RussianJunction/
│   ├── Grid2by2/
│   ├── Birmingham_demo_0312/
│   └── ...
├── utils/                       # Utility functions
└── output/                      # Training outputs
    ├── models/                  # Saved models
    ├── sumo_tracks/            # Vehicle trajectories
    └── sumo_screenshots/       # Visualization frames
```

## Network Files

The project includes several pre-configured SUMO networks:

- **RussianJunction**: Single complex intersection
- **Grid2by2**, **Grid3by3**: Grid networks with multiple intersections
- **TwoJunction**: Two connected intersections
- **ThreeLaneJunction**: Three-way intersection
- **Birmingham_demo_0312**: Real-world Birmingham road network

### Creating Custom Networks

Use [SUMO netedit](https://sumo.dlr.de/docs/Netedit/index.html):
```bash
netedit
# Create your network, save as .net.xml file
# Use with trafficrl.py
```

## Training Tips

### Hyperparameter Tuning
- **Start with high epsilon** (0.3-0.5) for exploration, decay to 0.01-0.05
- **Use larger buffers** (500k+) for complex networks
- **Adjust sumo-timestep**: Higher = faster training but coarser control
- **Network size**: Start with 512x512, scale up for complex intersections

### Multi-Agent Considerations
- Assign neighboring lights to same agent for coordination
- Balance agent loads (similar number of lights per agent)
- Expect longer training times with more agents

### Baseline Comparison
Always compare against:
1. **Random policy**: `--random-action`
2. **Greedy policy**: `--greedy-prob 1.0`
3. **Fixed timing**: Modify SUMO network with fixed-time programs

## Troubleshooting

### SUMO Connection Issues
```python
# Check SUMO_HOME
import os
print(os.environ.get('SUMO_HOME'))

# Verify TraCI
import traci
```

### Memory Issues
- Reduce `--replay-buffer-size`
- Decrease `--episode-length`
- Use smaller network or fewer agents

### Training Not Converging
- Increase `--num-episodes`
- Tune `--epsilon` decay schedule
- Try different `--network-layers`
- Add `--greedy-prob 0.1` for curriculum learning

### GUI Not Showing
- Ensure `--use-gui` flag is set
- Install `sumo-gui` package
- Check X11 forwarding if using SSH

## Advanced Features

### Distillation and Imitation Learning
Record expert trajectories (greedy policy) for supervised pre-training:
```bash
python bin/trafficrl.py sumo_data/RussianJunction/RussianJunction.net.xml \
  --greedy-prob 1.0 \
  --record-input-output \
  --num-episodes 100
# Creates agent_io_*.csv files for supervised learning
```

### Coordinate Conversion for Real-World Deployment
```python
from sumo2real import SumoToRealConverter

converter = SumoToRealConverter(
    net_file='sumo_data/Birmingham_demo_0312/network.net.xml',
    bbox_file='bbox.pkl'
)

# Convert SUMO coordinates to GPS
lat, lon = converter.sumo_to_gps(x=1000, y=500)
```

## Contributing

When adding new features:
1. Follow existing code structure
2. Add type hints for function signatures
3. Document new CLI options in `trafficrl.py`
4. Test with multiple network configurations

## Citation

If you use this code in your research, please cite:
```
[Add your citation information here]
```

## License

[Add license information]

## References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI API](https://sumo.dlr.de/docs/TraCI.html)
- [DQN Paper](https://www.nature.com/articles/nature14236)
