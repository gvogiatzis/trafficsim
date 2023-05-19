import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if os.path.basename(sys.argv[0]) != "trafficrl.py":
    print("running from jupyter")
    sys.argv=["trafficrl.py", "train", "--net", "sumo_data/TwoJunction.net.xml"]


import typer
from typing import Optional as Opt, List, Tuple
from typing_extensions import Annotated as Ann
from types import SimpleNamespace


app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)
state = SimpleNamespace() # state variable that will hold common set of options




@app.callback()
def main(vehicle_spawn_rate: Ann[Opt[float], typer.Option(help="The average rate at which new vehicles are being spawned")] = 0.05, 

         episode_length:Ann[Opt[int], typer.Option(help='the number of timesteps for each episode')] = 100,
         
         use_gui:Ann[Opt[bool], typer.Option(help="If set, performs the simulation using the sumo-gui command, i.e. with a graphical interface")] = False,
         
         save_tracks:Ann[Opt[bool], typer.Option(help="If set, will save sumo vehicle tracks during each simulation step in [OUTPUT_PATH]/sumo_tracks.")] = False,
         
         sumo_timestep:Ann[Opt[int], typer.Option(help='the number of sumo timesteps between RL timesteps (i.e. when actions are taken)')] = 10,  
         
         seed:Ann[Opt[int], typer.Option(help='Random seed to be passed to sumo. This guarantees reproducible results. If not given, a different seed is chosen each time.')] = None,
         
         step_length:Ann[Opt[float], typer.Option(help='The length of a single timestep in the simulation in seconds. Set to <1.0 for finer granularity and >1.0 for speed (and less accuracy)')] = 1.0,  

         car_length:Ann[Opt[float], typer.Option(help='The length of a car in sumo units. Increase to ensure cars stay away from each other when converted into the real world')] = 5.0,          
         
         output_path:Ann[Opt[str], typer.Option(help='The output path for saving all outputs')] = "output",
         
         num_episodes:Ann[Opt[int], typer.Option(help='The number of episodes to train the agent')] = 50,

         input:Ann[Opt[str], typer.Option(help='filename of a previously saved agent model which will be used as a starting point for further training. If not set, a new network is initialised according to the network-layers option.')] = None,

         network_layers:Ann[Opt[str], typer.Option(help="A string of integers separated by 'x' chars, denoting the size and number of hidden layers of the network architecture. E.g. '512x512x256' would create three hidden layers of dims 512,512 and 256. Ignored if 'input' option is set.")] = "1024x1024",
         
         plot_reward: Ann[Opt[bool], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] = True,

         cuda: Ann[Opt[bool], typer.Option(help="If set (and if CUDA is available), will use GPU acceleration.")] = True
         ):

    state.__dict__.update(locals())


@app.command("train")
def train(net_fname: Ann[str, typer.Option("--net", help="the filename of the sumo network to use")],
          gamma: Ann[Opt[float], typer.Option(help='the discount factor for training models')] 
          = 0.99,
 
          epsilon: Ann[Opt[float], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] 
          = 0.1,

          batch_size: Ann[Opt[int], typer.Option(help='the sample batch size for optimizing the models')] 
          = 128,

          replay_buffer_size: Ann[Opt[int], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] 
          = 5000,

          lr: Ann[Opt[float], typer.Option(help="The learning rate of the networks.")] 
          = 0.0001,

          save_intermediate: Ann[Opt[bool], typer.Option(help="If set, saves the trained model after every epoch at {output_path}/model/model{epoch:04d}.pt")] 
          = False):
    """
    Train a RL agent to perform traffic control on SUMO using the DQN algorithm. The final model is saved at {output_path}/model/model_final.pt """
    state.__dict__.update(locals())

    print(state)
    # <CODE> this snippet must be repeated in both train and test commands
    # no other way to use the Typer framework for CLI
    from collections import deque
    from random import random,sample
    from sumoenv import TrafficControlEnv
    from sumoenv.models import MLPnet, loadModel, saveModel
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    state.network_layers = [int(s) for s in state.network_layers.split("x")]
    env = TrafficControlEnv(net_fname=net_fname, vehicle_spawn_rate=state.vehicle_spawn_rate, state_wrapper=lambda x:torch.tensor(x,dtype=torch.float),episode_length=state.episode_length,use_gui=state.use_gui,sumo_timestep=state.sumo_timestep, seed=state.seed, step_length=state.step_length, output_path=state.output_path,save_tracks=state.save_tracks,car_length=state.car_length)
    num_actions = env.get_num_actions()
    state_size = env.get_obs_dim()
    num_episodes = state.num_episodes
    plot_reward = state.plot_reward
    input = state.input    
    device = torch.device("cuda" if state.cuda and torch.cuda.is_available() else "cpu")
    if input is not None:
        qnet = loadModel(input)
    else:
        qnet = MLPnet(state_size, *state.network_layers, num_actions).to(device)    
    # </CODE>

    if not os.path.exists(f"{state.output_path}/models/"):
        os.makedirs(f"{state.output_path}/models/")
    

    def updateQNet_batch(qnet, batch, optimizer, loss):
        s,a,r,s_new,d = batch
        
        s=torch.stack(s)
        s_new=torch.stack(s_new)
        r=torch.tensor(r,device=device,dtype=torch.float)
        d=torch.tensor(d,device=device,dtype=torch.bool)
        a=torch.tensor(a,device=device,dtype=torch.int64)

        with torch.no_grad():
            qmax,_ = qnet(s_new).view(-1,num_actions).max(dim=1)
            target = torch.where(d, r, r + gamma * qmax).view(-1,1)
        L = loss(qnet(s).gather(1,a.view(-1,1)),target)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()


    replaybuffer = deque(maxlen=replay_buffer_size)

    optim = torch.optim.RMSprop(qnet.parameters(), lr= lr)
    loss = nn.MSELoss()
    rewards=[]
    for e in range(state.num_episodes):
        done = False
        S = env.reset()
        tot_reward=0

        # epsilon = max(0.1, epsilon*0.99)
        while not done:
            # Epsilon-greedy strategy
            A = env.sample_action() if random()<epsilon else qnet(S).argmax().item()

            # Executing action, receiving reward and new state
            S_new, R, done = env.step(A)

            # adding the latest experience onto the replay buffer
            replaybuffer.append((S,A,R,S_new,done))

            S = S_new
            if len(replaybuffer)>=batch_size:
                batch = sample(replaybuffer, batch_size)
                batch = zip(*batch)    
                updateQNet_batch(qnet, batch, optim, loss)

            tot_reward += R

        rewards.append(tot_reward)
        print(f"{e+1}/{num_episodes} tot_reward={tot_reward}",end='\n')

        if save_intermediate:
            saveModel(qnet, f"{state.output_path}/models/model{e:04d}.pt")

    saveModel(qnet, f"{state.output_path}/models/model_final.pt")
    if plot_reward:
        print('plotting reward')
        plt.plot(rewards,'-')
        plt.show()    
    env.close()


@app.command("test")
def test(net_fname: Ann[str, typer.Option("--net", help="the filename of the sumo network to use")],):
    """
    Test a previously trained RL agent on a sumo network.
    """
    state.__dict__.update(locals())
    print(state)
    # <CODE> this snippet must be repeated in both train and test commands
    # no other way to use the Typer framework for CLI
    from collections import deque
    from random import random,sample
    from sumoenv import TrafficControlEnv
    from sumoenv.models import MLPnet, loadModel, saveModel
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    state.network_layers = [int(s) for s in state.network_layers.split("x")]
    env = TrafficControlEnv(net_fname=net_fname, vehicle_spawn_rate=state.vehicle_spawn_rate, state_wrapper=lambda x:torch.tensor(x,dtype=torch.float),episode_length=state.episode_length,use_gui=state.use_gui,sumo_timestep=state.sumo_timestep, seed=state.seed, step_length=state.step_length, output_path=state.output_path,save_tracks=state.save_tracks,car_length=state.car_length)
    num_actions = env.get_num_actions()
    state_size = env.get_obs_dim()
    num_episodes = state.num_episodes
    plot_reward = state.plot_reward
    input = state.input    
    device = torch.device("cuda" if state.cuda and torch.cuda.is_available() else "cpu")
    if input is not None:
        qnet = loadModel(input)
    else:
        qnet = MLPnet(state_size, *state.network_layers, num_actions).to(device)    
    # </CODE>


    rewards=[]
    for e in range(num_episodes):
        done = False
        S = env.reset()
        tot_reward=0
        while not done:
            A = qnet(S).argmax().item()
            # Executing action, receiving reward and new state
            S_new, R, done = env.step(A)
            S=S_new
            tot_reward += R
        rewards.append(tot_reward)
        print(f"{e+1}/{num_episodes} tot_reward={tot_reward}",end='\n')

    env.close()
    if plot_reward:
        print('plotting reward')
        plt.plot(rewards,'-')
        plt.show()



if __name__ == "__main__":
    app()
