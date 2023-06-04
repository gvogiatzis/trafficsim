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




@app.command()
def main(net_fname: Ann[str, typer.Argument(help="the filename of the sumo network to use")],
         vehicle_spawn_rate: Ann[Opt[float], typer.Option(help="The average rate at which new vehicles are being spawned")] = 0.05, 

         episode_length:Ann[Opt[int], typer.Option(help='the number of timesteps for each episode')] = 100,
         
         use_gui:Ann[Opt[bool], typer.Option(help="If set, performs the simulation using the sumo-gui command, i.e. with a graphical interface")] = False,
         
         sumo_timestep:Ann[Opt[int], typer.Option(help='the number of sumo timesteps between RL timesteps (i.e. when actions are taken)')] = 20,  
         
         seed:Ann[Opt[int], typer.Option(help='Random seed to be passed to sumo. This guarantees reproducible results. If not given, a different seed is chosen each time.')] = None,
         
         step_length:Ann[Opt[float], typer.Option(help='The length of a single timestep in the simulation in seconds. Set to <1.0 for finer granularity and >1.0 for speed (and less accuracy)')] = 1.0,  

         car_length:Ann[Opt[float], typer.Option(help='The length of a car in sumo units. Increase to ensure cars stay away from each other when converted into the real world')] = 5.0,          
         
         output_path:Ann[Opt[str], typer.Option(help='The output path for saving all outputs')] = "output",
         
         num_episodes:Ann[Opt[int], typer.Option(help='The number of episodes to train the agent')] = 50,

         input:Ann[Opt[str], typer.Option(help='filename of a previously saved agent model which will be used as a starting point for further training. If not set, a new network is initialised according to the network-layers option.')] = None,

         network_layers:Ann[Opt[str], typer.Option(help="A string of integers separated by 'x' chars, denoting the size and number of hidden layers of the network architecture. E.g. '512x512x256' would create three hidden layers of dims 512,512 and 256. Ignored if 'input' option is set.")] = "1024x1024",
         
         plot_reward: Ann[Opt[bool], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] = False,

         cuda: Ann[Opt[bool], typer.Option(help="If set (and if CUDA is available), will use GPU acceleration.")] = True,

         gui_config_file: Ann[Opt[str], typer.Option(help="A filename of a viewsettings configuration file.")] = None,

         real_routes_file: Ann[Opt[str], typer.Option(help="The real routes file saved by routegui. If set, will restrict vehicle generation in sumo to the routes that appear in that file. Use if you want to avoid certain difficult routes in your junction.")] = None,

         record_screenshots: Ann[Opt[bool], typer.Option(help="If set, will record a screenshot per timestep in [OUTPUT_PATH]/sumo_screenshots.")] = False,

         record_tracks:Ann[Opt[bool], typer.Option(help="If set, will save sumo vehicle tracks during each simulation step in [OUTPUT_PATH]/sumo_tracks.")] = False,

         greedy_action:Ann[Opt[bool], typer.Option(help="If set, will apply action that shows the green light to the maximum number of cars. This is a useful benchmark.")] = False,

         random_action:Ann[Opt[bool], typer.Option(help="If set, will apply a random action. This is a useful benchmark.")] = False,

         gamma: Ann[Opt[float], typer.Option(help='the discount factor for training models')] 
          = 0.99,
 
         epsilon: Ann[Opt[float], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] 
          = 0.1,

         batch_size: Ann[Opt[int], typer.Option(help='the sample batch size for optimizing the models')] 
          = 128,

         replay_buffer_size: Ann[Opt[int], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] 
          = 500000,

         lr: Ann[Opt[float], typer.Option(help="The learning rate of the networks.")] 
          = 0.0001,

         model_fname: Ann[Opt[str], typer.Option(help="If set, gives the filename to use when saving the trained model. If not set, the name of the network is used with a .pt extension")] = None,

         save_intermediate: Ann[Opt[bool], typer.Option(help="If set, saves the trained model after every epoch at {output_path}/model/model{epoch:04d}.pt")] 
          = False,

         test: Ann[Opt[bool], typer.Option(help="If set, performs only testing of a pre-trained agent model.")] = False):

    # print(locals())

    if model_fname is None:
        model_fname = f"{os.path.splitext(os.path.basename(net_fname))[0]}.pt"

    from collections import deque
    from random import random,sample
    from sumoenv import TrafficControlEnv
    from sumoenv.models import MLPnet, loadModel, saveModel
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    network_layers = [int(s) for s in network_layers.split("x")]
    env = TrafficControlEnv(net_fname=net_fname, vehicle_spawn_rate=vehicle_spawn_rate, state_wrapper=lambda x:torch.tensor(x,dtype=torch.float),episode_length=episode_length,use_gui=use_gui,sumo_timestep=sumo_timestep, seed=seed, step_length=step_length, output_path=output_path,record_tracks=record_tracks,car_length=car_length,record_screenshots = record_screenshots, gui_config_file = gui_config_file, real_routes_file = real_routes_file, greedy_action=greedy_action,random_action=random_action )
    num_actions = env.get_num_actions()
    state_size = env.get_obs_dim()
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    if input is not None:
        qnet = loadModel(input)
    else:
        qnet = MLPnet(state_size, *network_layers, num_actions).to(device)    
    state.__dict__.update(locals())
    
    if not test: # we do training
        if not os.path.exists(f"{output_path}/models/"):
            os.makedirs(f"{output_path}/models/")

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
        # optim = torch.optim.Adam(qnet.parameters(), lr= lr)
        loss = nn.MSELoss()
        rewards=[]
        for e in range(num_episodes):
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
            print(f"Training: {e+1}/{num_episodes} tot_reward={tot_reward}",end='\n')

            if save_intermediate:
                save_fname = os.path.join(output_path, 'models', f"{os.path.splitext(model_fname)[0]}{e:04d}.pt")
                saveModel(qnet, save_fname)

        saveModel(qnet, os.path.join(output_path, 'models', model_fname))
    else: # we do simple testing of an existing model
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
            print(f"Testing: {e+1}/{num_episodes} tot_reward={tot_reward}",end='\n')


    if plot_reward:
        print('plotting reward')
        plt.plot(rewards,'-')
        plt.show()  
    print("closing env")
    env.close()




if __name__ == "__main__":
    app()
