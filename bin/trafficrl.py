#!/usr/bin/env python
import sys
import os.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# if os.path.basename(sys.argv[0]) != "trafficrl.py":
#     print("running from jupyter")
#     sys.argv=["trafficrl.py", "train", "--net", "sumo_data/TwoJunction.net.xml"]


import typer
from typing import Optional as Opt, List, Tuple
from typing_extensions import Annotated as Ann
from random import random

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=True)


@app.command()
def main(net_fname: Ann[str, typer.Argument(help="the filename of the sumo network to use")],
         vehicle_spawn_rate: Ann[Opt[float], typer.Option(help="The average rate at which new vehicles are being spawned")] = 0.05, 

         episode_length:Ann[Opt[int], typer.Option(help='the number of timesteps for each episode')] = 20,
         
         use_gui:Ann[Opt[bool], typer.Option(help="If set, performs the simulation using the sumo-gui command, i.e. with a graphical interface")] = False,
         
         sumo_timestep:Ann[Opt[int], typer.Option(help='the number of sumo timesteps between RL timesteps (i.e. when actions are taken)')] = 20,  
         
         seed:Ann[Opt[int], typer.Option(help='Random seed to be passed to sumo. This guarantees reproducible results. If not given, a different seed is chosen each time.')] = None,
         
         step_length:Ann[Opt[float], typer.Option(help='The length of a single timestep in the simulation in seconds. Set to <1.0 for finer granularity and >1.0 for speed (and less accuracy)')] = 1.0,  

         car_length:Ann[Opt[float], typer.Option(help='The length of a car in sumo units. Increase to ensure cars stay away from each other when converted into the real world')] = 5.0,          
         
         output_path:Ann[Opt[str], typer.Option(help='The output path for saving all outputs')] = "output",
         
         num_episodes:Ann[Opt[int], typer.Option(help='The number of episodes to train the agent')] = 50,

         in_model_fname:Ann[Opt[str], typer.Option(help='filename of a previously saved agent model which will be used as a starting point for further training. If not set, a new network is initialised according to the network-layers option.')] = None,

         network_layers:Ann[Opt[str], typer.Option(help="A string of integers separated by 'x' chars, denoting the size and number of hidden layers of the network architecture. E.g. '512x512x256' would create three hidden layers of dims 512,512 and 256. Ignored if 'in_model_fname' option is set.")] = "1024x1024",
         
         plot_reward: Ann[Opt[bool], typer.Option(help="If set, will plot the reward vs episode number at the end of all episodes.")] = False,

         cuda: Ann[Opt[bool], typer.Option(help="If set (and if CUDA is available), will use GPU acceleration.")] = True,

         gui_config_file: Ann[Opt[str], typer.Option(help="A filename of a viewsettings configuration file.")] = None,

         real_routes_file: Ann[Opt[str], typer.Option(help="The real routes file saved by routegui. If set, will restrict vehicle generation in sumo to the routes that appear in that file. Use if you want to avoid certain difficult routes in your junction.")] = None,

         record_screenshots: Ann[Opt[bool], typer.Option(help="If set, will record a screenshot per timestep in [OUTPUT_PATH]/sumo_screenshots.")] = False,

         record_tracks:Ann[Opt[bool], typer.Option(help="If set, will save sumo vehicle tracks during each simulation step in [OUTPUT_PATH]/sumo_tracks.")] = False,

        #  greedy_action:Ann[Opt[bool], typer.Option(help="If set, will apply action that shows the green light to the maximum number of cars. This is a useful benchmark. If used in conjunction with training, will act as imitation RL where the agent is shown only the greedy actions being applied.")] = False,
         
         greedy_prob:Ann[Opt[float], typer.Option(help="A number between 0.0 and 1.0.  The probability of choosing the greedy action in each timestep.")] = 0.0,

         random_action:Ann[Opt[bool], typer.Option(help="If set, will apply a random action. This is a useful benchmark. If used in conjunction with training, will act as imitation RL where the agent is shown only the random actions being applied. Effectively equivalent to lambda = 0.")] = False,

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

         out_model_fname: Ann[Opt[str], typer.Option(help="If set, gives the filename to use when saving the trained model. If not set, the name of the network is used with a .pt extension")] = None,

         save_intermediate: Ann[Opt[bool], typer.Option(help="If set, saves the trained model after every epoch at {output_path}/model/model{epoch:04d}.pt")] 
          = False,

         test: Ann[Opt[bool], typer.Option(help="If set, performs only testing of a pre-trained agent model.")] = False,
         
         agent_lights_file:Ann[Opt[str], typer.Option(help='filename consisting of the TLs that are assigned to each RL agent. Each line corresponds to a different agent consists of a comma-separated list of TL ids to be controlled by that agent. A file with N lines corresponds to N agents. If not given then a single agent controlling all TLs is created.')] = None,
         record_input_output:Ann[Opt[bool], typer.Option(help="If set, records detailed input (state) output (action) pairs for each agent as a set of csv files.")] = False):
    """
    This is the main function of the traffic rl script. It does all the command line processing.
    """

    if out_model_fname is None:
        out_model_fname = f"{os.path.splitext(os.path.basename(net_fname))[0]}.pt"

    from sumoenv import TrafficControlEnv

    import matplotlib.pyplot as plt
    import numpy as np

    network_layers = [int(s) for s in network_layers.split("x") if s.isnumeric()]



    env = TrafficControlEnv(net_fname=net_fname, vehicle_spawn_rate=vehicle_spawn_rate, state_wrapper=None, episode_length=episode_length,use_gui=use_gui,sumo_timestep=sumo_timestep, seed=seed, step_length=step_length, output_path=output_path,record_tracks=record_tracks,car_length=car_length,record_screenshots = record_screenshots, gui_config_file = gui_config_file, real_routes_file = real_routes_file, random_action=random_action,agent_lights_file=agent_lights_file)



    rewards = rl_loop(env=env, cuda=cuda, network_layers=network_layers, output_path=output_path, gamma=gamma, replay_buffer_size=replay_buffer_size, num_episodes=num_episodes, test=test, lr=lr, epsilon=epsilon, epsilon_final=epsilon_final, batch_size=batch_size, save_intermediate=save_intermediate, in_model_fname = in_model_fname, out_model_fname=out_model_fname, update_freq=update_freq,greedy_prob=greedy_prob, record_input_output=record_input_output)

    if test:
        print(f"Average reward is: {np.mean(rewards):0.1f} \u00B1 {np.std(rewards):0.1f}")

    if plot_reward:
        print('plotting reward')
        plt.plot(rewards,'-')
        plt.show()  
    print("closing env")
    env.close()



def rl_loop(env, cuda, network_layers, output_path, gamma, replay_buffer_size, num_episodes, test, lr, epsilon, epsilon_final, batch_size, save_intermediate, in_model_fname, out_model_fname,update_freq,greedy_prob,record_input_output):
    from rl import DQNEnsemble

    mini_schema = env.get_action_breakdown()
    # print(mini_schema)
    # print(env.schema)

    # we create a csv file for each agend and write out the headers (x0,x1,x2,...,xn,a) 
    # where x0,...,xn is the state and a is the action taken by the agent
    if record_input_output:
        files={}
        for agentID, (obs_dim, num_actions) in mini_schema.items():
            files[agentID] = open(f"{output_path}/agent_io_"+str(agentID)+".csv", "w")
            files[agentID].write(",".join(map(lambda x: "x"+str(x),range(obs_dim))))
            files[agentID].write(",a\n")
        
    dqn_agent = DQNEnsemble(schema=mini_schema, network_layers=network_layers, learning_rate=lr, discount_factor=gamma, epsilon=epsilon, epsilon_decay=(epsilon_final/epsilon)**(1/(num_episodes-1)),batch_size=batch_size, memory_capacity=replay_buffer_size)

    print(f"We are using {len(dqn_agent.agents)} agents")

    if in_model_fname is not None:
        dqn_agent.load_from_file(in_model_fname)
    
    if not os.path.exists(f"{output_path}/models/"):
        os.makedirs(f"{output_path}/models/")

    rewards=[]
    steps_to_update = update_freq
    for e in range(num_episodes):
        done = False
        S_new = env.reset()

        tot_reward=0

        while not done:
            S = S_new # Update the current state

            if random()<=greedy_prob:
                A = env.choose_greedy_action()
            else:
                A = dqn_agent.choose_action(S, deterministic = test)

            # at this point, if needed, we record the state,action pair for each agent
            if record_input_output:
                # record S,A in table
                for agentID,state in S.items():
                    files[agentID].write(",".join(map(str,state)))
                    files[agentID].write(",")
                    files[agentID].write(str(A[agentID])+"\n")
                
               
            S_new, R, done = env.step(action=A)

            tot_R = sum(r for agID,r in R.items())
            tot_reward += tot_R  

            if not test:
                dqn_agent.remember(S, A, R, S_new, done)  # Remember the experience
                dqn_agent.replay()  # Train the agent by replaying experiences
                # dqn_agent.replay_supervised(target_fun= lambda x: x @ M)

                steps_to_update -= 1
                if steps_to_update == 0:
                    dqn_agent.update_target_model()
                    steps_to_update = update_freq
                    
        dqn_agent.decay_epsilon()
        # print(f"epsilon = {dqn_agent.agents[0].epsilon}")
            
        rewards.append(tot_reward)
        print(f"Training: {e+1}/{num_episodes} tot_reward={tot_reward}",end='\n')

        if save_intermediate:
            save_fname = os.path.join(output_path, 'models', f"{os.path.splitext(out_model_fname)[0]}{e:04d}.pt")
            dqn_agent.save_to_file(save_fname)

    final_save_name = os.path.join(output_path, 'models', out_model_fname)
    dqn_agent.save_to_file(final_save_name)

    # remember to close all the files
    if record_input_output:
        for agentID,f in files.items():
            f.close()
    return rewards

if __name__ == "__main__":
    app()
