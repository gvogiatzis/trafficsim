if __name__ != "__main__":
    print("leaving")
    exit(0)

from utils.arguments import get_args
import sys
import os.path

if os.path.basename(sys.argv[0]) != "trafficrl.py":
    print("running from jupyter")
    sys.argv=["trafficrl.py", "train", "--net", "sumo_data/TwoJunction.net.xml"]

args, parser = get_args()
if len(sys.argv) == 1:
    parser.print_usage()
    exit(0)
print(args.net)


from collections import deque
from random import random,sample
from sumoenv import TrafficControlEnv
from models import MLPnet, loadModel, saveModel
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

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


# env = TrafficControlEnv(net_fname="sumo_data/ThreeLaneJunction.net.xml",vehicle_spawn_rate=0.1, state_wrapper=lambda x:torch.tensor(x,dtype=torch.float),episode_length=100)
env = TrafficControlEnv(net_fname=args.net, vehicle_spawn_rate=args.spawn_rate, state_wrapper=lambda x:torch.tensor(x,dtype=torch.float),episode_length=args.episode_length,use_gui=args.use_gui,sumo_timestep=args.sumo_timestep, seed=args.seed)

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

rewards=[]
num_episodes=args.num_episodes
num_actions = env.get_num_actions()
state_size = env.get_obs_dim()

if args.cmd=="train":
    gamma = args.gamma
    epsilon = args.random_eps
    batchsize=args.batch_size
    replaybuffer = deque(maxlen=args.replay_buffer_size)

# qnet = MLPnet(state_size,512,512,512,num_actions).to(device)
# loadModel(args.)
if args.input is not None:
    qnet = loadModel(args.input)
else:
    qnet = MLPnet(state_size, *args.network_layers, num_actions).to(device)

# print(qnet)
# exit(0)
# qnet = MLPnet(state_size,512,512,512,num_actions).to(device)
# qnet.load_state_dict(torch.load("TwoJunction.dqn.pt"))

# optim = torch.optim.Adam(qnet.parameters(), lr= 0.001)
if args.cmd=="train":
    optim = torch.optim.RMSprop(qnet.parameters(), lr= args.lr)
    loss = nn.MSELoss()

for e in range(num_episodes):
    done = False
    S = env.reset()
    tot_reward=0

    # epsilon = max(0.1, epsilon*0.99)
    while not done:
        # Epsilon-greedy strategy
        if args.cmd == "train":
            A = env.sample_action() if random()<epsilon else qnet(S).argmax().item()
        else:
            A = qnet(S).argmax().item()

        # Executing action, receiving reward and new state
        S_new, R, done = env.step(A)

        if args.cmd == "train":
            # adding the latest experience onto the replay buffer
            replaybuffer.append((S,A,R,S_new,done))

            S = S_new
            if len(replaybuffer)>=batchsize:
                batch = sample(replaybuffer, batchsize)
                batch = zip(*batch)    
                updateQNet_batch(qnet, batch, optim, loss)
        else:
            S=S_new


        tot_reward += R

    rewards.append(tot_reward)
    # print(f"\r{e+1}/{num_episodes} tot_reward={tot_reward}",end='')
    print(f"{e+1}/{num_episodes} tot_reward={tot_reward}",end='\n')

    if args.cmd=="train" and args.save_intermediate:
        saveModel(qnet, args.output)

env.close()

if args.cmd == "train":
    saveModel(qnet, args.output)

if args.plot_reward:
    print('plotting reward')
    # plt.plot(np.convolve(rewards,[0.01]*100,'valid'),'-')
    plt.plot(rewards,'-')
    plt.show()