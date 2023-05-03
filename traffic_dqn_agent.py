from collections import deque
from random import random,sample
from sumoenv import TrafficControlEnv
from models import MLPnet
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


env = TrafficControlEnv(net_fname="sumo_data/ThreeLaneJunction.net.xml",vehicle_spawn_rate=0.1, state_wrapper=lambda x:torch.tensor(x,dtype=torch.float),episode_length=100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon = 0.1
rewards=[]
gamma = 0.99
num_episodes=50
num_actions = env.get_num_actions()
state_size = env.get_obs_dim()
batchsize=64

# replaybuffer = deque(maxlen=100000)
replaybuffer = deque(maxlen=5000)

qnet = MLPnet(state_size,512,512,512,num_actions).to(device)
# optim = torch.optim.Adam(qnet.parameters(), lr= 0.001)
optim = torch.optim.RMSprop(qnet.parameters(), lr= 0.0001)

loss = nn.MSELoss()

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
        if len(replaybuffer)>=batchsize:
            batch = sample(replaybuffer, batchsize)
            batch = zip(*batch)    
            updateQNet_batch(qnet, batch, optim, loss)

        tot_reward += R

    rewards.append(tot_reward)
    # print(f"\r{e+1}/{num_episodes} tot_reward={tot_reward}",end='')
    print(f"{e+1}/{num_episodes} tot_reward={tot_reward}",end='\n')



env.close()
# plt.plot(np.convolve(rewards,[0.01]*100,'valid'),'-')
plt.plot(rewards,'-')
