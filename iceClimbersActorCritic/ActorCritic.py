#adaptado de https://www.programmersought.com/article/1642476675/
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display
import numpy as np
import gym
import time
import retro
from stable_baselines.common.vec_env import DummyVecEnv

Actor_Lr = 0.001
Critic_lr = 0.002
GAMMA = 0.99
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01
env = retro.make(game='IceClimber-Nes')
env = DummyVecEnv([lambda: env])
RENDER = False

states_len = env.observation_space.shape[0]
antions_len = env.action_space.shape[0]
print(env.action_space)
a_bound = env.action_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('Entrenamiento')
    plt.xlabel('Episodio')
    plt.ylabel('score')
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episodio", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython:
        display.clear_output(wait = True)


#caula el promedio una vez se alcancen las 100 iteraciones, antes hace plot de 0
def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    #se verifica que la longitud de los valores sea lo suficientemente grande como para calcular el promedio
    #para el periodo dado
    if len(values)  >= period:
        moving_avg = values.unfold(dimension = 0, size = period, step = 1).mean(dim=1).flatten(start_dim=0)
        #.cat concatena la secuencia dada de tensores segun la dimensión especificada
        #se concantena con tensores poblados con 0, esto porque antes de alcanzar el periodo deseado
        #no tiene sentido calcular la media para ese periodo
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

#clase que modela el actor
class Actor(nn.Module):
    def __init__(self,states_len,antions_len):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(states_len,30)
        self.out = nn.Linear(30,antions_len)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return torch.tanh(x)*2

    def getStateFromTheDic(self, key):
        return self.state_dict()[key]

#clase que modela el crítico
class Critic(nn.Module):
    def __init__(self,states_len,antions_len):
        super(Critic,self).__init__()
        self.fcs = nn.Linear(states_len,30)
        self.fca = nn.Linear(antions_len,30)
        self.out = nn.Linear(30,1)

    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value

    def getStateFromTheDic(self, key):
        return self.state_dict()[key]


class DDPG(object):
    def __init__(self, antions_len, states_len, a_bound,):
        self.antions_len, self.states_len, self.a_bound = antions_len, states_len, a_bound,
        #se inicializa la memoria como un arreglo de ceros
        self.memory = np.zeros((MEMORY_CAPACITY, states_len * 2 + antions_len + 1), dtype=np.float32)
        self.pointer = 0
        self.Actor_target = Actor(states_len,antions_len)
        self.Actor_policy = Actor(states_len,antions_len)
        self.Critic_policy = Critic(states_len,antions_len)
        self.Critic_target = Critic(states_len,antions_len)
        self.ctrain = torch.optim.Adam(self.Critic_policy.parameters(),lr=Critic_lr)
        self.atrain = torch.optim.Adam(self.Actor_target.parameters(),lr=Actor_Lr)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_target(s)[0].detach()

    def learn(self):

        for x in self.Actor_policy.state_dict().keys():
            self.Actor_policy.getStateFromTheDic(x).data.mul_((1 - TAU))
            self.Actor_policy.getStateFromTheDic(x).data.add_(TAU*self.Actor_target.getStateFromTheDic(x).data)
        for x in self.Critic_target.state_dict().keys():
            self.Critic_target.getStateFromTheDic(x).data.mul_((1-TAU))
            self.Critic_target.getStateFromTheDic(x).data.add_(TAU*self.Critic_policy.getStateFromTheDic(x).data)

        # soft target replacement

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.states_len])
        ba = torch.FloatTensor(bt[:, self.states_len: self.states_len + self.antions_len])
        br = torch.FloatTensor(bt[:, -self.states_len - 1: -self.states_len])
        bs_ = torch.FloatTensor(bt[:, -self.states_len:])

        a = self.Actor_target(bs)
        q = self.Critic_policy(bs,a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_policy(bs_)
        q_ = self.Critic_target(bs_,a_)
        q_target = br+GAMMA*q_
        q_v = self.Critic_policy(bs,ba)
        td_error = self.loss_td(q_target,q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

#entrenamiento y gráficado

episode_durations = []
ddpg = DDPG(antions_len, states_len, a_bound)
steps = 500
var = 4  # control exploration
t1 = time.time()
for i in range(500):
    s = env.reset()
    ep_reward = 0
    for j in range(steps):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()

        s = s_
        ep_reward += r

        if j == steps-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            episode_durations.append(int(ep_reward))
            plot(episode_durations, 100)
            if ep_reward > -200 and i > 50 :RENDER = True
            break
print('Running time: ', time.time() - t1)