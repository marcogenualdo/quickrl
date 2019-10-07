import sys
sys.path.append('../')

import quickrl
from quickrl import Agent
from quickrl.interactions import gym_reset, gym_step
from quickrl.value_functions import SparseLinear
from quickrl.policies import MaxValue, TorchNet
import quickrl.learning_algorithms as algos
from quickrl.memories import ListOfTuples
from quickrl.logger import Logger
from quickrl.tile_coding import IHT, tiles

import gym
import torch
from torch import nn


env = gym.make('CartPole-v1')

# processing function
iht = IHT(1024)
tiler = lambda s: torch.LongTensor(tiles(iht, 8, tuple(s)))

net = nn.Sequential(
    nn.Linear(4,24),
    nn.ReLU(),
    nn.Linear(24,2),
    nn.Softmax(dim=0)
)

# agent parts
value_function = SparseLinear('v', 1024, processing=tiler)
policy = TorchNet(net)
memory = ListOfTuples()
value_function.optimizer = torch.optim.SGD([value_function.weights], lr=0.1)
policy.optimizer = torch.optim.Adam(net.parameters())
value_function.loss_func = nn.MSELoss()

agent = Agent(env,
              gym_reset,
              gym_step,
              value_function=value_function,
              policy=policy,
              memory=memory
             )


def eps_gen (episode, episodes):
    return max(0.01, 1. - episode / (episodes - 30))

algos.actor_critic(agent, episodes=300, epsilon_schedule=eps_gen)
print('average performace =', torch.tensor(agent.logger.total_rewards[-30:]).mean())
print('best =', max(agent.logger.total_rewards))
env.close()
