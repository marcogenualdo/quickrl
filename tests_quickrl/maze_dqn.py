import sys
sys.path.append('../')

import quickrl
from quickrl import Agent
from quickrl.interactions import gym_reset, gym_step
from quickrl.value_functions import SparseLinear, TorchNet
from quickrl.policies import MaxValue
import quickrl.learning_algorithms as algos
from quickrl.memories import ListOfTuples, PriorityQueue
from quickrl.logger import Logger

from maze_env import Maze

import gym
import torch
from torch import nn


env = gym.make('CartPole-v1')

net = nn.Sequential(
    nn.Linear(*env.observation_space.shape, 24),
    nn.ReLU(),
    nn.Linear(24,24),
    nn.ReLU(),
    nn.Linear(24, env.action_space.n)
)


# agent parts
vf = TorchNet('q', net)
policy = MaxValue(torch.LongTensor([0,1]), vf)
#memory = PriorityQueue(threshold=0.1)
memory = ListOfTuples()

vf.loss_func = nn.MSELoss(reduction='none')
vf.optimizer = torch.optim.Adam(net.parameters())

agent = Agent(env,
              gym_reset,
              gym_step,
              gamma=0.95,
              action_value_function=vf,
              policy=policy,
              memory=memory,
              plot_frequency=10
             )

def eps_gen (episode, episodes):
    return max(0.01, 1. - episode / (episodes - 30))


#algos.prioritized_memory_sample(agent, 5, 300, 500, eps_gen)
algos.uniform_memory_sample(agent, 5, 300, 300, eps_gen)

print('average performace =', torch.tensor(agent.logger.total_rewards[-30:]).mean())
print('best =', max(agent.logger.total_rewards))
env.close()
