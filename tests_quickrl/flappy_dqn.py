import sys
sys.path.append('../')
sys.path.append('./')

import quickrl
from quickrl import Agent
from quickrl.interactions import flappy_reset, flappy_step
from quickrl.value_functions import SparseLinear, TorchNet
from quickrl.policies import MaxValue
import quickrl.learning_algorithms as algos
from quickrl.memories import ListOfTuples, PriorityQueue
from quickrl.logger import Logger

from flappy_bird_game import game
import torch
from torch import nn


env = game()

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8,4),
    nn.ReLU(),
    nn.Linear(4, 2)
)


# agent parts
vf = TorchNet('q', net)
policy = MaxValue(torch.LongTensor([0,1]), vf)
#memory = PriorityQueue(threshold=0.1)
memory = ListOfTuples(int(1e+6))

vf.loss_func = nn.MSELoss(reduction='none')
vf.optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.6)

agent = Agent(env,
              flappy_reset,
              flappy_step,
              gamma=0.95,
              action_value_function=vf,
              policy=policy,
              memory=memory,
              plot_frequency=10
             )

def eps_gen (episode, episodes):
    return max(0.02, 1. - episode / (episodes * 0.4))


#algos.prioritized_memory_sample(agent, 5, 300, 500, eps_gen)
algos.uniform_memory_sample(
    agent=agent,
    batch_size=32, 
    update_time=300, 
    episodes=700, 
    epsilon_schedule=eps_gen,
    frames_to_skip=10
)

print('average performace =', torch.tensor(agent.logger.total_rewards[-30:]).mean())
print('best =', max(agent.logger.total_rewards))
env.close()
