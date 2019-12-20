from .logger import Logger
from random import random, randint

import torch


class Agent:
    def __init__ (self,
                  env,
                  reset_function,
                  step_function,
                  gamma=1.,
                  
                  value_function=None,
                  action_value_function=None,
                  policy=None,
                  
                  memory=None,
                  plot_frequency=50
                 ):

        # general setup
        self.env = env
        self.gamma = gamma 

        self.reset_func = reset_function 
        self.step_func = step_function 

        self.V = value_function
        self.Q = action_value_function
        self.policy = policy
        self.memory = memory

        self.logger = Logger(self, plot_frequency)


    def reset (self, *args, **kwargs):
        return self.reset_func(self, *args, **kwargs)
    
    
    def step (self, action, *args, **kwargs):
        return self.step_func(self, action, *args, **kwargs)


    def epsilon_greedy_action (self, epsilon):
        if random() < epsilon:
            try:
                return torch.LongTensor([self.env.action_space.sample()])
            except:
                return torch.LongTensor([randint(0,1)])
        return self.policy.best_action(self.memory.current_state)
