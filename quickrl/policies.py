from random import random, choice
import torch
from torch import nn

from .value_functions import SparseLinear


class Policy:
    def __init__(
        self, 
        processing=None
    ):

        self.processing_func = processing


    def process (function):
        def wrapper (self, state, action=None, *args, **kwargs):
            processed = kwargs.pop('processed', False)
            if self.processing_func is None or processed:
                if action is None:
                    return function(self, state, *args, **kwargs)
                else:
                    return function(self, state, action, *args, **kwargs)

            if action is None:
                pair = self.processing_func(state)
            else:
                pair = self.processing_func(state, action)
            return function(self, pair, *args, **kwargs)

        return wrapper


class MaxValue (Policy):
    def __init__(self, actions, action_value_function, processing=None):
        super().__init__(processing)
        if isinstance(actions, list):
            self.actions = torch.LongTensor(actions)
        else:
            self.actions = actions
        self.Q = action_value_function


    @Policy.process
    def best_action (self, state):
        if isinstance(self.Q, SparseLinear):
            return max([a for a in self.actions],
                       key=lambda x: self.Q(state, x))

        return torch.argmax(self.Q(state)).unsqueeze(0)


    @Policy.process
    def update (self, *args, **kwargs):
        pass


class TorchNet (Policy):
    def __init__(self,
                 network, 
                 processing=None
                ):

        super().__init__(processing)
        self.function = network

        self.loss_func = None
        self.optimizer = None
    
    
    def parameters (self):
        return self.function.parameters()


    @Policy.process
    def __call__ (self, state, action=None):
        if action is None:
            return self.function(state)

        axis = 0 if not action.shape or action.shape[0] <= 1 else 1 
        if axis: action = action.view(-1,1)
        return self.function(state).gather(axis, action).view(-1)


    @Policy.process
    def best_action (self, state):
        return torch.argmax(self.function(state))
    
    
    @Policy.process
    def update (self, state, action, target):
        prediction = self(state, action)
        losses = self.loss_func(target, prediction)
        
        losses.mean().backward()
        self.optimizer.step()
        self.function.zero_grad()
        return losses
