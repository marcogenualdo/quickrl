from collections import namedtuple
from heapq import heappop, heappush
import torch
import random
import pandas as pd

from .sumtree import SumTree


class ListOfTuples:
    def __init__ (self, size=None):
        self.size = size
   
        self.fields = ['state', 'action', 'reward']
        self.event = namedtuple('event', (
            'episode',
            'step',
            *self.fields
        ))

        self.data = []
        self.current_episode = 0
        self.current_step = 0
        self.current_state = None
        
        
    def remember (self, state, action, reward, done):
        self.data.append(self.event(
            self.current_episode,
            self.current_step,
            self.current_state,
            action,
            reward
        ))
        self.current_state = state

        if self.size is not None and len(self.data) > self.size: 
            del self.data[0]

        self.current_step += 1
        if done:
            self.remember(self.current_state, torch.LongTensor([0]), torch.zeros(1), False)
            self.current_episode += 1
            self.current_step = 0


    def __len__ (self):
       return len(self.data)
    
    
    def get_last (self, n=1):
        if len(self.data) >= n:
            return self.data[-n:]

        return None


    def get_by_position (self, addresses):
        returns = []
        for n in addresses:
            if n <= len(self.data):
                returns.append(self.data[n])
        return returns


    def sample (self, n):
        indices = torch.randint(0, len(self.data), n)
        return self.get_by_position(indices)


    def sample_transitions (self, n, length=2):
        if len(self.data) < length * n: return []

        paths = []
        for k in range(n):
            start_index = random.randint(0, len(self.data) - length)
            path = [self.data[start_index]]
            index = start_index + 1
            while (
                index - start_index < length 
                and self.data[index-1].episode == self.data[index].episode
            ):
                path.append(self.data[index])
                index += 1
            paths.append(path)
        return paths


    def __str__ (self, start=0):
        memory_string = "Agent memory \n Episode \t Step \t State \t Action \t Reward \n"
        for memory in self.data[start:]:
            memory_string = memory_string + \
                f"{memory.episode} \t {memory.step} \
                \t {memory.state} \
                \t {memory.action} \
                \t {memory.reward}\n\n"

        return memory_string


class PrioritizedListOfTuples (ListOfTuples):
    def __init__ (self, size, distribution_exponent, minimum_selection_chance):
        super().__init__(size)
        self.tree = SumTree(size)
        self.pexp = distribution_exponent
        self.epsilon = minimum_selection_chance

    
    def push (self, loss):
        probability = (loss + self.epsilon) ** self.pexp
        self.tree.push(probability)


    def update_priority (self, index, loss):
        probability = (loss + self.epsilon) ** self.pexp
        self.tree.update(index, probability)


    def sample_transitions (self, n, length=2):
        if len(self.data) < length * n: return [], []

        indexes = [self.tree.get(random.random() * self.tree.total()) for k in range(n)]

        paths = []
        for start_index in indexes:
            path = [self.data[start_index]]
            index = start_index + 1
            while (
                index - start_index < length 
                and index < len(self.data) 
                and self.data[index-1].episode == self.data[index].episode
            ):
                path.append(self.data[index])
                index += 1
            paths.append(path)

        return paths, indexes


###### OBSOLETE MEMORIES ######

class PriorityQueue (ListOfTuples):
    """
    A unit of the self.queue heap is composed by a tuple = (-priority, old_event, new_event)
    + the negative sign on the priority transforms the python min-heap into a max-heap
    + old_event, new_event are namedtuples as defined in the ListOfTuples class
    """

    def __init__ (self, size=None, threshold=0.):
        super().__init__(size)
        self.queue = []
        self.threshold = threshold


    def pop (self, n=1):
        if n > len(self.queue): return [], []

        returns = [heappop(self.queue) for k in range(n)]
        old_events = [r[1] for r in returns]
        new_events = [r[2] for r in returns]
        return old_events, new_events


    def push (self, old_events, new_events, losses):
        for old, new, loss in zip(old_events, new_events, losses):
            if loss > self.threshold:
                heappush(self.queue, (-loss.item(), old, new))
            


class Pandas:
    def __init__ (self, size=None):
        self.size = size

        self.fields = ['state', 'action', 'reward']
        data_index = pd.MultiIndex(levels=[[],[]], labels=[[],[]], names= ['current_episode', 'current_step'])
        self.data = pd.DataFrame(index=data_index, columns=self.fields)
        self.last_index = (0,-1)

        self.current_state = None
    
    
    def remember (self, state, action, reward, done):
        event = [self.current_state, action, reward]
        self.current_state = state

        this_index = (self.last_index[0], self.last_index[1] + 1)
        new_index = pd.MultiIndex.from_tuples([this_index])
        self.last_index = this_index 
        
        self.data = self.data.append(
            pd.DataFrame([event], index=new_index, columns=self.fields))

        if done:
            self.remember(self.current_state, 0, 0., False)
            self.last_index = (this_index[0] + 1, -1)

        #if self.data_size is not None and len(self.data.index) > self.capacity:
        #    self.data.drop(self.data.index[0], inplace=True)


    def pop (self, n=1, requested_fields=None):
        if requested_fields is not None:
            return self.data[-n:][requested_fields] 
        return self.data[-n:]
