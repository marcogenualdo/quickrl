from collections import deque
from copy import copy
from itertools import cycle
import torch
from torch import nn

from .policies import MaxValue


def n_td_step (agent, paths, n=1):
    """
    Updates agent.V (value function) based on transitions contained in 'paths'.
    'paths' is a list of transitions of length n+1.

    Taking these paths, the function builds a vector of all the initial states (old_states) and last states (new_states)
    and a matrix of rewards where rewards(i,j) is the reward given at the j-th step of transition i.
    the vector of total predicted rewards is then

        target = rewards * d + gamma ^ n * value(new_states)

    where d = (1, gamma, gamma ^ 2, ..., gamma ^ (n-1)) is the discount vector.
    The values of new states are masked, i.e. multiplied by a 0-1 vector 'mask' which is 0 if the episode finishes before n steps,
    that is len(path(i)) < n.
    """

    if not paths: return
    if not isinstance(paths[0], list): paths = [paths]
   
    # transposing memory
    discount = torch.tensor([agent.gamma ** k for k in range(n)])
    old_states, new_states = [], []
    rewards, mask = [], []

    for path in paths:
        old_states.append(path[0].state)
        new_states.append(path[-1].state)

        l = 1 if len(path) == 1 else len(path) - 1
        mask.append(1. if len(path) == n+1 else 0.)

        rewards.append(torch.cat(
            [event.reward for event in path[:l]] 
            + [torch.zeros(n - l)]
        ))
    
    old_states = torch.stack(old_states)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)

    # biulding target = R * d + gamma^n * masked(V)
    with torch.no_grad():
        mask = agent.gamma ** n * torch.tensor(mask)
        targets = torch.mv(rewards, discount) + mask * agent.target_V(new_states)

    # updating value function
    loss = agent.V.update(old_states, targets)
    return loss


def sarsa_lambda_reset (agent):
    for w in agent.Q.parameters():
        if w.grad is not None:
            w.grad.zero_()


def sarsa_lambda_step (agent, state, action, lb = 0.8):
    last_event = agent.memory.get_last()
    if last_event.step < 1: return

    old_state, old_action = last_event.state, last_event.action
    reward = last_event.reward

    prediction = agent.Q(last_event.state, last_event.action)
    with torch.no_grad():
        delta = reward + agent.gamma * agent.Q(state, action) - prediction
    agent.Q.loss_func(delta, prediction).backward()
    agent.Q.optimizer.step()

    for w in agent.Q.parameters():
        w.grad *= agent.gamma * lb

    
def sarsa_lambda (agent, lb = 0.8, episodes = 1000, epsilon_schedule=None):
    assert agent.Q is not None, "sarsa requires the agent to have a Q-function"
    assert isinstance(agent.policy, MaxValue), "sarsa requires the agent to have a 'MaxValue' type policy"
    
    agent.Q.loss_func = lambda delta, x: - delta * x

    for episode in range(episodes):
        eps = epsilon_schedule(episode, episodes)
        state = agent.reset()
        action = agent.epsilon_greedy_action(eps)

        done = False 
        while not done:
            state, reward, done, info = agent.step(action)
            action = agent.epsilon_greedy_action(eps)

            last_event = agent.memory.get_last()
            prediction = agent.Q(last_event.state, last_event.action)
            with torch.no_grad():
                delta = reward + agent.gamma * agent.Q(state, action) - prediction
            agent.Q.loss_func(delta, prediction).backward()
            agent.Q.optimizer.step()

            for w in agent.Q.parameters():
                w.grad *= agent.gamma * lb

        for w in agent.Q.parameters():
            w.grad.zero_()


def n_sarsa_step (agent, paths, n=1):
    """
    Updates agent.Q (action-value function) based on transitions contained in 'paths'.
    'paths' is a list of transitions of length n+1.

    Taking these paths, the function builds a vector of all the initial states, actions (old_states, old_actions), last states, actions, (new_states, new_actions) and a matrix of rewards where rewards(i,j) is the reward given at the j-th step of transition i.
    The vector of total predicted rewards is then

        target = rewards * d + gamma ^ n * value(new_states, new_actions)

    where d = (1, gamma, gamma ^ 2, ..., gamma ^ (n-1)) is the discount vector.
    The values of new states are masked, i.e. multiplied by a 0-1 vector 'mask' which is 0 if the episode finishes before n steps,
    that is len(path(i)) < n.
    """

    if not paths: return torch.empty(0)
    if not isinstance(paths[0], list): paths = [paths]
   
    # transposing paths
    discount = torch.tensor([agent.gamma ** k for k in range(n)])
    old_states, old_actions, new_states, new_actions = [], [], [], []
    rewards, mask = [], []

    for path in paths:
        old_states.append(path[0].state)
        old_actions.append(path[0].action)
        new_states.append(path[-1].state)
        new_actions.append(path[-1].action)

        l = 1 if len(path) == 1 else len(path) - 1
        mask.append(1. if len(path) == n+1 else 0.)

        rewards.append(torch.cat(
            [event.reward for event in path[:l]] 
            + [torch.zeros(n - l)]
        ))
   
    old_states = torch.stack(old_states).squeeze()
    old_actions = torch.cat(old_actions).squeeze()
    new_states = torch.stack(new_states).squeeze()
    new_actions = torch.cat(new_actions).squeeze()
    rewards = torch.stack(rewards)

    # building target vector
    with torch.no_grad():
        mask = agent.gamma ** n * torch.tensor(mask)
        targets = torch.mv(rewards, discount) + mask * agent.target_Q(new_states, new_actions)

    # updating value function
    losses = agent.Q.update(old_states, old_actions, targets)
    return losses


def n_sarsa (agent, n = 1, episodes = 1000, epsilon_schedule=None, frames_to_skip=0):
    assert agent.Q is not None, "n-sarsa requires the agent to have a Q-function"
    assert isinstance(agent.policy, MaxValue), "n-sarsa requires the agent to have a 'MaxValue' type policy"

    path = deque()

    for episode in range(episodes):
        eps = epsilon_schedule(episode, episodes)
        path.clear()

        state = agent.reset()
        action = agent.epsilon_greedy_action(eps) 
        path.append((state, action, torch.zeros(1)))
        
        done = False 
        while not done:
            state, reward, done, info = agent.step(action, frames_to_skip)
            action = agent.epsilon_greedy_action(eps)

            with torch.no_grad():
                rewards = [r[2] for r in path] + [agent.Q(state, action)]
                gain = torch.tensor([r * agent.gamma ** k for k,r in enumerate(rewards)]).sum()

            if len(path) > n:
                (st_old, act_old, _) = path.popleft()
                agent.Q.update(st_old, act_old, gain)
            path.append((state, action, reward))


def q_step (agent, paths, n=1):
    """
    Updates agent.Q (action-value function) based on transitions contained in 'paths'.
    'paths' is a list of transitions of length n+1.

    Taking these paths, the function builds a vector of all the initial states, actions (old_states, old_actions), last states, actions, (new_states, new_actions) and a matrix of rewards where rewards(i,j) is the reward given at the j-th step of transition i.
    The vector of total predicted rewards is then

        target = rewards * d + gamma ^ n * qvalue(new_states)

    where d = (1, gamma, gamma ^ 2, ..., gamma ^ (n-1)) is the discount vector.
    The values of new states are masked, i.e. multiplied by a 0-1 vector 'mask' which is 0 if the episode finishes before n steps,
    that is len(path(i)) < n, and 1 otherwise.
    """
    
    if not paths: return []
    if not isinstance(paths[0], list): paths = [paths]
   
    # transposing paths
    discount = torch.tensor([agent.gamma ** k for k in range(n)])
    old_states, old_actions, new_states = [], [], []
    rewards, mask = [], []

    for path in paths:
        old_states.append(path[0].state)
        old_actions.append(path[0].action)
        new_states.append(path[-1].state)

        l = 1 if len(path) == 1 else len(path) - 1
        mask.append(1. if len(path) == n+1 else 0.)

        rewards.append(torch.cat(
            [event.reward for event in path[:l]] 
            + [torch.zeros(n - l)]
        ))
    
    old_states = torch.stack(old_states)
    old_actions = torch.cat(old_actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)

    # building target vector
    with torch.no_grad():
        mask = agent.gamma ** n * torch.tensor(mask)
        targets = torch.mv(rewards, discount) + mask * agent.target_Q(new_states).max(1)[0]

    # updating value function
    losses = agent.Q.update(old_states, old_actions, targets)
    return losses


def actor_critic (agent, episodes, epsilon_schedule, frames_to_skip=0):
    assert agent.V is not None, "actor-critic requires the agent to have a V-function"
    assert agent.policy is not None, "actor-critic requires the agent to have a policy"
    
    agent.policy.loss_func = lambda delta, x: - delta * torch.log(x)

    for episode in range(episodes):
        eps = epsilon_schedule(episode, episodes)
        old_state = agent.reset()

        discount = 1.
        done = False
        while not done:
            # interaction with the environemnt
            action = agent.epsilon_greedy_action(eps)
            state, reward, done, info = agent.step(action, frames_to_skip)

            # updating value weights
            target = reward + agent.gamma * agent.V(state)
            agent.V.update(old_state, target)

            # updating policy weights
            with torch.no_grad():
                delta = discount * (target - agent.V(old_state))
            agent.policy.update(old_state, action, delta)
            old_state = state
            discount *= agent.gamma


def uniform_memory_sample (
    agent, 
    batch_size, 
    update_time,
    episodes, 
    epsilon_schedule,
    transition_length=2,
    frames_to_skip=0
):
    assert agent.Q is not None, "uniform_memory_sample requires the agent to have a Q-function"
    assert isinstance(agent.policy, MaxValue), "uniform_memory_sample requires the agent to have a 'MaxValue' type policy"

    update_cycle = cycle(range(update_time)) 
    for episode in range(episodes):
        eps = epsilon_schedule(episode, episodes)
        state = agent.reset()

        done = False
        while not done: 
            if not next(update_cycle): agent.target_Q = copy(agent.Q)

            action = agent.epsilon_greedy_action(eps)
            state, reward, done, info = agent.step(action, frames_to_skip)

            paths = agent.memory.sample_transitions(batch_size, transition_length)
            #n_sarsa_step(agent, paths, transition_length - 1)
            q_step(agent, paths, transition_length - 1)


def prioritized_memory_sample (
    agent, 
    batch_size, 
    update_time,
    episodes, 
    epsilon_schedule,
    transition_length=2,
    frames_to_skip=0
):
    assert agent.Q.loss_func.reduction == 'none', "prioritized_memory_sample requires agent.Q.loss_func to have 'none' reduction"

    update_cycle = cycle(range(update_time))
    for episode in range(episodes):
        eps = epsilon_schedule(episode, episodes)
        state = agent.reset()
        action = agent.epsilon_greedy_action(eps)

        done = False
        while not done: 
            if not next(update_cycle): agent.target_Q = copy(agent.Q)

            # real experience
            state, reward, done, info = agent.step(action, frames_to_skip)
            action = agent.epsilon_greedy_action(eps)

            last_event = agent.memory.event(0, 0, state, action, None) 
            last_path = [agent.memory.data[-1], last_event]
            loss = n_sarsa_step(agent, last_path)
            agent.memory.push(loss.item())

            # replaying past experiences
            paths, indexes = agent.memory.sample_transitions(batch_size, transition_length)

            #losses = n_sarsa_step(agent, paths, transition_length - 1)
            losses = q_step(agent, paths, transition_length - 1)
            
            for loss, index in zip(losses, indexes):
                agent.memory.update_priority(index, loss.item())
