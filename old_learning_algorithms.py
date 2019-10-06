from collections import deque
from copy import copy
import torch
from torch import nn

from .policies import MaxValue


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
    if agent.memory.data[-1].step < n: return None

    path = agent.memory.get_last(n)
    old_state, old_action = path[0].state, path[0].action

    with torch.no_grad():
        rewards = [e.reward for e in path] + [agent.Q(state, action)]
        gain = torch.tensor([r * agent.gamma ** k for k,r in enumerate(rewards)]).sum()

    loss = agent.Q.update(old_state, old_action, gain)
    return loss


def n_sarsa (agent, n = 1, episodes = 1000, epsilon_schedule=None):
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
            state, reward, done, info = agent.step(action)
            action = agent.epsilon_greedy_action(eps)

            with torch.no_grad():
                rewards = [r[2] for r in path] + [agent.Q(state, action)]
                gain = torch.tensor([r * agent.gamma ** k for k,r in enumerate(rewards)]).sum()

            if len(path) > n:
                (st_old, act_old, _) = path.popleft()
                agent.Q.update(st_old, act_old, gain)
            path.append((state, action, reward))


def actor_critic (agent, episodes, epsilon_schedule):
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
            state, reward, done, info = agent.step(action)

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
    epsilon_schedule
):
    assert agent.Q is not None, "uniform_ememory_sample requires the agent to have a Q-function"
    assert isinstance(agent.policy, MaxValue), "uniform_memory_sample requires the agent to have a 'MaxValue' type policy"

    t = 0
    for episode in range(episodes):
        eps = epsilon_schedule(episode, episodes)
        state = agent.reset()

        done = False
        while not done: 
            if not t % update_time: agent.target_Q = copy(agent.Q)
            t += 1

            action = agent.epsilon_greedy_action(eps)
            state, reward, done, info = agent.step(action)

            old, new = agent.memory.sample_transitions(batch_size)
            if old and new:
                with torch.no_grad():
                    old_states = torch.stack([e.state for e in old])
                    old_actions = torch.LongTensor([e.action for e in old])
                    new_states = torch.stack([e.state for e in new])
                    new_actions = torch.LongTensor([e.action for e in new])

                    mask = torch.tensor([1. if o.episode == n.episode else 0. for o,n in zip(old, new) ])
                    values = agent.target_Q(new_states, new_actions)
                    targets = torch.cat([e.reward for e in old]) + agent.gamma * values * mask

                agent.Q.update(old_states, old_actions, targets)


def prioritized_memory_sample (
    agent, 
    batch_size, 
    update_time,
    episodes, 
    epsilon_schedule
):
    assert agent.Q.loss_func.reduction == 'none', "prioritized_memory_sample requires agent.Q.loss_func to have 'none' reduction"

    t = 0
    for episode in range(episodes):
        eps = epsilon_schedule(episode, episodes)
        state = agent.reset()
        action = agent.epsilon_greedy_action(eps)

        done = False
        while not done: 
            if not t % update_time: agent.target_Q = copy(agent.Q)
            t += 1

            state, reward, done, info = agent.step(action)
            action = agent.epsilon_greedy_action(eps)

            loss = n_sarsa_step(agent, state, action)
            if loss is not None:
                agent.memory.push(
                    [agent.memory.data[-1]],
                    [agent.memory.event(
                        agent.memory.current_episode,
                        agent.memory.current_step,
                        state,
                        action,
                        None)],
                    [loss]
                )

            #print(agent.memory.queue)
            old_events, new_events = agent.memory.pop(batch_size)
            if old_events and new_events:
                with torch.no_grad():
                    old_states = torch.stack([e.state for e in old_events])
                    old_actions = torch.LongTensor([e.action for e in old_events])
                    new_states = torch.stack([e.state for e in new_events])
                    new_actions = torch.LongTensor([e.action for e in new_events])

                    mask = torch.tensor([1. if o.episode == n.episode else 0. for o,n in zip(old_events, new_events)])
                    values = agent.target_Q(new_states, new_actions)
                    targets = torch.cat([e.reward for e in old_events]) + agent.gamma * mask * values
                
                losses = agent.Q.update(old_states, old_actions, targets)
                agent.memory.push(old_events, new_events, losses)
