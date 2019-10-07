import torch
from torchvision import transforms as T


def gym_reset (agent, *args, **kwargs):
    state = agent.env.reset()
    state = torch.as_tensor(state, dtype=torch.float32)
    
    agent.memory.current_state = state
    agent.logger.reset()
    return state


def gym_step (agent, action, frames_to_skip=0):
    for k in range(frames_to_skip + 1):
        state, reward, done, info = agent.env.step(action.item())
    state = torch.as_tensor(state, dtype=torch.float32)
    reward = torch.tensor([reward], dtype=torch.float32)

    if agent.memory.data and agent.memory.data[-1].episode >= 720:
        agent.env.render()

    agent.memory.remember(state, action, reward, done)
    agent.logger.step()
    return state, reward, done, info


def gym_screen_reset (agent):
    agent.env.reset()
    state = agent.env.render(mode='rgb_array')

    # processing states (if any processing function is present)
    try: state = agent.process_state(state)
    except AttributeError: state = torch.as_tensor(state, dtype=torch.float32)
    
    agent.memory.current_state = state
    agent.logger.reset()
    return state


def gym_screen_step (agent, action, frames_to_skip=0):
    for k in range(frames_to_skip + 1):
        state, reward, done, info = agent.env.step(action.item())
        state = agent.env.render(mode='rgb_array')

    # processing states (if any processing function is present)
    try: state = agent.process_state(state)
    except AttributeError: state = torch.as_tensor(state, dtype=torch.float32)
    reward = torch.tensor([reward])

    agent.memory.remember(state, action, reward, done)
    agent.logger.step()
    return state, reward, done, info


def atari_reset (agent, frames_to_skip=0, state_frames=2):
    frame = agent.env.reset()[::2,::2]
    state = [torch.as_tensor(frame, dtype=torch.float32).mean(2)]
    
    # a state may be composed of multiple frames
    for n in range(state_frames - 1):
        # skipping some frames
        for k in range(frames_to_skip + 1):
            frame, r, done, info = agent.env.step(0)

        frame = frame[::2,::2]
        state.append(torch.as_tensor(frame, dtype=torch.float32).mean(2))

    # putting the frames side by side in one picture
    state = torch.stack(state) / 255
    
    agent.memory.current_state = state
    agent.logger.reset()
    return state


def atari_step (agent, action, frames_to_skip=0, state_frames=2):
    state = []
    reward = 0.

    # a state may be composed of multiple frames
    for n in range(state_frames):
        # skipping some frames
        for k in range(frames_to_skip + 1):
            frame, r, done, info = agent.env.step(action.item())
        
        frame = frame[::2,::2]
        state.append(torch.as_tensor(frame, dtype=torch.float32).mean(2))
        reward += r

    # putting the frames side by side in one picture
    state = torch.stack(state) / 255
    reward = torch.tensor([reward])

    agent.memory.remember(state, action, reward, done)
    agent.logger.step()
    return state, reward, done, info


def flappy_reset (agent, *args, **kwargs):
    state = agent.env.reset()
    state = torch.as_tensor(state, dtype=torch.float32)
    
    agent.memory.current_state = state
    agent.logger.reset()
    return state


def flappy_step (agent, action, frames_to_skip=0):
    state, reward, done, info = agent.env.step(action.item(), frames_to_skip)
    state = torch.as_tensor(state, dtype=torch.float32)
    reward = torch.tensor([reward], dtype=torch.float32)

    agent.memory.remember(state, action, reward, done)
    agent.logger.step()
    return state, reward, done, info
