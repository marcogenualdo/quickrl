# quickrl
## A reinforcement learning framework built with modularity as the core concept.

**What's inside:**
- the framework's files are stored in the `./quickrl` directory
- tests and examples of the framework put to use can be found in the `./tests_quickrl` directory.

**How to use:**

This framework is built with ease of use, generality and modularity in mind. At the moment this is achieved at the expense of performance, since all tasks are executed sequencially (no concurrent processes), and by the processor(s). This framework, although it is built using PyTorch, does not make any use of the graphics card.

Ideally you won't have to implement any function to make the agent learn in a new environment. You'll just have to *build* the agent giving it the parts already available in the framework. Some of these parts are typical of any RL agent, while others are characteristic of this framework. Here I present a scheme of the typical `quickrl` agent, folowed by a list in which I'll go into more detail.

![quickrl_agent_scheme](/images/agent_scheme.jpg)

- one or more *(action)-value functions*. This is a Python class able to assign a value to a state (or state-action pair) through a `__call__` function and learn from an `update` function.
- one or more *policies*. This is a Python class able to decide which action is best to take, often (but not necessarily) according to the value computed by the action-value function above.
- a *memory*. This can be either a list of tuples or a more complicated data structure that stores the past events, so that the agent can recall them and learn from the past rewards.
- a *learning algorithm*. This is a Python function that controls the agent behaviour while it's actively learning and playing.
- two *interaction functions: `step` and `reset`*. These come into play every time the agent has to take a step in the environment, or reset the game. They allow a standard code of communication between an arbitrary game environment and the agent parts.
- a *logger*. A Python class that records important information the user may want to keep track of, such as the total reward over each episode or the total number of game losses.

Each one of these parts can be implemented by the user, if none of the already implemented ones seem fit for the task.
The agent interaction scheme is based upon the OpenAI Gym environments, since thoe are the ones I used to benchmark the performances of the learning algorithms. The algorithms I implemented so far are classics of the RL literature, namely:
- Deep Q-Network, associated with random or prioritized search through memory.
- n-SARSA with arbitrary n, also associated with random or prioritized recall.
- Actor critic (typically using a neural network for the policy and a linear classifier for the value function).
- Reinforce
