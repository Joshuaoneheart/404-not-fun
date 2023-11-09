from mingpt.model import GPT
import numpy as np
from collections import deque
from gridworld import GridWorld
import torch
import torch.nn as nn
import random
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space
        self.policy_index = np.zeros(self.state_space, dtype=int)

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index

    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values

class Q_Learning(DynamicProgramming):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        indexes = np.random.choice(range(len(self.buffer)), min(self.sample_batch_size, len(self.buffer)), replace = False)
        return np.array(self.buffer)[indexes]

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        loss = r + self.discount_factor * (1 - is_done) * self.q_values[int(s2)].max() - self.q_values[int(s), int(a)]
        self.q_values[int(s), int(a)] += self.lr * loss
        for j in range(self.action_space):
            if j == self.q_values[int(s)].argmax():
                self.policy[int(s), j] = self.epsilon / self.action_space + 1 - self.epsilon
            else:
                self.policy[int(s), j] = self.epsilon / self.action_space

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        is_done = False
        transition_count = 0
        reward_trace = []
        loss_trace = []
        episodic_reward = []
        episodic_loss = []
        """run = wandb.init(project = "reinforcement learning 2", config={
            "method": "Q_Learning",
            "step_reward": STEP_REWARD,
            "goal_reward":GOAL_REWARD,
            "trap_reward":TRAP_REWARD,
            "discount_factor":DISCOUNT_FACTOR,
            "learning_rate": LEARNING_RATE,
            "epsilon": EPSILON,
            "buffer_size": BUFFER_SIZE,
            "update_frequency": UPDATE_FREQUENCY,
            "sample_batch_size": SAMPLE_BATCH_SIZE
        })"""
        trajectories = []
        trajectory = [current_state + 1]
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            action = np.random.choice(np.arange(0, 4), p = self.policy[current_state])
            next_state, reward, is_done = self.grid_world.step(action)
            loss = reward + self.discount_factor * (1 - is_done) * self.q_values[next_state].max() - self.q_values[current_state, action]
            loss_trace.append(abs(loss))
            reward_trace.append(reward)
            trajectory.append(current_state + 1)
            if is_done:
                iter_episode += 1
                episodic_reward.append(np.mean(reward_trace))
                episodic_loss.append(np.mean(loss_trace))
                # run.log({"last_10_episodic_reward": np.mean(episodic_reward[-10:]), "last_10_estimation_loss": np.mean(episodic_loss[-10:])})
                reward_trace = []
                loss_trace = []
                assert (np.array(trajectory) != 0).all()
                trajectories.append(trajectory)
                trajectory = [next_state + 1]
            self.add_buffer(current_state, action, reward, next_state, is_done)
            transition_count += 1
            B = []
            if transition_count % self.update_frequency == 0:
                B = self.sample_batch()
            # for s, a, r, s_, is_done in B:
                # self.policy_eval_improve(s, a, r, s_, is_done)
            current_state = next_state
        # run.finish()
        return trajectories

STEP_REWARD       = -0.1
GOAL_REWARD       = 1.0
TRAP_REWARD       = -1.0
DISCOUNT_FACTOR   = 0.99
LEARNING_RATE     = 0.01
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500


def bold(s):
    return "\033[1m" + str(s) + "\033[0m"

def underline(s):
    return "\033[4m" + str(s) + "\033[0m"

def green(s):
    return "\033[92m" + str(s) + "\033[0m"

def red(s):
    return "\033[91m" + str(s) + "\033[0m"

def init_grid_world(maze_file: str = "maze.txt"):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world

def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world,
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    trajectories = policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
            f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()
    return trajectories

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, device):
        self.x = trajectories
        self.device = device
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = x[1:]
        y.append(0)
        while len(x) < 1000:
            x.append(0)
            y.append(0)
        if len(x) > 1000:
            x = x[:1000]
            y = y[:1000]
        assert len(x) == 1000
        return torch.LongTensor(x).to(self.device), torch.LongTensor(y).to(self.device)
device="cuda:1"
batch_size = 32
epoch_num = 0
num_workers = 0
lr = 5e-4
update_frequency = 200
action_space = 4
train_episode_num = 500
max_steps = 1000
epsilon = 0.1
method = "DQN"
run = wandb.init(project = "reinforcement learning final", config={
            "method": "Q_Learning",
            "step_reward": STEP_REWARD,
            "goal_reward":GOAL_REWARD,
            "trap_reward":TRAP_REWARD,
            "discount_factor":DISCOUNT_FACTOR,
            "learning_rate": LEARNING_RATE,
            "epsilon": EPSILON,
            "buffer_size": BUFFER_SIZE,
            "update_frequency": UPDATE_FREQUENCY,
            "sample_batch_size": SAMPLE_BATCH_SIZE
        })
grid_world = init_grid_world()
trajectories = run_Q_Learning(grid_world, 1)
train_dataset = TrajectoryDataset(trajectories, device)
model_config = GPT.get_default_config()
model_config.model_type = 'gpt-nano'
model_config.vocab_size = 23
model_config.block_size = 1000
model = GPT(model_config, action_space)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
for epoch in tqdm(range(epoch_num)):
    losses = []
    for x, y in tqdm(train_loader):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Losses: {np.mean(losses)}")

class EpisodeBuffer:
    
    def __init__(self, buffer_size=1000, batch_size=32, max_steps=1000):
        self.sample_batch_size = batch_size
        self.state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.done_buf = []
        self.ptr = 0
        self.size = buffer_size
        self.full = False

    def append(self, states, actions, rewards, dones):
        self.state_buf.append(states)
        self.action_buf.append(actions)
        self.reward_buf.append(rewards)
        self.done_buf.append(dones)
        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0
            self.full = True

    def sample(self):
        indexes = np.random.choice(range(len(self)), min(self.sample_batch_size, len(self)), replace = False)
        state_list = []
        next_state_list = []
        action_list = []
        reward_list = []
        done_list = []
        for idx in indexes:
            next_state_list.append(self.state_buf[idx][1:])
            state_list.append(self.state_buf[idx][:-1])
            action_list.append(self.action_buf[idx])
            reward_list.append(self.reward_buf[idx])
            done_list.append(self.done_buf[idx])

        return state_list, action_list, next_state_list, reward_list, done_list
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.ptr
buffer = EpisodeBuffer(1000, 1)
if method == "GPT":
    model.QMode(True)
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = 23
    model_config.block_size = 1000
    target_model = GPT(model_config, action_space).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.QMode(True)
elif method == "DQN":
    from DQN import Agent
    from Memory import ReplayMemory
    QL = Agent(device)
    mem = ReplayMemory(1000000)
total_step = 0
for episode in tqdm(range(train_episode_num)):
    current_state = grid_world.reset()
    state_list = [current_state]
    action_list = []
    reward_list = []
    done_list = []
    losses = []
    step = 0
    while step < max_steps:
        if method == "GPT":
            with torch.no_grad():
                logits, _ = model(torch.LongTensor(state_list).unsqueeze(0).to(device))
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(logits[0, -1, :].cpu().numpy())
        elif method == "DQN":
            state = np.zeros((22,))
            state[current_state] = 1
            action = QL.act_e_greedy(torch.from_numpy(state).float().unsqueeze(0).to(device), 0.1)
        next_state, reward, is_done = grid_world.step(action)
        if method == "DQN":
            next_state_e = np.zeros((22,))
            next_state_e[next_state] = 1
            mem.append(state, action, next_state_e, reward, is_done)
        state_list.append(next_state)
        action_list.append(action)
        reward_list.append(reward)
        done_list.append(is_done)
        if method == "DQN":
            losses.append(QL.learn(mem, total_step))
        step += 1
        if method == "DQN":
            total_step += 1
        if is_done:
            print(f"Done in {step} steps")
            break
        current_state = next_state
    buffer.append(state_list, action_list, reward_list, done_list)
    if len(buffer) >= 1 and method == "GPT":
        for batch in range(32):
            states, actions, next_states, rewards, dones = buffer.sample()
            states = torch.LongTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            state_action_values, _ = model(states)
            state_action_values=state_action_values.gather(2, actions.unsqueeze(2))
            next_states = torch.LongTensor(next_states).to(device)
            best_action, _ = model(next_states)
            best_action = best_action.argmax(2).unsqueeze(2)
            next_state_values, _ = target_model(next_states)
            next_state_values = next_state_values.gather(2, best_action)
            dones = 1 - torch.FloatTensor(dones).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            TDtargets = next_state_values.squeeze(2) * DISCOUNT_FACTOR * dones + rewards
            criterion = nn.MSELoss()
            
            loss = criterion(state_action_values.squeeze(2), TDtargets)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step += 1
            if total_step % update_frequency == 0:
                target_model.load_state_dict(model.state_dict())
    run.log({"loss": np.mean(losses), "step": step})

run.finish()
