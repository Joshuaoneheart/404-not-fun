import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

class Agent():
    def __init__(self, device):
        self.device = device
        self.batch_size = 32
        self.discount = 0.99
        self.losses = []
        self.target_update = 10000
        self.policy_net = DQN(22, 4,
                              (128, 128), device).to(self.device)
        self.target_net = DQN(22, 4,
                              (128, 128), device).to(self.device)
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=5e-4)
        self.target_net.eval()

    def act_e_greedy(self, state, epsilon):
        sample = random.random()

        if sample > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].item()
        else:
            return random.randrange(4)
    def learn(self, memory, steps_done):
        if steps_done < self.batch_size:
            return 0

        if steps_done % self.target_update == 0:
            self.update_target_net()

        # Sample a mini-batch from the replay memory
        idxs, states, actions, next_states, returns, dones, weights = memory.sample(self.batch_size)
        uses = 1 - dones
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        use_batch = torch.FloatTensor(uses).to(self.device).squeeze()
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        reward_batch = torch.FloatTensor(returns).to(self.device).squeeze()
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        best_actions = self.policy_net(
            next_state_batch).argmax(1).unsqueeze(1)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values = self.target_net(
            next_state_batch).gather(1, best_actions).squeeze().detach()

        TDtargets = (next_state_values * self.discount) * use_batch + reward_batch
        # for i, nv in enumerate(TDtargets):
            # if nv.item() == 0 and batch.nextState[i][0] != None:
                # print(f"target net output: {self.target_net(non_final_next_states)} next state: {non_final_next_states}, best action: {best_actions}")

        TDerrors = TDtargets.unsqueeze(1) - state_action_values
        criterion = nn.MSELoss(reduction="none")

        loss = criterion(state_action_values, TDtargets.unsqueeze(1)).squeeze()
        for i, reward in enumerate(reward_batch):
            if reward == 1:
                print(f"reward {reward} weights: {None if type(weights) == type(None) else weights[i].item()} use: {uses[i].item()} TDerrors: {TDerrors[i].item()} Action Value: {state_action_values[i].item()} Target: {TDtargets[i].item()} loss: {loss[i].item()} action: {action_batch[i].item()} next_state: {next_states[i]}")

        self.optimizer.zero_grad()

        self.losses.append(loss.mean().item())
        if type(weights) != type(None):
            weights = torch.FloatTensor(weights).to(self.device)
            (loss).mean().backward()
            memory.update_priorities(idxs, loss.detach().cpu().numpy() + 1e-6)
        else:
            loss.mean().backward()

        self.optimizer.step()
        return self.losses[-1]

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DQN(nn.Module):

    def __init__(self, inputSize, numActions, hiddenLayerSize=(256, 128), device="cpu"):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenLayerSize[0])
        self.fc2 = nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1])
        self.fc3 = nn.Linear(hiddenLayerSize[1], hiddenLayerSize[1])
        self.fc_A = nn.Linear(hiddenLayerSize[1], hiddenLayerSize[1])
        self.fc_V = nn.Linear(hiddenLayerSize[1], hiddenLayerSize[1])
        self.A = nn.Linear(hiddenLayerSize[1], numActions)
        self.V = nn.Linear(hiddenLayerSize[1], 1)
        self.device = device

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v = self.V(F.relu(self.fc_V(x)))
        a = self.A(F.relu(self.fc_A(x)))
        return v + a - a.mean()

