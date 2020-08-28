import torch
import torch.nn as nn
import numpy as np
import random

class NaiveDQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        super(NaiveDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon)

        self.criteria = nn.MSELoss()

    def choose_action(self, state):
        qs = self.qnet(state)  # Notice that qs is 2d tensor [batch x action]

        if self.train:  # epsilon-greedy policy
            #prob = np.random.uniform(0.0, 1.0, 1)
            #if torch.from_numpy(prob).float() <= self.epsilon:  # random
            if random.random() <= self.epsilon: # random
                action = np.random.choice(range(self.action_dim))
            else:  # greedy
                action = qs.argmax(dim=-1)
        else:  # greedy policy
            action = qs.argmax(dim=-1)
        return int(action)

    def learn(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state
        # Q-Learning target
        q_max, _ = self.qnet(next_state).max(dim=-1)
        q_target = r + self.gamma * q_max * (1 - done)

        # Don't forget to detach `td_target` from the computational graph
        q_target = q_target.detach()

        # Or you can follow a better practice as follows:
        """
        with torch.no_grad():
            q_max, _ = self.qnet(next_state).max(dim=-1)
            q_target = r + self.gamma * q_max * (1 - done)
        """

        loss = self.criteria(self.qnet(s)[0, action], q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss
