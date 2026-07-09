import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import os
from collections import deque

# --- DEVICE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. THE REPLAY BUFFER (Memory)
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        """Wipes the memory buffer to prevent poisoning during transfer learning."""
        self.buffer.clear()

# ==============================================================================
# 2. THE NETWORKS (Brain)
# ==============================================================================
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, log_sig_min=-20, log_sig_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean

# ==============================================================================
# 3. THE AGENT (Coordinator)
# ==============================================================================
class SACAgent:
    def __init__(self, num_inputs, num_actions, hidden_size=64, lr=0.0003, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma = gamma 
        self.tau = tau      
        self.alpha = alpha  

        self.critic = ValueNetwork(num_inputs, num_actions, hidden_size).to(device)
        self.critic_target = ValueNetwork(num_inputs, num_actions, hidden_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict()) 

        self.actor = PolicyNetwork(num_inputs, num_actions, hidden_size).to(device)

        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.target_entropy = -torch.prod(torch.Tensor([num_actions]).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.log_alpha.exp() * next_state_log_pi
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch) 
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.log_alpha.exp() * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return qf1_loss.item(), actor_loss.item()
        
    def save_weights(self, path):
        """Saves the neural network weights for transfer learning."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'log_alpha': self.log_alpha
        }, path)
        print(f"   [Agent] Brain successfully saved to {path}")

    def load_weights(self, path):
        """Loads pre-trained weights to jumpstart zero-shot TS searches."""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.log_alpha = checkpoint['log_alpha']
            print(f"   [Agent] Pre-trained brain successfully loaded from {path}")
        else:
            print(f"   [Agent] Warning: Checkpoint {path} not found. Starting fresh.")