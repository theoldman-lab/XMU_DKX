import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from .base_q_learning import BaseQLearning

# 自动检测并使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """
    深度 Q 网络模型
    """
    def __init__(self, input_size=2, hidden_size=128, output_size=4):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


class DQNModel(BaseQLearning):
    """
    DQN 模型实现

    注意：奖励计算由 OptimizedQLearningTrainer 提供
    """
    def __init__(self,
                 buffer_size=10000,
                 batch_size=32,
                 lr=1e-3,
                 tau=1e-3,
                 update_every=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.tau = tau  # 软更新参数
        self.update_every = update_every

        # 神经网络（自动使用 GPU）
        self.qnetwork_local = DQN().to(device)
        self.qnetwork_target = DQN().to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # 经验回放池
        self.memory = deque(maxlen=self.buffer_size)

        # 更新计数器
        self.t_step = 0

    def calculate_reward(self, t_start, t_end, eeg_data, labels):
        """
        计算奖励（占位符实现）
        
        注意：实际训练中，Trainer 会提供自己的奖励计算实现
        """
        return 0.0

    def _state_to_tensor(self, state_idx):
        """将状态索引转换为神经网络输入张量"""
        if state_idx == -1:
            return torch.zeros(2, device=device)

        state = self.index_to_state(state_idx)
        if state is None:
            return torch.zeros(2, device=device)

        t_start, t_end = state
        return torch.tensor([t_start, t_end], dtype=torch.float32, device=device)

    def select_action(self, state_idx, epsilon):
        """ε-greedy 动作选择"""
        state_tensor = self._state_to_tensor(state_idx).unsqueeze(0)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def store_experience(self, state_idx, action, reward, next_state_idx):
        """存储经验到回放缓冲区"""
        state_tensor = self._state_to_tensor(state_idx)
        next_state_tensor = self._state_to_tensor(next_state_idx)

        self.memory.append((state_tensor, action, torch.tensor(reward, dtype=torch.float32, device=device),
                           next_state_tensor, torch.tensor(1.0, device=device)))

    def update_q_values(self, state_idx, action, reward, next_state_idx):
        """更新 DQN 的 Q 值"""
        # 存储经验
        self.store_experience(state_idx, action, reward, next_state_idx)

        # 每 update_every 步更新一次网络
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self._sample_experiences()
            self._learn(experiences)

    def _sample_experiences(self):
        """从经验回放缓冲区中采样"""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.stack([e[0] for e in experiences]).to(device)
        actions = torch.tensor([e[1] for e in experiences], dtype=torch.long, device=device)
        rewards = torch.stack([e[2] for e in experiences]).to(device)
        next_states = torch.stack([e[3] for e in experiences]).to(device)
        dones = torch.stack([e[4] for e in experiences]).to(device)

        return (states, actions, rewards, next_states, dones)

    def _learn(self, experiences):
        """从经验中学习"""
        states, actions, rewards, next_states, dones = experiences

        # 计算当前 Q 值
        current_q_values = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))

        # 计算目标 Q 值
        next_q_values = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # 对于无效状态（全零张量），next_q_values 接近 0，相当于没有未来奖励
        target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * dones.unsqueeze(1))

        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        self._soft_update()

    def _soft_update(self):
        """软更新目标网络"""
        for target_param, local_param in zip(self.qnetwork_target.parameters(),
                                           self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def get_q_value(self, state_idx, action):
        """获取 Q 值"""
        state_tensor = self._state_to_tensor(state_idx).unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        return action_values[0, action].item()

    def get_optimal_policy(self):
        """获取最优策略对应的状态"""
        # 由于是连续状态空间，我们可以通过采样找到最优状态
        best_value = -np.inf
        best_state_idx = -1

        # 在状态空间中进行密集采样
        for t_start in np.arange(self.t_start_min, self.t_start_max, 0.2):
            for t_end in np.arange(self.t_end_min, self.t_end_max, 0.2):
                if t_end - t_start < self.min_window_len:
                    continue

                state_idx = self.state_to_index(t_start, t_end)
                if state_idx != -1:
                    state_tensor = self._state_to_tensor(state_idx).unsqueeze(0)

                    self.qnetwork_local.eval()
                    with torch.no_grad():
                        action_values = self.qnetwork_local(state_tensor)
                    self.qnetwork_local.train()

                    max_q = torch.max(action_values).item()
                    if max_q > best_value:
                        best_value = max_q
                        best_state_idx = state_idx

        return self.index_to_state(best_state_idx)
