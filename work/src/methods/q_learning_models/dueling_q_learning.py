import numpy as np
from .base_q_learning import BaseQLearning


class DuelingQLearning(BaseQLearning):
    """
    Dueling Q-Learning 模型
    
    注意：奖励计算由 OptimizedQLearningTrainer 提供
    """
    def __init__(self, alpha_v=0.05, alpha_a=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha_v = alpha_v
        self.alpha_a = alpha_a
        # 初始化价值函数 V 和优势函数 A
        self.V = np.zeros(self.n_states, dtype=np.float64)
        self.A = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def calculate_reward(self, t_start, t_end, eeg_data, labels):
        """
        计算奖励（占位符实现）
        
        注意：实际训练中，Trainer 会提供自己的奖励计算实现
        """
        return 0.0

    def compute_q_values(self, state_idx):
        """计算 Dueling 架构下的 Q 值"""
        if state_idx == -1:
            return np.full(self.n_actions, -np.inf)
        mean_A = np.mean(self.A[state_idx])
        return self.V[state_idx] + self.A[state_idx] - mean_A

    def select_action(self, state_idx, epsilon):
        """ε-greedy 动作选择"""
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_vals = self.compute_q_values(state_idx)
            return int(np.argmax(q_vals))

    def update_q_values(self, state_idx, action, reward, next_state_idx):
        """
        Dueling Q-learning 更新规则
        
        改进：处理无效 next_state_idx 的情况
        """
        # 处理无效状态
        if next_state_idx == -1:
            # 无效状态：不使用未来奖励
            td_target = reward
            current_q = self.V[state_idx] + self.A[state_idx, action] - np.mean(self.A[state_idx])
            td_error = td_target - current_q
        else:
            # 正常更新
            next_q_vals = self.compute_q_values(next_state_idx)
            td_target = reward + self.gamma * np.max(next_q_vals)
            current_q = self.V[state_idx] + self.A[state_idx, action] - np.mean(self.A[state_idx])
            td_error = td_target - current_q

        # 更新 V 和 A
        self.V[state_idx] += self.alpha_v * td_error
        self.A[state_idx, action] += self.alpha_a * td_error

    def get_q_value(self, state_idx, action):
        """获取 Q 值"""
        q_vals = self.compute_q_values(state_idx)
        return q_vals[action]

    def get_optimal_policy(self):
        """获取最优策略对应的状态"""
        best_value = -np.inf
        best_state_idx = -1
        for idx in range(self.n_states):
            state = self.index_to_state(idx)
            if state is None:
                continue
            q_vals = self.compute_q_values(idx)
            val = np.max(q_vals)
            if val > best_value:
                best_value = val
                best_state_idx = idx

        return self.index_to_state(best_state_idx)
