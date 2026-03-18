import numpy as np
from .base_q_learning import BaseQLearning


class StandardQLearning(BaseQLearning):
    """
    标准 Q-Learning 模型
    
    注意：奖励计算由 OptimizedQLearningTrainer 提供
    """
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        # 初始化 Q 表
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def calculate_reward(self, t_start, t_end, eeg_data, labels):
        """
        计算奖励（占位符实现）
        
        注意：实际训练中，Trainer 会提供自己的奖励计算实现
        """
        return 0.0

    def select_action(self, state_idx, epsilon):
        """ε-greedy 动作选择"""
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            return int(np.argmax(self.Q[state_idx]))

    def update_q_values(self, state_idx, action, reward, next_state_idx):
        """
        标准 Q-learning 更新规则
        
        改进：
        1. 处理无效 next_state_idx 的情况
        2. 对于无效状态，不使用折扣未来奖励
        """
        # 处理无效状态
        if next_state_idx == -1:
            # 无效状态：TD Target = reward（没有未来奖励）
            td_target = reward
            td_error = td_target - self.Q[state_idx, action]
            self.Q[state_idx, action] += self.alpha * td_error
        else:
            # 正常状态：TD Target = reward + γ * max(Q(next_state))
            td_target = reward + self.gamma * np.max(self.Q[next_state_idx])
            td_error = td_target - self.Q[state_idx, action]
            self.Q[state_idx, action] += self.alpha * td_error

    def get_q_value(self, state_idx, action):
        """获取 Q 值"""
        return self.Q[state_idx, action]

    def get_optimal_policy(self):
        """获取最优策略对应的状态"""
        best_value = -np.inf
        best_state_idx = -1
        for idx in range(self.n_states):
            state = self.index_to_state(idx)
            if state is None:
                continue
            val = np.max(self.Q[idx])
            if val > best_value:
                best_value = val
                best_state_idx = idx

        return self.index_to_state(best_state_idx)
