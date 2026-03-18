import numpy as np
from .base_q_learning import BaseQLearning


class DoubleQLearning(BaseQLearning):
    """
    Double Q-Learning 模型
    
    注意：奖励计算由 OptimizedQLearningTrainer 提供
    """
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        # 初始化两套 Q 表
        self.Q1 = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.Q2 = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def calculate_reward(self, t_start, t_end, eeg_data, labels):
        """
        计算奖励（占位符实现）
        
        注意：实际训练中，Trainer 会提供自己的奖励计算实现
        """
        return 0.0

    def select_action(self, state_idx, epsilon):
        """ε-greedy 动作选择，使用两表平均值"""
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            avg_q = (self.Q1[state_idx] + self.Q2[state_idx]) / 2.0
            return int(np.argmax(avg_q))

    def update_q_values(self, state_idx, action, reward, next_state_idx):
        """
        Double Q-learning 更新规则
        
        改进：处理无效 next_state_idx 的情况
        """
        # 处理无效状态
        if next_state_idx == -1:
            # 无效状态：不使用未来奖励
            td_target = reward
            
            # 更新 Q1
            td_error1 = td_target - self.Q1[state_idx, action]
            self.Q1[state_idx, action] += self.alpha * td_error1
            
            # 更新 Q2
            td_error2 = td_target - self.Q2[state_idx, action]
            self.Q2[state_idx, action] += self.alpha * td_error2
        else:
            # 随机选择更新哪个 Q 表
            if np.random.rand() < 0.5:
                # 更新 Q1，使用 Q2 选择动作
                a_star = int(np.argmax(self.Q1[next_state_idx]))
                td_target = reward + self.gamma * self.Q2[next_state_idx, a_star]
                td_error = td_target - self.Q1[state_idx, action]
                self.Q1[state_idx, action] += self.alpha * td_error
            else:
                # 更新 Q2，使用 Q1 选择动作
                a_star = int(np.argmax(self.Q2[next_state_idx]))
                td_target = reward + self.gamma * self.Q1[next_state_idx, a_star]
                td_error = td_target - self.Q2[state_idx, action]
                self.Q2[state_idx, action] += self.alpha * td_error

    def get_q_value(self, state_idx, action):
        """获取 Q 值（返回两表平均值）"""
        return (self.Q1[state_idx, action] + self.Q2[state_idx, action]) / 2.0

    def get_optimal_policy(self):
        """获取最优策略对应的状态"""
        best_value = -np.inf
        best_state_idx = -1
        for idx in range(self.n_states):
            state = self.index_to_state(idx)
            if state is None:
                continue
            # 使用两表平均值
            avg_q = (self.Q1[idx] + self.Q2[idx]) / 2.0
            val = np.max(avg_q)
            if val > best_value:
                best_value = val
                best_state_idx = idx

        return self.index_to_state(best_state_idx)
