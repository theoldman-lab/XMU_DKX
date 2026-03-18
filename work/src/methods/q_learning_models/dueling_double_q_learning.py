import numpy as np
from .base_q_learning import BaseQLearning


class DuelingDoubleQLearning(BaseQLearning):
    """
    Dueling Double Q-Learning 模型
    
    注意：奖励计算由 OptimizedQLearningTrainer 提供
    """
    def __init__(self, alpha_v=0.05, alpha_a=0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha_v = alpha_v
        self.alpha_a = alpha_a
        # 初始化两套价值函数 V 和优势函数 A
        self.V1 = np.zeros(self.n_states, dtype=np.float64)
        self.A1 = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.V2 = np.zeros(self.n_states, dtype=np.float64)
        self.A2 = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def calculate_reward(self, t_start, t_end, eeg_data, labels):
        """
        计算奖励（占位符实现）
        
        注意：实际训练中，Trainer 会提供自己的奖励计算实现
        """
        return 0.0

    def compute_q_values(self, V, A, state_idx):
        """计算 Dueling 架构下的 Q 值"""
        if state_idx == -1:
            return np.full(self.n_actions, -np.inf)
        mean_A = np.mean(A[state_idx])
        return V[state_idx] + A[state_idx] - mean_A

    def select_action(self, state_idx, epsilon):
        """ε-greedy 动作选择，使用两网络平均值"""
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            q1 = self.compute_q_values(self.V1, self.A1, state_idx)
            q2 = self.compute_q_values(self.V2, self.A2, state_idx)
            q_avg = (q1 + q2) / 2.0
            return int(np.argmax(q_avg))

    def update_q_values(self, state_idx, action, reward, next_state_idx):
        """
        Dueling Double Q-learning 更新规则

        Double Q-Learning 核心思想：
        - 以 50% 概率更新网络 1：用网络 1 选择动作，用网络 2 评估
        - 以 50% 概率更新网络 2：用网络 2 选择动作，用网络 1 评估

        Dueling 架构：
        - Q(s,a) = V(s) + A(s,a) - mean(A(s))
        """
        # 处理无效状态
        if next_state_idx == -1:
            # 无效状态：不使用未来奖励
            target = reward

            # 同时更新两个网络（因为无效动作，两个网络都学习这个惩罚）
            # 更新网络 1
            current_q1 = self.V1[state_idx] + self.A1[state_idx, action] - np.mean(self.A1[state_idx])
            td_error1 = target - current_q1
            self.V1[state_idx] += self.alpha_v * td_error1
            self.A1[state_idx, action] += self.alpha_a * td_error1

            # 更新网络 2
            current_q2 = self.V2[state_idx] + self.A2[state_idx, action] - np.mean(self.A2[state_idx])
            td_error2 = target - current_q2
            self.V2[state_idx] += self.alpha_v * td_error2
            self.A2[state_idx, action] += self.alpha_a * td_error2
        else:
            # Double Q-Learning 交替更新机制
            if np.random.rand() < 0.5:
                # 更新网络 1：用网络 1 选择动作，用网络 2 评估
                # 步骤 1：用网络 1 的 Q 值选择最优动作
                q1_next = self.compute_q_values(self.V1, self.A1, next_state_idx)
                a_star = int(np.argmax(q1_next))

                # 步骤 2：用网络 2 的 Q 值评估该动作
                q2_next = self.compute_q_values(self.V2, self.A2, next_state_idx)
                target = reward + self.gamma * q2_next[a_star]

                # 步骤 3：计算网络 1 的当前 Q 值和 TD 误差
                current_q1 = self.V1[state_idx] + self.A1[state_idx, action] - np.mean(self.A1[state_idx])
                td_error = target - current_q1

                # 步骤 4：更新网络 1
                self.V1[state_idx] += self.alpha_v * td_error
                self.A1[state_idx, action] += self.alpha_a * td_error
            else:
                # 更新网络 2：用网络 2 选择动作，用网络 1 评估
                # 步骤 1：用网络 2 的 Q 值选择最优动作
                q2_next = self.compute_q_values(self.V2, self.A2, next_state_idx)
                a_star = int(np.argmax(q2_next))

                # 步骤 2：用网络 1 的 Q 值评估该动作
                q1_next = self.compute_q_values(self.V1, self.A1, next_state_idx)
                target = reward + self.gamma * q1_next[a_star]

                # 步骤 3：计算网络 2 的当前 Q 值和 TD 误差
                current_q2 = self.V2[state_idx] + self.A2[state_idx, action] - np.mean(self.A2[state_idx])
                td_error = target - current_q2

                # 步骤 4：更新网络 2
                self.V2[state_idx] += self.alpha_v * td_error
                self.A2[state_idx, action] += self.alpha_a * td_error

    def get_q_value(self, state_idx, action):
        """获取 Q 值（返回两网络平均值）"""
        q1_vals = self.compute_q_values(self.V1, self.A1, state_idx)
        q2_vals = self.compute_q_values(self.V2, self.A2, state_idx)
        return (q1_vals[action] + q2_vals[action]) / 2.0

    def get_optimal_policy(self):
        """获取最优策略对应的状态"""
        best_value = -np.inf
        best_state_idx = -1
        for idx in range(self.n_states):
            state = self.index_to_state(idx)
            if state is None:
                continue
            q1 = self.compute_q_values(self.V1, self.A1, idx)
            q2 = self.compute_q_values(self.V2, self.A2, idx)
            avg_q = (q1 + q2) / 2.0
            val = np.max(avg_q)
            if val > best_value:
                best_value = val
                best_state_idx = idx

        return self.index_to_state(best_state_idx)
