"""
Q-Learning 基础抽象类，定义通用接口和基本功能

注意：训练功能已移至 OptimizedQLearningTrainer 类
"""
from abc import ABC, abstractmethod


class BaseQLearning(ABC):
    """
    Q-Learning 基础抽象类，定义通用接口和基本功能
    
    训练功能由 OptimizedQLearningTrainer 提供
    """
    def __init__(self,
                 t_start_min=0.0,
                 t_start_max=3.0,
                 t_end_min=1.0,
                 t_end_max=4.0,
                 step_size=0.1,
                 min_window_len=0.5,
                 gamma=0.9,
                 epsilon_init=0.3,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 n_episodes=100,
                 max_steps_per_episode=30,
                 n_components=3,
                 lambda_efficiency=0.0,  # 效率惩罚系数（默认 0，不使用效率惩罚）
                 **kwargs):
        """
        初始化基础 Q-Learning 参数
        """
        # 时间窗参数
        self.t_start_min = t_start_min
        self.t_start_max = t_start_max
        self.t_end_min = t_end_min
        self.t_end_max = t_end_max
        self.step_size = step_size
        self.min_window_len = min_window_len
        self.lambda_efficiency = lambda_efficiency

        # RL 超参数
        self.gamma = gamma
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_episodes = n_episodes
        self.max_steps_per_episode = max_steps_per_episode

        # CSP 参数
        self.n_components = n_components

        # 动作空间定义
        self.n_actions = 4
        self.ACTION_SPACE = {
            0: "decrease_t_start",
            1: "increase_t_start",
            2: "decrease_t_end",
            3: "increase_t_end"
        }

        # 状态空间大小（离散化）
        self.start_states_count = int((t_start_max - t_start_min) / step_size) + 1
        self.end_states_count = int((t_end_max - t_end_min) / step_size) + 1
        self.n_states = self.start_states_count * self.end_states_count

        # 预计算有效状态索引
        self._precompute_valid_windows()

        # 训练日志
        self.training_logs = {
            'reward_history': [],
            'epsilon_history': [],
            'optimal_window_history': []
        }

    def _precompute_valid_windows(self):
        """预计算所有有效的状态索引，以便快速查找"""
        self.valid_state_indices = []
        for start_idx in range(self.start_states_count):
            for end_idx in range(self.end_states_count):
                t_start = self.t_start_min + start_idx * self.step_size
                t_end = self.t_end_min + end_idx * self.step_size
                if t_end - t_start >= self.min_window_len:
                    idx = int(start_idx * self.end_states_count + end_idx)
                    self.valid_state_indices.append(idx)

    def state_to_index(self, t_start, t_end):
        """将连续时间窗映射为离散状态索引"""
        if (not (self.t_start_min <= t_start <= self.t_start_max) or
            not (self.t_end_min <= t_end <= self.t_end_max) or
            t_end - t_start < self.min_window_len):
            return -1  # 无效状态

        start_idx = round((t_start - self.t_start_min) / self.step_size)
        end_idx = round((t_end - self.t_end_min) / self.step_size)

        # 确保索引在范围内
        start_idx = max(0, min(start_idx, self.start_states_count - 1))
        end_idx = max(0, min(end_idx, self.end_states_count - 1))

        return int(start_idx * self.end_states_count + end_idx)

    def index_to_state(self, idx):
        """将状态索引还原为时间窗参数"""
        if idx < 0 or idx >= self.n_states:
            return None

        start_idx = idx // self.end_states_count
        end_idx = idx % self.end_states_count

        t_start = self.t_start_min + start_idx * self.step_size
        t_end = self.t_end_min + end_idx * self.step_size

        if t_end - t_start < self.min_window_len:
            return None

        return (t_start, t_end)

    def apply_action(self, t_start, t_end, action):
        """确定性状态转移"""
        dt = self.step_size
        if action == 0:
            t_start = max(self.t_start_min, t_start - dt)
        elif action == 1:
            t_start = min(self.t_start_max, t_start + dt)
        elif action == 2:
            t_end = max(self.t_end_min, t_end - dt)
        elif action == 3:
            t_end = min(self.t_end_max, t_end + dt)

        return t_start, t_end

    @abstractmethod
    def calculate_reward(self, t_start, t_end, eeg_data, labels):
        """
        计算给定时间窗的奖励值（抽象方法）
        
        Parameters:
        -----------
        t_start, t_end : float
            时间窗参数
        eeg_data : array
            EEG 数据
        labels : array
            标签
            
        Returns:
        --------
        reward : float
            奖励值
        """
        pass

    @abstractmethod
    def select_action(self, state_idx, epsilon, **kwargs):
        """选择动作的抽象方法"""
        pass

    @abstractmethod
    def update_q_values(self, state_idx, action, reward, next_state_idx, **kwargs):
        """更新 Q 值的抽象方法"""
        pass

    @abstractmethod
    def get_q_value(self, state_idx, action, **kwargs):
        """获取 Q 值的抽象方法"""
        pass

    @abstractmethod
    def get_optimal_policy(self):
        """获取最优策略的抽象方法"""
        pass
