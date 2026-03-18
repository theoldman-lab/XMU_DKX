"""
DQN 专用训练器

与 trainer_simple 保持接口一致，便于控制变量实验
"""

import numpy as np
import torch
import torch.nn as nn
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from src.features.csp import extract_csp_features
from src.utils.config import Config


class DQNTrainer:
    """
    DQN 训练器
    
    设计原则：与 SimpleQTrainer 保持接口一致
    - 相同的奖励计算（准确率）
    - 相同的ε-greedy 探索
    - 相同的训练循环结构
    """

    def __init__(self, model, use_feature_cache=True):
        """
        Parameters:
        -----------
        model : DQNModel
            DQN 模型实例
        use_feature_cache : bool
            是否使用特征缓存
        """
        self.model = model
        self.use_feature_cache = use_feature_cache

        # 特征缓存
        self.feature_cache = {}
        self.window_indices = {}
        self.eeg_data = None
        self.labels = None

    def setup(self, eeg_data, labels_train, precompute=False):
        """设置训练器"""
        self.eeg_data = eeg_data
        self.labels = labels_train
        self.feature_cache.clear()
        self.window_indices.clear()

        if precompute:
            self._precompute_features()

    def _precompute_features(self):
        """预计算所有可能时间窗的 CSP 特征"""
        step = self.model.step_size
        t_starts = np.arange(self.model.t_start_min,
                            self.model.t_start_max + step, step)
        t_ends = np.arange(self.model.t_end_min,
                          self.model.t_end_max + step, step)

        for t_start in t_starts:
            for t_end in t_ends:
                if t_end - t_start >= self.model.min_window_len:
                    start_idx = int(t_start * 250)
                    end_idx = int(t_end * 250)
                    self.window_indices[(t_start, t_end)] = (start_idx, end_idx)

        for (t_start, t_end), (start_idx, end_idx) in tqdm(
            self.window_indices.items(), desc="预计算特征"
        ):
            try:
                windowed = self.eeg_data[:, :, start_idx:end_idx]
                features = extract_csp_features(
                    windowed, self.labels,
                    n_components=Config.CSP_PARAMS['n_components']
                )
                self.feature_cache[(t_start, t_end)] = features
            except Exception:
                continue

    def calculate_reward(self, t_start, t_end):
        """计算奖励（准确率）- 与 SimpleQTrainer 一致"""
        if self.eeg_data is None or self.labels is None:
            return -1.0

        window_length = t_end - t_start
        if t_end <= t_start:
            return -1.0
        if window_length < 0.5:
            return -1.0

        if (t_start, t_end) in self.feature_cache:
            features = self.feature_cache[(t_start, t_end)]
        else:
            start_idx = int(t_start * 250)
            end_idx = int(t_end * 250)
            start_idx = max(0, min(start_idx, self.eeg_data.shape[2]))
            end_idx = max(0, min(end_idx, self.eeg_data.shape[2]))

            if start_idx >= end_idx:
                return -1.0

            windowed = self.eeg_data[:, :, start_idx:end_idx]
            try:
                features = extract_csp_features(
                    windowed, self.labels,
                    n_components=Config.CSP_PARAMS['n_components']
                )
                if self.use_feature_cache:
                    self.feature_cache[(t_start, t_end)] = features
            except Exception:
                return -1.0

        if features.size == 0 or np.isnan(features).any():
            return -1.0

        min_samples = min([np.sum(self.labels == c) for c in np.unique(self.labels)])
        cv_folds = min(5, min_samples)

        if cv_folds < 2:
            return 0.5

        try:
            svm = SVC(
                kernel=Config.SVM_PARAMS['kernel'],
                C=Config.SVM_PARAMS['C'],
                random_state=Config.SVM_PARAMS['random_state']
            )
            # 过滤 UndefinedMetricWarning 警告（某些 fold 中类别样本太少导致）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                scores = cross_val_score(svm, features, self.labels,
                                        cv=cv_folds, scoring='accuracy')
            return np.mean(scores)
        except Exception:
            return -1.0

    def train(self, eeg_train, labels_train, verbose=True, precompute=True):
        """
        训练 DQN 模型 - 与 SimpleQTrainer 训练循环结构一致

        Parameters:
        -----------
        eeg_train, labels_train : array
            训练数据和标签
        verbose : bool
            是否显示进度
        precompute : bool
            是否预计算特征

        Returns:
        --------
        tuple : 最优时间窗 (t_start, t_end)
        """
        self.setup(eeg_train, labels_train, precompute=precompute)

        # 初始化日志
        self.model.training_logs = {
            'reward_history': [],
            'best_accuracy_history': []
        }

        # 训练循环
        best_accuracy = 0.0
        best_window = (self.model.t_start_min,
                       self.model.t_end_min + self.model.min_window_len)
        epsilon = self.model.epsilon_init

        pbar = tqdm(range(self.model.n_episodes), desc="训练进度") if verbose else range(self.model.n_episodes)

        for episode in pbar:
            # 自适应ε衰减 - 与 SimpleQTrainer 一致
            epsilon = self._decay_epsilon(episode)

            # 随机初始化有效状态
            t_start, t_end = self._random_valid_state()
            state_idx = self.model.state_to_index(t_start, t_end)
            episode_reward = 0

            for step in range(self.model.max_steps_per_episode):
                # ε-greedy 动作选择
                action = self.model.select_action(state_idx, epsilon)

                # 执行动作
                t_start_new, t_end_new = self.model.apply_action(t_start, t_end, action)
                next_state_idx_raw = self.model.state_to_index(t_start_new, t_end_new)

                if next_state_idx_raw == -1:
                    # 无效动作：惩罚，状态不变
                    reward = -1.0
                    # 关键修复：使用 -1 作为 next_state_idx，让模型知道这是无效动作
                    next_state_idx = -1
                    # 状态保持不变
                    t_start_new, t_end_new = t_start, t_end
                else:
                    # 有效动作：计算奖励并更新状态
                    reward = self.calculate_reward(t_start_new, t_end_new)
                    t_start, t_end = t_start_new, t_end_new
                    next_state_idx = next_state_idx_raw

                # Q 值更新（DQN 通过经验回放和梯度下降）
                self.model.update_q_values(state_idx, action, reward, next_state_idx)

                # 更新状态索引
                state_idx = next_state_idx if next_state_idx != -1 else state_idx
                episode_reward += reward

            # 记录日志
            self.model.training_logs['reward_history'].append(episode_reward)

            # 更新最优窗口 - 与 SimpleQTrainer 一致
            current_acc = self.calculate_reward(t_start, t_end)
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_window = (t_start, t_end)

            # 定期从 Q 表提取最优策略（每 50 轮）
            if (episode + 1) % 50 == 0:
                q_best_window = self.model.get_optimal_policy()
                if q_best_window:
                    q_best_acc = self.calculate_reward(*q_best_window)
                    if q_best_acc > best_accuracy:
                        best_accuracy = q_best_acc
                        best_window = q_best_window

            self.model.training_logs['best_accuracy_history'].append(best_accuracy)

            # 进度显示
            if verbose and (episode + 1) % 50 == 0:
                pbar.set_description(
                    f"Ep {episode+1}/{self.model.n_episodes}, "
                    f"ε={epsilon:.3f}, Acc={best_accuracy:.4f}"
                )

        return best_window if best_window else (self.model.t_start_min,
                                                 self.model.t_end_min + self.model.min_window_len)

    def _decay_epsilon(self, episode):
        """自适应ε衰减 - 与 SimpleQTrainer 一致"""
        return self.model.epsilon_init - \
               (self.model.epsilon_init - self.model.epsilon_min) * \
               (episode / self.model.n_episodes)

    def _random_valid_state(self):
        """随机生成有效起始状态 - 与 SimpleQTrainer 一致"""
        attempts = 0
        while attempts < 100:
            t_start = np.random.uniform(
                self.model.t_start_min,
                self.model.t_start_max
            )
            t_end = np.random.uniform(
                max(self.model.t_end_min, t_start + self.model.min_window_len),
                self.model.t_end_max
            )
            # 对齐到网格
            t_start = round(t_start / self.model.step_size) * self.model.step_size
            t_end = round(t_end / self.model.step_size) * self.model.step_size
            if t_end - t_start >= self.model.min_window_len:
                return t_start, t_end
            attempts += 1
        return self.model.t_start_min, self.model.t_end_min + self.model.min_window_len


def create_dqn_trainer(model, use_feature_cache=True):
    """
    工厂函数：创建 DQN 训练器

    Parameters:
    -----------
    model : DQNModel
        DQN 模型实例
    use_feature_cache : bool
        是否使用特征缓存

    Returns:
    --------
    DQNTrainer
        训练器实例
    """
    return DQNTrainer(model, use_feature_cache=use_feature_cache)
