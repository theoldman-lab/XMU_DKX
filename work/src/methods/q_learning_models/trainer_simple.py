"""
Q-Learning 简化训练器

精简设计，保留核心优化：
- 特征缓存（提升效率）
- 乐观初始化（加速收敛）
- 自适应ε衰减（平衡探索利用）

移除的次要优化：
- 经验回放（表格方法不需要）
- UCB 探索（ε-greedy 已足够）
- 奖励塑形（简化奖励计算）
- 先验知识引导（避免干扰）
- 两阶段搜索（单阶段即可）
"""

import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from src.data.preprocessing import extract_time_window
from src.features.csp import extract_csp_features
from src.utils.config import Config


class SimpleQTrainer:
    """
    简化 Q-Learning 训练器

    适用于：Standard Q, Double Q, Dueling Q, Dueling Double Q
    """

    def __init__(self, model, use_feature_cache=True):
        """
        Parameters:
        -----------
        model : BaseQLearning
            Q-Learning 模型实例
        use_feature_cache : bool
            是否使用特征缓存（默认 True，显著提升效率）
        """
        self.model = model
        self.use_feature_cache = use_feature_cache

        # 检测模型类型
        self.model_type = type(model).__name__
        self.is_dueling = 'Dueling' in self.model_type
        self.is_double = 'Double' in self.model_type

        # 特征缓存
        self.feature_cache = {}
        self.window_indices = {}
        self.eeg_data = None
        self.labels = None

    def setup(self, eeg_data, labels_train, precompute=False):
        """
        设置训练器

        Parameters:
        -----------
        eeg_data : array
            EEG 数据 (n_trials, n_channels, n_samples)
        labels_train : array
            训练标签
        precompute : bool
            是否预计算所有特征（默认 True，显著提升效率）
        """
        self.eeg_data = eeg_data
        self.labels = labels_train
        self.feature_cache.clear()
        self.window_indices.clear()

        # 预计算所有时间窗的特征
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

        # 预计算特征
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
        """
        计算奖励（仅使用准确率，简化版）

        Returns:
        --------
        float : 奖励值（准确率）
        """
        if self.eeg_data is None or self.labels is None:
            return -1.0

        # 检查时间窗有效性
        window_length = t_end - t_start
        if t_end <= t_start:
            return -1.0  # 严重惩罚无效状态
        if window_length < 0.5:
            # 过短窗口：严重惩罚，促使智能体选择有效窗口
            return -1.0

        # 使用缓存特征
        if (t_start, t_end) in self.feature_cache:
            features = self.feature_cache[(t_start, t_end)]
        else:
            # 动态计算时间窗索引
            start_idx = int(t_start * 250)
            end_idx = int(t_end * 250)
            
            # 边界检查
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

        # 检查特征有效性
        if features.size == 0 or np.isnan(features).any():
            return -1.0

        # 交叉验证计算准确率
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
        训练 Q-Learning 模型

        Parameters:
        -----------
        eeg_train : array
            训练数据
        labels_train : array
            训练标签
        verbose : bool
            是否显示进度
        precompute : bool
            是否预计算特征（默认 True）

        Returns:
        --------
        tuple : 最优时间窗 (t_start, t_end)
        """
        # 设置和预计算
        self.setup(eeg_train, labels_train, precompute=precompute)

        # 初始化日志
        self.model.training_logs = {
            'reward_history': [],
            'best_accuracy_history': [],
            'epsilon_history': [],
            'q_value_stats': []  # 新增：Q 值统计
        }

        # 乐观初始化 Q 值（加速早期探索）
        # 使用 0.5 作为初始值，显著高于随机水平 (0.25)，激励探索
        self._optimistic_init(0.5)

        # 训练循环
        best_accuracy = 0.0
        best_window = (self.model.t_start_min,
                       self.model.t_end_min + self.model.min_window_len)  # 默认有效窗口
        epsilon = self.model.epsilon_init

        pbar = tqdm(range(self.model.n_episodes), desc="训练进度") if verbose else range(self.model.n_episodes)

        for episode in pbar:
            # 自适应ε衰减
            epsilon = self._decay_epsilon(episode)

            # 随机初始化有效状态
            t_start, t_end = self._random_valid_state()
            state_idx = self.model.state_to_index(t_start, t_end)
            episode_reward = 0

            for step in range(self.model.max_steps_per_episode):
                # ε-greedy 动作选择
                action = self._select_action(state_idx, epsilon)

                # 执行动作
                t_start_new, t_end_new = self.model.apply_action(t_start, t_end, action)
                next_state_idx_raw = self.model.state_to_index(t_start_new, t_end_new)

                if next_state_idx_raw == -1:
                    # 无效动作：给予惩罚，状态保持不变
                    reward = -1.0
                    # 关键修复：使用 -1 作为 next_state_idx，让模型知道这是无效动作
                    # 模型会正确处理这种情况（不使用折扣未来奖励）
                    next_state_idx = -1
                    # 状态保持不变
                    t_start_new, t_end_new = t_start, t_end
                else:
                    # 有效动作：计算奖励并更新状态
                    reward = self.calculate_reward(t_start_new, t_end_new)
                    t_start, t_end = t_start_new, t_end_new
                    next_state_idx = next_state_idx_raw

                # Q 值更新
                self.model.update_q_values(state_idx, action, reward, next_state_idx)

                # 更新状态索引
                state_idx = next_state_idx if next_state_idx != -1 else state_idx

                episode_reward += reward

            # 记录日志
            self.model.training_logs['reward_history'].append(episode_reward)
            self.model.training_logs['epsilon_history'].append(epsilon)

            # 定期记录 Q 值统计（每 10 轮）
            if (episode + 1) % 10 == 0:
                q_stats = self._compute_q_value_statistics()
                self.model.training_logs['q_value_stats'].append({
                    'episode': episode + 1,
                    **q_stats
                })

            # 更新最优窗口 - 同时考虑历史最佳和 Q 表最佳
            # 1. 检查当前状态的准确率（每轮都检查）
            current_acc = self.calculate_reward(t_start, t_end)
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_window = (t_start, t_end)

            # 2. 定期从 Q 表提取最优策略进行评估
            if (episode + 1) % 20 == 0:
                q_best_window = self.model.get_optimal_policy()
                if q_best_window:
                    q_best_acc = self.calculate_reward(*q_best_window)
                    if q_best_acc > best_accuracy:
                        best_accuracy = q_best_acc
                        best_window = q_best_window

            self.model.training_logs['best_accuracy_history'].append(best_accuracy)

            # 进度显示（每 50 轮）
            if verbose and (episode + 1) % 50 == 0:
                pbar.set_description(
                    f"Ep {episode+1}/{self.model.n_episodes}, "
                    f"ε={epsilon:.3f}, Acc={best_accuracy:.4f}"
                )

        # 最终 Q 值统计
        final_q_stats = self._compute_q_value_statistics()
        self.model.training_logs['final_q_stats'] = final_q_stats

        return best_window if best_window else (self.model.t_start_min,
                                                 self.model.t_end_min + self.model.min_window_len)

    def _compute_q_value_statistics(self):
        """
        计算 Q 值统计信息

        Returns:
        --------
        dict : Q 值统计信息
        """
        stats = {
            'q_mean': None,
            'q_std': None,
            'q_min': None,
            'q_max': None,
            'q_nonzero_count': 0,
        }

        # Standard Q
        if hasattr(self.model, 'Q'):
            q_values = self.model.Q.flatten()
            non_zero = q_values[q_values != 0]
            stats.update({
                'q_mean': float(np.mean(q_values)) if len(q_values) > 0 else None,
                'q_std': float(np.std(q_values)) if len(q_values) > 0 else None,
                'q_min': float(np.min(q_values)) if len(q_values) > 0 else None,
                'q_max': float(np.max(q_values)) if len(q_values) > 0 else None,
                'q_nonzero_count': int(np.sum(q_values != 0)),
            })

        # Double Q
        if hasattr(self.model, 'Q1') and hasattr(self.model, 'Q2'):
            q1_values = self.model.Q1.flatten()
            q2_values = self.model.Q2.flatten()
            avg_q = (q1_values + q2_values) / 2.0
            non_zero = avg_q[avg_q != 0]
            stats.update({
                'q_mean': float(np.mean(avg_q)) if len(avg_q) > 0 else None,
                'q_std': float(np.std(avg_q)) if len(avg_q) > 0 else None,
                'q_min': float(np.min(avg_q)) if len(avg_q) > 0 else None,
                'q_max': float(np.max(avg_q)) if len(avg_q) > 0 else None,
                'q_nonzero_count': int(np.sum(avg_q != 0)),
            })

        # Dueling Q
        if hasattr(self.model, 'V') and not hasattr(self.model, 'V1'):
            stats.update({
                'v_mean': float(np.mean(self.model.V)) if len(self.model.V) > 0 else None,
                'v_std': float(np.std(self.model.V)) if len(self.model.V) > 0 else None,
                'a_mean': float(np.mean(self.model.A)) if len(self.model.A) > 0 else None,
                'a_std': float(np.std(self.model.A)) if len(self.model.A) > 0 else None,
            })

        # Dueling Double Q
        if hasattr(self.model, 'V1'):
            stats.update({
                'v1_mean': float(np.mean(self.model.V1)),
                'v2_mean': float(np.mean(self.model.V2)),
                'a1_mean': float(np.mean(self.model.A1)),
                'a2_mean': float(np.mean(self.model.A2)),
            })

        return stats

    def _optimistic_init(self, value=0.5):
        """
        乐观初始化 Q 值（加速早期探索）

        不同架构的初始化策略：
        - Standard/Double Q: Q 表初始化为正值
        - Dueling 架构：V 初始化为正值，A 初始化为 0（优势是相对值）
        """
        # Standard Q
        if hasattr(self.model, 'Q'):
            self.model.Q.fill(value)

        # Double Q
        if hasattr(self.model, 'Q1'):
            self.model.Q1.fill(value)
            self.model.Q2.fill(value)

        # Dueling Q: V 初始化为正值，A 初始化为 0
        if hasattr(self.model, 'V') and not hasattr(self.model, 'V1'):
            self.model.V.fill(value)
            self.model.A.fill(0.0)

        # Dueling Double Q: V1/V2 初始化为正值，A1/A2 初始化为 0
        if hasattr(self.model, 'V1'):
            self.model.V1.fill(value)
            self.model.A1.fill(0.0)
            self.model.V2.fill(value)
            self.model.A2.fill(0.0)

    def _decay_epsilon(self, episode):
        """自适应ε衰减"""
        return self.model.epsilon_init - \
               (self.model.epsilon_init - self.model.epsilon_min) * \
               (episode / self.model.n_episodes)

    def _random_valid_state(self):
        """随机生成有效起始状态"""
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

    def _select_action(self, state_idx, epsilon):
        """ε-greedy 动作选择"""
        return self.model.select_action(state_idx, epsilon)


def create_q_trainer(model, use_feature_cache=True):
    """
    工厂函数：创建 Q-Learning 训练器

    Parameters:
    -----------
    model : BaseQLearning
        Q-Learning 模型实例
    use_feature_cache : bool
        是否使用特征缓存

    Returns:
    --------
    SimpleQTrainer
        训练器实例
    """
    return SimpleQTrainer(model, use_feature_cache=use_feature_cache)
