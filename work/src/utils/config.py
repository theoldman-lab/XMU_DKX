"""
参数配置文件
统一管理项目中所有算法和实验的参数

所有配置参数都应在此文件中定义，便于调试和修改
"""


class Config:
    """全局参数配置类"""

    # ==================== EEG 数据参数 ====================
    EEG_PARAMS = {
        "sfreq": 250,  # 采样频率
        "n_channels": 22,  # 通道数
        "n_classes": 4,  # 类别数（运动想象任务数）
        "trial_duration": 4.0,  # 试次持续时间（秒）
        "low_freq": 8,  # 带通滤波低频截止
        "high_freq": 30,  # 带通滤波高频截止
        "notch_freq": 50,  # 陷波滤波频率
    }

    # ==================== Q-Learning 算法参数 ====================
    # 表格方法通用参数（Standard Q, Double Q, Dueling Q, Dueling Double Q）
    # 性能优化配置：增加训练、调整探索策略、提升收敛质量
    QLEARNING_PARAMS = {
        # 状态空间参数
        "t_start_min": 0.0,
        "t_start_max": 3.0,
        "t_end_min": 1.0,
        "t_end_max": 4.0,
        "step_size": 0.2,
        "min_window_len": 0.5,

        # 学习参数
        "gamma": 0.95,
        "alpha": 0.2,
        "alpha_v": 0.05,
        "alpha_a": 0.1,

        # 探索策略参数
        "epsilon_init": 0.9,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,

        # 训练参数
        "n_episodes": 100,
        "max_steps_per_episode": 40,

        # 奖励参数（当前未使用，仅保留接口）
        "lambda_efficiency": 0.0,  # 效率惩罚系数（0=不使用）
        "target_window_length": 2.0,  # 目标窗口长度（奖励塑形用，当前未使用）
        "use_prior_knowledge": False,  # 是否使用先验知识（当前未使用）

        # 特征参数
        "n_components": 4,
    }

    # DQN 参数（神经网络方法）
    DQN_PARAMS = {
        # 状态空间参数
        "t_start_min": 0.0,
        "t_start_max": 3.0,
        "t_end_min": 1.0,
        "t_end_max": 4.0,
        "step_size": 0.2,
        "min_window_len": 0.5,

        # 学习参数
        "gamma": 0.95,  # 折扣因子
        "lr": 1e-3,  # 学习率

        # 探索策略参数
        "epsilon_init": 0.9,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,

        # 训练参数
        "n_episodes": 100,
        "max_steps_per_episode": 40,

        # 经验回放参数
        "buffer_size": 10000,
        "batch_size": 32,
        "tau": 1e-3,  # 软更新参数
        "update_every": 4,  # 目标网络更新频率

        # 奖励参数
        "lambda_efficiency": 0.0,

        # 特征参数
        "n_components": 4,
    }

    # ==================== 实验参数 ====================
    EXPERIMENT_PARAMS = {
        "n_seeds": 5,  # 随机种子数量
        "demo_n_seeds": 2,  # 演示模式随机种子数量
        "demo_n_episodes": 50,  # 演示模式训练轮数
        "demo_max_steps_per_episode": 20,  # 演示模式每轮最大步数
        "demo_epsilon_decay": 0.95,  # 演示模式探索衰减

        # 受试者列表
        "subjects_list": [
            "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
        ],
        "demo_subjects_list": ["S01"],  # 演示模式受试者列表

        # 基线方法参数
        "grid_search_step": 0.2,  # 网格搜索步长
        "random_search_n": 50,  # 随机搜索次数
    }

    # ==================== 分类器配置（固定参数） ====================
    # 所有实验方法统一使用固定参数的 SVM 分类器
    # 避免分类器参数变化对时间窗评估产生干扰
    CLASSIFIER_CONFIG = {
        'kernel': 'linear',       # 固定线性核
        'C': 1.0,                 # 固定 C 参数
        'use_normalization': True, # 使用 Z-score 标准化
        'cv_folds': 5,            # 5 折交叉验证
    }

    # ==================== CSP 参数 ====================
    CSP_PARAMS = {
        "n_components": 4,  # CSP 分量数量
        "reg": 1e-6,  # 协方差矩阵正则化参数
        "log_var": True,  # 是否使用对数方差作为特征
    }

    # ==================== SVM 参数 ====================
    # 固定参数 SVM（用于评估）
    SVM_PARAMS = {
        "kernel": "linear",
        "C": 1.0,
        "random_state": 0,
    }

    # ==================== 交叉验证参数 ====================
    CV_PARAMS = {
        "cv_folds": 5,  # 交叉验证折数
    }

    # ==================== 统计分析参数 ====================
    STATISTICAL_PARAMS = {
        "alpha": 0.05,  # 显著性水平
        "alternative": "two-sided",  # 统计检验备择假设
    }

    # ==================== 可视化参数 ====================
    VISUALIZATION_PARAMS = {
        "figsize": (12, 8),  # 图表尺寸
        "dpi": 300,  # 图像分辨率
        "font_size": 12,  # 字体大小
        "colors": [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ],  # 颜色方案
    }

    # ==================== 参数获取方法 ====================
    @classmethod
    def get_qlearning_params(cls, demo_mode=False):
        """获取 Q-learning 参数，根据是否为演示模式调整"""
        params = cls.QLEARNING_PARAMS.copy()
        if demo_mode:
            params.update({
                "n_episodes": cls.EXPERIMENT_PARAMS["demo_n_episodes"],
                "max_steps_per_episode": cls.EXPERIMENT_PARAMS["demo_max_steps_per_episode"],
                "epsilon_decay": cls.EXPERIMENT_PARAMS["demo_epsilon_decay"],
            })
        return params

    @classmethod
    def get_dqn_params(cls, demo_mode=False):
        """获取 DQN 参数，根据是否为演示模式调整"""
        params = cls.DQN_PARAMS.copy()
        if demo_mode:
            params.update({
                "n_episodes": cls.EXPERIMENT_PARAMS["demo_n_episodes"],
                "max_steps_per_episode": cls.EXPERIMENT_PARAMS["demo_max_steps_per_episode"],
                "epsilon_decay": cls.EXPERIMENT_PARAMS["demo_epsilon_decay"],
            })
        return params

    @classmethod
    def get_experiment_params(cls, demo_mode=False):
        """获取实验参数，根据是否为演示模式调整"""
        params = cls.EXPERIMENT_PARAMS.copy()
        if demo_mode:
            params["n_seeds"] = cls.EXPERIMENT_PARAMS["demo_n_seeds"]
        return params

    @classmethod
    def get_classifier_config(cls):
        """获取分类器配置"""
        return cls.CLASSIFIER_CONFIG.copy()
