"""
Q-Learning 模型包装器
提供统一的训练接口，屏蔽不同模型实现的差异
"""
import numpy as np
from drop.trainer import OptimizedQLearningTrainer


class QLearningModelWrapper:
    """
    Q-Learning 模型包装器
    
    提供统一的 train 接口，自动使用 OptimizedQLearningTrainer 进行训练
    屏蔽不同 Q-Learning 变体之间的差异
    """
    
    def __init__(self, model, use_feature_cache=True, precompute=True):
        """
        Parameters:
        -----------
        model : BaseQLearning
            Q-Learning 模型实例
        use_feature_cache : bool
            是否使用特征缓存
        precompute : bool
            是否预计算特征
        """
        self.model = model
        self.trainer = OptimizedQLearningTrainer(
            model, 
            use_feature_cache=use_feature_cache
        )
        self.precompute = precompute
        self.reward_history = []
        
    def train(self, eeg_data, labels, verbose=False):
        """
        训练模型（统一接口）
        
        Parameters:
        -----------
        eeg_data : array
            EEG 数据 (n_trials, n_channels, n_times)
        labels : array
            标签
        verbose : bool
            是否显示训练进度
            
        Returns:
        --------
        optimal_window : tuple
            最优时间窗 (t_start, t_end)
        """
        optimal_window = self.trainer.train(
            eeg_data, labels, 
            verbose=verbose, 
            precompute=self.precompute
        )
        
        # 同步训练日志到包装器
        self.reward_history = self.model.training_logs.get('reward_history', [])
        
        return optimal_window
    
    def get_training_logs(self):
        """获取训练日志"""
        return self.model.training_logs
    
    def __getattr__(self, name):
        """代理其他属性访问到内部模型"""
        return getattr(self.model, name)


def create_ql_model(model_type, **kwargs):
    """
    创建 Q-Learning 模型的工厂函数
    
    Parameters:
    -----------
    model_type : str
        模型类型：'standard', 'double', 'dueling', 'dueling_double', 'dqn'
    **kwargs : dict
        模型参数
        
    Returns:
    --------
    QLearningModelWrapper
        包装后的模型实例
    """
    from src.methods.q_learning_models.model_factory import QLearningModelFactory
    
    # 创建基础模型
    model = QLearningModelFactory.create_model(model_type, **kwargs)
    
    # 包装并返回
    return QLearningModelWrapper(model)
