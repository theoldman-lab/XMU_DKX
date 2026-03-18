"""
Q-Learning 模型管理器

负责创建和管理 Q-Learning 模型实例
与实验流程解耦，便于独立调试和维护

使用方法：
---------
from src.experiments.q_models import create_ql_model

# 创建模型
model = create_ql_model('double', n_episodes=100)

# 训练
optimal_window = model.train(eeg_data, labels, verbose=True)
"""

from src.methods.q_learning_models.model_factory import QLearningModelFactory
from src.methods.q_learning_models.trainer_simple import create_q_trainer
from src.methods.q_learning_models.trainer_dqn import create_dqn_trainer
from src.utils.config import Config


# 模型类型映射
QL_METHOD_TYPE_MAP = {
    'Dueling_Double_Q': 'dueling_double',
    'Standard_Q': 'standard',
    'Double_Q': 'double',
    'Dueling_Q': 'dueling',
    'DQN': 'dqn',
}


def get_default_params(model_type: str) -> dict:
    """
    获取模型默认参数
    
    Parameters:
    -----------
    model_type : str
        模型类型 ('standard', 'double', 'dueling', 'dueling_double', 'dqn')
    
    Returns:
    --------
    dict : 默认参数字典
    """
    if model_type == 'dqn':
        return {
            'gamma': Config.DQN_PARAMS['gamma'],
            'n_episodes': Config.DQN_PARAMS['n_episodes'],
            'max_steps_per_episode': Config.DQN_PARAMS['max_steps_per_episode'],
            'epsilon_init': Config.DQN_PARAMS['epsilon_init'],
            'epsilon_min': Config.DQN_PARAMS['epsilon_min'],
            'alpha': Config.DQN_PARAMS.get('alpha', 0.1),
            'lr': Config.DQN_PARAMS['lr'],
            'buffer_size': Config.DQN_PARAMS['buffer_size'],
            'batch_size': Config.DQN_PARAMS['batch_size'],
            'tau': Config.DQN_PARAMS['tau'],
            'step_size': Config.DQN_PARAMS['step_size'],
            't_start_min': Config.DQN_PARAMS['t_start_min'],
            't_start_max': Config.DQN_PARAMS['t_start_max'],
            't_end_min': Config.DQN_PARAMS['t_end_min'],
            't_end_max': Config.DQN_PARAMS['t_end_max'],
            'min_window_len': Config.DQN_PARAMS['min_window_len'],
            'n_components': Config.DQN_PARAMS['n_components'],
        }
    else:
        return {
            'gamma': Config.QLEARNING_PARAMS['gamma'],
            'n_episodes': Config.QLEARNING_PARAMS['n_episodes'],
            'max_steps_per_episode': Config.QLEARNING_PARAMS['max_steps_per_episode'],
            'epsilon_init': Config.QLEARNING_PARAMS['epsilon_init'],
            'epsilon_min': Config.QLEARNING_PARAMS['epsilon_min'],
            'epsilon_decay': Config.QLEARNING_PARAMS['epsilon_decay'],
            'alpha': Config.QLEARNING_PARAMS['alpha'],
            'alpha_v': Config.QLEARNING_PARAMS['alpha_v'],
            'alpha_a': Config.QLEARNING_PARAMS['alpha_a'],
            'step_size': Config.QLEARNING_PARAMS['step_size'],
            't_start_min': Config.QLEARNING_PARAMS['t_start_min'],
            't_start_max': Config.QLEARNING_PARAMS['t_start_max'],
            't_end_min': Config.QLEARNING_PARAMS['t_end_min'],
            't_end_max': Config.QLEARNING_PARAMS['t_end_max'],
            'min_window_len': Config.QLEARNING_PARAMS['min_window_len'],
            'n_components': Config.QLEARNING_PARAMS['n_components'],
        }


def create_ql_model(model_type: str, **kwargs) -> object:
    """
    创建 Q-Learning 训练器
    
    Parameters:
    -----------
    model_type : str
        模型类型：'standard', 'double', 'dueling', 'dueling_double', 'dqn'
    **kwargs : dict
        额外参数，会覆盖默认参数
    
    Returns:
    --------
    object : QModelWrapper 实例
    """
    # 获取默认参数
    default_params = get_default_params(model_type)
    
    # 覆盖额外参数
    default_params.update(kwargs)
    
    # 创建模型
    model = QLearningModelFactory.create_model(model_type, **default_params)
    
    # 根据模型类型选择训练器
    if model_type == 'dqn':
        trainer = create_dqn_trainer(model, use_feature_cache=True)
    else:
        trainer = create_q_trainer(model, use_feature_cache=True)
    
    # 包装为统一接口
    class QModelWrapper:
        """
        Q-Learning 模型包装器
        
        提供统一的训练接口，隐藏模型内部细节
        """
        
        def __init__(self, model, trainer):
            self.model = model
            self.trainer = trainer
            self.training_logs = {}
            self.reward_history = []
        
        def train(self, eeg_data, labels, verbose=True):
            """
            训练模型
            
            Parameters:
            -----------
            eeg_data : array
                EEG 数据 (n_trials, n_channels, n_samples)
            labels : array
                标签 (n_trials,)
            verbose : bool
                是否显示训练进度
            
            Returns:
            --------
            tuple : 最优时间窗 (t_start, t_end)
            """
            optimal_window = self.trainer.train(
                eeg_data, labels, 
                verbose=verbose, 
                precompute=True  # 预计算特征，提升效率
            )
            self.training_logs = self.trainer.model.training_logs
            self.reward_history = self.training_logs.get('reward_history', [])
            return optimal_window
        
        def get_training_logs(self):
            """获取训练日志"""
            return self.training_logs
        
        def get_reward_history(self):
            """获取奖励历史"""
            return self.reward_history
    
    return QModelWrapper(model, trainer)


def get_available_methods() -> list:
    """
    获取可用的 Q-Learning 方法列表
    
    Returns:
    --------
    list : 方法名称列表
    """
    return list(QL_METHOD_TYPE_MAP.keys())


def is_ql_method(method_name: str) -> bool:
    """
    判断是否为 Q-Learning 方法
    
    Parameters:
    -----------
    method_name : str
        方法名称
    
    Returns:
    --------
    bool : 是否为 Q-Learning 方法
    """
    return method_name in QL_METHOD_TYPE_MAP


def get_model_type(method_name: str) -> str:
    """
    获取模型类型
    
    Parameters:
    -----------
    method_name : str
        方法名称
    
    Returns:
    --------
    str : 模型类型
    """
    return QL_METHOD_TYPE_MAP.get(method_name)
