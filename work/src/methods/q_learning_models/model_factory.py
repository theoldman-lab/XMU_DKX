"""
Q-Learning 模型工厂
提供统一的模型创建接口
"""
from src.methods.q_learning_models.standard_q_learning import StandardQLearning
from src.methods.q_learning_models.double_q_learning import DoubleQLearning
from src.methods.q_learning_models.dueling_q_learning import DuelingQLearning
from src.methods.q_learning_models.dueling_double_q_learning import DuelingDoubleQLearning
from src.methods.q_learning_models.dqn_model import DQNModel


class QLearningModelFactory:
    """Q-Learning 模型工厂类"""
    
    # 模型类型映射
    MODEL_TYPES = {
        'standard': StandardQLearning,
        'double': DoubleQLearning,
        'dueling': DuelingQLearning,
        'dueling_double': DuelingDoubleQLearning,
        'dqn': DQNModel
    }
    
    @classmethod
    def create_model(cls, model_type, **kwargs):
        """
        创建 Q-Learning 模型实例
        
        Parameters:
        -----------
        model_type : str
            模型类型：'standard', 'double', 'dueling', 'dueling_double', 'dqn'
        **kwargs : dict
            模型参数
            
        Returns:
        --------
        BaseQLearning
            Q-Learning 模型实例
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(f"未知的模型类型：{model_type}")
        
        model_class = cls.MODEL_TYPES[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_types(cls):
        """获取可用的模型类型列表"""
        return list(cls.MODEL_TYPES.keys())
