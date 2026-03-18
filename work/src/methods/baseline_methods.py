"""
基线方法模块（简化版）

包含：
1. 固定时间窗方法（长窗、短窗）
2. 网格搜索方法
3. 随机搜索方法

所有方法使用固定参数的 SVM 分类器（线性核，C=1.0）
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from src.data.preprocessing import extract_time_window
from src.features.csp import extract_csp_features
from src.features.classifier import StandardSVMClassifier
from src.utils.config import Config


def _evaluate_window(windowed_data, labels, use_normalization=None, cv_folds=None):
    """
    评估时间窗性能（内部辅助函数）

    :param windowed_data: 时间窗 EEG 数据
    :param labels: 标签
    :param use_normalization: 是否使用特征标准化（默认从 Config 读取）
    :param cv_folds: 交叉验证折数（默认从 Config 读取）
    :return: 交叉验证准确率
    """
    if use_normalization is None:
        use_normalization = Config.CLASSIFIER_CONFIG['use_normalization']
    if cv_folds is None:
        cv_folds = Config.CV_PARAMS['cv_folds']

    # CSP 特征提取
    csp_features = extract_csp_features(
        windowed_data,
        labels,
        n_components=Config.CSP_PARAMS['n_components']
    )

    # SVM 评估（固定参数）
    clf = StandardSVMClassifier(
        C=Config.SVM_PARAMS['C'],
        use_normalization=use_normalization,
        cv_folds=cv_folds
    )
    clf.fit(csp_features, labels, verbose=False)

    return clf.get_cv_score()



def fixed_window_group_experiment(eeg_train, labels_train,
                                  window_length=2.0,
                                  eeg_test=None, labels_test=None):
    """
    固定窗口长度多组实验方法
    
    在给定窗口长度下，遍历所有满足时间窗限制的组合，对每个窗口进行评估
    
    时间窗限制：
        - t_start 范围：[0.0, 3.0]
        - t_end 范围：[1.0, 4.0]
        - 窗口长度固定为 window_length
    
    :param eeg_train, labels_train: EEG 训练数据和标签
    :param window_length: 固定窗口长度（秒），支持 1.0, 1.5, 2.0
    :param eeg_test, labels_test: EEG 测试数据和标签（未使用）
    :return: (最优窗口，所有窗口评估结果列表)
             最优窗口格式：(t_start, t_end)
             评估结果列表格式：[{'window': (t_start, t_end), 'accuracy': acc}, ...]
    """
    # 时间窗限制
    t_start_min = 0.0
    t_start_max = 3.0
    t_end_min = 1.0
    t_end_max = 4.0
    
    # 生成所有满足窗口长度的时间窗组合
    step = 0.5
    t_start_values = np.arange(t_start_min, t_start_max + step/2, step)
    
    # 收集所有有效窗口
    valid_windows = []
    for t_start in t_start_values:
        t_end = t_start + window_length
        if t_start_min <= t_start <= t_start_max and t_end_min <= t_end <= t_end_max:
            valid_windows.append((t_start, t_end))
    
    # 评估每个窗口
    results = []
    best_accuracy = -1
    best_params = valid_windows[0] if valid_windows else (t_start_min, t_start_min + window_length)
    
    for t_start, t_end in valid_windows:
        try:
            windowed_data = extract_time_window(eeg_train, t_start, t_end)
            accuracy = _evaluate_window(windowed_data, labels_train)
            
            results.append({
                'window': (t_start, t_end),
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (t_start, t_end)
                
        except Exception:
            results.append({
                'window': (t_start, t_end),
                'accuracy': 0.0
            })
    
    return best_params, results




def fixed_2s_group_experiment(eeg_train, labels_train,
                              eeg_test=None, labels_test=None):
    """
    2s 窗口组实验方法
    
    窗口长度固定为 2.0s，遍历所有满足时间窗限制的组合：
    - [0.0, 2.0], [0.5, 2.5], [1.0, 3.0], [1.5, 3.5], [2.0, 4.0]
    
    :return: (最优窗口，所有窗口评估结果列表)
    """
    return fixed_window_group_experiment(eeg_train, labels_train, 
                                         window_length=2.0,
                                         eeg_test=eeg_test, labels_test=labels_test)




def fixed_1_5s_group_experiment(eeg_train, labels_train,
                                eeg_test=None, labels_test=None):
    """
    1.5s 窗口组实验方法
    
    窗口长度固定为 1.5s，遍历所有满足时间窗限制的组合：
    - [0.0, 1.5], [0.5, 2.0], [1.0, 2.5], [1.5, 3.0], [2.0, 3.5], [2.5, 4.0]
    
    :return: (最优窗口，所有窗口评估结果列表)
    """
    return fixed_window_group_experiment(eeg_train, labels_train, 
                                         window_length=1.5,
                                         eeg_test=eeg_test, labels_test=labels_test)




def fixed_1s_group_experiment(eeg_train, labels_train,
                              eeg_test=None, labels_test=None):
    """
    1s 窗口组实验方法
    
    窗口长度固定为 1.0s，遍历所有满足时间窗限制的组合：
    - [0.0, 1.0], [0.5, 1.5], [1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5], [3.0, 4.0]
    
    :return: (最优窗口，所有窗口评估结果列表)
    """
    return fixed_window_group_experiment(eeg_train, labels_train, 
                                         window_length=1.0,
                                         eeg_test=eeg_test, labels_test=labels_test)


def get_fixed_window_candidates(window_length=2.0):
    """
    获取固定窗口长度的所有候选窗口（不进行评估）
    
    用于批量实验，返回所有满足时间窗限制的窗口列表
    
    时间窗限制：
        - t_start 范围：[0.0, 3.0]
        - t_end 范围：[1.0, 4.0]
    
    :param window_length: 固定窗口长度（秒），支持 1.0, 1.5, 2.0
    :return: 候选窗口列表 [(t_start, t_end), ...]
    """
    t_start_min = 0.0
    t_start_max = 3.0
    t_end_min = 1.0
    t_end_max = 4.0
    
    step = 0.5
    t_start_values = np.arange(t_start_min, t_start_max + step/2, step)
    
    valid_windows = []
    for t_start in t_start_values:
        t_end = t_start + window_length
        if t_start_min <= t_start <= t_start_max and t_end_min <= t_end <= t_end_max:
            valid_windows.append((round(t_start, 1), round(t_end, 1)))
    
    return valid_windows


def grid_search_method(eeg_train, labels_train,
                       step=None,
                       return_search_log=False,
                       cv_folds=None,
                       use_normalization=True):
    """
    网格搜索方法 - 遍历所有可能的时间窗组合

    :param eeg_train, labels_train: EEG 训练数据和标签
    :param step: 搜索步长（默认从 Config 读取）
    :param return_search_log: 是否返回搜索日志
    :param cv_folds: 交叉验证折数（默认从 Config 读取）
    :param use_normalization: 是否使用特征标准化（默认 True）
    :return: 最优时间窗 (t_start, t_end)
    """
    if step is None:
        step = Config.EXPERIMENT_PARAMS['grid_search_step']
    if cv_folds is None:
        cv_folds = Config.CV_PARAMS['cv_folds']

    # 生成搜索网格
    t_start_range = np.arange(0.0, 3.1, step)
    t_end_range = np.arange(1.0, 4.1, step)

    best_accuracy = -1
    best_params = (0.0, 1.0)
    search_log = []

    # 网格搜索
    for t_start in t_start_range:
        for t_end in t_end_range:
            # 跳过无效窗口（长度<0.5 秒）
            if t_end - t_start < 0.5:
                continue

            try:
                # 提取时间窗
                windowed_data = extract_time_window(eeg_train, t_start, t_end)

                # 评估
                accuracy = _evaluate_window(
                    windowed_data, labels_train,
                    use_normalization=use_normalization,
                    cv_folds=cv_folds
                )

                # 记录
                search_log.append({
                    't_start': t_start,
                    't_end': t_end,
                    'accuracy': accuracy
                })

                # 更新最优
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (t_start, t_end)

            except Exception as e:
                search_log.append({
                    't_start': t_start,
                    't_end': t_end,
                    'accuracy': 0.0,
                    'error': str(e)
                })

    if return_search_log:
        return best_params, {
            'best_accuracy': best_accuracy,
            'best_window': best_params,
            'all_results': search_log
        }

    return best_params


def random_search_method(eeg_train, labels_train,
                         n_searches=None,
                         random_state=None,
                         return_search_log=False,
                         cv_folds=None,
                         use_normalization=True):
    """
    随机搜索方法 - 随机采样时间窗组合

    :param eeg_train, labels_train: EEG 训练数据和标签
    :param n_searches: 搜索次数（默认从 Config 读取）
    :param random_state: 随机种子
    :param return_search_log: 是否返回搜索日志
    :param cv_folds: 交叉验证折数（默认从 Config 读取）
    :param use_normalization: 是否使用特征标准化（默认 True）
    :return: 最优时间窗 (t_start, t_end)
    """
    if n_searches is None:
        n_searches = Config.EXPERIMENT_PARAMS['random_search_n']
    if cv_folds is None:
        cv_folds = Config.CV_PARAMS['cv_folds']
    if use_normalization is None:
        use_normalization = True

    if random_state is not None:
        np.random.seed(random_state)

    best_accuracy = -1
    best_params = (0.0, 1.0)
    search_log = []

    # 随机搜索
    for i in range(n_searches):
        try:
            # 随机生成时间窗
            t_start = np.random.uniform(0.0, 3.0)
            t_end = np.random.uniform(max(1.0, t_start + 0.5), 4.0)

            # 提取时间窗
            windowed_data = extract_time_window(eeg_train, t_start, t_end)

            # 评估
            accuracy = _evaluate_window(
                windowed_data, labels_train,
                use_normalization=use_normalization,
                cv_folds=cv_folds
            )

            # 记录
            search_log.append({
                'iteration': i,
                't_start': t_start,
                't_end': t_end,
                'accuracy': accuracy
            })

            # 更新最优
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (t_start, t_end)

        except Exception as e:
            search_log.append({
                'iteration': i,
                'accuracy': 0.0,
                'error': str(e)
            })

    if return_search_log:
        return best_params, {
            'best_accuracy': best_accuracy,
            'best_window': best_params,
            'all_results': search_log
        }

    return best_params
