"""
实验结果评估模块

用于评估时间窗优化方法在测试集上的性能
与训练模块完全解耦，可独立使用

功能：
1. 评估单个时间窗的性能
2. 汇总多个种子的评估结果
3. 生成详细的性能指标报告

使用方法：
---------
# 方式 1：作为工具函数调用
from src.experiments.evaluator import evaluate_time_window, aggregate_results

metrics = evaluate_time_window(
    optimal_window=(2.0, 4.0),
    eeg_train=eeg_train,
    labels_train=labels_train,
    eeg_test=eeg_test,
    labels_test=labels_test
)

# 方式 2：使用评估器类
evaluator = Evaluator(eeg_train, labels_train, eeg_test, labels_test)
metrics = evaluator.evaluate(optimal_window=(2.0, 4.0))
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report

from src.data.preprocessing import extract_time_window
from src.features.csp import CSP
from src.features.classifier import StandardSVMClassifier, create_svm_classifier
from src.utils.config import Config


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    accuracy: float
    kappa: float
    f1: float
    cv_score: float
    t_start: float
    t_end: float
    window_length: float
    per_class_metrics: Dict[str, Dict[str, float]]
    classifier_params: Dict[str, Any]
    use_normalization: bool


class Evaluator:
    """
    评估器类
    
    提供统一的评估接口，支持批量评估和结果汇总
    """
    
    def __init__(self,
                 eeg_train: np.ndarray,
                 labels_train: np.ndarray,
                 eeg_test: np.ndarray,
                 labels_test: np.ndarray,
                 use_normalization: bool = None,
                 cv_folds: int = None):
        """
        初始化评估器
        
        Parameters:
        -----------
        eeg_train : np.ndarray
            训练集 EEG 数据 (n_trials, n_channels, n_samples)
        labels_train : np.ndarray
            训练集标签
        eeg_test : np.ndarray
            测试集 EEG 数据
        labels_test : np.ndarray
            测试集标签
        use_normalization : bool, optional
            是否使用标准化，默认从 Config 读取
        cv_folds : int, optional
            交叉验证折数，默认从 Config 读取
        """
        self.eeg_train = eeg_train
        self.labels_train = labels_train
        self.eeg_test = eeg_test
        self.labels_test = labels_test
        
        self.use_normalization = use_normalization if use_normalization is not None \
            else Config.CLASSIFIER_CONFIG['use_normalization']
        self.cv_folds = cv_folds if cv_folds is not None \
            else Config.CV_PARAMS['cv_folds']
        
        # 缓存 CSP 变换器
        self._csp = None
    
    def evaluate(self,
                 optimal_window: Tuple[float, float],
                 verbose: bool = False) -> EvaluationResult:
        """
        评估给定时间窗的性能
        
        Parameters:
        -----------
        optimal_window : tuple
            时间窗 (t_start, t_end)
        verbose : bool
            是否打印详细信息
        
        Returns:
        --------
        EvaluationResult : 评估结果
        """
        if optimal_window is None:
            raise ValueError("optimal_window 不能为 None")
        
        t_start, t_end = optimal_window
        
        if verbose:
            print(f"评估时间窗：[{t_start:.2f}, {t_end:.2f}] 秒")
        
        # 1. 截取时间窗
        eeg_train_windowed = extract_time_window(self.eeg_train, t_start, t_end)
        eeg_test_windowed = extract_time_window(self.eeg_test, t_start, t_end)
        
        # 2. CSP 特征提取
        csp = CSP(n_components=Config.CSP_PARAMS['n_components'])
        csp.fit(eeg_train_windowed, self.labels_train)
        
        csp_features_train = csp.transform(eeg_train_windowed)
        csp_features_test = csp.transform(eeg_test_windowed)
        
        if verbose:
            print(f"CSP 特征维度：{csp_features_train.shape}")
        
        # 3. 训练 SVM 分类器
        classifier = create_svm_classifier(
            use_normalization=self.use_normalization,
            cv_folds=self.cv_folds
        )
        classifier.fit(csp_features_train, self.labels_train, verbose=False)
        
        # 4. 预测测试集
        pred = classifier.predict(csp_features_test)
        
        # 5. 计算评估指标
        acc = accuracy_score(self.labels_test, pred)
        kappa = cohen_kappa_score(self.labels_test, pred)
        f1 = f1_score(self.labels_test, pred, average='macro')
        
        # 6. 生成详细分类报告
        class_names = np.unique(self.labels_train)
        report = classification_report(
            self.labels_test, pred,
            labels=class_names,
            output_dict=True,
            zero_division=0
        )
        
        per_class_metrics = {}
        for cls_name in class_names:
            if str(cls_name) in report:
                per_class_metrics[cls_name] = {
                    'precision': report[str(cls_name)]['precision'],
                    'recall': report[str(cls_name)]['recall'],
                    'f1-score': report[str(cls_name)]['f1-score'],
                    'support': int(report[str(cls_name)]['support'])
                }
        
        return EvaluationResult(
            accuracy=float(acc),
            kappa=float(kappa),
            f1=float(f1),
            cv_score=float(classifier.get_cv_score()),
            t_start=float(t_start),
            t_end=float(t_end),
            window_length=float(t_end - t_start),
            per_class_metrics=per_class_metrics,
            classifier_params=classifier.get_params(),
            use_normalization=self.use_normalization
        )
    
    def evaluate_multiple_windows(self,
                                   windows: List[Tuple[float, float]],
                                   verbose: bool = False) -> List[EvaluationResult]:
        """
        评估多个时间窗的性能
        
        Parameters:
        -----------
        windows : list
            时间窗列表 [(t_start1, t_end1), (t_start2, t_end2), ...]
        verbose : bool
            是否打印详细信息
        
        Returns:
        --------
        list : 评估结果列表
        """
        results = []
        for window in windows:
            result = self.evaluate(window, verbose=verbose)
            results.append(result)
        return results
    
    def compare_windows(self,
                        windows: List[Tuple[float, float]],
                        verbose: bool = False) -> Dict[str, Any]:
        """
        比较多个时间窗的性能
        
        Parameters:
        -----------
        windows : list
            时间窗列表
        verbose : bool
            是否打印详细信息
        
        Returns:
        --------
        dict : 比较结果
        """
        results = self.evaluate_multiple_windows(windows, verbose=verbose)
        
        # 找到最佳时间窗
        best_result = max(results, key=lambda r: r.accuracy)
        
        # 计算统计信息
        accuracies = [r.accuracy for r in results]
        
        comparison = {
            'best_window': (best_result.t_start, best_result.t_end),
            'best_accuracy': best_result.accuracy,
            'best_result': best_result,
            'all_results': results,
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'accuracy_min': float(np.min(accuracies)),
            'accuracy_max': float(np.max(accuracies)),
        }
        
        if verbose:
            print(f"\n最佳时间窗：[{best_result.t_start:.2f}, {best_result.t_end:.2f}]")
            print(f"最佳准确率：{best_result.accuracy:.4f}")
            print(f"平均准确率：{comparison['accuracy_mean']:.4f} ± {comparison['accuracy_std']:.4f}")
        
        return comparison


def evaluate_time_window(optimal_window: Tuple[float, float],
                         eeg_train: np.ndarray,
                         labels_train: np.ndarray,
                         eeg_test: np.ndarray,
                         labels_test: np.ndarray,
                         use_normalization: bool = None,
                         classifier_type: str = 'svm',  # 保留参数，向后兼容
                         verbose: bool = False) -> Dict[str, Any]:
    """
    评估最优时间窗在测试集上的性能（函数接口，保持向后兼容）
    
    Parameters:
    -----------
    optimal_window : tuple
        时间窗 (t_start, t_end)
    eeg_train : np.ndarray
        训练集 EEG 数据
    labels_train : np.ndarray
        训练集标签
    eeg_test : np.ndarray
        测试集 EEG 数据
    labels_test : np.ndarray
        测试集标签
    use_normalization : bool, optional
        是否使用标准化
    classifier_type : str, optional
        分类器类型（保留参数，仅支持 'svm'）
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    dict : 性能指标字典
    """
    # classifier_type 参数保留用于向后兼容，但仅支持 'svm'
    if classifier_type != 'svm':
        raise ValueError(f"不支持的分类器类型：{classifier_type}，仅支持 'svm'")
    
    evaluator = Evaluator(
        eeg_train=eeg_train,
        labels_train=labels_train,
        eeg_test=eeg_test,
        labels_test=labels_test,
        use_normalization=use_normalization
    )
    
    result = evaluator.evaluate(optimal_window, verbose=verbose)
    
    return {
        "accuracy": result.accuracy,
        "kappa": result.kappa,
        "f1": result.f1,
        "cv_score": result.cv_score,
        "classifier_params": result.classifier_params,
        "use_normalization": result.use_normalization,
        "per_class_metrics": result.per_class_metrics,
        "t_start": result.t_start,
        "t_end": result.t_end,
        "window_length": result.window_length,
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    汇总多个评估结果
    
    Parameters:
    -----------
    results : list
        评估结果列表
    
    Returns:
    --------
    dict : 汇总结果
    """
    if not results:
        return {}
    
    accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
    kappas = [r['kappa'] for r in results if 'kappa' in r]
    f1_scores_list = [r['f1'] for r in results if 'f1' in r]
    
    aggregated = {
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'kappa_mean': float(np.mean(kappas)),
        'kappa_std': float(np.std(kappas)),
        'f1_mean': float(np.mean(f1_scores_list)),
        'f1_std': float(np.std(f1_scores_list)),
        'all_results': results
    }
    
    return aggregated


def evaluate_multiple_seeds(evaluator: Evaluator,
                            get_optimal_window_func,
                            n_seeds: int = 5,
                            verbose: bool = False) -> List[Dict[str, Any]]:
    """
    在多个随机种子下评估模型性能
    
    Parameters:
    -----------
    evaluator : Evaluator
        评估器实例
    get_optimal_window_func : callable
        获取最优时间窗的函数
    n_seeds : int
        随机种子数量
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    list : 多个种子下的性能指标列表
    """
    results = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Seed {seed}")
            print(f"{'='*50}")
        
        np.random.seed(seed)
        
        # 获取最优时间窗
        optimal_window = get_optimal_window_func()
        
        # 评估性能
        result = evaluator.evaluate(optimal_window, verbose=verbose)
        
        metrics = {
            'seed': seed,
            'optimal_window': (result.t_start, result.t_end),
            'accuracy': result.accuracy,
            'kappa': result.kappa,
            'f1': result.f1,
            'cv_score': result.cv_score,
        }
        
        results.append(metrics)
    
    return results
