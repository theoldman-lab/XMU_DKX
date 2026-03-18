"""
分类器模块（简化版）

固定参数设计，避免对时间窗评估产生干扰：
- 固定使用线性核 SVM
- 固定 C 参数
- 固定使用 Z-score 标准化
"""

import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import UndefinedMetricWarning


class StandardSVMClassifier:
    """
    标准 SVM 分类器（固定参数）
    
    设计原则：
    - 固定核函数：linear（避免核函数选择干扰时间窗评估）
    - 固定 C 参数：1.0（标准设置）
    - 固定标准化：Z-score（消除特征量纲影响）
    - 不进行搜索：确保评估一致性
    """
    
    def __init__(self, C=1.0, use_normalization=True, cv_folds=5):
        """
        :param C: SVM 正则化参数（默认 1.0）
        :param use_normalization: 是否使用 Z-score 标准化（默认 True）
        :param cv_folds: 交叉验证折数（默认 5）
        """
        self.C = C
        self.use_normalization = use_normalization
        self.cv_folds = cv_folds
        
        self.normalizer = None
        self.svm_ = None
        self.cv_score_ = None
    
    def fit(self, X, y, verbose=False):
        """
        训练 SVM 分类器
        
        :param X: 特征矩阵 (n_samples, n_features)
        :param y: 标签向量 (n_samples,)
        :param verbose: 是否打印信息
        :return: self
        """
        # 1. Z-score 标准化
        if self.use_normalization:
            self.normalizer = StandardScaler()
            X_normalized = self.normalizer.fit_transform(X)
            if verbose:
                print("已应用 Z-score 标准化")
        else:
            X_normalized = X
        
        # 2. 交叉验证评估
        self.svm_ = SVC(kernel='linear', C=self.C, random_state=0)
        # 过滤 UndefinedMetricWarning 警告（某些 fold 中类别样本太少导致）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            scores = cross_val_score(
                self.svm_, X_normalized, y,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1
            )
        self.cv_score_ = np.mean(scores)
        
        if verbose:
            print(f"交叉验证准确率：{self.cv_score_:.4f} (+/- {np.std(scores):.4f})")
        
        # 3. 用全部数据训练最终模型
        self.svm_.fit(X_normalized, y)
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.svm_ is None:
            raise ValueError("模型未训练")
        
        if self.use_normalization and self.normalizer is not None:
            X = self.normalizer.transform(X)
        
        return self.svm_.predict(X)
    
    def get_cv_score(self):
        """获取交叉验证分数"""
        return self.cv_score_
    
    def get_params(self):
        """获取模型参数"""
        return {
            'kernel': 'linear',
            'C': self.C,
            'use_normalization': self.use_normalization
        }


def create_svm_classifier(use_normalization=True, cv_folds=5):
    """
    工厂函数：创建标准 SVM 分类器
    
    :param use_normalization: 是否使用标准化
    :param cv_folds: 交叉验证折数
    :return: StandardSVMClassifier 实例
    """
    return StandardSVMClassifier(
        C=1.0,
        use_normalization=use_normalization,
        cv_folds=cv_folds
    )
