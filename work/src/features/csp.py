"""
CSP 特征提取模块

功能：
1. 标准 CSP 特征提取
2. 支持多类别 CSP
3. 可选特征标准化
4. 正则化协方差估计
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


class CSP(BaseEstimator, TransformerMixin):
    """
    共空间模式 (Common Spatial Pattern) 特征提取器

    改进版：
    1. 修复 n_components 计算逻辑
    2. 增强数值稳定性
    3. 支持多类别 CSP
    4. 添加正则化协方差估计
    """
    def __init__(self, n_components=4, reg=1e-6, log_var=True):
        """
        :param n_components: CSP 滤波器数量（通常为偶数，如 2, 4, 6）
                           对于 4 类任务，建议使用 4 或 6
        :param reg: 协方差矩阵正则化参数
        :param log_var: 是否使用对数方差作为特征
        """
        self.n_components = n_components
        self.reg = reg
        self.log_var = log_var
        self.filters_ = None
        self.patterns_ = None

    def fit(self, X, y):
        """
        训练 CSP 滤波器
        :param X: EEG 数据，形状为 (n_trials, n_channels, n_times)
        :param y: 标签，形状为 (n_trials,)
        """
        X = np.copy(X)
        labels = np.unique(y)
        n_channels = X.shape[1]

        # 确保至少有 2 个类别
        if len(labels) < 2:
            raise ValueError(f"CSP 需要至少 2 个类别，但只找到{len(labels)}个")

        # 计算各类别的协方差矩阵
        covs = []
        for label in labels:
            x_class = X[y == label]
            if len(x_class) == 0:
                continue
            cov = np.zeros((n_channels, n_channels))
            for trial in x_class:
                trial_centered = trial - np.mean(trial, axis=1, keepdims=True)
                cov += np.dot(trial_centered, trial_centered.T)
            cov /= x_class.shape[0]
            covs.append(cov)

        if len(covs) < 2:
            raise ValueError(f"需要至少 2 个类别有数据")

        # 计算复合协方差矩阵
        C = sum(covs)

        # 正则化
        C_reg = C + self.reg * np.eye(n_channels)

        # 对协方差矩阵进行归一化（迹归一化）
        for i in range(len(covs)):
            trace_i = np.trace(covs[i])
            if trace_i > 0:
                covs[i] = covs[i] / trace_i

        # 多类别 CSP：使用近似对角化
        if len(covs) == 2:
            # 二分类
            C1 = covs[0] + self.reg * np.eye(n_channels)
            C2 = covs[1] + self.reg * np.eye(n_channels)
            try:
                eigen_values, eigen_vectors = eigh(C1, C1 + C2)
            except np.linalg.LinAlgError:
                eigen_values, eigen_vectors = self._whitening_method(C1, C2, self.reg)
        else:
            # 多类别
            eigen_values, eigen_vectors = self._approximate_diagonalization(covs, self.reg)

        # 按特征值排序
        idx = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, idx]

        # 选择滤波器（确保是偶数）
        n_comp = min(self.n_components, len(eigen_vectors) - 1)
        if n_comp % 2 == 1:
            n_comp -= 1

        n_pairs = n_comp // 2
        pick_idx = np.concatenate([
            np.arange(0, n_pairs),
            np.arange(len(idx) - n_pairs, len(idx))
        ])

        self.filters_ = eigen_vectors.T[pick_idx]
        self.patterns_ = np.linalg.pinv(eigen_vectors[:, pick_idx])

        return self

    def _whitening_method(self, C1, C2, reg):
        """白化方法求解广义特征值问题"""
        C = C1 + C2
        E, U = eigh(C)
        idx = np.argsort(E)[::-1]
        E = E[idx]
        U = U[:, idx]
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(E, reg)))
        W = D_inv_sqrt @ U.T
        C1_white = W @ C1 @ W.T
        E2, U2 = eigh(C1_white)
        eigen_vectors = U @ D_inv_sqrt @ U2
        return E2, eigen_vectors

    def _approximate_diagonalization(self, covs, reg):
        """近似对角化方法（用于多类别 CSP）"""
        n = len(covs)
        C_avg = sum(covs) / n
        E, U = eigh(C_avg)
        idx = np.argsort(E)[::-1]
        E = E[idx]
        U = U[:, idx]
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(E, reg)))
        W = D_inv_sqrt @ U.T
        covs_white = [W @ c @ W.T for c in covs]
        E2, U2 = eigh(covs_white[0])
        eigen_vectors = U @ D_inv_sqrt @ U2
        return E2, eigen_vectors

    def transform(self, X):
        """
        应用 CSP 滤波器提取特征
        :param X: EEG 数据，形状为 (n_trials, n_channels, n_times)
        :return: CSP 特征，形状为 (n_trials, n_components)
        """
        if self.filters_ is None:
            raise ValueError("CSP 未训练")

        features = []
        for trial in X:
            trial_filtered = np.dot(self.filters_, trial)
            variances = np.var(trial_filtered, axis=1)
            variances = np.clip(variances, 1e-10, None)
            
            if self.log_var:
                log_var = np.log(variances)
                features.append(log_var)
            else:
                features.append(variances)

        return np.array(features)


class CSPWithNormalization(BaseEstimator, TransformerMixin):
    """
    带标准化的 CSP 特征提取器
    
    将 CSP 特征提取和标准化整合到一个流水线中
    """
    
    def __init__(self, n_components=4, reg=1e-6, 
                 normalization_method='zscore',
                 use_normalization=True):
        """
        :param n_components: CSP 滤波器数量
        :param reg: 协方差矩阵正则化参数
        :param normalization_method: 标准化方法 (zscore, robust, minmax)
        :param use_normalization: 是否使用标准化
        """
        self.n_components = n_components
        self.reg = reg
        self.normalization_method = normalization_method
        self.use_normalization = use_normalization
        
        self.csp_ = CSP(n_components=n_components, reg=reg)
        self.scaler_ = None
    
    def fit(self, X, y):
        """
        训练 CSP 并拟合标准化器
        :param X: EEG 数据，形状为 (n_trials, n_channels, n_times)
        :param y: 标签
        """
        # 训练 CSP
        self.csp_.fit(X, y)
        
        # 提取特征
        features = self.csp_.transform(X)
        
        # 拟合标准化器
        if self.use_normalization:
            if self.normalization_method == 'zscore':
                self.scaler_ = StandardScaler()
            elif self.normalization_method == 'robust':
                self.scaler_ = RobustScaler()
            elif self.normalization_method == 'minmax':
                self.scaler_ = MinMaxScaler()
            else:
                raise ValueError(f"未知的标准化方法：{self.normalization_method}")
            
            self.scaler_.fit(features)
        
        return self
    
    def transform(self, X):
        """
        提取 CSP 特征并应用标准化
        :param X: EEG 数据
        :return: 标准化后的 CSP 特征
        """
        features = self.csp_.transform(X)
        
        if self.use_normalization and self.scaler_ is not None:
            features = self.scaler_.transform(features)
        
        return features


def extract_csp_features(windowed_data, labels_train, n_components=4):
    """
    从时间窗 EEG 数据中提取 CSP 特征（标准版，不带标准化）
    :param windowed_data: 时间窗 EEG 数据，形状为 (n_trials, n_channels, n_samples)
    :param labels_train: 训练标签
    :param n_components: CSP 分量数量
    :return: CSP 特征，形状为 (n_trials, n_components)
    """
    csp = CSP(n_components=n_components)
    features = csp.fit(windowed_data, labels_train).transform(windowed_data)
    return features


def extract_csp_features_with_normalization(windowed_data, labels_train, 
                                             n_components=4,
                                             normalization_method='zscore'):
    """
    从时间窗 EEG 数据中提取 CSP 特征（带标准化）
    :param windowed_data: 时间窗 EEG 数据
    :param labels_train: 训练标签
    :param n_components: CSP 分量数量
    :param normalization_method: 标准化方法
    :return: 标准化后的 CSP 特征
    """
    csp_norm = CSPWithNormalization(
        n_components=n_components,
        normalization_method=normalization_method,
        use_normalization=True
    )
    features = csp_norm.fit(windowed_data, labels_train).transform(windowed_data)
    return features
