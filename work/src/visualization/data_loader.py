"""
实验数据加载器

负责从 experiments 目录加载和解析实验数据
"""

import json
import zipfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class SeedResult:
    """单个种子的实验结果"""
    seed: int
    subject: str
    method: str
    
    # 性能指标
    accuracy: float
    kappa: float
    f1: float
    cv_score: float
    
    # 时间窗
    t_start: Optional[float]
    t_end: Optional[float]
    window_length: Optional[float]
    
    # 训练历史
    reward_history: List[float] = field(default_factory=list)
    best_accuracy_history: List[float] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    
    # 元数据
    timestamp: str = ''
    success: bool = True
    error: Optional[str] = None


@dataclass
class MethodSummary:
    """方法汇总统计"""
    method: str
    category: str
    
    # 性能指标统计
    accuracy_mean: float
    accuracy_std: float
    accuracy_min: float
    accuracy_max: float
    
    kappa_mean: float
    kappa_std: float
    
    f1_mean: float
    f1_std: float
    
    # 时间窗统计
    t_start_mean: float
    t_start_std: float
    t_end_mean: float
    t_end_std: float
    window_length_mean: float
    window_length_std: float
    
    # 原始数据
    seed_results: List[SeedResult] = field(default_factory=list)
    all_accuracies: List[float] = field(default_factory=list)
    
    @property
    def n_seeds(self) -> int:
        return len(self.seed_results)
    
    @property
    def n_subjects(self) -> int:
        subjects = set()
        for r in self.seed_results:
            subjects.add(r.subject)
        return len(subjects)


class ExperimentDataLoader:
    """
    实验数据加载器
    
    从 experiments 目录加载 .expdata.zip 格式的实验数据
    """
    
    # 方法类别定义
    METHOD_CATEGORIES = {
        'baseline': ['FL-2s', 'FL-1.5s', 'FL-1s', 'Grid_Search', 'Random_Search'],
        'q_learning': ['Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q'],
        'dqn': ['DQN'],
    }
    
    def __init__(self, experiments_dir: str):
        self.experiments_dir = Path(experiments_dir)
        
        # 存储加载的数据
        self.seed_results: List[SeedResult] = []
        self.method_summaries: Dict[str, MethodSummary] = {}
        self.results_df: Optional[pd.DataFrame] = None
        
        # 按 subject 分组的数据
        self.subject_data: Dict[str, List[SeedResult]] = {}
    
    def load_all(self, aggregate_by_method: bool = True) -> int:
        """
        加载所有实验数据
        
        Parameters:
        -----------
        aggregate_by_method : bool
            是否按方法聚合（每个方法只保留最新文件）
        
        Returns:
        --------
        int : 加载的种子总数
        """
        if not self.experiments_dir.exists():
            raise FileNotFoundError(f"实验目录不存在：{self.experiments_dir}")
        
        # 收集所有实验文件
        exp_files = self._scan_experiment_files(aggregate_by_method)
        
        # 加载每个文件
        total_seeds = 0
        for exp_file in exp_files:
            seeds = self._load_expdata_file(exp_file)
            total_seeds += len(seeds)
        
        # 构建 DataFrame
        self._build_dataframe()
        
        # 计算汇总统计
        if aggregate_by_method:
            self._compute_summaries()
        
        return total_seeds
    
    def _scan_experiment_files(self, aggregate_by_method: bool) -> List[Path]:
        """
        扫描实验文件
        
        Returns:
        --------
        List[Path] : 实验文件列表
        """
        method_subject_files: Dict[Tuple[str, str], List[Path]] = {}
        
        for date_dir in sorted(self.experiments_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name == 'aggregated':
                continue
            
            for exp_file in date_dir.glob('*.expdata.zip'):
                filename = exp_file.name
                parsed = self._parse_filename(filename)
                
                if parsed is None:
                    continue
                
                method, subject = parsed
                key = (method, subject)
                
                if key not in method_subject_files:
                    method_subject_files[key] = []
                method_subject_files[key].append(exp_file)
        
        # 选择文件
        selected_files = []
        for (method, subject), files in method_subject_files.items():
            if aggregate_by_method:
                # 按文件名排序，选择最新的
                files.sort(key=lambda f: f.name, reverse=True)
                selected_files.append(files[0])
            else:
                selected_files.extend(files)
        
        return selected_files
    
    def _parse_filename(self, filename: str) -> Optional[Tuple[str, str]]:
        """
        解析文件名
        
        格式：{method}_{subject}_{n_seeds}_{timestamp}.expdata.zip
        例如：Standard_Q_S01_5seeds_20260308_195601.expdata.zip
        
        Returns:
        --------
        Optional[Tuple[str, str]] : (method, subject) 或 None
        """
        name = filename.replace('.expdata.zip', '')
        parts = name.split('_')
        
        if len(parts) < 4:
            return None
        
        # 查找 subject (Sxx 格式)
        subject = None
        method_end_idx = 0
        
        for i, part in enumerate(parts):
            if part.startswith('S') and part[1:].isdigit():
                subject = part
                method_end_idx = i
                break
        
        if subject is None:
            return None
        
        method = '_'.join(parts[:method_end_idx])
        return (method, subject)
    
    def _load_expdata_file(self, filepath: Path) -> List[SeedResult]:
        """
        加载单个 .expdata.zip 文件
        
        Returns:
        --------
        List[SeedResult] : 种子结果列表
        """
        seeds = []
        
        with zipfile.ZipFile(filepath, 'r') as zipf:
            json_filename = filepath.stem
            
            with zipf.open(json_filename) as jsonf:
                data = json.load(jsonf)
            
            metadata = data.get('metadata', {})
            method = metadata.get('method', 'Unknown')
            seed_results = data.get('seed_results', [])
            training_logs = data.get('training_logs', {})
            
            for seed_result in seed_results:
                metrics = seed_result.get('metrics', {})
                optimal_window = seed_result.get('optimal_window')
                
                # 计算时间窗长度
                window_length = None
                if optimal_window and len(optimal_window) == 2:
                    if optimal_window[0] is not None and optimal_window[1] is not None:
                        window_length = optimal_window[1] - optimal_window[0]
                
                result = SeedResult(
                    seed=seed_result.get('seed', 0),
                    subject=metadata.get('subject', 'S01'),
                    method=method,
                    accuracy=metrics.get('accuracy', 0),
                    kappa=metrics.get('kappa', 0),
                    f1=metrics.get('f1', 0),
                    cv_score=metrics.get('cv_score', 0),
                    t_start=optimal_window[0] if optimal_window else None,
                    t_end=optimal_window[1] if optimal_window else None,
                    window_length=window_length,
                    reward_history=seed_result.get('reward_history', []),
                    best_accuracy_history=training_logs.get('best_accuracy_history', []),
                    epsilon_history=training_logs.get('epsilon_history', []),
                    timestamp=seed_result.get('timestamp', ''),
                    success=seed_result.get('success', True),
                    error=seed_result.get('error'),
                )
                
                seeds.append(result)
                self.seed_results.append(result)
        
        return seeds
    
    def _build_dataframe(self):
        """构建 pandas DataFrame"""
        rows = []
        
        for result in self.seed_results:
            rows.append({
                'method': result.method,
                'subject': result.subject,
                'seed': result.seed,
                'accuracy': result.accuracy,
                'kappa': result.kappa,
                'f1': result.f1,
                'cv_score': result.cv_score,
                't_start': result.t_start,
                't_end': result.t_end,
                'window_length': result.window_length,
                'success': result.success,
            })
        
        self.results_df = pd.DataFrame(rows)
        
        # 按 subject 分组
        self.subject_data = {}
        for subject in self.results_df['subject'].unique():
            subject_results = [r for r in self.seed_results if r.subject == subject]
            self.subject_data[subject] = subject_results
    
    def _compute_summaries(self):
        """计算每个方法的汇总统计"""
        if self.results_df is None:
            return
        
        for method in self.results_df['method'].unique():
            method_data = self.results_df[self.results_df['method'] == method]
            method_results = [r for r in self.seed_results if r.method == method]
            
            # 确定类别
            category = self._get_method_category(method)
            
            # 性能指标统计
            accuracies = method_data['accuracy'].values
            
            # 时间窗统计
            t_starts = method_data['t_start'].dropna()
            t_ends = method_data['t_end'].dropna()
            window_lengths = method_data['window_length'].dropna()
            
            summary = MethodSummary(
                method=method,
                category=category,
                accuracy_mean=float(np.mean(accuracies)),
                accuracy_std=float(np.std(accuracies)),
                accuracy_min=float(np.min(accuracies)),
                accuracy_max=float(np.max(accuracies)),
                kappa_mean=float(method_data['kappa'].mean()),
                kappa_std=float(method_data['kappa'].std()),
                f1_mean=float(method_data['f1'].mean()),
                f1_std=float(method_data['f1'].std()),
                t_start_mean=float(t_starts.mean()) if len(t_starts) > 0 else 0,
                t_start_std=float(t_starts.std()) if len(t_starts) > 0 else 0,
                t_end_mean=float(t_ends.mean()) if len(t_ends) > 0 else 0,
                t_end_std=float(t_ends.std()) if len(t_ends) > 0 else 0,
                window_length_mean=float(window_lengths.mean()) if len(window_lengths) > 0 else 0,
                window_length_std=float(window_lengths.std()) if len(window_lengths) > 0 else 0,
                seed_results=method_results,
                all_accuracies=accuracies.tolist(),
            )
            
            self.method_summaries[method] = summary
    
    def _get_method_category(self, method: str) -> str:
        """获取方法类别"""
        for category, methods in self.METHOD_CATEGORIES.items():
            if method in methods:
                return category
        return 'unknown'
    
    def get_method_names(self) -> List[str]:
        """获取所有方法名称"""
        return list(self.method_summaries.keys())
    
    def get_subjects(self) -> List[str]:
        """获取所有被试"""
        return list(self.subject_data.keys())
    
    def get_results_by_subject(self, subject: str) -> pd.DataFrame:
        """获取指定被试的结果"""
        if self.results_df is None:
            return pd.DataFrame()
        return self.results_df[self.results_df['subject'] == subject].copy()
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        生成方法对比表格

        Returns:
        --------
        pd.DataFrame : 对比表格
        """
        rows = []

        # 按固定顺序排列方法
        from .performance_analyzer import ALL_METHODS_ORDER
        
        ordered_methods = [m for m in ALL_METHODS_ORDER if m in self.method_summaries]
        
        for method in ordered_methods:
            summary = self.method_summaries[method]
            rows.append({
                'Method': method,
                'Category': summary.category,
                'Accuracy (%)': f'{summary.accuracy_mean * 100:.2f} ± {summary.accuracy_std * 100:.2f}',
                'Kappa': f'{summary.kappa_mean:.4f} ± {summary.kappa_std:.4f}',
                'F1': f'{summary.f1_mean:.4f} ± {summary.f1_std:.4f}',
                't_start (s)': f'{summary.t_start_mean:.2f} ± {summary.t_start_std:.2f}' if summary.t_start_mean > 0 else 'N/A',
                't_end (s)': f'{summary.t_end_mean:.2f} ± {summary.t_end_std:.2f}' if summary.t_end_mean > 0 else 'N/A',
                'Window (s)': f'{summary.window_length_mean:.2f} ± {summary.window_length_std:.2f}' if summary.window_length_mean > 0 else 'N/A',
                'Seeds': summary.n_seeds,
                'Subjects': summary.n_subjects,
            })

        return pd.DataFrame(rows)
