"""
实验数据管理器（优化版）

设计原则：
1. 完整收集：训练期间的所有关键数据
2. 结构化存储：清晰的层次结构
3. 智能命名：便于后续分析和取用
4. 高效压缩：减少存储空间

文件命名规范：
    {method}_{subject}_{n_seeds}seeds_{timestamp}.expdata
    
目录结构：
results/
└── experiments/
    ├── 2026-03-08/                    # 日期分组
    │   ├── Standard_Q_S01_5seeds_163000.expdata
    │   ├── Double_Q_S01_5seeds_164500.expdata
    │   └── summary_2026-03-08.json    # 每日汇总
    └── aggregated/
        └── all_methods_comparison.csv
"""

import os
import json
import pickle
import zipfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd


def convert_numpy_types(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj


def generate_experiment_id(method: str, subject: str, n_seeds: int) -> str:
    """生成实验 ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{method}_{subject}_{n_seeds}seeds_{timestamp}"


def generate_filename(method: str, subject: str, n_seeds: int, timestamp: str = None) -> str:
    """生成标准化的文件名"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{method}_{subject}_{n_seeds}seeds_{timestamp}.expdata"


class ExperimentDataCollector:
    """
    实验数据收集器
    
    完整收集实验期间的所有关键数据
    """
    
    def __init__(self, method_name: str, subject: str, n_seeds: int):
        self.method_name = method_name
        self.subject = subject
        self.n_seeds = n_seeds
        self.start_time = datetime.now()
        
        # 核心数据
        self.seed_results = []
        self.training_logs = {
            'reward_history': [],
            'best_accuracy_history': [],
            'epsilon_history': [],
            'q_value_stats': [],
        }
        
        # 元数据
        self.metadata = {
            'method': method_name,
            'subject': subject,
            'n_seeds': n_seeds,
            'start_time': self.start_time.isoformat(),
            'config': self._get_config(),
        }
    
    def _get_config(self) -> Dict[str, Any]:
        """获取实验配置"""
        from src.utils.config import Config
        return {
            'eeg_params': Config.EEG_PARAMS,
            'csp_params': Config.CSP_PARAMS,
            'classifier_config': Config.CLASSIFIER_CONFIG,
            'qlearning_params': Config.QLEARNING_PARAMS,
        }
    
    def add_seed_result(self, 
                        seed: int,
                        optimal_window: Optional[tuple],
                        metrics: Dict[str, Any],
                        reward_history: List[float] = None,
                        training_logs: Dict[str, Any] = None,
                        success: bool = True,
                        error: str = None):
        """
        添加单个种子的实验结果
        
        Parameters:
        -----------
        seed : int
            随机种子编号
        optimal_window : tuple, optional
            最优时间窗 (t_start, t_end)
        metrics : dict
            性能指标 {accuracy, kappa, f1, cv_score, per_class_metrics}
        reward_history : list, optional
            训练奖励历史
        training_logs : dict, optional
            训练日志（Q 值统计、epsilon 历史等）
        success : bool
            是否成功完成
        error : str, optional
            错误信息（如果失败）
        """
        result = {
            'seed': seed,
            'optimal_window': list(optimal_window) if optimal_window else None,
            'metrics': convert_numpy_types(metrics),
            'reward_history': convert_numpy_types(reward_history or []),
            'training_logs': convert_numpy_types(training_logs or {}),
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat(),
        }
        self.seed_results.append(result)
        
        # 聚合训练日志
        if training_logs:
            if 'reward_history' in training_logs:
                self.training_logs['reward_history'].extend(training_logs['reward_history'])
            if 'best_accuracy_history' in training_logs:
                self.training_logs['best_accuracy_history'].extend(training_logs['best_accuracy_history'])
    
    def get_summary(self) -> Dict[str, Any]:
        """获取实验摘要"""
        successful = [r for r in self.seed_results if r.get('success')]
        
        if not successful:
            return {'success_rate': 0.0}
        
        accuracies = [r['metrics'].get('accuracy', 0) for r in successful]
        kappas = [r['metrics'].get('kappa', 0) for r in successful]
        f1s = [r['metrics'].get('f1', 0) for r in successful]
        
        windows = [r['optimal_window'] for r in successful if r['optimal_window']]
        t_starts = [w[0] for w in windows]
        t_ends = [w[1] for w in windows]
        
        return {
            'success_rate': len(successful) / self.n_seeds * 100,
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'kappa_mean': float(np.mean(kappas)),
            'kappa_std': float(np.std(kappas)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            't_start_mean': float(np.mean(t_starts)) if t_starts else None,
            't_end_mean': float(np.mean(t_ends)) if t_ends else None,
            'window_length_mean': float(np.mean([e-s for s,e in windows])) if windows else None,
        }
    
    def package(self) -> Dict[str, Any]:
        """打包完整数据"""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['duration_seconds'] = (
            datetime.fromisoformat(self.metadata['end_time']) - 
            self.start_time
        ).total_seconds()
        
        return {
            'version': '1.0',
            'metadata': self.metadata,
            'seed_results': self.seed_results,
            'training_logs': self.training_logs,
            'summary': self.get_summary(),
        }


class ExperimentDataManager:
    """
    实验数据管理器
    
    负责保存、加载和聚合实验数据
    """
    
    def __init__(self, base_dir: str = 'results/experiments'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 按日期分组
        self.today_dir = self.base_dir / datetime.now().strftime('%Y-%m-%d')
        self.today_dir.mkdir(exist_ok=True)
        
        self.aggregated_dir = self.base_dir / 'aggregated'
        self.aggregated_dir.mkdir(exist_ok=True)
    
    def save(self, collector: ExperimentDataCollector, 
             compress: bool = True) -> str:
        """
        保存实验数据
        
        Parameters:
        -----------
        collector : ExperimentDataCollector
            数据收集器
        compress : bool
            是否压缩保存（推荐 True）
        
        Returns:
        --------
        str : 保存的文件路径
        """
        data = collector.package()
        filename = generate_filename(
            collector.method_name, 
            collector.subject, 
            collector.n_seeds
        )
        
        if compress:
            # 保存为压缩文件
            filepath = self.today_dir / (filename + '.zip')
            temp_json = self.today_dir / filename
            
            # 先保存 JSON
            with open(temp_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # 压缩
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(temp_json, filename)
            
            # 清理临时文件
            temp_json.unlink()
        else:
            # 保存为 JSON
            filepath = self.today_dir / (filename + '.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # 更新每日汇总
        self._update_daily_summary()
        
        return str(filepath)
    
    def _update_daily_summary(self):
        """更新每日汇总"""
        summary_file = self.today_dir / 'summary.json'
        
        # 加载现有汇总
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {'date': datetime.now().strftime('%Y-%m-%d'), 'experiments': []}
        
        # 扫描所有实验文件
        experiments = []
        for f in self.today_dir.glob('*.expdata.zip'):
            try:
                with zipfile.ZipFile(f, 'r') as zipf:
                    with zipf.open(f.stem) as jsonf:
                        data = json.load(jsonf)
                    experiments.append({
                        'file': f.name,
                        'method': data['metadata']['method'],
                        'subject': data['metadata']['subject'],
                        'n_seeds': data['metadata']['n_seeds'],
                        'accuracy': data['summary'].get('accuracy_mean'),
                        'timestamp': data['metadata']['end_time'],
                    })
            except Exception as e:
                print(f"读取失败 {f}: {e}")
        
        summary['experiments'] = experiments
        summary['total_experiments'] = len(experiments)
        summary['updated_at'] = datetime.now().isoformat()
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """加载实验数据"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zipf:
                with zipf.open(filepath.stem) as jsonf:
                    return json.load(jsonf)
        elif filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的文件格式：{filepath.suffix}")
    
    def load_all(self, date: str = None) -> List[Dict[str, Any]]:
        """
        加载所有实验数据
        
        Parameters:
        -----------
        date : str, optional
            日期（YYYY-MM-DD），默认加载今天的所有数据
        
        Returns:
        --------
        list : 实验数据列表
        """
        if date:
            target_dir = self.base_dir / date
        else:
            target_dir = self.today_dir
        
        experiments = []
        for f in target_dir.glob('*.expdata.zip'):
            try:
                data = self.load(f)
                experiments.append(data)
            except Exception as e:
                print(f"加载失败 {f}: {e}")
        
        print(f"✓ 已加载 {len(experiments)} 个实验数据")
        return experiments
    
    def aggregate_all(self) -> pd.DataFrame:
        """
        聚合所有实验数据
        
        Returns:
        --------
        pd.DataFrame : 聚合结果 DataFrame
        """
        all_experiments = []
        
        # 扫描所有日期目录
        for date_dir in sorted(self.base_dir.iterdir()):
            if not date_dir.is_dir() or date_dir.name == 'aggregated':
                continue
            
            for exp_file in date_dir.glob('*.expdata.zip'):
                try:
                    data = self.load(exp_file)
                    metadata = data['metadata']
                    summary = data['summary']
                    
                    for seed_result in data['seed_results']:
                        metrics = seed_result['metrics']
                        window = seed_result['optimal_window']
                        
                        all_experiments.append({
                            'date': date_dir.name,
                            'method': metadata['method'],
                            'subject': metadata['subject'],
                            'seed': seed_result['seed'],
                            'accuracy': metrics.get('accuracy'),
                            'kappa': metrics.get('kappa'),
                            'f1': metrics.get('f1'),
                            'cv_score': metrics.get('cv_score'),
                            't_start': window[0] if window else None,
                            't_end': window[1] if window else None,
                            'window_length': (window[1] - window[0]) if window and window[0] and window[1] else None,
                            'success': seed_result['success'],
                        })
                except Exception as e:
                    print(f"聚合失败 {exp_file}: {e}")
        
        df = pd.DataFrame(all_experiments)
        
        # 保存聚合结果
        if len(df) > 0:
            output_file = self.aggregated_dir / 'all_experiments.csv'
            df.to_csv(output_file, index=False)
            print(f"✓ 聚合结果保存到：{output_file}")
            
            # 生成方法对比表
            comparison = df.groupby('method').agg({
                'accuracy': ['mean', 'std'],
                'kappa': ['mean', 'std'],
                'f1': ['mean', 'std'],
                't_start': ['mean', 'std'],
                't_end': ['mean', 'std'],
            }).round(4)
            comparison.columns = ['_'.join(col).strip() for col in comparison.columns]
            comparison_file = self.aggregated_dir / 'method_comparison.csv'
            comparison.reset_index().to_csv(comparison_file, index=False)
            print(f"✓ 方法对比保存到：{comparison_file}")
        
        return df


def create_collector(method: str, subject: str, n_seeds: int) -> ExperimentDataCollector:
    """创建数据收集器"""
    return ExperimentDataCollector(method, subject, n_seeds)


def create_manager(base_dir: str = 'results/experiments') -> ExperimentDataManager:
    """创建数据管理器"""
    return ExperimentDataManager(base_dir)
