"""
训练性能分析模块

用于分析和可视化 Q 方法训练期间的性能表现
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .data_loader import ExperimentDataLoader


# 方法显示名称
METHOD_DISPLAY_NAMES = {
    'Standard_Q': 'Standard Q-Learning',
    'Double_Q': 'Double Q-Learning',
    'Dueling_Q': 'Dueling Q-Learning',
    'Dueling_Double_Q': 'Dueling Double Q-Learning',
    'DQN': 'DQN',
}

# Q 方法专属颜色（固定顺序，确保所有图表一致）
Q_METHOD_COLORS = {
    'Standard_Q': '#2E86AB',       # 标准蓝
    'Double_Q': '#56B4E9',         # 天蓝
    'Dueling_Q': '#0072B2',        # 深蓝
    'Dueling_Double_Q': '#009E73', # 蓝绿
    'DQN': '#D55E00',              # 朱红
}

# 固定的方法展示顺序（与 performance_analyzer.py 保持一致）
Q_METHODS_ORDER = ['Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q', 'DQN']


def get_method_display_name(method: str) -> str:
    """获取方法显示名称"""
    return METHOD_DISPLAY_NAMES.get(method, method)


def get_method_color(method: str) -> str:
    """获取方法颜色"""
    return Q_METHOD_COLORS.get(method, '#333333')


def get_ordered_methods(available_methods: List[str]) -> List[str]:
    """
    按固定顺序返回方法列表
    
    Parameters:
    -----------
    available_methods : list
        实际可用的方法列表
    
    Returns:
    --------
    list : 按固定顺序排列的方法列表
    """
    return [m for m in Q_METHODS_ORDER if m in available_methods]


@dataclass
class TrainingMetrics:
    """训练指标"""
    method: str
    subject: str
    seed: int
    
    # 训练历史
    reward_history: List[float] = field(default_factory=list)
    best_accuracy_history: List[float] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    q_value_stats: List[dict] = field(default_factory=list)
    
    # 统计信息
    final_reward: float = 0
    avg_reward: float = 0
    final_accuracy: float = 0
    best_accuracy: float = 0
    convergence_episode: Optional[int] = None


class TrainingPerformanceAnalyzer:
    """
    训练性能分析器

    分析 Q 方法训练期间的性能表现
    """

    def __init__(self,
                 data_loader: ExperimentDataLoader,
                 output_dir: str = 'figures',
                 style: str = 'paper'):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.style = style

        # 创建输出目录
        (self.output_dir / 'training').mkdir(parents=True, exist_ok=True)

        self._setup_style(style)

        # 训练数据
        self.training_metrics: Dict[str, TrainingMetrics] = {}
        self.method_training_data: Dict[str, List[TrainingMetrics]] = {}
        
        # 按固定顺序排列的可用方法
        self.available_methods = []

        # 加载训练数据
        self._load_training_data()
    
    def _setup_style(self, style: str):
        """设置绘图样式"""
        if style == 'paper':
            sns.set_style('whitegrid')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['font.size'] = 11
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['axes.titlesize'] = 13
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rcParams['legend.fontsize'] = 10
        elif style == 'presentation':
            sns.set_style('white')
            plt.rcParams['figure.figsize'] = (16, 9)
            plt.rcParams['font.size'] = 14
        else:
            sns.set_style('darkgrid')
    
    def _load_training_data(self):
        """从数据加载器中提取训练数据"""
        for result in self.data_loader.seed_results:
            if result.method not in Q_METHODS_ORDER:
                continue

            # 从原始数据中获取训练历史
            method = result.method
            subject = result.subject
            seed = result.seed

            # 查找对应的训练历史
            metrics = TrainingMetrics(
                method=method,
                subject=subject,
                seed=seed,
                reward_history=result.reward_history,
                best_accuracy_history=result.best_accuracy_history,
                epsilon_history=result.epsilon_history,
                final_reward=np.mean(result.reward_history[-10:]) if result.reward_history else 0,
                avg_reward=np.mean(result.reward_history) if result.reward_history else 0,
                final_accuracy=result.best_accuracy_history[-1] if result.best_accuracy_history else 0,
                best_accuracy=max(result.best_accuracy_history) if result.best_accuracy_history else 0,
                convergence_episode=self._find_convergence_episode(result.best_accuracy_history),
            )

            key = f"{method}_{subject}_seed{seed}"
            self.training_metrics[key] = metrics

            if method not in self.method_training_data:
                self.method_training_data[method] = []
            self.method_training_data[method].append(metrics)

        # 按固定顺序初始化可用方法列表
        self.available_methods = get_ordered_methods(list(self.method_training_data.keys()))

    def _find_convergence_episode(self, accuracy_history,
                                   threshold: float = 0.95, 
                                   window_size: int = 10) -> Optional[int]:
        """
        找到收敛的 episode
        
        Parameters:
        -----------
        accuracy_history : list
            最佳准确率历史
        threshold : float
            收敛阈值（相对于最终准确率的百分比）
        window_size : int
            滑动窗口大小
        """
        if not accuracy_history or len(accuracy_history) < window_size:
            return None
        
        final_acc = accuracy_history[-1]
        target_acc = final_acc * threshold
        
        for i in range(window_size, len(accuracy_history)):
            window = accuracy_history[i-window_size:i]
            if min(window) >= target_acc:
                return i - window_size
        
        return None
    
    def plot_all(self, prefix: str = '02') -> Dict[str, str]:
        """
        生成所有训练性能图表
        
        Returns:
        --------
        Dict[str, str] : 生成的文件路径
        """
        saved_files = {}
        
        print("生成训练性能分析图表...")
        print("-" * 50)
        
        # 1. 训练曲线对比
        print("  [1/4] 生成训练曲线对比图...")
        path = self.plot_training_curves(prefix=prefix)
        saved_files['training_curves'] = path
        
        # 2. 收敛速度对比
        print("  [2/4] 生成收敛速度对比图...")
        path = self.plot_convergence_comparison(prefix=prefix)
        saved_files['convergence'] = path
        
        # 3. 奖励分布箱线图
        print("  [3/4] 生成奖励分布图...")
        path = self.plot_reward_distribution(prefix=prefix)
        saved_files['reward_dist'] = path
        
        # 4. 训练热图
        print("  [4/4] 生成训练热图...")
        path = self.plot_training_heatmap(prefix=prefix)
        saved_files['heatmap'] = path
        
        print("-" * 50)
        print(f"✓ 所有图表已保存到：{self.output_dir / 'training'}")
        
        return saved_files
    
    def plot_training_curves(self, 
                             metric: str = 'accuracy',
                             smooth_window: int = 10,
                             prefix: str = '02') -> str:
        """
        生成训练曲线对比图
        
        Parameters:
        -----------
        metric : str
            指标类型 ('accuracy' 或 'reward')
        smooth_window : int
            平滑窗口大小
        prefix : str
            文件名前缀
        
        Returns:
        --------
        str : 保存的文件路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 使用固定顺序的方法列表
        methods = self.available_methods

        if not methods:
            print("  ⚠ 没有 Q 方法训练数据，跳过此图")
            return ''

        # (a) 平均训练曲线
        ax1 = axes[0, 0]
        self._plot_average_training_curve(ax1, methods, metric, smooth_window)
        ax1.set_title('(a) Average Training Curve', fontweight='bold')

        # (b) 训练曲线阴影图（标准差）
        ax2 = axes[0, 1]
        self._plot_training_curve_with_std(ax2, methods, metric, smooth_window)
        ax2.set_title('(b) Training Curve with Std Deviation', fontweight='bold')

        # (c) 各种子训练曲线
        ax3 = axes[1, 0]
        self._plot_all_seed_curves(ax3, methods, metric, smooth_window)
        ax3.set_title('(c) All Seed Training Curves', fontweight='bold')

        # (d) 最终性能对比
        ax4 = axes[1, 1]
        self._plot_final_performance(ax4, methods, metric)
        ax4.set_title('(d) Final Performance Comparison', fontweight='bold')

        plt.tight_layout()

        filename = f'{prefix}_training_curves.png'
        filepath = self.output_dir / 'training' / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def _plot_average_training_curve(self, ax, methods: List[str],
                                      metric: str, smooth_window: int):
        """绘制平均训练曲线"""
        for method in methods:
            all_data = self.method_training_data[method]

            if metric == 'accuracy':
                histories = [m.best_accuracy_history for m in all_data if m.best_accuracy_history]
            else:
                histories = [m.reward_history for m in all_data if m.reward_history]

            if not histories:
                continue

            # 对齐长度
            min_len = min(len(h) for h in histories)
            aligned = [h[:min_len] for h in histories]

            # 计算平均值
            mean_curve = np.mean(aligned, axis=0)

            # 平滑
            if smooth_window > 1 and len(mean_curve) > smooth_window:
                mean_curve = pd.Series(mean_curve).rolling(window=smooth_window).mean().values

            ax.plot(mean_curve, color=get_method_color(method),
                   linewidth=2, label=get_method_display_name(method))

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def _plot_training_curve_with_std(self, ax, methods: List[str],
                                       metric: str, smooth_window: int):
        """绘制带标准差的训练曲线"""
        # 只显示前 3 个方法，避免过于拥挤
        methods_to_show = methods[:3]

        for method in methods_to_show:
            all_data = self.method_training_data[method]

            if metric == 'accuracy':
                histories = [m.best_accuracy_history for m in all_data if m.best_accuracy_history]
            else:
                histories = [m.reward_history for m in all_data if m.reward_history]

            if not histories:
                continue

            # 对齐长度
            min_len = min(len(h) for h in histories)
            aligned = [h[:min_len] for h in histories]

            # 计算平均值和标准差
            mean_curve = np.mean(aligned, axis=0)
            std_curve = np.std(aligned, axis=0)
            episodes = range(len(mean_curve))

            # 平滑
            if smooth_window > 1 and len(mean_curve) > smooth_window:
                mean_curve = pd.Series(mean_curve).rolling(window=smooth_window).mean().values
                std_curve = pd.Series(std_curve).rolling(window=smooth_window).mean().values

            color = get_method_color(method)
            ax.plot(mean_curve, color=color, linewidth=2,
                   label=get_method_display_name(method))
            ax.fill_between(episodes,
                           mean_curve - std_curve,
                           mean_curve + std_curve,
                           alpha=0.2, color=color)

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def _plot_all_seed_curves(self, ax, methods: List[str],
                               metric: str, smooth_window: int):
        """绘制所有种子的训练曲线"""
        for method in methods:
            all_data = self.method_training_data[method]
            color = get_method_color(method)

            for metrics in all_data[:3]:  # 每个方法只显示前 3 个种子
                if metric == 'accuracy':
                    history = metrics.best_accuracy_history
                else:
                    history = metrics.reward_history

                if not history:
                    continue

                # 平滑
                if smooth_window > 1 and len(history) > smooth_window:
                    history = pd.Series(history).rolling(window=smooth_window).mean().values

                ax.plot(history, color=color, linewidth=1, alpha=0.5)

        # 添加平均曲线
        for method in methods:
            all_data = self.method_training_data[method]

            if metric == 'accuracy':
                histories = [m.best_accuracy_history for m in all_data if m.best_accuracy_history]
            else:
                histories = [m.reward_history for m in all_data if m.reward_history]

            if not histories:
                continue

            min_len = min(len(h) for h in histories)
            aligned = [h[:min_len] for h in histories]
            mean_curve = np.mean(aligned, axis=0)

            if smooth_window > 1 and len(mean_curve) > smooth_window:
                mean_curve = pd.Series(mean_curve).rolling(window=smooth_window).mean().values

            ax.plot(mean_curve, color=get_method_color(method),
                   linewidth=3, label=get_method_display_name(method))

        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def _plot_final_performance(self, ax, methods: List[str], metric: str):
        """绘制最终性能对比"""
        final_values = []
        std_values = []
        colors = []
        labels = []

        for method in methods:
            all_data = self.method_training_data[method]

            if metric == 'accuracy':
                values = [m.final_accuracy for m in all_data]
            else:
                values = [m.final_reward for m in all_data]

            final_values.append(np.mean(values))
            std_values.append(np.std(values))
            colors.append(get_method_color(method))
            labels.append(get_method_display_name(method))

        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, final_values, yerr=std_values, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)

        # 添加数值标签
        for bar, value in zip(bars, final_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def plot_convergence_comparison(self, prefix: str = '02') -> str:
        """
        生成收敛速度对比图

        Returns:
        --------
        str : 保存的文件路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 使用固定顺序的方法列表
        methods = self.available_methods

        if not methods:
            print("  ⚠ 没有 Q 方法训练数据，跳过此图")
            return ''

        # (a) 收敛 episode 对比
        ax1 = axes[0]
        convergence_data = []
        colors = []
        labels = []

        for method in methods:
            all_data = self.method_training_data[method]
            convergence_episodes = [m.convergence_episode for m in all_data
                                    if m.convergence_episode is not None]

            if convergence_episodes:
                convergence_data.append(convergence_episodes)
                colors.append(get_method_color(method))
                labels.append(get_method_display_name(method))

        if convergence_data:
            bp = ax1.boxplot(convergence_data, patch_artist=True, labels=labels)

            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')

            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.set_ylabel('Episode', fontweight='bold')
            ax1.set_title('(a) Convergence Speed Distribution', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
        else:
            ax1.text(0.5, 0.5, 'No convergence data', ha='center', va='center',
                    transform=ax1.transAxes)

        # (b) 收敛率统计
        ax2 = axes[1]

        convergence_rates = []
        for method in methods:
            all_data = self.method_training_data[method]
            n_converged = sum(1 for m in all_data if m.convergence_episode is not None)
            rate = n_converged / len(all_data) * 100 if all_data else 0
            convergence_rates.append(rate)

        x_pos = np.arange(len(methods))
        bars = ax2.bar(x_pos, convergence_rates,
                      color=[get_method_color(m) for m in methods],
                      edgecolor='black', alpha=0.8, linewidth=1.2)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([get_method_display_name(m) for m in methods],
                           rotation=45, ha='right')
        ax2.set_ylabel('Convergence Rate (%)', fontweight='bold')
        ax2.set_title('(b) Convergence Rate', fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        for bar, rate in zip(bars, convergence_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        filename = f'{prefix}_convergence_comparison.png'
        filepath = self.output_dir / 'training' / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def plot_reward_distribution(self, prefix: str = '02') -> str:
        """
        生成奖励分布图

        Returns:
        --------
        str : 保存的文件路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 使用固定顺序的方法列表
        methods = self.available_methods
        
        if not methods:
            print("  ⚠ 没有 Q 方法训练数据，跳过此图")
            return ''
        
        # (a) 平均奖励箱线图
        ax1 = axes[0]

        reward_data = []
        colors = []
        labels = []

        for method in methods:
            all_data = self.method_training_data[method]
            avg_rewards = [m.avg_reward for m in all_data]
            reward_data.append(avg_rewards)
            colors.append(get_method_color(method))
            labels.append(get_method_display_name(method))

        bp = ax1.boxplot(reward_data, patch_artist=True, labels=labels)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Average Reward', fontweight='bold')
        ax1.set_title('(a) Average Reward Distribution', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) 奖励趋势对比
        ax2 = axes[1]

        for method in methods:
            all_data = self.method_training_data[method]

            # 对齐所有奖励历史
            histories = [m.reward_history for m in all_data if m.reward_history]
            if not histories:
                continue

            min_len = min(len(h) for h in histories)
            aligned = [h[:min_len] for h in histories]
            mean_curve = np.mean(aligned, axis=0)

            # 平滑
            smooth_window = 20
            if len(mean_curve) > smooth_window:
                mean_curve = pd.Series(mean_curve).rolling(window=smooth_window).mean().values

            ax2.plot(mean_curve, color=get_method_color(method),
                    linewidth=2, label=get_method_display_name(method))

        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Average Reward (smoothed)', fontweight='bold')
        ax2.set_title('(b) Reward Trend Comparison', fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f'{prefix}_reward_distribution.png'
        filepath = self.output_dir / 'training' / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def plot_training_heatmap(self, prefix: str = '02') -> str:
        """
        生成训练热图

        Returns:
        --------
        str : 保存的文件路径
        """
        # 使用固定顺序的方法列表
        methods = self.available_methods
        
        if not methods:
            print("  ⚠ 没有 Q 方法训练数据，跳过此图")
            return ''
        
        # 创建热图数据
        # 行：方法，列：性能指标
        heatmap_data = []
        row_labels = []

        for method in methods:
            all_data = self.method_training_data[method]

            # 计算各项统计
            final_accuracies = [m.final_accuracy for m in all_data]
            best_accuracies = [m.best_accuracy for m in all_data]
            avg_rewards = [m.avg_reward for m in all_data]
            final_rewards = [m.final_reward for m in all_data]

            # 收敛率
            n_converged = sum(1 for m in all_data if m.convergence_episode is not None)
            convergence_rate = n_converged / len(all_data) * 100 if all_data else 0

            heatmap_data.append([
                np.mean(final_accuracies) * 100,  # 最终准确率
                np.mean(best_accuracies) * 100,   # 最佳准确率
                np.mean(avg_rewards),              # 平均奖励
                np.mean(final_rewards),            # 最终奖励
                convergence_rate,                  # 收敛率
            ])
            row_labels.append(get_method_display_name(method))

        col_labels = ['Final Acc (%)', 'Best Acc (%)', 'Avg Reward',
                      'Final Reward', 'Conv. Rate (%)']
        
        # 归一化到 0-100
        heatmap_data = np.array(heatmap_data)
        for i in range(heatmap_data.shape[1]):
            col = heatmap_data[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                heatmap_data[:, i] = (col - col_min) / (col_max - col_min) * 100
        
        # 绘制热图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        
        # 添加数值
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                              ha='center', va='center', color='black', fontsize=10)
        
        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title('Training Performance Heatmap', fontweight='bold', pad=15)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Score', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_training_heatmap.png'
        filepath = self.output_dir / 'training' / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def generate_training_summary(self) -> pd.DataFrame:
        """
        生成训练性能摘要表格

        Returns:
        --------
        pd.DataFrame : 摘要表格
        """
        rows = []

        # 使用固定顺序的方法列表
        for method in self.available_methods:
            all_data = self.method_training_data[method]

            final_accuracies = [m.final_accuracy for m in all_data]
            best_accuracies = [m.best_accuracy for m in all_data]
            avg_rewards = [m.avg_reward for m in all_data]
            convergence_episodes = [m.convergence_episode for m in all_data
                                    if m.convergence_episode is not None]

            rows.append({
                'Method': get_method_display_name(method),
                'Final Accuracy (%)': f'{np.mean(final_accuracies)*100:.2f} ± {np.std(final_accuracies)*100:.2f}',
                'Best Accuracy (%)': f'{np.mean(best_accuracies)*100:.2f} ± {np.std(best_accuracies)*100:.2f}',
                'Avg Reward': f'{np.mean(avg_rewards):.2f} ± {np.std(avg_rewards):.2f}',
                'Convergence Episode': f'{np.mean(convergence_episodes):.1f} ± {np.std(convergence_episodes):.1f}' if convergence_episodes else 'N/A',
                'Convergence Rate (%)': f'{len(convergence_episodes)/len(all_data)*100:.1f}' if all_data else '0',
            })

        return pd.DataFrame(rows)
