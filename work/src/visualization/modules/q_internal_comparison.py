"""
Q 方法内部对比分析模块

对比 5 个 Q 方法之间的性能：
1. 表格型 Q 方法与 DQN 的对比
2. 各方面性能指标分析
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ..data_loader import ExperimentDataLoader


# 方法定义
TABLE_Q_METHODS = ['Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q']
DQN_METHODS = ['DQN']
ALL_Q_METHODS = TABLE_Q_METHODS + DQN_METHODS

# 颜色方案
Q_COLORS = {
    'Standard_Q': '#2E86AB',
    'Double_Q': '#56B4E9',
    'Dueling_Q': '#0072B2',
    'Dueling_Double_Q': '#009E73',
    'DQN': '#D55E00',
}

# 方法显示名称
DISPLAY_NAMES = {
    'Standard_Q': 'Standard Q-Learning',
    'Double_Q': 'Double Q-Learning',
    'Dueling_Q': 'Dueling Q-Learning',
    'Dueling_Double_Q': 'Dueling Double Q-Learning',
    'DQN': 'DQN',
}


class QInternalComparisonAnalyzer:
    """
    Q 方法内部对比分析器
    """
    
    def __init__(self,
                 data_loader: ExperimentDataLoader,
                 output_dir: str = 'figures',
                 style: str = 'paper'):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir) / 'q_internal'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        self._setup_style(style)
        
        self.q_methods = [m for m in ALL_Q_METHODS if m in self.data_loader.method_summaries]
        self.table_q_methods = [m for m in TABLE_Q_METHODS if m in self.data_loader.method_summaries]
        self.dqn_methods = [m for m in DQN_METHODS if m in self.data_loader.method_summaries]
    
    def _setup_style(self, style: str):
        """设置绘图样式"""
        if style == 'paper':
            sns.set_style('whitegrid')
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['font.size'] = 11
        elif style == 'presentation':
            sns.set_style('white')
            plt.rcParams['figure.figsize'] = (16, 9)
            plt.rcParams['font.size'] = 14
        else:
            sns.set_style('darkgrid')
    
    def _get_color(self, method: str) -> str:
        """获取方法颜色"""
        return Q_COLORS.get(method, '#333333')
    
    def plot_all(self, prefix: str = '03') -> Dict[str, str]:
        """生成所有对比图表"""
        saved_files = {}
        
        print("生成 Q 方法内部对比图表...")
        print("-" * 50)
        
        print("  [1/4] 生成性能对比图...")
        path = self.plot_performance_comparison(prefix=prefix)
        saved_files['performance'] = path
        
        print("  [2/4] 生成表格型 Q vs DQN 对比图...")
        path = self.plot_table_q_vs_dqn(prefix=prefix)
        saved_files['table_vs_dqn'] = path
        
        print("  [3/4] 生成训练性能对比图...")
        path = self.plot_training_performance(prefix=prefix)
        saved_files['training'] = path
        
        print("  [4/4] 生成综合对比表格...")
        path = self.generate_comparison_table(prefix=prefix)
        saved_files['table'] = path
        
        print("-" * 50)
        print(f"✓ Q 方法内部对比图表已保存到：{self.output_dir}")
        
        return saved_files
    
    def plot_performance_comparison(self, prefix: str = '03') -> str:
        """生成性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        methods = self.q_methods
        
        if len(methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 准确率对比
        ax1 = axes[0, 0]
        self._plot_accuracy_bar(ax1, methods, summaries)
        ax1.set_title('(a) Accuracy Comparison', fontweight='bold')
        
        # (b) 性能排名
        ax2 = axes[0, 1]
        self._plot_ranking(ax2, methods, summaries)
        ax2.set_title('(b) Performance Ranking', fontweight='bold')
        
        # (c) 多指标对比
        ax3 = axes[1, 0]
        self._plot_multi_metric(ax3, methods, summaries)
        ax3.set_title('(c) Multi-metric Comparison', fontweight='bold')
        
        # (d) 稳定性对比
        ax4 = axes[1, 1]
        self._plot_stability(ax4, methods, summaries)
        ax4.set_title('(d) Stability Comparison', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_performance_comparison.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_accuracy_bar(self, ax, methods: List[str], summaries: Dict):
        """绘制准确率柱状图"""
        x_pos = np.arange(len(methods))
        means = [summaries[m].accuracy_mean * 100 for m in methods]
        stds = [summaries[m].accuracy_std * 100 for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(means) * 1.2)
    
    def _plot_ranking(self, ax, methods: List[str], summaries: Dict):
        """绘制性能排名"""
        sorted_methods = sorted(methods, key=lambda m: summaries[m].accuracy_mean, reverse=True)
        
        means = [summaries[m].accuracy_mean * 100 for m in sorted_methods]
        stds = [summaries[m].accuracy_std * 100 for m in sorted_methods]
        colors = [self._get_color(m) for m in sorted_methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in sorted_methods]
        
        x_pos = np.arange(len(sorted_methods))
        bars = ax.barh(x_pos, means, xerr=stds, capsize=3,
                      color=colors, edgecolor='black', alpha=0.8)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{mean:.1f}%', va='center', fontsize=9)
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    def _plot_multi_metric(self, ax, methods: List[str], summaries: Dict):
        """绘制多指标对比"""
        x_pos = np.arange(len(methods))
        width = 0.25
        
        acc = [summaries[m].accuracy_mean * 100 for m in methods]
        kappa = [summaries[m].kappa_mean * 100 for m in methods]
        f1 = [summaries[m].f1_mean * 100 for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in methods]
        
        ax.bar(x_pos - width, acc, width, label='Accuracy', color=colors, alpha=0.8)
        ax.bar(x_pos, kappa, width, label='Kappa (×100)', color=colors, alpha=0.8, hatch='//')
        ax.bar(x_pos + width, f1, width, label='F1 (×100)', color=colors, alpha=0.8, hatch='\\\\')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Score', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_stability(self, ax, methods: List[str], summaries: Dict):
        """绘制稳定性对比"""
        data_to_plot = []
        colors = []
        labels = []
        
        for method in methods:
            s = summaries[method]
            data_to_plot.append([x * 100 for x in s.all_accuracies])
            colors.append(self._get_color(method))
            labels.append(DISPLAY_NAMES.get(method, method))
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_table_q_vs_dqn(self, prefix: str = '03') -> str:
        """生成表格型 Q vs DQN 对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        
        if not self.table_q_methods or not self.dqn_methods:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 性能对比
        ax1 = axes[0, 0]
        self._plot_table_vs_dqn_accuracy(ax1, summaries)
        ax1.set_title('(a) Table Q vs DQN: Accuracy', fontweight='bold')
        
        # (b) 效应量分析
        ax2 = axes[0, 1]
        self._plot_effect_size(ax2, summaries)
        ax2.set_title('(b) Effect Size Analysis', fontweight='bold')
        
        # (c) 时间窗对比
        ax3 = axes[1, 0]
        self._plot_window_comparison(ax3, summaries)
        ax3.set_title('(c) Time Window Comparison', fontweight='bold')
        
        # (d) 综合性能雷达图
        ax4 = axes[1, 1]
        self._plot_radar_comparison(ax4, summaries)
        ax4.set_title('(d) Multi-metric Radar', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_table_q_vs_dqn.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_table_vs_dqn_accuracy(self, ax, summaries: Dict):
        """绘制表格型 Q vs DQN 准确率"""
        all_methods = self.table_q_methods + self.dqn_methods
        
        x_pos = np.arange(len(all_methods))
        means = [summaries[m].accuracy_mean * 100 for m in all_methods]
        stds = [summaries[m].accuracy_std * 100 for m in all_methods]
        colors = [self._get_color(m) for m in all_methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in all_methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_effect_size(self, ax, summaries: Dict):
        """绘制效应量分析"""
        # 以 Standard_Q 为基准
        baseline = summaries['Standard_Q']
        
        effect_sizes = []
        colors = []
        labels = []
        
        for method in self.q_methods:
            s = summaries[method]
            pooled_std = np.sqrt((baseline.accuracy_std**2 + s.accuracy_std**2) / 2)
            cohens_d = (s.accuracy_mean - baseline.accuracy_mean) / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(cohens_d)
            colors.append(self._get_color(method))
            labels.append(DISPLAY_NAMES.get(method, method))
        
        x_pos = np.arange(len(self.q_methods))
        
        bar_colors = ['#27AE60' if es >= 0.5 else '#F39C12' if es >= 0.2 else '#E74C3C' for es in effect_sizes]
        bars = ax.bar(x_pos, effect_sizes, color=bar_colors, edgecolor='black', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, label='Small (0.2)')
        ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, label='Medium (0.5)')
        
        for bar, es in zip(bars, effect_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'd={es:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Cohen's d (vs Standard Q)", fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_window_comparison(self, ax, summaries: Dict):
        """绘制时间窗对比"""
        valid_methods = [m for m in self.q_methods if summaries[m].t_start_mean > 0]
        
        x_pos = np.arange(len(valid_methods))
        width = 0.25
        
        t_starts = [summaries[m].t_start_mean for m in valid_methods]
        t_ends = [summaries[m].t_end_mean for m in valid_methods]
        lengths = [summaries[m].window_length_mean for m in valid_methods]
        colors = [self._get_color(m) for m in valid_methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in valid_methods]
        
        ax.bar(x_pos - width, t_starts, width, label='t_start', color=colors, alpha=0.7)
        ax.bar(x_pos, t_ends, width, label='t_end', color=colors, alpha=0.7, hatch='//')
        ax.bar(x_pos + width, lengths, width, label='Length', color=colors, alpha=0.7, hatch='\\\\')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Time (s)', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_radar_comparison(self, ax, summaries: Dict):
        """绘制雷达图对比"""
        # 归一化
        def normalize(values):
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return [50] * len(values)
            return [(v - min_v) / (max_v - min_v) * 100 for v in values]
        
        categories = ['Accuracy', 'Kappa', 'F1', 'Stability']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        acc_vals = [summaries[m].accuracy_mean * 100 for m in self.q_methods]
        kappa_vals = [summaries[m].kappa_mean * 100 for m in self.q_methods]
        f1_vals = [summaries[m].f1_mean * 100 for m in self.q_methods]
        stab_vals = [100 - summaries[m].accuracy_std * 100 for m in self.q_methods]
        
        for i, method in enumerate(self.q_methods):
            values = [
                normalize(acc_vals)[i],
                normalize(kappa_vals)[i],
                normalize(f1_vals)[i],
                normalize(stab_vals)[i],
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2,
                   color=self._get_color(method),
                   label=DISPLAY_NAMES.get(method, method))
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def plot_training_performance(self, prefix: str = '03') -> str:
        """生成训练性能对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # (a) 收敛速度对比
        ax1 = axes[0]
        self._plot_convergence_speed(ax1)
        ax1.set_title('(a) Convergence Speed', fontweight='bold')
        
        # (b) 训练曲线对比
        ax2 = axes[1]
        self._plot_training_curves(ax2)
        ax2.set_title('(b) Training Curves', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_training_performance.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_convergence_speed(self, ax):
        """绘制收敛速度对比"""
        # 这里需要从训练日志中获取收敛数据
        # 简化版本：使用最终准确率作为代理
        summaries = self.data_loader.method_summaries
        
        x_pos = np.arange(len(self.q_methods))
        final_accs = [summaries[m].accuracy_mean * 100 for m in self.q_methods]
        colors = [self._get_color(m) for m in self.q_methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in self.q_methods]
        
        bars = ax.bar(x_pos, final_accs, color=colors, edgecolor='black', alpha=0.8)
        
        for bar, acc in zip(bars, final_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Final Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_training_curves(self, ax):
        """绘制训练曲线对比"""
        # 从训练日志中获取数据
        for method in self.q_methods:
            # 获取该方法的训练历史
            for result in self.data_loader.seed_results:
                if result.method == method and result.best_accuracy_history:
                    history = result.best_accuracy_history[:200]  # 前 200 轮
                    ax.plot(history, color=self._get_color(method), alpha=0.3, linewidth=1)
                    break
        
        # 绘制平均曲线
        for method in self.q_methods:
            histories = []
            for result in self.data_loader.seed_results:
                if result.method == method and result.best_accuracy_history:
                    histories.append(result.best_accuracy_history[:200])
            
            if histories:
                min_len = min(len(h) for h in histories)
                aligned = [h[:min_len] for h in histories]
                mean_curve = np.mean(aligned, axis=0)
                ax.plot(mean_curve, color=self._get_color(method), linewidth=2,
                       label=DISPLAY_NAMES.get(method, method))
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Best Accuracy', fontweight='bold')
        ax.legend(loc='lower right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    def generate_comparison_table(self, prefix: str = '03') -> str:
        """生成综合对比表格"""
        summaries = self.data_loader.method_summaries
        
        rows = []
        for method in self.q_methods:
            if method not in summaries:
                continue
            s = summaries[method]
            
            method_type = 'Table Q' if method in TABLE_Q_METHODS else 'Deep Q'
            
            rows.append({
                'Method': DISPLAY_NAMES.get(method, method),
                'Type': method_type,
                'Accuracy (%)': f'{s.accuracy_mean*100:.2f} ± {s.accuracy_std*100:.2f}',
                'Kappa': f'{s.kappa_mean:.4f} ± {s.kappa_std:.4f}',
                'F1': f'{s.f1_mean:.4f} ± {s.f1_std:.4f}',
                'Std Dev (%)': f'{s.accuracy_std*100:.2f}',
                't_start (s)': f'{s.t_start_mean:.2f}' if s.t_start_mean > 0 else 'N/A',
                't_end (s)': f'{s.t_end_mean:.2f}' if s.t_end_mean > 0 else 'N/A',
                'Window (s)': f'{s.window_length_mean:.2f}' if s.window_length_mean > 0 else 'N/A',
            })
        
        df = pd.DataFrame(rows)
        
        csv_path = self.output_dir / f'{prefix}_q_internal_comparison.csv'
        df.to_csv(csv_path, index=False)
        
        tex_path = self.output_dir / f'{prefix}_q_internal_comparison.tex'
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, column_format='lcccccccc'))
        
        print(f"    ✓ 表格已保存到：{csv_path}")
        
        return str(csv_path)
