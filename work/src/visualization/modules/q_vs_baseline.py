"""
Q 方法与基线方法对比分析模块

对比 5 个 Q 方法与 Random_Search、Grid_Search：
1. 展示 Q 方法的有效性
2. 对比各方面性能指标
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ..data_loader import ExperimentDataLoader


# 方法定义
Q_METHODS = ['Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q', 'DQN']
BASELINE_METHODS = ['Random_Search', 'Grid_Search']

# 颜色方案
Q_COLORS = {
    'Standard_Q': '#2E86AB',
    'Double_Q': '#56B4E9',
    'Dueling_Q': '#0072B2',
    'Dueling_Double_Q': '#009E73',
    'DQN': '#D55E00',
}

BASELINE_COLORS = {
    'Random_Search': '#FF7F0E',
    'Grid_Search': '#17BECF',
}

# 方法显示名称
DISPLAY_NAMES = {
    'Standard_Q': 'Standard Q-Learning',
    'Double_Q': 'Double Q-Learning',
    'Dueling_Q': 'Dueling Q-Learning',
    'Dueling_Double_Q': 'Dueling Double Q-Learning',
    'DQN': 'DQN',
    'Random_Search': 'Random-Search',
    'Grid_Search': 'Grid-Search',
}


class QVsBaselineAnalyzer:
    """
    Q 方法与基线方法对比分析器
    """
    
    def __init__(self,
                 data_loader: ExperimentDataLoader,
                 output_dir: str = 'figures',
                 style: str = 'paper'):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir) / 'q_vs_baseline'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        self._setup_style(style)
        
        # 获取可用方法
        self.q_methods = [m for m in Q_METHODS if m in self.data_loader.method_summaries]
        self.baseline_methods = [m for m in BASELINE_METHODS if m in self.data_loader.method_summaries]
    
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
        if method in Q_COLORS:
            return Q_COLORS[method]
        if method in BASELINE_COLORS:
            return BASELINE_COLORS[method]
        return '#333333'
    
    def plot_all(self, prefix: str = '02') -> Dict[str, str]:
        """生成所有对比图表"""
        saved_files = {}
        
        print("生成 Q 方法与基线对比图表...")
        print("-" * 50)
        
        print("  [1/4] 生成性能对比图...")
        path = self.plot_performance_comparison(prefix=prefix)
        saved_files['performance'] = path
        
        print("  [2/4] 生成统计显著性分析图...")
        path = self.plot_statistical_analysis(prefix=prefix)
        saved_files['statistical'] = path
        
        print("  [3/4] 生成时间窗性能对比图...")
        path = self.plot_window_performance(prefix=prefix)
        saved_files['window'] = path
        
        print("  [4/4] 生成综合对比表格...")
        path = self.generate_comparison_table(prefix=prefix)
        saved_files['table'] = path
        
        print("-" * 50)
        print(f"✓ Q vs 基线对比图表已保存到：{self.output_dir}")
        
        return saved_files
    
    def plot_performance_comparison(self, prefix: str = '02') -> str:
        """
        生成性能对比图
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        all_methods = self.baseline_methods + self.q_methods
        
        if len(all_methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 准确率对比
        ax1 = axes[0, 0]
        self._plot_accuracy_bar(ax1, all_methods, summaries)
        ax1.set_title('(a) Accuracy Comparison', fontweight='bold')
        
        # (b) 相对于最佳基线的提升
        ax2 = axes[0, 1]
        self._plot_improvement_over_baseline(ax2, all_methods, summaries)
        ax2.set_title('(b) Improvement over Best Baseline', fontweight='bold')
        
        # (c) Kappa 系数对比
        ax3 = axes[1, 0]
        self._plot_kappa_comparison(ax3, all_methods, summaries)
        ax3.set_title('(c) Kappa Coefficient', fontweight='bold')
        
        # (d) 综合性能雷达图
        ax4 = axes[1, 1]
        self._plot_radar_chart(ax4, all_methods, summaries)
        ax4.set_title('(d) Multi-metric Radar Chart', fontweight='bold')
        
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
    
    def _plot_improvement_over_baseline(self, ax, methods: List[str], summaries: Dict):
        """绘制相对于最佳基线的提升"""
        # 找到最佳基线
        baseline_accs = [(m, summaries[m].accuracy_mean) for m in self.baseline_methods]
        best_baseline = max(baseline_accs, key=lambda x: x[1])
        best_baseline_acc = best_baseline[1] * 100
        
        improvements = []
        colors = []
        labels = []
        
        for method in methods:
            acc = summaries[method].accuracy_mean * 100
            improvements.append(acc - best_baseline_acc)
            colors.append(self._get_color(method))
            labels.append(DISPLAY_NAMES.get(method, method))
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, improvements, color=colors, edgecolor='black', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
        
        for bar, imp in zip(bars, improvements):
            color = 'white' if imp < 0 else 'black'
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + (0.5 if imp > 0 else -1.5),
                   f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
                   ha='center', fontsize=8, color=color)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(f'Improvement over {DISPLAY_NAMES.get(best_baseline[0], best_baseline[0])} (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_kappa_comparison(self, ax, methods: List[str], summaries: Dict):
        """绘制 Kappa 系数对比"""
        x_pos = np.arange(len(methods))
        means = [summaries[m].kappa_mean for m in methods]
        stds = [summaries[m].kappa_std for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Kappa Coefficient', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_radar_chart(self, ax, methods: List[str], summaries: Dict):
        """绘制雷达图"""
        # 归一化函数
        def normalize(values):
            min_v, max_v = min(values), max(values)
            if max_v == min_v:
                return [50] * len(values)
            return [(v - min_v) / (max_v - min_v) * 100 for v in values]
        
        categories = ['Accuracy', 'Kappa', 'F1', 'Stability']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # 计算各指标
        acc_vals = [summaries[m].accuracy_mean * 100 for m in methods]
        kappa_vals = [summaries[m].kappa_mean * 100 for m in methods]
        f1_vals = [summaries[m].f1_mean * 100 for m in methods]
        stab_vals = [100 - summaries[m].accuracy_std * 100 for m in methods]  # 稳定性用 100-std
        
        for i, method in enumerate(methods):
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
    
    def plot_statistical_analysis(self, prefix: str = '02') -> str:
        """
        生成统计显著性分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        all_methods = self.baseline_methods + self.q_methods
        
        if len(all_methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 效应量分析 (Cohen's d)
        ax1 = axes[0, 0]
        self._plot_effect_size(ax1, all_methods, summaries)
        ax1.set_title('(a) Effect Size (Cohen\'s d)', fontweight='bold')
        
        # (b) 性能排名
        ax2 = axes[0, 1]
        self._plot_performance_ranking(ax2, all_methods, summaries)
        ax2.set_title('(b) Performance Ranking', fontweight='bold')
        
        # (c) 稳定性对比
        ax3 = axes[1, 0]
        self._plot_stability_comparison(ax3, all_methods, summaries)
        ax3.set_title('(c) Stability Comparison', fontweight='bold')
        
        # (d) 成功率对比
        ax4 = axes[1, 1]
        self._plot_success_rate(ax4, all_methods, summaries)
        ax4.set_title('(d) Success Rate', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_statistical_analysis.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_effect_size(self, ax, methods: List[str], summaries: Dict):
        """绘制效应量分析"""
        # 计算合并标准差
        all_stds = [summaries[m].accuracy_std for m in self.baseline_methods]
        pooled_std = np.sqrt(np.mean([s**2 for s in all_stds]))
        baseline_mean = np.mean([summaries[m].accuracy_mean for m in self.baseline_methods])
        
        effect_sizes = []
        colors = []
        labels = []
        
        for method in methods:
            s = summaries[method]
            cohens_d = (s.accuracy_mean - baseline_mean) / pooled_std
            effect_sizes.append(cohens_d)
            colors.append(self._get_color(method))
            labels.append(DISPLAY_NAMES.get(method, method))
        
        x_pos = np.arange(len(methods))
        
        # 效应量颜色编码
        bar_colors = []
        for es in effect_sizes:
            if es >= 0.8:
                bar_colors.append('#27AE60')
            elif es >= 0.5:
                bar_colors.append('#F39C12')
            else:
                bar_colors.append('#E74C3C')
        
        bars = ax.bar(x_pos, effect_sizes, color=bar_colors, edgecolor='black', alpha=0.8)
        
        ax.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, label='Small (0.2)')
        ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, label='Medium (0.5)')
        ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1, label='Large (0.8)')
        
        for bar, es in zip(bars, effect_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'd={es:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Cohen's d Effect Size", fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_performance_ranking(self, ax, methods: List[str], summaries: Dict):
        """绘制性能排名"""
        # 按准确率排序
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
        ax.set_title('Methods Ranked by Accuracy', fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    def _plot_stability_comparison(self, ax, methods: List[str], summaries: Dict):
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
    
    def _plot_success_rate(self, ax, methods: List[str], summaries: Dict):
        """绘制成功率对比"""
        x_pos = np.arange(len(methods))
        success_rates = []
        colors = []
        labels = []
        
        for method in methods:
            s = summaries[method]
            success_rates.append(s.get('success_rate', 100) if isinstance(s, dict) else 100)
            colors.append(self._get_color(method))
            labels.append(DISPLAY_NAMES.get(method, method))
        
        bars = ax.bar(x_pos, success_rates, color=colors, edgecolor='black', alpha=0.8)
        
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Success Rate (%)', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_window_performance(self, prefix: str = '02') -> str:
        """
        生成时间窗性能对比图
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        summaries = self.data_loader.method_summaries
        all_methods = self.baseline_methods + self.q_methods
        
        # 过滤有有效时间窗数据的方法
        valid_methods = [m for m in all_methods
                        if summaries[m].t_start_mean > 0]
        
        if not valid_methods:
            print("    ⚠ 没有时间窗数据，跳过此图")
            return ''
        
        # (a) 时间窗位置对比
        ax1 = axes[0]
        self._plot_window_position(ax1, valid_methods, summaries)
        ax1.set_title('(a) Time Window Position', fontweight='bold')
        
        # (b) 时间窗长度与性能关系
        ax2 = axes[1]
        self._plot_window_length_vs_performance(ax2, valid_methods, summaries)
        ax2.set_title('(b) Window Length vs. Performance', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_window_performance.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_window_position(self, ax, methods: List[str], summaries: Dict):
        """绘制时间窗位置对比"""
        x_pos = np.arange(len(methods))
        width = 0.35
        
        t_starts = [summaries[m].t_start_mean for m in methods]
        t_ends = [summaries[m].t_end_mean for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in methods]
        
        ax.bar(x_pos - width/2, t_starts, width, label='t_start', color=colors, alpha=0.7)
        ax.bar(x_pos + width/2, t_ends, width, label='t_end', color=colors, alpha=0.7, hatch='//')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Time (s)', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_window_length_vs_performance(self, ax, methods: List[str], summaries: Dict):
        """绘制时间窗长度与性能关系"""
        window_lengths = [summaries[m].window_length_mean for m in methods]
        accuracies = [summaries[m].accuracy_mean * 100 for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [DISPLAY_NAMES.get(m, m) for m in methods]
        
        scatter = ax.scatter(window_lengths, accuracies, s=150, c=colors,
                            alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # 添加方法标签
        for i, method in enumerate(methods):
            ax.annotate(DISPLAY_NAMES.get(method, method),
                       (window_lengths[i], accuracies[i]),
                       fontsize=8, ha='center', va='bottom')
        
        ax.set_xlabel('Window Length (s)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def generate_comparison_table(self, prefix: str = '02') -> str:
        """生成综合对比表格"""
        summaries = self.data_loader.method_summaries
        
        rows = []
        all_methods = self.baseline_methods + self.q_methods
        
        for method in all_methods:
            if method not in summaries:
                continue
            s = summaries[method]
            
            # 计算相对于最佳基线的提升
            baseline_accs = [summaries[m].accuracy_mean for m in self.baseline_methods]
            best_baseline = max(baseline_accs) if baseline_accs else 0
            improvement = ((s.accuracy_mean - best_baseline) / best_baseline) * 100 if best_baseline > 0 else 0
            
            # 计算效应量
            baseline_stds = [summaries[m].accuracy_std for m in self.baseline_methods]
            pooled_std = np.sqrt(np.mean([x**2 for x in baseline_stds])) if baseline_stds else 1
            baseline_mean = np.mean([summaries[m].accuracy_mean for m in self.baseline_methods]) if baseline_accs else 0
            effect_size = (s.accuracy_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            rows.append({
                'Method': DISPLAY_NAMES.get(method, method),
                'Category': 'Q-Learning' if method in Q_METHODS else 'Baseline',
                'Accuracy (%)': f'{s.accuracy_mean*100:.2f} ± {s.accuracy_std*100:.2f}',
                'Kappa': f'{s.kappa_mean:.4f} ± {s.kappa_std:.4f}',
                'F1': f'{s.f1_mean:.4f} ± {s.f1_std:.4f}',
                'Improvement (%)': f'{improvement:+.2f}',
                'Effect Size (d)': f'{effect_size:.2f}',
                't_start (s)': f'{s.t_start_mean:.2f}' if s.t_start_mean > 0 else 'N/A',
                't_end (s)': f'{s.t_end_mean:.2f}' if s.t_end_mean > 0 else 'N/A',
                'Window (s)': f'{s.window_length_mean:.2f}' if s.window_length_mean > 0 else 'N/A',
            })
        
        df = pd.DataFrame(rows)
        
        csv_path = self.output_dir / f'{prefix}_q_vs_baseline.csv'
        df.to_csv(csv_path, index=False)
        
        tex_path = self.output_dir / f'{prefix}_q_vs_baseline.tex'
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, column_format='lcccccccccc'))
        
        print(f"    ✓ 表格已保存到：{csv_path}")
        
        return str(csv_path)
