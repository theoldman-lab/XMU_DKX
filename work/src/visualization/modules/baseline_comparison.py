"""
基线方法对比分析模块

对比基线方法 (FL-2s, FL-1.5s, FL-1s, Random_Search, Grid_Search)
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


# 基线方法定义
BASELINE_METHODS = ['FL-2s', 'FL-1.5s', 'FL-1s', 'Random_Search', 'Grid_Search']

# 基线方法颜色（暖色调/中性色）
BASELINE_COLORS = {
    'FL-2s': '#A67C52',         # 浅棕色
    'FL-1.5s': '#B8958A',       # 灰棕色
    'FL-1s': '#CDB4A8',         # 米棕色
    'Random_Search': '#FF7F0E', # 橙色
    'Grid_Search': '#17BECF',   # 青色
}

# 方法显示名称
BASELINE_DISPLAY_NAMES = {
    'FL-2s': 'Fixed-2s',
    'FL-1.5s': 'Fixed-1.5s',
    'FL-1s': 'Fixed-1s',
    'Random_Search': 'Random-Search',
    'Grid_Search': 'Grid-Search',
}


class BaselineComparisonAnalyzer:
    """
    基线方法对比分析器
    """
    
    def __init__(self, 
                 data_loader: ExperimentDataLoader,
                 output_dir: str = 'figures',
                 style: str = 'paper'):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir) / 'baseline'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        self._setup_style(style)
        
        # 获取基线方法数据
        self.baseline_methods = [m for m in BASELINE_METHODS 
                                  if m in self.data_loader.method_summaries]
    
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
    
    def plot_all(self, prefix: str = '01') -> Dict[str, str]:
        """生成所有基线对比图表"""
        saved_files = {}
        
        print("生成基线方法对比图表...")
        print("-" * 50)
        
        print("  [1/4] 生成性能对比图...")
        path = self.plot_performance_comparison(prefix=prefix)
        saved_files['performance'] = path
        
        print("  [2/4] 生成稳定性分析图...")
        path = self.plot_stability_analysis(prefix=prefix)
        saved_files['stability'] = path
        
        print("  [3/4] 生成被试间性能分布图...")
        path = self.plot_subject_distribution(prefix=prefix)
        saved_files['subject_dist'] = path
        
        print("  [4/4] 生成综合对比表格...")
        path = self.generate_comparison_table(prefix=prefix)
        saved_files['table'] = path
        
        print("-" * 50)
        print(f"✓ 基线对比图表已保存到：{self.output_dir}")
        
        return saved_files
    
    def plot_performance_comparison(self, prefix: str = '01') -> str:
        """
        生成性能对比图
        展示 Random/Grid 相对于 FL 的性能提升
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        methods = self.baseline_methods
        
        if len(methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 准确率对比柱状图
        ax1 = axes[0, 0]
        self._plot_accuracy_bar(ax1, methods, summaries)
        ax1.set_title('(a) Accuracy Comparison', fontweight='bold')
        
        # (b) 相对于 FL-Short 的提升
        ax2 = axes[0, 1]
        self._plot_improvement_over_fixed(ax2, methods, summaries)
        ax2.set_title('(b) Improvement over Fixed-Window', fontweight='bold')
        
        # (c) Kappa 系数对比
        ax3 = axes[1, 0]
        self._plot_kappa_comparison(ax3, methods, summaries)
        ax3.set_title('(c) Kappa Coefficient Comparison', fontweight='bold')
        
        # (d) F1 分数对比
        ax4 = axes[1, 1]
        self._plot_f1_comparison(ax4, methods, summaries)
        ax4.set_title('(d) F1 Score Comparison', fontweight='bold')
        
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
        colors = [BASELINE_COLORS.get(m, '#333333') for m in methods]
        labels = [BASELINE_DISPLAY_NAMES.get(m, m) for m in methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)
        
        # 添加数值标签
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(means) * 1.2)
    
    def _plot_improvement_over_fixed(self, ax, methods: List[str], summaries: Dict):
        """绘制相对于固定窗口的提升"""
        # 选择可用的 FL 方法作为基准（优先 FL-2s，其次 FL-1.5s，再次 FL-1s）
        baseline_method = None
        for candidate in ['FL-2s', 'FL-1.5s', 'FL-1s']:
            if candidate in summaries:
                baseline_method = candidate
                break
        
        if baseline_method is None:
            # 没有可用的 FL 方法，使用所有方法的平均作为基准
            baseline_acc = np.mean([summaries[m].accuracy_mean for m in methods]) * 100
            baseline_name = 'Average'
        else:
            baseline_acc = summaries[baseline_method].accuracy_mean * 100
            baseline_name = BASELINE_DISPLAY_NAMES.get(baseline_method, baseline_method)

        improvements = []
        colors = []
        labels = []

        for method in methods:
            if method == baseline_method:
                improvements.append(0)
            else:
                acc = summaries[method].accuracy_mean * 100
                improvements.append(acc - baseline_acc)

            colors.append(BASELINE_COLORS.get(method, '#333333'))
            labels.append(BASELINE_DISPLAY_NAMES.get(method, method))

        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, improvements, color=colors, edgecolor='black', alpha=0.8)

        # 零基线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)

        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            color = 'white' if imp < 0 else 'black'
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + (0.5 if imp > 0 else -1.5),
                   f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%',
                   ha='center', fontsize=9, color=color)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy Improvement (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_kappa_comparison(self, ax, methods: List[str], summaries: Dict):
        """绘制 Kappa 系数对比"""
        x_pos = np.arange(len(methods))
        means = [summaries[m].kappa_mean for m in methods]
        stds = [summaries[m].kappa_std for m in methods]
        colors = [BASELINE_COLORS.get(m, '#333333') for m in methods]
        labels = [BASELINE_DISPLAY_NAMES.get(m, m) for m in methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Kappa Coefficient', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_f1_comparison(self, ax, methods: List[str], summaries: Dict):
        """绘制 F1 分数对比"""
        x_pos = np.arange(len(methods))
        means = [summaries[m].f1_mean for m in methods]
        stds = [summaries[m].f1_std for m in methods]
        colors = [BASELINE_COLORS.get(m, '#333333') for m in methods]
        labels = [BASELINE_DISPLAY_NAMES.get(m, m) for m in methods]
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8)
        
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('F1 Score', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_stability_analysis(self, prefix: str = '01') -> str:
        """
        生成稳定性分析图
        展示 Random/Grid 的不稳定性缺陷
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        methods = self.baseline_methods
        
        if len(methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 准确率标准差对比
        ax1 = axes[0, 0]
        self._plot_std_comparison(ax1, methods, summaries)
        ax1.set_title('(a) Accuracy Standard Deviation', fontweight='bold')
        
        # (b) 变异系数对比
        ax2 = axes[0, 1]
        self._plot_cv_comparison(ax2, methods, summaries)
        ax2.set_title('(b) Coefficient of Variation', fontweight='bold')
        
        # (c) 性能分布箱线图
        ax3 = axes[1, 0]
        self._plot_boxplot(ax3, methods, summaries)
        ax3.set_title('(c) Performance Distribution', fontweight='bold')
        
        # (d) 最差/最佳性能对比
        ax4 = axes[1, 1]
        self._plot_min_max_comparison(ax4, methods, summaries)
        ax4.set_title('(d) Min/Max Performance', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_stability_analysis.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_std_comparison(self, ax, methods: List[str], summaries: Dict):
        """绘制标准差对比"""
        x_pos = np.arange(len(methods))
        stds = [summaries[m].accuracy_std * 100 for m in methods]
        colors = [BASELINE_COLORS.get(m, '#333333') for m in methods]
        labels = [BASELINE_DISPLAY_NAMES.get(m, m) for m in methods]
        
        # 标准差越低越好，所以用倒序颜色
        bars = ax.bar(x_pos, stds, color=colors, edgecolor='black', alpha=0.8)
        
        for bar, std in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{std:.2f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy Std Dev (%)', fontweight='bold')
        ax.set_ylim(0, max(stds) * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_cv_comparison(self, ax, methods: List[str], summaries: Dict):
        """绘制变异系数对比 (CV = std/mean)"""
        x_pos = np.arange(len(methods))
        cvs = []
        
        for method in methods:
            s = summaries[method]
            cv = (s.accuracy_std / s.accuracy_mean) * 100
            cvs.append(cv)
        
        colors = [BASELINE_COLORS.get(m, '#333333') for m in methods]
        labels = [BASELINE_DISPLAY_NAMES.get(m, m) for m in methods]
        
        bars = ax.bar(x_pos, cvs, color=colors, edgecolor='black', alpha=0.8)
        
        for bar, cv in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{cv:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_boxplot(self, ax, methods: List[str], summaries: Dict):
        """绘制箱线图"""
        data_to_plot = []
        colors = []
        labels = []
        
        for method in methods:
            s = summaries[method]
            data_to_plot.append([x * 100 for x in s.all_accuracies])
            colors.append(BASELINE_COLORS.get(method, '#333333'))
            labels.append(BASELINE_DISPLAY_NAMES.get(method, method))
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_min_max_comparison(self, ax, methods: List[str], summaries: Dict):
        """绘制最差/最佳性能对比"""
        x_pos = np.arange(len(methods))
        width = 0.35
        
        mins = []
        maxs = []
        
        for method in methods:
            s = summaries[method]
            accs = s.all_accuracies
            mins.append(min(accs) * 100)
            maxs.append(max(accs) * 100)
        
        colors = [BASELINE_COLORS.get(m, '#333333') for m in methods]
        labels = [BASELINE_DISPLAY_NAMES.get(m, m) for m in methods]
        
        ax.bar(x_pos - width/2, mins, width, label='Min', color=colors, alpha=0.7)
        ax.bar(x_pos + width/2, maxs, width, label='Max', color=colors, alpha=0.7, hatch='//')
        
        # 添加范围线
        for i in range(len(methods)):
            ax.plot([i, i], [mins[i], maxs[i]], 'k-', linewidth=2, alpha=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_subject_distribution(self, prefix: str = '01') -> str:
        """
        生成被试间性能分布图
        """
        methods = self.baseline_methods

        if len(methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''

        # 根据方法数量动态创建子图
        n_methods = len(methods)
        n_cols = 2
        n_rows = (n_methods + n_cols - 1) // n_cols  # 向上取整
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes = axes.flatten() if n_methods > 1 else [axes]

        # 为每个方法绘制被试分布
        for idx, method in enumerate(methods):
            ax = axes[idx]

            # 获取该方法的被试数据
            subject_data = {}
            for result in self.data_loader.seed_results:
                if result.method == method:
                    subject = result.subject
                    if subject not in subject_data:
                        subject_data[subject] = []
                    subject_data[subject].append(result.accuracy * 100)

            # 绘制散点图
            positions = []
            values = []
            for subject in sorted(subject_data.keys()):
                positions.extend([list(subject_data.keys()).index(subject)] * len(subject_data[subject]))
                values.extend(subject_data[subject])

            ax.scatter(positions, values, alpha=0.7, s=80,
                      color=BASELINE_COLORS.get(method, '#333333'),
                      edgecolors='black', linewidth=1)

            # 绘制均值线
            all_vals = [v for vals in subject_data.values() for v in vals]
            mean_val = np.mean(all_vals)
            ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}%')

            ax.set_xticks(range(len(subject_data)))
            ax.set_xticklabels(sorted(subject_data.keys()))
            ax.set_xlabel('Subject', fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title(f'{BASELINE_DISPLAY_NAMES.get(method, method)}', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 100)

        # 隐藏多余的子图
        for idx in range(n_methods, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        filename = f'{prefix}_subject_distribution.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)
    
    def generate_comparison_table(self, prefix: str = '01') -> str:
        """生成综合对比表格"""
        summaries = self.data_loader.method_summaries

        # 选择可用的 FL 方法作为基准（优先 FL-2s，其次 FL-1.5s，再次 FL-1s）
        baseline_method = None
        for candidate in ['FL-2s', 'FL-1.5s', 'FL-1s']:
            if candidate in summaries:
                baseline_method = candidate
                break

        rows = []
        for method in self.baseline_methods:
            if method not in summaries:
                continue
            s = summaries[method]

            # 计算变异系数
            cv = (s.accuracy_std / s.accuracy_mean) * 100 if s.accuracy_mean > 0 else 0

            # 计算相对于基准 FL 方法的提升
            baseline = summaries.get(baseline_method)
            if baseline and baseline_method:
                improvement = ((s.accuracy_mean - baseline.accuracy_mean) / baseline.accuracy_mean) * 100
            else:
                improvement = 0

            # 动态生成列名
            baseline_display = BASELINE_DISPLAY_NAMES.get(baseline_method, baseline_method) if baseline_method else 'Average'
            improvement_col = f'Improvement over {baseline_display} (%)'

            rows.append({
                'Method': BASELINE_DISPLAY_NAMES.get(method, method),
                'Accuracy (%)': f'{s.accuracy_mean*100:.2f} ± {s.accuracy_std*100:.2f}',
                'Kappa': f'{s.kappa_mean:.4f} ± {s.kappa_std:.4f}',
                'F1': f'{s.f1_mean:.4f} ± {s.f1_std:.4f}',
                'Std Dev (%)': f'{s.accuracy_std*100:.2f}',
                'CV (%)': f'{cv:.2f}',
                improvement_col: f'{improvement:+.2f}',
                'Min Acc (%)': f'{min(s.all_accuracies)*100:.2f}',
                'Max Acc (%)': f'{max(s.all_accuracies)*100:.2f}',
            })

        df = pd.DataFrame(rows)

        # 保存 CSV
        csv_path = self.output_dir / f'{prefix}_baseline_comparison.csv'
        df.to_csv(csv_path, index=False)

        # 保存 LaTeX
        tex_path = self.output_dir / f'{prefix}_baseline_comparison.tex'
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, column_format='lcccccccc'))

        print(f"    ✓ 表格已保存到：{csv_path}")

        return str(csv_path)
