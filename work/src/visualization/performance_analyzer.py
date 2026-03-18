"""
性能分析模块

用于生成学术论文级别的性能分析图表
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .data_loader import ExperimentDataLoader, MethodSummary


# ==================== 学术样式配置 ====================

PAPER_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.linewidth': 1,
}

# 方法显示名称
METHOD_DISPLAY_NAMES = {
    'FL-2s': 'Fixed-2s',
    'FL-1.5s': 'Fixed-1.5s',
    'FL-1s': 'Fixed-1s',
    'Grid_Search': 'Grid-Search',
    'Random_Search': 'Random-Search',
    'Standard_Q': 'Standard Q-Learning',
    'Double_Q': 'Double Q-Learning',
    'Dueling_Q': 'Dueling Q-Learning',
    'Dueling_Double_Q': 'Dueling Double Q-Learning',
    'DQN': 'DQN',
}

# 固定的方法展示顺序（按基线方法 → Q 方法 → DQN 排序）
ALL_METHODS_ORDER = [
    'FL-2s', 'FL-1.5s', 'FL-1s', 'Random_Search', 'Grid_Search',  # 基线方法
    'Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q',  # Q 方法
    'DQN'  # 深度学习方法
]

# 方法专属颜色方案（高区分度，适合学术出版）
# 基线方法使用暖色调/中性色，Q 方法使用蓝色系，DQN 使用突出色
METHOD_COLORS = {
    # 基线方法 - 使用暖色调/中性色
    'FL-2s': '#A67C52',         # 浅棕色
    'FL-1.5s': '#B8958A',       # 灰棕色
    'FL-1s': '#CDB4A8',         # 米棕色
    'Random_Search': '#FF7F0E', # 橙色
    'Grid_Search': '#17BECF',   # 青色

    # Q 学习方法 - 使用蓝色系渐变（体现方法演进）
    'Standard_Q': '#2E86AB',    # 标准蓝
    'Double_Q': '#56B4E9',      # 天蓝
    'Dueling_Q': '#0072B2',     # 深蓝
    'Dueling_Double_Q': '#009E73',  # 蓝绿

    # DQN
    'DQN': '#D55E00',           # 朱红
}

# 类别定义
METHOD_CATEGORIES = {
    'baseline': ['FL-2s', 'FL-1.5s', 'FL-1s', 'Random_Search', 'Grid_Search'],
    'q_learning': ['Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q'],
    'dqn': ['DQN'],
}

# 类别颜色方案（用于按类别分组展示）
CATEGORY_COLORS = {
    'baseline': '#95A5A6',      # 灰色 - 基线方法
    'q_learning': '#3498DB',    # 蓝色 - Q 学习
    'dqn': '#E74C3C',           # 红色 - DQN
}


def get_method_color(method: str, categories: Dict[str, List[str]] = None) -> str:
    """
    获取方法颜色
    
    Parameters:
    -----------
    method : str
        方法名称
    categories : dict, optional
        类别定义（用于向后兼容）
    
    Returns:
    --------
    str : 颜色代码
    """
    # 优先使用方法专属颜色
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    
    # 回退到类别颜色
    if categories:
        for category, methods in categories.items():
            if method in methods:
                return CATEGORY_COLORS.get(category, '#333333')
    
    return '#333333'


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
    return [m for m in ALL_METHODS_ORDER if m in available_methods]


def get_method_category(method: str) -> str:
    """
    获取方法类别
    
    Parameters:
    -----------
    method : str
        方法名称
    
    Returns:
    --------
    str : 类别名称
    """
    for category, methods in METHOD_CATEGORIES.items():
        if method in methods:
            return category
    return 'unknown'


class PerformanceAnalyzer:
    """
    性能分析器

    生成学术论文级别的性能分析图表
    """

    def __init__(self,
                 data_loader: ExperimentDataLoader,
                 output_dir: str = 'figures',
                 style: str = 'paper'):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.style = style

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)

        self._setup_style(style)
        
        # 按固定顺序排列的可用方法
        self.available_methods = get_ordered_methods(
            list(self.data_loader.method_summaries.keys())
        )

    def _setup_style(self, style: str):
        """设置绘图样式"""
        if style == 'paper':
            sns.set_style('whitegrid')
            for key, value in PAPER_STYLE.items():
                plt.rcParams[key] = value
        elif style == 'presentation':
            sns.set_style('white')
            plt.rcParams['figure.figsize'] = (16, 9)
            plt.rcParams['font.size'] = 14
        else:
            sns.set_style('darkgrid')

    def plot_all(self, prefix: str = '01') -> Dict[str, str]:
        """
        生成所有性能分析图表

        Parameters:
        -----------
        prefix : str
            文件名前缀

        Returns:
        --------
        Dict[str, str] : 生成的文件路径
        """
        saved_files = {}

        print("生成性能分析图表...")
        print("-" * 50)

        # 1. 主性能对比图
        print("  [1/6] 生成主性能对比图...")
        path = self.plot_main_comparison(prefix=prefix)
        saved_files['main_comparison'] = path

        # 2. Q 方法与基线方法对比
        print("  [2/6] 生成 Q 方法与基线方法对比图...")
        path = self.plot_q_vs_baseline(prefix=prefix)
        saved_files['q_vs_baseline'] = path

        # 3. 各被试性能对比
        print("  [3/6] 生成各被试性能对比图...")
        path = self.plot_subject_wise_comparison(prefix=prefix)
        saved_files['subject_wise'] = path

        # 4. 性能分布图
        print("  [4/6] 生成性能分布图...")
        path = self.plot_performance_distribution(prefix=prefix)
        saved_files['distribution'] = path

        # 5. 雷达图
        print("  [5/6] 生成多指标雷达图...")
        path = self.plot_radar_chart(prefix=prefix)
        saved_files['radar'] = path

        # 6. 时间窗分析图
        print("  [6/6] 生成时间窗分析图...")
        path = self.plot_window_analysis(prefix=prefix)
        saved_files['window_analysis'] = path

        print("-" * 50)
        print(f"✓ 所有图表已保存到：{self.output_dir}")

        return saved_files
    
    def plot_main_comparison(self,
                             metric: str = 'accuracy',
                             sort_by: str = 'accuracy_mean',
                             prefix: str = '01') -> str:
        """
        生成主性能对比图

        Parameters:
        -----------
        metric : str
            性能指标 ('accuracy', 'kappa', 'f1')
        sort_by : str
            排序依据
        prefix : str
            文件名前缀

        Returns:
        --------
        str : 保存的文件路径
        """
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

        method_summaries = self.data_loader.method_summaries
        
        # 使用固定顺序的方法列表
        methods = self.available_methods
        categories = METHOD_CATEGORIES

        if not methods:
            print("  ⚠ 没有数据，跳过此图")
            return ''

        # (a) 柱状图
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_bar_chart(ax1, methods, metric)
        ax1.set_title('(a) Performance Comparison', fontweight='bold')

        # (b) 箱线图
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_boxplot(ax2, methods, metric)
        ax2.set_title('(b) Performance Distribution', fontweight='bold')

        # (c) 排序柱状图
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_sorted_bar_chart(ax3, methods, sort_by)
        ax3.set_title('(c) Ranked Performance', fontweight='bold')

        plt.tight_layout()
        
        filename = f'{prefix}_main_comparison.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def _plot_bar_chart(self, ax, methods: List[str], metric: str):
        """绘制柱状图"""
        summaries = self.data_loader.method_summaries

        means = []
        stds = []
        colors = []
        display_names = []

        for method in methods:
            s = summaries[method]
            if metric == 'accuracy':
                means.append(s.accuracy_mean * 100)
                stds.append(s.accuracy_std * 100)
            elif metric == 'kappa':
                means.append(s.kappa_mean)
                stds.append(s.kappa_std)
            else:
                means.append(s.f1_mean)
                stds.append(s.f1_std)

            colors.append(get_method_color(method))
            display_names.append(METHOD_DISPLAY_NAMES.get(method, method))

        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)

        # 添加数值标签
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        ax.set_ylabel(f'{metric.capitalize()} ' + ('(%)' if metric == 'accuracy' else ''),
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加图例（使用方法专属颜色）
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=METHOD_COLORS.get(method, '#333333'), label=METHOD_DISPLAY_NAMES.get(method, method))
            for method in methods
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=3)

    def _plot_boxplot(self, ax, methods: List[str], metric: str):
        """绘制箱线图"""
        summaries = self.data_loader.method_summaries

        data_to_plot = []
        colors = []
        labels = []

        for method in methods:
            s = summaries[method]
            data_to_plot.append(s.all_accuracies)
            colors.append(get_method_color(method))
            labels.append(METHOD_DISPLAY_NAMES.get(method, method))

        bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def _plot_sorted_bar_chart(self, ax, methods: List[str], sort_by: str):
        """绘制排序柱状图"""
        summaries = self.data_loader.method_summaries

        # 按指定指标排序
        sorted_methods = sorted(
            methods,
            key=lambda m: getattr(summaries[m], sort_by),
            reverse=True
        )

        means = [summaries[m].accuracy_mean * 100 for m in sorted_methods]
        stds = [summaries[m].accuracy_std * 100 for m in sorted_methods]
        colors = [get_method_color(m) for m in sorted_methods]
        display_names = [METHOD_DISPLAY_NAMES.get(m, m) for m in sorted_methods]

        x_pos = np.arange(len(sorted_methods))
        bars = ax.barh(x_pos, means, xerr=stds, capsize=3,
                      color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)

        ax.set_yticks(x_pos)
        ax.set_yticklabels(display_names)
        ax.set_xlabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Methods Ranked by Accuracy', fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # 添加数值标签
        for bar, mean in zip(bars, means):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f'{mean:.2f}%', va='center', fontsize=9)

    def plot_q_vs_baseline(self, prefix: str = '01') -> str:
        """
        生成 Q 方法与基线方法对比分析图
        
        包含：
        1. 性能提升对比（准确率）
        2. 统计显著性分析（效应量）
        3. 稳定性对比（标准差）
        4. 综合性能雷达图

        Returns:
        --------
        str : 保存的文件路径
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig)

        baseline_methods = METHOD_CATEGORIES['baseline']
        q_methods = METHOD_CATEGORIES['q_learning']
        
        # 获取实际可用的方法
        available_baseline = [m for m in baseline_methods if m in self.available_methods]
        available_q = [m for m in q_methods if m in self.available_methods]

        if not available_baseline or not available_q:
            print("  ⚠ 数据不足，跳过 Q vs 基线对比图")
            return ''

        # (a) 性能提升对比
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_performance_improvement(ax1, available_baseline, available_q)
        ax1.set_title('(a) Performance Improvement over Baseline', fontweight='bold')

        # (b) 统计显著性分析
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_statistical_significance(ax2, available_baseline, available_q)
        ax2.set_title('(b) Statistical Significance Analysis', fontweight='bold')

        # (c) 稳定性对比
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_stability_comparison(ax3, available_baseline, available_q)
        ax3.set_title('(c) Stability Comparison', fontweight='bold')

        # (d) 类别性能对比
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_category_comparison(ax4, available_baseline, available_q)
        ax4.set_title('(d) Category-wise Performance', fontweight='bold')

        plt.tight_layout()

        filename = f'{prefix}_q_vs_baseline.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)

    def _plot_performance_improvement(self, ax, baseline_methods: List[str], q_methods: List[str]):
        """绘制性能提升对比图"""
        summaries = self.data_loader.method_summaries
        
        # 计算基线方法平均性能
        baseline_acc = np.mean([summaries[m].accuracy_mean for m in baseline_methods])
        
        # Q 方法相对于基线的提升
        improvements = []
        colors = []
        labels = []
        
        for method in q_methods:
            s = summaries[method]
            improvement = (s.accuracy_mean - baseline_acc) / baseline_acc * 100
            improvements.append(improvement)
            colors.append(get_method_color(method))
            labels.append(METHOD_DISPLAY_NAMES.get(method, method))
        
        x_pos = np.arange(len(q_methods))
        bars = ax.bar(x_pos, improvements, color=colors, edgecolor='black', alpha=0.8)
        
        # 添加基线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, label='Baseline Avg')
        
        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            color = 'white' if imp < 0 else 'black'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'+{imp:.1f}%', ha='center', va='bottom', fontsize=9, color=color)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Improvement over Baseline (%)', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def _plot_statistical_significance(self, ax, baseline_methods: List[str], q_methods: List[str]):
        """绘制统计显著性分析图（效应量）"""
        summaries = self.data_loader.method_summaries
        
        # 计算基线方法合并标准差
        baseline_stds = [summaries[m].accuracy_std for m in baseline_methods]
        pooled_std = np.sqrt(np.mean([s**2 for s in baseline_stds]))
        baseline_mean = np.mean([summaries[m].accuracy_mean for m in baseline_methods])
        
        # 计算 Cohen's d 效应量
        effect_sizes = []
        colors = []
        labels = []
        
        for method in q_methods:
            s = summaries[method]
            cohens_d = (s.accuracy_mean - baseline_mean) / pooled_std
            effect_sizes.append(cohens_d)
            colors.append(get_method_color(method))
            labels.append(METHOD_DISPLAY_NAMES.get(method, method))
        
        x_pos = np.arange(len(q_methods))
        
        # 效应量颜色编码
        bar_colors = []
        for es in effect_sizes:
            if es >= 0.8:
                bar_colors.append('#27AE60')  # 大效应 - 绿色
            elif es >= 0.5:
                bar_colors.append('#F39C12')  # 中等效应 - 橙色
            else:
                bar_colors.append('#E74C3C')  # 小效应 - 红色
        
        bars = ax.bar(x_pos, effect_sizes, color=bar_colors, edgecolor='black', alpha=0.8)
        
        # 效应量阈值线
        ax.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, label='Small (0.2)')
        ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, label='Medium (0.5)')
        ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1, label='Large (0.8)')
        
        # 添加数值标签
        for bar, es in zip(bars, effect_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'd={es:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Cohen's d Effect Size", fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def _plot_stability_comparison(self, ax, baseline_methods: List[str], q_methods: List[str]):
        """绘制稳定性对比图"""
        summaries = self.data_loader.method_summaries
        
        # 收集数据
        baseline_stds = [summaries[m].accuracy_std * 100 for m in baseline_methods]
        q_stds = [summaries[m].accuracy_std * 100 for m in q_methods]
        
        data_to_plot = [baseline_stds, q_stds]
        colors = ['#95A5A6', '#3498DB']
        labels = ['Baseline', 'Q-Learning']
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        # 添加各个方法的标准差点
        all_stds = []
        all_labels = []
        for method in baseline_methods + q_methods:
            s = summaries[method]
            all_stds.append(s.accuracy_std * 100)
            all_labels.append(METHOD_DISPLAY_NAMES.get(method, method))
        
        ax.set_ylabel('Accuracy Std Dev (%)', fontweight='bold')
        ax.set_title('Stability Comparison (Lower is Better)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    def _plot_category_comparison(self, ax, baseline_methods: List[str], q_methods: List[str]):
        """绘制类别性能对比图"""
        summaries = self.data_loader.method_summaries
        
        # 计算类别平均性能
        categories = ['Baseline', 'Q-Learning', 'DQN']
        
        baseline_mean = np.mean([summaries[m].accuracy_mean * 100 for m in baseline_methods])
        q_mean = np.mean([summaries[m].accuracy_mean * 100 for m in q_methods])
        
        dqn_methods = METHOD_CATEGORIES['dqn']
        available_dqn = [m for m in dqn_methods if m in self.available_methods]
        if available_dqn:
            dqn_mean = np.mean([summaries[m].accuracy_mean * 100 for m in available_dqn])
        else:
            dqn_mean = 0
        
        category_means = [baseline_mean, q_mean, dqn_mean]
        colors = ['#95A5A6', '#3498DB', '#E74C3C']
        
        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, category_means, color=colors, edgecolor='black', alpha=0.8)
        
        # 添加数值标签
        for bar, mean in zip(bars, category_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{mean:.2f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Category-wise Average Performance', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(category_means) * 1.2)

    def plot_subject_wise_comparison(self, prefix: str = '01') -> str:
        """
        生成各被试性能对比图

        Returns:
        --------
        str : 保存的文件路径
        """
        subjects = self.data_loader.get_subjects()
        if not subjects:
            print("  ⚠ 没有被试数据，跳过此图")
            return ''

        n_subjects = len(subjects)
        n_cols = 3
        n_rows = (n_subjects + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_subjects > 1 else [axes]

        method_summaries = self.data_loader.method_summaries

        for idx, subject in enumerate(sorted(subjects)):
            ax = axes[idx]

            # 获取该被试的数据
            subject_df = self.data_loader.get_results_by_subject(subject)

            if subject_df.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            # 绘制该被试的方法对比
            methods = subject_df['method'].unique()
            means = []
            stds = []
            colors = []

            for method in methods:
                method_data = subject_df[subject_df['method'] == method]
                means.append(method_data['accuracy'].mean() * 100)
                stds.append(method_data['accuracy'].std() * 100)
                colors.append(get_method_color(method))

            x_pos = np.arange(len(methods))
            ax.bar(x_pos, means, yerr=stds, capsize=3,
                  color=colors, edgecolor='black', alpha=0.8)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods],
                             rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title(f'{subject}', fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 100)

        # 隐藏多余的子图
        for idx in range(n_subjects, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        filename = f'{prefix}_subject_wise_comparison.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_performance_distribution(self, prefix: str = '01') -> str:
        """
        生成性能分布图（小提琴图 + 散点图）

        Returns:
        --------
        str : 保存的文件路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        method_summaries = self.data_loader.method_summaries
        # 使用固定顺序的方法列表
        methods = self.available_methods

        if not methods:
            print("  ⚠ 没有数据，跳过此图")
            return ''

        # (a) 小提琴图 - 使用方法专属颜色
        ax1 = axes[0]
        data_to_plot = []
        colors = []
        labels = []

        for method in methods:
            s = method_summaries[method]
            data_to_plot.append([x * 100 for x in s.all_accuracies])
            colors.append(get_method_color(method))
            labels.append(METHOD_DISPLAY_NAMES.get(method, method))

        parts = ax1.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                              showmeans=True, showmedians=True, showextrema=True)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('(a) Distribution (Violin Plot)', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) 散点图（每个种子的结果）
        ax2 = axes[1]

        for i, method in enumerate(methods):
            s = method_summaries[method]
            accuracies = [x * 100 for x in s.all_accuracies]

            # 添加抖动
            x_vals = np.random.normal(i, 0.04, len(accuracies))
            ax2.scatter(x_vals, accuracies, alpha=0.7, s=80,
                       color=get_method_color(method),
                       edgecolors='black', linewidth=1,
                       label=METHOD_DISPLAY_NAMES.get(method, method))

            # 绘制均值线
            ax2.hlines(np.mean(accuracies), i-0.2, i+0.2,
                      colors='black', linestyles='dashed', linewidth=2)

        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in methods],
                           rotation=45, ha='right')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('(b) Individual Results', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 100)

        plt.tight_layout()

        filename = f'{prefix}_performance_distribution.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)
    
    def plot_radar_chart(self, prefix: str = '01') -> str:
        """
        生成多指标雷达图

        Returns:
        --------
        str : 保存的文件路径
        """
        method_summaries = self.data_loader.method_summaries
        # 使用固定顺序的方法列表
        methods = self.available_methods

        if not methods:
            print("  ⚠ 没有数据，跳过此图")
            return ''

        # 归一化指标到 0-100
        def normalize(value, min_val, max_val):
            if max_val == min_val:
                return 50
            return (value - min_val) / (max_val - min_val) * 100

        # 计算归一化参数
        all_acc = [s.accuracy_mean for s in method_summaries.values()]
        all_kappa = [s.kappa_mean for s in method_summaries.values()]
        all_f1 = [s.f1_mean for s in method_summaries.values()]

        acc_range = (min(all_acc), max(all_acc))
        kappa_range = (min(all_kappa), max(all_kappa))
        f1_range = (min(all_f1), max(all_f1))

        # 雷达图类别
        radar_categories = ['Accuracy', 'Kappa', 'F1']
        N = len(radar_categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for method in methods:
            s = method_summaries[method]
            color = get_method_color(method)

            values = [
                normalize(s.accuracy_mean * 100, acc_range[0]*100, acc_range[1]*100),
                normalize(s.kappa_mean, kappa_range[0], kappa_range[1]),
                normalize(s.f1_mean, f1_range[0], f1_range[1]),
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                   color=color,
                   label=METHOD_DISPLAY_NAMES.get(method, method))
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_categories, fontsize=11)
        ax.set_title('Multi-metric Performance Comparison',
                    fontweight='bold', pad=20, fontsize=13)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()

        filename = f'{prefix}_radar_chart.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)
    
    def plot_window_analysis(self, prefix: str = '01') -> str:
        """
        生成时间窗分析图

        Returns:
        --------
        str : 保存的文件路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        method_summaries = self.data_loader.method_summaries
        # 使用固定顺序的方法列表
        methods = self.available_methods

        # 过滤有有效时间窗数据的方法
        valid_methods = [m for m in methods
                        if method_summaries[m].t_start_mean > 0]

        if not valid_methods:
            print("  ⚠ 没有时间窗数据，跳过此图")
            return ''

        # (a) 时间窗范围图
        ax1 = axes[0]

        t_starts = [method_summaries[m].t_start_mean for m in valid_methods]
        t_ends = [method_summaries[m].t_end_mean for m in valid_methods]
        lengths = [method_summaries[m].window_length_mean for m in valid_methods]

        x_pos = np.arange(len(valid_methods))
        width = 0.25

        # 使用方法专属颜色
        colors = [get_method_color(m) for m in valid_methods]
        
        ax1.bar(x_pos - width, t_starts, width, label='t_start',
               color=colors, alpha=0.8, edgecolor='black')
        ax1.bar(x_pos, t_ends, width, label='t_end',
               color=colors, alpha=0.8, edgecolor='black', hatch='//')
        ax1.bar(x_pos + width, lengths, width, label='Length',
               color=colors, alpha=0.8, edgecolor='black', hatch='\\\\')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([METHOD_DISPLAY_NAMES.get(m, m) for m in valid_methods],
                           rotation=45, ha='right')
        ax1.set_ylabel('Time (s)', fontweight='bold')
        ax1.set_title('(a) Optimal Time Window', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        # (b) 时间窗与性能关系散点图
        ax2 = axes[1]

        for method in valid_methods:
            s = method_summaries[method]

            # 散点：x=window_length, y=accuracy, size=kappa
            scatter_size = s.kappa_mean * 500 + 50

            ax2.scatter(s.window_length_mean, s.accuracy_mean * 100,
                       s=scatter_size,
                       c=[get_method_color(method)],
                       alpha=0.7, edgecolors='black', linewidth=1.5,
                       label=METHOD_DISPLAY_NAMES.get(method, method))

        ax2.set_xlabel('Window Length (s)', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('(b) Window Length vs. Performance', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f'{prefix}_window_analysis.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)
    
    def generate_comparison_table(self, filename: str = 'performance_comparison.csv') -> str:
        """
        生成性能对比表格
        
        Returns:
        --------
        str : 保存的文件路径
        """
        table_df = self.data_loader.get_comparison_table()
        
        filepath = self.output_dir / 'tables' / filename
        table_df.to_csv(filepath, index=False)
        
        print(f"  ✓ 对比表格已保存到：{filepath}")
        
        return str(filepath)
    
    def generate_latex_table(self, filename: str = 'performance_table.tex') -> str:
        """
        生成 LaTeX 格式的性能对比表格
        
        Returns:
        --------
        str : 保存的文件路径
        """
        lines = []
        lines.append(r"% 性能对比表")
        lines.append(r"% 生成时间：" + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(r"  \caption{Performance Comparison of Different Methods}")
        lines.append(r"  \label{tab:performance_comparison}")
        lines.append(r"  \begin{tabular}{lcccccc}")
        lines.append(r"    \toprule")
        lines.append(r"    \textbf{Method} & \textbf{Accuracy (\%)} & \textbf{Kappa} & \textbf{F1} & \textbf{t\_start (s)} & \textbf{t\_end (s)} & \textbf{Window (s)} \\")
        lines.append(r"    \midrule")
        
        summaries = self.data_loader.method_summaries
        
        # 按准确率排序
        sorted_methods = sorted(
            summaries.keys(),
            key=lambda m: summaries[m].accuracy_mean,
            reverse=True
        )
        
        for method in sorted_methods:
            s = summaries[method]
            display_name = METHOD_DISPLAY_NAMES.get(method, method)
            
            t_start_str = f'{s.t_start_mean:.2f}±{s.t_start_std:.2f}' if s.t_start_mean > 0 else 'N/A'
            t_end_str = f'{s.t_end_mean:.2f}±{s.t_end_std:.2f}' if s.t_end_mean > 0 else 'N/A'
            window_str = f'{s.window_length_mean:.2f}±{s.window_length_std:.2f}' if s.window_length_mean > 0 else 'N/A'
            
            lines.append(
                f"    {display_name} & "
                f"{s.accuracy_mean*100:.2f}±{s.accuracy_std*100:.2f} & "
                f"{s.kappa_mean:.4f}±{s.kappa_std:.4f} & "
                f"{s.f1_mean:.4f}±{s.f1_std:.4f} & "
                f"{t_start_str} & "
                f"{t_end_str} & "
                f"{window_str} \\\\"
            )

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

        filepath = self.output_dir / 'tables' / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  ✓ LaTeX 表格已保存到：{filepath}")

        return str(filepath)

    def generate_training_latex_table(self, 
                                       training_analyzer: 'TrainingPerformanceAnalyzer' = None,
                                       filename: str = 'training_performance_table.tex') -> str:
        """
        生成 LaTeX 格式的训练性能对比表格

        Parameters:
        -----------
        training_analyzer : TrainingPerformanceAnalyzer, optional
            训练性能分析器
        filename : str
            输出文件名

        Returns:
        --------
        str : 保存的文件路径
        """
        if training_analyzer is None:
            print("  ⚠ 未提供训练分析器，跳过训练性能表格")
            return ''

        lines = []
        lines.append(r"% 训练性能对比表")
        lines.append(r"% 生成时间：" + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        lines.append("")
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"  \centering")
        lines.append(r"  \caption{Training Performance Comparison of Q-Learning Methods}")
        lines.append(r"  \label{tab:training_performance}")
        lines.append(r"  \begin{tabular}{lccccc}")
        lines.append(r"    \toprule")
        lines.append(r"    \textbf{Method} & \textbf{Final Acc (\%)} & \textbf{Best Acc (\%)} & \textbf{Avg Reward} & \textbf{Conv. Episode} & \textbf{Conv. Rate (\%)} \\")
        lines.append(r"    \midrule")

        summary_df = training_analyzer.generate_training_summary()

        for _, row in summary_df.iterrows():
            lines.append(
                f"    {row['Method']} & "
                f"{row['Final Accuracy (%)']} & "
                f"{row['Best Accuracy (%)']} & "
                f"{row['Avg Reward']} & "
                f"{row['Convergence Episode']} & "
                f"{row['Convergence Rate (%)']} \\\\"
            )

        lines.append(r"    \bottomrule")
        lines.append(r"  \end{tabular}")
        lines.append(r"\end{table}")

        filepath = self.output_dir / 'tables' / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  ✓ 训练性能 LaTeX 表格已保存到：{filepath}")

        return str(filepath)
