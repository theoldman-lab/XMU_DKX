"""
消融实验分析模块

对比 4 个表格型 Q 方法的演进：
Standard_Q → Double_Q → Dueling_Q → Dueling_Double_Q
展示每种改进带来的性能提升
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


# 消融实验方法顺序（按演进顺序）
ABLATION_METHODS = ['Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q']

# 方法改进说明
ABLATION_DESCRIPTIONS = {
    'Standard_Q': 'Baseline',
    'Double_Q': '+ Double Q',
    'Dueling_Q': '+ Dueling',
    'Dueling_Double_Q': '+ Both',
}

# 颜色方案（蓝色系渐变，体现演进）
ABLATION_COLORS = {
    'Standard_Q': '#2E86AB',       # 标准蓝 - 基线
    'Double_Q': '#56B4E9',         # 天蓝 - Double Q 改进
    'Dueling_Q': '#0072B2',        # 深蓝 - Dueling 改进
    'Dueling_Double_Q': '#009E73', # 蓝绿 - 双重改进
}

# 方法显示名称
DISPLAY_NAMES = {
    'Standard_Q': 'Standard Q-Learning',
    'Double_Q': 'Double Q-Learning',
    'Dueling_Q': 'Dueling Q-Learning',
    'Dueling_Double_Q': 'Dueling Double Q-Learning',
}


class AblationStudyAnalyzer:
    """
    消融实验分析器
    """
    
    def __init__(self,
                 data_loader: ExperimentDataLoader,
                 output_dir: str = 'figures',
                 style: str = 'paper'):
        self.data_loader = data_loader
        self.output_dir = Path(output_dir) / 'ablation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        self._setup_style(style)
        
        self.ablation_methods = [m for m in ABLATION_METHODS 
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
    
    def _get_color(self, method: str) -> str:
        """获取方法颜色"""
        return ABLATION_COLORS.get(method, '#333333')
    
    def plot_all(self, prefix: str = '04') -> Dict[str, str]:
        """生成所有消融实验图表"""
        saved_files = {}
        
        print("生成消融实验图表...")
        print("-" * 50)
        
        print("  [1/4] 生成性能演进图...")
        path = self.plot_performance_evolution(prefix=prefix)
        saved_files['evolution'] = path
        
        print("  [2/4] 生成改进幅度分析图...")
        path = self.plot_improvement_analysis(prefix=prefix)
        saved_files['improvement'] = path
        
        print("  [3/4] 生成时间窗演进图...")
        path = self.plot_window_evolution(prefix=prefix)
        saved_files['window'] = path
        
        print("  [4/4] 生成消融实验表格...")
        path = self.generate_ablation_table(prefix=prefix)
        saved_files['table'] = path
        
        print("-" * 50)
        print(f"✓ 消融实验图表已保存到：{self.output_dir}")
        
        return saved_files
    
    def plot_performance_evolution(self, prefix: str = '04') -> str:
        """
        生成性能演进图
        展示从 Standard_Q 到 Dueling_Double_Q 的性能演进
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        methods = self.ablation_methods
        
        if len(methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 准确率演进
        ax1 = axes[0, 0]
        self._plot_accuracy_evolution(ax1, methods, summaries)
        ax1.set_title('(a) Accuracy Evolution', fontweight='bold')
        
        # (b) Kappa 系数演进
        ax2 = axes[0, 1]
        self._plot_kappa_evolution(ax2, methods, summaries)
        ax2.set_title('(b) Kappa Coefficient Evolution', fontweight='bold')
        
        # (c) F1 分数演进
        ax3 = axes[1, 0]
        self._plot_f1_evolution(ax3, methods, summaries)
        ax3.set_title('(c) F1 Score Evolution', fontweight='bold')
        
        # (d) 综合性能演进
        ax4 = axes[1, 1]
        self._plot_composite_evolution(ax4, methods, summaries)
        ax4.set_title('(d) Composite Score Evolution', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_performance_evolution.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_accuracy_evolution(self, ax, methods: List[str], summaries: Dict):
        """绘制准确率演进图"""
        x_pos = np.arange(len(methods))
        means = [summaries[m].accuracy_mean * 100 for m in methods]
        stds = [summaries[m].accuracy_std * 100 for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [ABLATION_DESCRIPTIONS.get(m, m) for m in methods]
        
        # 使用折线图 + 误差棒
        ax.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5,
                   color=colors[0], linewidth=2, markersize=8,
                   ecolor='gray', elinewidth=2)
        
        # 填充颜色
        for i, method in enumerate(methods):
            ax.scatter([i], [means[i]], s=150, c=colors[i], 
                      edgecolors='black', linewidth=1.5, zorder=5)
        
        # 添加数值标签
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 1, f'{mean:.1f}±{std:.1f}%',
                   ha='center', va='bottom', fontsize=9)
        
        # 添加改进标注
        for i in range(1, len(methods)):
            improvement = means[i] - means[i-1]
            ax.annotate(f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                       xy=(i-0.5, (means[i] + means[i-1])/2),
                       fontsize=8, ha='center', va='bottom',
                       color='green' if improvement > 0 else 'red',
                       fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(means) * 1.15)
    
    def _plot_kappa_evolution(self, ax, methods: List[str], summaries: Dict):
        """绘制 Kappa 系数演进图"""
        x_pos = np.arange(len(methods))
        means = [summaries[m].kappa_mean for m in methods]
        stds = [summaries[m].kappa_std for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [ABLATION_DESCRIPTIONS.get(m, m) for m in methods]
        
        ax.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5,
                   color=colors[0], linewidth=2, markersize=8,
                   ecolor='gray', elinewidth=2)
        
        for i, method in enumerate(methods):
            ax.scatter([i], [means[i]], s=150, c=colors[i],
                      edgecolors='black', linewidth=1.5, zorder=5)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Kappa Coefficient', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_f1_evolution(self, ax, methods: List[str], summaries: Dict):
        """绘制 F1 分数演进图"""
        x_pos = np.arange(len(methods))
        means = [summaries[m].f1_mean for m in methods]
        stds = [summaries[m].f1_std for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [ABLATION_DESCRIPTIONS.get(m, m) for m in methods]
        
        ax.errorbar(x_pos, means, yerr=stds, fmt='o-', capsize=5,
                   color=colors[0], linewidth=2, markersize=8,
                   ecolor='gray', elinewidth=2)
        
        for i, method in enumerate(methods):
            ax.scatter([i], [means[i]], s=150, c=colors[i],
                      edgecolors='black', linewidth=1.5, zorder=5)
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('F1 Score', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_composite_evolution(self, ax, methods: List[str], summaries: Dict):
        """绘制综合性能演进图"""
        x_pos = np.arange(len(methods))
        
        # 综合分数 = 0.5*accuracy + 0.25*kappa*100 + 0.25*f1*100
        composite_scores = []
        for method in methods:
            s = summaries[method]
            score = (s.accuracy_mean * 50 + 
                    s.kappa_mean * 25 + 
                    s.f1_mean * 25)
            composite_scores.append(score)
        
        colors = [self._get_color(m) for m in methods]
        labels = [ABLATION_DESCRIPTIONS.get(m, m) for m in methods]
        
        bars = ax.bar(x_pos, composite_scores, color=colors,
                     edgecolor='black', alpha=0.8, linewidth=1.2)
        
        for bar, score in zip(bars, composite_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 添加改进百分比
        baseline = composite_scores[0]
        for i in range(1, len(methods)):
            improvement = ((composite_scores[i] - baseline) / baseline) * 100
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(i, composite_scores[i] + 1),
                       fontsize=8, ha='center', va='bottom',
                       color='green', fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Composite Score', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_improvement_analysis(self, prefix: str = '04') -> str:
        """
        生成改进幅度分析图
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        summaries = self.data_loader.method_summaries
        methods = self.ablation_methods
        
        if len(methods) < 2:
            print("    ⚠ 数据不足，跳过此图")
            return ''
        
        # (a) 相对基线的提升百分比
        ax1 = axes[0, 0]
        self._plot_relative_improvement(ax1, methods, summaries)
        ax1.set_title('(a) Relative Improvement over Baseline', fontweight='bold')
        
        # (b) 效应量分析
        ax2 = axes[0, 1]
        self._plot_effect_size(ax2, methods, summaries)
        ax2.set_title('(b) Effect Size Analysis', fontweight='bold')
        
        # (c) 稳定性对比
        ax3 = axes[1, 0]
        self._plot_stability_comparison(ax3, methods, summaries)
        ax3.set_title('(c) Stability Comparison', fontweight='bold')
        
        # (d) 性能分布箱线图
        ax4 = axes[1, 1]
        self._plot_distribution(ax4, methods, summaries)
        ax4.set_title('(d) Performance Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_improvement_analysis.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_relative_improvement(self, ax, methods: List[str], summaries: Dict):
        """绘制相对基线的提升百分比"""
        baseline_acc = summaries['Standard_Q'].accuracy_mean * 100
        
        improvements = []
        colors = []
        labels = []
        
        for method in methods:
            acc = summaries[method].accuracy_mean * 100
            improvement = ((acc - baseline_acc) / baseline_acc) * 100
            improvements.append(improvement)
            colors.append(self._get_color(method))
            labels.append(ABLATION_DESCRIPTIONS.get(method, method))
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, improvements, color=colors, edgecolor='black', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
        
        for bar, imp in zip(bars, improvements):
            color = 'white' if imp < 0 else 'black'
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + (0.2 if imp > 0 else -0.5),
                   f'+{imp:.2f}%' if imp > 0 else f'{imp:.2f}%',
                   ha='center', fontsize=9, color=color)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Improvement over Standard Q (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_effect_size(self, ax, methods: List[str], summaries: Dict):
        """绘制效应量分析"""
        baseline = summaries['Standard_Q']
        
        effect_sizes = []
        colors = []
        labels = []
        
        for method in methods:
            s = summaries[method]
            pooled_std = np.sqrt((baseline.accuracy_std**2 + s.accuracy_std**2) / 2)
            cohens_d = (s.accuracy_mean - baseline.accuracy_mean) / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(cohens_d)
            colors.append(self._get_color(method))
            labels.append(ABLATION_DESCRIPTIONS.get(method, method))
        
        x_pos = np.arange(len(methods))
        
        bar_colors = ['#27AE60' if es >= 0.5 else '#F39C12' if es >= 0.2 else '#E74C3C' for es in effect_sizes]
        bars = ax.bar(x_pos, effect_sizes, color=bar_colors, edgecolor='black', alpha=0.8)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=0.2, color='gray', linestyle=':', linewidth=1, label='Small (0.2)')
        ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=1, label='Medium (0.5)')
        ax.axhline(y=0.8, color='green', linestyle=':', linewidth=1, label='Large (0.8)')
        
        for bar, es in zip(bars, effect_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'd={es:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel("Cohen's d (vs Standard Q)", fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_stability_comparison(self, ax, methods: List[str], summaries: Dict):
        """绘制稳定性对比"""
        x_pos = np.arange(len(methods))
        stds = [summaries[m].accuracy_std * 100 for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [ABLATION_DESCRIPTIONS.get(m, m) for m in methods]
        
        # 稳定性用标准差表示，越低越好
        bars = ax.bar(x_pos, stds, color=colors, edgecolor='black', alpha=0.8)
        
        for bar, std in zip(bars, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{std:.2f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy Std Dev (%)', fontweight='bold')
        ax.set_title('Stability (Lower is Better)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def _plot_distribution(self, ax, methods: List[str], summaries: Dict):
        """绘制性能分布箱线图"""
        data_to_plot = []
        colors = []
        labels = []
        
        for method in methods:
            s = summaries[method]
            data_to_plot.append([x * 100 for x in s.all_accuracies])
            colors.append(self._get_color(method))
            labels.append(ABLATION_DESCRIPTIONS.get(method, method))
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
        
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    def plot_window_evolution(self, prefix: str = '04') -> str:
        """生成时间窗演进图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        summaries = self.data_loader.method_summaries
        methods = self.ablation_methods
        
        # 过滤有有效时间窗数据的方法
        valid_methods = [m for m in methods if summaries[m].t_start_mean > 0]
        
        if not valid_methods:
            print("    ⚠ 没有时间窗数据，跳过此图")
            return ''
        
        # (a) 时间窗位置演进
        ax1 = axes[0]
        self._plot_window_position_evolution(ax1, valid_methods, summaries)
        ax1.set_title('(a) Time Window Position Evolution', fontweight='bold')
        
        # (b) 时间窗长度与性能关系
        ax2 = axes[1]
        self._plot_window_length_vs_performance(ax2, valid_methods, summaries)
        ax2.set_title('(b) Window Length vs. Performance', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f'{prefix}_window_evolution.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _plot_window_position_evolution(self, ax, methods: List[str], summaries: Dict):
        """绘制时间窗位置演进"""
        x_pos = np.arange(len(methods))
        width = 0.25
        
        t_starts = [summaries[m].t_start_mean for m in methods]
        t_ends = [summaries[m].t_end_mean for m in methods]
        colors = [self._get_color(m) for m in methods]
        labels = [ABLATION_DESCRIPTIONS.get(m, m) for m in methods]
        
        ax.bar(x_pos - width/2, t_starts, width, label='t_start', color=colors, alpha=0.7)
        ax.bar(x_pos + width/2, t_ends, width, label='t_end', color=colors, alpha=0.7, hatch='//')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha='right')
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
        
        # 添加方法标签和演进箭头
        for i, method in enumerate(methods):
            ax.annotate(DISPLAY_NAMES.get(method, method),
                       (window_lengths[i], accuracies[i]),
                       fontsize=8, ha='center', va='bottom')
            
            # 添加演进箭头
            if i > 0:
                ax.annotate('',
                           xy=(window_lengths[i], accuracies[i]),
                           xytext=(window_lengths[i-1], accuracies[i-1]),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.5))
        
        ax.set_xlabel('Window Length (s)', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def generate_ablation_table(self, prefix: str = '04') -> str:
        """生成消融实验表格"""
        summaries = self.data_loader.method_summaries
        
        rows = []
        baseline_acc = summaries['Standard_Q'].accuracy_mean if 'Standard_Q' in summaries else 0
        
        for method in self.ablation_methods:
            if method not in summaries:
                continue
            s = summaries[method]
            
            # 计算相对于基线的提升
            improvement = ((s.accuracy_mean - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
            
            # 计算效应量
            baseline_std = summaries['Standard_Q'].accuracy_std if 'Standard_Q' in summaries else 0.01
            pooled_std = np.sqrt((baseline_std**2 + s.accuracy_std**2) / 2)
            effect_size = (s.accuracy_mean - baseline_acc) / pooled_std if pooled_std > 0 else 0
            
            rows.append({
                'Method': DISPLAY_NAMES.get(method, method),
                'Improvement': ABLATION_DESCRIPTIONS.get(method, method),
                'Accuracy (%)': f'{s.accuracy_mean*100:.2f} ± {s.accuracy_std*100:.2f}',
                'Improvement (%)': f'{improvement:+.2f}',
                'Effect Size (d)': f'{effect_size:.2f}',
                'Kappa': f'{s.kappa_mean:.4f} ± {s.kappa_std:.4f}',
                'F1': f'{s.f1_mean:.4f} ± {s.f1_std:.4f}',
                'Std Dev (%)': f'{s.accuracy_std*100:.2f}',
                't_start (s)': f'{s.t_start_mean:.2f}' if s.t_start_mean > 0 else 'N/A',
                't_end (s)': f'{s.t_end_mean:.2f}' if s.t_end_mean > 0 else 'N/A',
                'Window (s)': f'{s.window_length_mean:.2f}' if s.window_length_mean > 0 else 'N/A',
            })
        
        df = pd.DataFrame(rows)
        
        csv_path = self.output_dir / f'{prefix}_ablation_study.csv'
        df.to_csv(csv_path, index=False)
        
        tex_path = self.output_dir / f'{prefix}_ablation_study.tex'
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(df.to_latex(index=False, column_format='lccccccccccc'))
        
        print(f"    ✓ 表格已保存到：{csv_path}")
        
        return str(csv_path)
