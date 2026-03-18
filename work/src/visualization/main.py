"""
可视化模块主入口

提供命令行接口用于生成学术论文级别的可视化图表
支持分模块生成：
- baseline: 基线方法对比
- q-baseline: Q 方法与基线对比
- q-internal: Q 方法内部对比
- ablation: 消融实验
- all: 生成所有图表
"""

import argparse
import sys
from pathlib import Path

from .data_loader import ExperimentDataLoader
from .modules import (
    BaselineComparisonAnalyzer,
    QVsBaselineAnalyzer,
    QInternalComparisonAnalyzer,
    AblationStudyAnalyzer,
)


def main():
    """命令行主入口"""
    parser = argparse.ArgumentParser(
        description='实验结果可视化分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成所有图表
  python -m src.visualization --experiments-dir results/experiments

  # 仅生成基线方法对比
  python -m src.visualization --experiments-dir results/experiments --module baseline

  # 仅生成 Q 方法与基线对比
  python -m src.visualization --experiments-dir results/experiments --module q-baseline

  # 仅生成 Q 方法内部对比
  python -m src.visualization --experiments-dir results/experiments --module q-internal

  # 仅生成消融实验
  python -m src.visualization --experiments-dir results/experiments --module ablation

  # 指定输出目录
  python -m src.visualization --experiments-dir results/experiments --output-dir figures/paper
        """
    )

    parser.add_argument('--experiments-dir', type=str, default='results/experiments',
                       help='实验数据目录')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='输出图表目录')
    parser.add_argument('--style', type=str, default='paper',
                       choices=['paper', 'presentation', 'default'],
                       help='图表样式')
    parser.add_argument('--module', type=str, default='all',
                       choices=['all', 'baseline', 'q-baseline', 'q-internal', 'ablation'],
                       help='生成的模块')
    parser.add_argument('--prefix', type=str, default=None,
                       help='输出文件名前缀 (默认按模块编号)')

    args = parser.parse_args()

    # 打印头部信息
    print("=" * 70)
    print("实验结果可视化分析工具")
    print("=" * 70)
    print(f"实验数据目录：{args.experiments_dir}")
    print(f"输出目录：{args.output_dir}")
    print(f"图表样式：{args.style}")
    print(f"生成模块：{args.module}")
    print("=" * 70)

    # 检查实验目录
    experiments_path = Path(args.experiments_dir)
    if not experiments_path.exists():
        print(f"\n✗ 错误：实验目录不存在：{args.experiments_dir}")
        sys.exit(1)

    # 加载数据
    print("\n[步骤 1/2] 加载实验数据...")
    try:
        loader = ExperimentDataLoader(args.experiments_dir)
        n_seeds = loader.load_all()
        print(f"✓ 成功加载 {n_seeds} 个种子结果")
        print(f"  - 方法数量：{len(loader.method_summaries)}")
        print(f"  - 被试数量：{len(loader.get_subjects())}")
    except Exception as e:
        print(f"✗ 加载数据失败：{e}")
        sys.exit(1)

    # 显示数据摘要
    print("\n数据摘要:")
    table_df = loader.get_comparison_table()
    print(table_df[['Method', 'Accuracy (%)', 'Kappa']].to_string(index=False))

    # 创建分析器并生成图表
    print(f"\n[步骤 2/2] 生成可视化图表...")
    
    output_base = Path(args.output_dir)
    prefix = args.prefix
    
    if args.module == 'all' or args.module == 'baseline':
        print("\n" + "=" * 50)
        print("模块 1: 基线方法对比分析")
        print("=" * 50)
        analyzer = BaselineComparisonAnalyzer(
            data_loader=loader,
            output_dir=str(output_base),
            style=args.style
        )
        analyzer.plot_all(prefix=prefix or '01')

    if args.module == 'all' or args.module == 'q-baseline':
        print("\n" + "=" * 50)
        print("模块 2: Q 方法与基线方法对比分析")
        print("=" * 50)
        analyzer = QVsBaselineAnalyzer(
            data_loader=loader,
            output_dir=str(output_base),
            style=args.style
        )
        analyzer.plot_all(prefix=prefix or '02')

    if args.module == 'all' or args.module == 'q-internal':
        print("\n" + "=" * 50)
        print("模块 3: Q 方法内部对比分析")
        print("=" * 50)
        analyzer = QInternalComparisonAnalyzer(
            data_loader=loader,
            output_dir=str(output_base),
            style=args.style
        )
        analyzer.plot_all(prefix=prefix or '03')

    if args.module == 'all' or args.module == 'ablation':
        print("\n" + "=" * 50)
        print("模块 4: 消融实验分析")
        print("=" * 50)
        analyzer = AblationStudyAnalyzer(
            data_loader=loader,
            output_dir=str(output_base),
            style=args.style
        )
        analyzer.plot_all(prefix=prefix or '04')

    # 完成
    print("\n" + "=" * 70)
    print("✓ 可视化分析完成!")
    print("=" * 70)
    print(f"\n图表保存在：{output_base.absolute()}")
    print("=" * 70)


if __name__ == '__main__':
    main()
