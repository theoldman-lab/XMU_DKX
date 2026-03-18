"""
简化的实验流程控制模块

设计原则：
1. 简单：只保留核心功能
2. 清晰：代码逻辑一目了然
3. 实用：支持单次实验和多次实验数据整合

使用流程：
1. 运行单个方法：python -m src.experiments.experiment_simple --method Double_Q --subjects S01 --seeds 5
2. 运行另一个方法：python -m src.experiments.experiment_simple --method Dueling_Q --subjects S01 --seeds 5
3. 整合所有结果：python -m src.experiments.experiment_simple --aggregate

注意：模型创建逻辑已移至 src/experiments/q_models.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.data.data_loader import load_bci_competition_data_npz
from src.data.preprocessing import preprocess_eeg
from src.experiments.evaluator import evaluate_time_window, aggregate_results
from src.experiments.data_manager import create_collector, create_manager
from src.experiments.q_models import create_ql_model, is_ql_method, get_model_type
from src.methods.baseline_methods import (
    fixed_2s_group_experiment, fixed_1_5s_group_experiment, fixed_1s_group_experiment,
    get_fixed_window_candidates, grid_search_method, random_search_method,
)


# def compute_optimal_window_statistics(optimal_windows):
#     """计算最优时间窗的统计信息"""
#     if not optimal_windows:
#         return {
#             't_start_mean': None, 't_start_std': None,
#             't_end_mean': None, 't_end_std': None,
#             'window_length_mean': None, 'window_length_std': None,
#         }

#     t_starts = [w[0] for w in optimal_windows]
#     t_ends = [w[1] for w in optimal_windows]
#     window_lengths = [w[1] - w[0] for w in optimal_windows]

#     return {
#         't_start_mean': float(np.mean(t_starts)),
#         't_start_std': float(np.std(t_starts)),
#         't_end_mean': float(np.mean(t_ends)),
#         't_end_std': float(np.std(t_ends)),
#         'window_length_mean': float(np.mean(window_lengths)),
#         'window_length_std': float(np.std(window_lengths)),
#     }


# 基线方法注册表
# FL-2s, FL-1.5s, FL-1s 会在训练时返回所有候选窗口，由 train_method 对每个窗口进行完整实验
BASELINE_METHODS = {
    'FL-2s': fixed_2s_group_experiment,
    'FL-1.5s': fixed_1_5s_group_experiment,
    'FL-1s': fixed_1s_group_experiment,
    'Grid_Search': grid_search_method,
    'Random_Search': random_search_method,
}


def _get_baseline_windows(method_name):
    """获取基线方法的候选窗口列表（用于批量实验）"""
    if method_name == 'FL-2s':
        return get_fixed_window_candidates(2.0)
    elif method_name == 'FL-1.5s':
        return get_fixed_window_candidates(1.5)
    elif method_name == 'FL-1s':
        return get_fixed_window_candidates(1.0)
    else:
        return None  # 其他方法返回 None，表示只返回单个最优窗口


def _train_ql_method(method_name, eeg_train, labels_train, verbose=True):
    """训练 Q-learning 方法"""
    model_type = get_model_type(method_name)
    model = create_ql_model(model_type)
    optimal_window = model.train(eeg_train, labels_train, verbose=verbose)
    training_logs = model.get_training_logs()
    reward_history = model.get_reward_history()
    return optimal_window, reward_history, training_logs


def _train_baseline_method(method_name, eeg_train, labels_train):
    """
    训练基线方法
    
    对于 FL-2s/FL-1.5s/FL-1s 方法，返回所有候选窗口及其评估结果
    对于其他方法，返回最优窗口
    """
    method_func = BASELINE_METHODS.get(method_name)
    if method_func is None:
        raise ValueError(f"未知的基线方法：{method_name}")
    
    # 对于 FL 组方法，method_func 返回 (optimal_window, all_results)
    if method_name in ['FL-2s', 'FL-1.5s', 'FL-1s']:
        optimal_window, all_results = method_func(eeg_train, labels_train)
        return optimal_window, [], {'all_window_results': all_results}
    else:
        # 其他方法直接返回最优窗口
        optimal_window = method_func(eeg_train, labels_train)
        return optimal_window, [], {}


def train_method(method_name, eeg_train, labels_train, eeg_test, labels_test,
                 n_seeds=5, subject='S01'):
    """
    训练单个方法并收集完整数据
    
    对于 FL-2s/FL-1.5s/FL-1s 方法：
    - 每个候选窗口都会在所有种子上进行评估
    - 每个种子独立评估所有窗口
    
    对于其他方法：
    - 只在单个最优窗口上评估

    Parameters:
    -----------
    method_name : str
        方法名称
    eeg_train, labels_train, eeg_test, labels_test : array
        EEG 数据和标签
    n_seeds : int
        随机种子数量
    subject : str
        受试者 ID

    Returns:
    --------
    dict : 实验数据包
    """
    print(f"\n{'='*70}")
    print(f"训练方法：{method_name}")
    print(f"种子数：{n_seeds}")
    print(f"{'='*70}")

    # 创建数据收集器
    collector = create_collector(method_name, subject, n_seeds)

    eeg_train_processed = preprocess_eeg(eeg_train)
    eeg_test_processed = preprocess_eeg(eeg_test)

    # 检查是否为 FL 组方法（需要对每个窗口进行实验）
    is_fl_group = method_name in ['FL-2s', 'FL-1.5s', 'FL-1s']
    
    if is_fl_group:
        # FL 组方法：对每个候选窗口进行完整实验
        candidate_windows = _get_baseline_windows(method_name)
        print(f"候选窗口数：{len(candidate_windows)}")
        print(f"每个窗口将独立进行 {n_seeds} 个种子的实验")
        print(f"总实验次数：{len(candidate_windows) * n_seeds}")
        print()
        
        for window_idx, (t_start, t_end) in enumerate(candidate_windows):
            print(f"\n--- 窗口 {window_idx + 1}/{len(candidate_windows)}: [{t_start:.1f}s, {t_end:.1f}s] ---")
            
            for seed in range(n_seeds):
                print(f"[种子 {seed+1}/{n_seeds}] ", end="", flush=True)
                np.random.seed(seed)
                
                try:
                    # 评估时间窗
                    metrics = evaluate_time_window(
                        optimal_window=(t_start, t_end),
                        eeg_train=eeg_train_processed,
                        labels_train=labels_train,
                        eeg_test=eeg_test_processed,
                        labels_test=labels_test,
                        use_normalization=True,
                        classifier_type='svm',
                        verbose=False
                    )
                    
                    # 收集数据
                    collector.add_seed_result(
                        seed=seed,
                        optimal_window=(t_start, t_end),
                        metrics=metrics,
                        reward_history=[],
                        training_logs={'window_index': window_idx},
                        success=True
                    )
                    
                    print(f"Acc={metrics['accuracy']:.4f}")
                    
                except Exception as e:
                    print(f"✗ 失败：{e}")
                    import traceback
                    traceback.print_exc()
                    
                    collector.add_seed_result(
                        seed=seed,
                        optimal_window=(t_start, t_end),
                        metrics={'accuracy': 0.0, 'kappa': 0.0, 'f1': 0.0},
                        reward_history=[],
                        training_logs={'window_index': window_idx},
                        success=False,
                        error=str(e)
                    )
    else:
        # 其他方法：正常流程
        for seed in range(n_seeds):
            print(f"[种子 {seed+1}/{n_seeds}] ", end="")
            np.random.seed(seed)
            
            try:
                # 判断方法类型并训练
                if is_ql_method(method_name):
                    optimal_window, reward_history, model_logs = _train_ql_method(
                        method_name, eeg_train_processed, labels_train, verbose=True
                    )
                else:
                    optimal_window, reward_history, model_logs = _train_baseline_method(
                        method_name, eeg_train_processed, labels_train
                    )
                
                # 评估性能
                metrics = evaluate_time_window(
                    optimal_window,
                    eeg_train_processed, labels_train,
                    eeg_test_processed, labels_test,
                    use_normalization=True,
                    classifier_type='svm',
                    verbose=True
                )
                
                # 收集数据
                collector.add_seed_result(
                    seed=seed,
                    optimal_window=optimal_window,
                    metrics=metrics,
                    reward_history=reward_history,
                    training_logs=model_logs,
                    success=True
                )
                
                print(f"Acc={metrics['accuracy']:.4f}, Window={optimal_window}")
                
            except Exception as e:
                print(f"✗ 失败：{e}")
                import traceback
                traceback.print_exc()

                collector.add_seed_result(
                    seed=seed,
                    optimal_window=None,
                    metrics={'accuracy': 0.0, 'kappa': 0.0, 'f1': 0.0},
                    reward_history=[],
                    training_logs={},
                    success=False,
                    error=str(e)
                )

    # 保存数据
    data_manager = create_manager()
    saved_path = data_manager.save(collector, compress=True)
    print(f"\n✓ 数据保存到：{saved_path}")

    # 返回摘要
    summary = collector.get_summary()
    print(f"\n实验摘要:")
    print(f"  成功率：{summary.get('success_rate', 0):.1f}%")
    print(f"  平均准确率：{summary.get('accuracy_mean', 0):.4f} ± {summary.get('accuracy_std', 0):.4f}")
    print(f"  平均 Kappa: {summary.get('kappa_mean', 0):.4f} ± {summary.get('kappa_std', 0):.4f}")

    return collector.package()


def run_experiment(method_name, subjects_list, n_seeds=5, save_dir='results'):
    """运行单个方法的完整实验"""
    print(f"\n{'='*70}")
    print(f"开始实验：{method_name}")
    print(f"受试者：{subjects_list}")
    print(f"{'='*70}")

    all_rows = []

    for subject_id in subjects_list:
        print(f"\n[受试者 {subject_id}]")
        eeg_train, labels_train, eeg_test, labels_test = load_subject_data(subject_id)

        result = train_method(
            method_name, eeg_train, labels_train, eeg_test, labels_test,
            n_seeds=n_seeds, subject=subject_id
        )

        # 汇总到 DataFrame
        for seed_result in result['seed_results']:
            metrics = seed_result.get('metrics', {})
            window = seed_result.get('optimal_window')
            row = {
                'method': method_name,
                'subject': subject_id,
                'seed': seed_result.get('seed', 0),
                'accuracy': metrics.get('accuracy', 0),
                'kappa': metrics.get('kappa', 0),
                'f1': metrics.get('f1', 0),
                't_start': window[0] if window and len(window) >= 2 else None,
                't_end': window[1] if window and len(window) >= 2 else None,
            }
            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    
    # 保存汇总结果
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(save_dir, f"{method_name}_summary_{timestamp}.csv")
    results_df.to_csv(summary_file, index=False)
    print(f"\n✓ 汇总结果保存到：{summary_file}")

    return results_df


def load_subject_data(subject_id, verbose=False, use_train_as_test=True):
    """加载受试者数据"""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'bcidatasetIV2a-master'
    )
    subj_num = int(subject_id.replace("S0", "").replace("S", "")) if isinstance(subject_id, str) else int(subject_id)

    if verbose:
        print(f"加载受试者 {subj_num:02d} 数据...")

    eeg_train, labels_train, fs = load_bci_competition_data_npz(data_path, subj_num, session='T')

    if use_train_as_test:
        eeg_test = eeg_train.copy()
        labels_test = labels_train.copy()
    else:
        try:
            eeg_test, _, _ = load_bci_competition_data_npz(data_path, subj_num, session='E')
            labels_test = np.zeros(len(eeg_test), dtype=int)
        except Exception as e:
            print(f"警告：无法加载评估集，使用训练集代替：{e}")
            eeg_test = eeg_train.copy()
            labels_test = labels_train.copy()

    return eeg_train, labels_train, eeg_test, labels_test


def aggregate_all_results(results_dir='results/experiments'):
    """聚合所有实验结果"""
    print(f"\n{'='*70}")
    print("聚合所有实验结果")
    print(f"{'='*70}")
    
    data_manager = create_manager(results_dir)
    df = data_manager.aggregate_all()
    
    if df is not None and len(df) > 0:
        print("\n方法性能对比:")
        print(df.groupby('method')['accuracy', 'kappa', 'f1'].mean().round(4).to_string())
    
    return df


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='实验脚本 - 简化版')
    parser.add_argument('--method', type=str,
                       choices=['FL-2s', 'FL-1.5s', 'FL-1s',
                               'Grid_Search', 'Random_Search',
                               'Standard_Q', 'Double_Q', 'Dueling_Q', 'Dueling_Double_Q', 'DQN'],
                       help='要运行的方法')
    parser.add_argument('--subjects', type=str, nargs='+', default=['S01','S02','S03','S04','S05','S06','S07','S08','S09'],
                       help='受试者列表')
    parser.add_argument('--seeds', type=int, default=5, help='随机种子数量')
    parser.add_argument('--save-dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--aggregate', action='store_true', help='聚合所有实验结果')

    args = parser.parse_args()

    print("="*70)
    print("实验脚本 - 简化版")
    print("="*70)

    if args.aggregate:
        aggregate_all_results(os.path.join(args.save_dir, 'experiments'))
    elif args.method:
        run_experiment(
            args.method, args.subjects,
            n_seeds=args.seeds, save_dir=args.save_dir
        )
    else:
        print("请指定 --method 或 --aggregate")
        print("\n示例:")
        print("  python -m src.experiments.experiment_simple --method Double_Q --subjects S01 --seeds 5")
        print("  python -m src.experiments.experiment_simple --aggregate")


if __name__ == "__main__":
    main()
