import numpy as np
import mne
from scipy import signal

mne.set_log_level("WARNING")


def preprocess_eeg(raw_eeg_data, sfreq=250):
    """
    EEG数据预处理流水线
    :param raw_eeg_data: 原始EEG数据，形状为 (n_trials, n_channels, n_samples)
    :param sfreq: 采样频率，默认250Hz
    :return: 预处理后的EEG数据
    """
    # 验证输入数据
    if raw_eeg_data.size == 0:
        raise ValueError("Input EEG data is empty")

    if raw_eeg_data.ndim != 3:
        raise ValueError(
            f"Expected 3D EEG data (n_trials, n_channels, n_samples), got {raw_eeg_data.ndim}D"
        )

    # 创建通道信息
    ch_names = [f"EEG{i:02d}" for i in range(raw_eeg_data.shape[1])]  # 生成通道名称
    ch_types = ["eeg"] * raw_eeg_data.shape[1]  # 为每个通道指定类型

    # 创建info对象
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    processed_trials = []
    for i, trial in enumerate(raw_eeg_data):
        try:
            # 验证单个试验数据
            if trial.size == 0:
                print(f"Warning: Empty trial at index {i}, skipping...")
                continue

            # 创建MNE Raw对象
            raw = mne.io.RawArray(trial / 1e6, info)  # 转换为伏特单位

            # 平均重参考
            raw.set_eeg_reference(ref_channels="average")

            # 8-30Hz带通滤波
            raw.filter(l_freq=8, h_freq=30, method="fir", fir_design="firwin")

            # 50Hz陷波滤波
            raw.notch_filter(freqs=50, method="iir")

            # 转回numpy数组
            processed_trial = raw.get_data() * 1e6  # 转回微伏单位
            processed_trials.append(processed_trial)

        except Exception as e:
            print(f"Error processing trial {i}: {e}")
            # 添加原始数据作为后备方案
            processed_trials.append(trial)

    if len(processed_trials) == 0:
        raise ValueError("All trials failed to process")

    return np.array(processed_trials)


def extract_time_window(eeg_data, t_start, t_end, sfreq=250):
    """
    截取指定时间窗内的EEG数据
    :param eeg_data: EEG数据，形状为 (n_trials, n_channels, n_samples)
    :param t_start: 开始时间 (秒)
    :param t_end: 结束时间 (秒)
    :param sfreq: 采样频率
    :return: 截取后的时间窗数据
    """
    # 验证输入数据
    if eeg_data.size == 0:
        raise ValueError("Input EEG data is empty")

    if eeg_data.ndim != 3:
        raise ValueError(
            f"Expected 3D EEG data (n_trials, n_channels, n_samples), got {eeg_data.ndim}D"
        )

    # 验证时间参数
    if t_start >= t_end:
        raise ValueError(f"Start time ({t_start}) must be less than end time ({t_end})")

    if t_start < 0:
        raise ValueError(f"Start time ({t_start}) must be non-negative")

    start_idx = int(t_start * sfreq)
    end_idx = int(t_end * sfreq)

    # 确保索引在范围内
    start_idx = max(0, start_idx)
    end_idx = min(eeg_data.shape[2], end_idx)

    # 再次验证索引
    if start_idx >= end_idx:
        raise ValueError(
            f"Time window results in empty selection: start_idx {start_idx} >= end_idx {end_idx}"
        )

    return eeg_data[:, :, start_idx:end_idx]
