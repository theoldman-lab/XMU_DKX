import numpy as np
import mne
import os
from scipy.io import loadmat


def load_bci_competition_data_npz(file_path, subject_id, session='T'):
    """
    加载 BCI Competition IV 2a 数据集 (.npz 格式)
    
    Parameters:
    -----------
    file_path : str
        数据文件路径
    subject_id : int
        受试者 ID (1-9)
    session : str
        会话类型 ('T' for training, 'E' for evaluation)
    
    Returns:
    --------
    eeg_data : array, shape (n_trials, n_channels, n_times)
        EEG 数据
    labels : array, shape (n_trials,)
        试次标签
    fs : int
        采样频率
    
    Notes:
    ------
    数据格式说明:
    - 's': 原始信号数据，形状为 (时间点，通道)
    - 'etyp': 事件类型，768=trial start, 769=left, 770=right, 771=foot, 772=tongue
    - 'epos': 事件位置索引
    - 'edur': 事件持续时间
    """
    # 文件名格式 (A01T.npz, A01E.npz 等)
    if session == 'T':
        file_name = f"A{subject_id:02d}T.npz"
    else:
        file_name = f"A{subject_id:02d}E.npz"
    
    full_path = os.path.join(file_path, file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file does not exist: {full_path}")
    
    try:
        # 加载 npz 文件
        data = np.load(full_path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load npz file: {e}")
    
    # 验证数据是否包含必要的键
    required_keys = ['s', 'etyp', 'epos', 'edur']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key '{key}' in data file {full_path}")
    
    # 提取数据
    signal = data['s'].T  # 转置为 (通道，时间点)
    
    # 验证信号数据形状
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal data, got {signal.ndim}D")
    
    events_type = data['etyp'].T[0]  # 事件类型
    events_position = data['epos'].T[0]  # 事件位置
    events_duration = data['edur'].T[0]  # 事件持续时间
    
    # 验证事件数据长度一致
    if not (len(events_type) == len(events_position) == len(events_duration)):
        raise ValueError("Event data arrays have inconsistent lengths")
    
    # 运动想象任务类型映射
    mi_types = {
        769: 0,  # left hand -> class 0
        770: 1,  # right hand -> class 1
        771: 2,  # foot -> class 2
        772: 3   # tongue -> class 3
    }
    
    # 提取所有 trial
    trials = []
    labels = []
    
    # 找到所有 trial start 事件 (代码 768)
    starttrial_code = 768
    starttrial_events = events_type == starttrial_code
    idxs = [i for i, x in enumerate(starttrial_events) if x]
    
    if len(idxs) == 0:
        raise ValueError("No trial start events found in data")
    
    for index in idxs:
        try:
            # 确保不会越界
            if index + 1 >= len(events_type):
                continue
                
            # 获取 trial 类型 (下一个事件的类型)
            # 获取 trial 类型 (下一个事件的类型)
            type_e = events_type[index + 1]
            
            # 跳过未知类型
            if type_e not in mi_types:
                continue
            
            class_e = mi_types[type_e]
            labels.append(class_e)
            
            # 提取 trial 信号 (所有通道)
            start = int(events_position[index])
            stop = start + int(events_duration[index])
            
            # 验证索引范围
            if start < 0 or stop > signal.shape[1]:
                continue
                
            trial = signal[:, start:stop]  # (通道，时间)
            
            # 验证 trial 形状
            if trial.shape[0] == 0 or trial.shape[1] == 0:
                continue
                
            # 确保 trial 长度一致
            expected_length = int(4.0 * 250)  # 4 秒 * 250Hz
            if trial.shape[1] < expected_length:
                # 如果 trial 太短，进行填充
                padding = np.zeros((trial.shape[0], expected_length - trial.shape[1]))
                trial = np.hstack([trial, padding])
            elif trial.shape[1] > expected_length:
                # 如果 trial 太长，进行截断
                trial = trial[:, :expected_length]
            
            trials.append(trial)
            
        except Exception as e:
            # 跳过无效的 trial
            print(f"Skipping invalid trial at index {index}: {e}")
            continue
    
    if len(trials) == 0:
        raise ValueError("No valid trials found in data")
    
    # 转换为 numpy 数组
    eeg_data = np.array(trials)  # (n_trials, n_channels, n_times)
    mapped_labels = np.array(labels)  # (n_trials,)
    
    # 验证数据形状
    if eeg_data.shape[0] != len(mapped_labels):
        raise ValueError("Number of trials does not match number of labels")
    
    # 采样频率
    fs = 250  # BCI Competition IV 2a 采样率为 250Hz
    
    return eeg_data, mapped_labels, fs


def load_bci_competition_data(file_path, subject_id, session='T', use_npz=False):
    """
    加载 BCI Competition IV 2a 数据集 (支持.mat 和.npz 格式)
    
    Parameters:
    -----------
    file_path : str
        数据文件路径
    subject_id : int
        受试者 ID (1-9)
    session : str
        会话类型 ('T' for training, 'E' for evaluation)
    use_npz : bool
        是否使用 npz 格式，默认 False 使用 mat 格式
    
    Returns:
    --------
    eeg_data : array, shape (n_trials, n_channels, n_times)
        EEG 数据
    labels : array, shape (n_trials,)
        试次标签
    fs : int
        采样频率
    """
    if use_npz:
        return load_bci_competition_data_npz(file_path, subject_id, session)
    
    # 文件名格式
    if session == 'T':
        file_name = f"S{subject_id:02d}T.mat"
    else:
        file_name = f"S{subject_id:02d}E.mat"
    
    full_path = os.path.join(file_path, file_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Data file does not exist: {full_path}")
    
    try:
        # 加载 MAT 文件
        mat_data = loadmat(full_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load mat file: {e}")
    
    # 提取数据
    try:
        if session == 'T':
            eeg_data = mat_data['data_train']['x'][0][0]
            labels = mat_data['data_train']['y_dec'][0][0].flatten()
        else:
            eeg_data = mat_data['data_test']['x'][0][0]
            labels = mat_data['data_test']['y_dec'][0][0].flatten()
    except KeyError as e:
        raise KeyError(f"Missing expected keys in mat file {full_path}: {e}")
    
    # 修正：使用固定的标签映射，与NPZ文件加载保持一致
    # 假设MAT文件中的标签也是按照相同的顺序：769=left, 770=right, 771=foot, 772=tongue
    # 但是MAT文件中可能使用的是实际标签值而不是事件编码
    # 所以我们先对标签进行标准化映射
    unique_labels = np.unique(labels)
    # 确保标签是连续的从0开始的整数
    label_map = {}
    for i, label in enumerate(sorted(unique_labels)):
        label_map[label] = i
    mapped_labels = np.array([label_map[label] for label in labels])
    
    # 验证标签数量
    if len(np.unique(mapped_labels)) != 4:
        print(f"Warning: Expected 4 classes but found {len(np.unique(mapped_labels))} classes for subject {subject_id}")
    
    # 重新塑形数据 (trial, channel, time)
    if len(eeg_data.shape) == 3:
        # 如果已经是 (trial, channel, time) 格式
        pass
    else:
        # 需要根据具体数据格式进行调整
        # 这里假设数据是 (channel, time, trial) 格式
        eeg_data = np.transpose(eeg_data, (2, 0, 1))
    
    # 采样频率
    fs = 250  # BCI Competition IV 2a 采样率为 250Hz
    
    return eeg_data, mapped_labels, fs


def create_epochs_from_raw(eeg_data, labels, fs=250, t_min=0, t_max=4):
    """
    从原始 EEG 数据创建 Epochs
    
    Parameters:
    -----------
    eeg_data : array, shape (n_trials, n_channels, n_times)
        EEG 数据
    labels : array, shape (n_trials,)
        试次标签
    fs : int
        采样频率
    t_min, t_max : float
        时间窗口边界 (秒)
    
    Returns:
    --------
    epochs_data : array, shape (n_trials, n_channels, n_times_in_window)
        epoch 数据
    """
    # 验证输入数据
    if eeg_data.size == 0:
        raise ValueError("Input EEG data is empty")
    
    if len(eeg_data) != len(labels):
        raise ValueError(f"Mismatch between number of trials and labels: {len(eeg_data)} vs {len(labels)}")
    
    start_idx = int(t_min * fs)
    end_idx = int(t_max * fs)
    
    # 确保索引在范围内
    end_idx = min(end_idx, eeg_data.shape[2])
    start_idx = max(0, start_idx)
    
    if start_idx >= end_idx:
        raise ValueError(f"Invalid time window: start_idx {start_idx} >= end_idx {end_idx}")
    
    epochs_data = eeg_data[:, :, start_idx:end_idx]
    
    return epochs_data


def load_subject_data(data_path, subject_id, sessions=['T', 'E']):
    """
    加载指定受试者的完整数据
    
    Parameters:
    -----------
    data_path : str
        数据根目录路径
    subject_id : int
        受试者 ID
    sessions : list
        要加载的会话列表
    
    Returns:
    --------
    data_dict : dict
        包含训练和测试数据的字典
    """
    data_dict = {}
    
    for session in sessions:
        try:
            eeg_data, labels, fs = load_bci_competition_data(data_path, subject_id, session)
            
            # 验证加载的数据
            if eeg_data.size == 0:
                print(f"Warning: Empty data for subject {subject_id}, session {session}")
                continue
                
            if len(np.unique(labels)) < 2:
                print(f"Warning: Less than 2 classes in data for subject {subject_id}, session {session}")
                continue
            
            # 创建 epochs
            epochs_data = create_epochs_from_raw(eeg_data, labels, fs)
            
            if session == 'T':
                data_dict['train_data'] = epochs_data
                data_dict['train_labels'] = labels
            else:
                data_dict['test_data'] = epochs_data
                data_dict['test_labels'] = labels
        except Exception as e:
            print(f"Failed to load data for subject {subject_id}, session {session}: {e}")
    
    return data_dict


def get_complete_dataset(data_path, subjects_list):
    """
    获取完整数据集
    
    Parameters:
    -----------
    data_path : str
        数据根目录路径
    subjects_list : list
        受试者 ID 列表
    
    Returns:
    --------
    dataset : dict
        包含所有受试者数据的字典
    """
    dataset = {}
    
    for subj_id in subjects_list:
        # 将字符串 ID 转换为整数
        if isinstance(subj_id, str):
            subj_num = int(subj_id.replace("S0", "").replace("S", ""))
        else:
            subj_num = subj_id
            
        try:
            subj_data = load_subject_data(data_path, subj_num)
            if len(subj_data) > 0:  # 只有当数据成功加载时才存储
                dataset[f"S{subj_num:02d}"] = subj_data
                print(f"Successfully loaded data for subject S{subj_num:02d}")
            else:
                print(f"No valid data found for subject S{subj_num:02d}")
        except Exception as e:
            print(f"Failed to load data for subject S{subj_num:02d}: {str(e)}")
            continue
    
    return dataset


def preprocess_raw_eeg(eeg_data, fs=250):
    """
    预处理原始 EEG 数据
    
    Parameters:
    -----------
    eeg_data : array, shape (n_trials, n_channels, n_times)
        原始 EEG 数据
    fs : int
        采样频率
    
    Returns:
    --------
    processed_data : array
        预处理后的数据
    """
    # 这里使用与 preprocessing.py 中类似的方法
    # 但在实际应用中，可能会有更复杂的预处理
    from src.data.preprocessing import preprocess_eeg
    
    processed_data = preprocess_eeg(eeg_data, fs)
    
    return processed_data