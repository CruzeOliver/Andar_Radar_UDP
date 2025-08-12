import numpy as np
import scipy.io
from datetime import datetime

def save_to_mat(frame_data, sample_number, chirp_number, filename="raw_data.mat"):
    try:
        # 获取当前时间戳，确保每一帧有唯一的变量名
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        # 计算预期的数据大小：4通道，I/Q每个16bit，每个数据点2字节
        num_antennas = 4  # 4通道
        num_iq = 2        # I/Q 每个16bit = 2字节
        expected_size = sample_number * chirp_number * num_antennas * num_iq * np.dtype(np.int16).itemsize

        # 检查数据的大小
        if len(frame_data) != expected_size:
            print(f"Error: Unexpected buffer size! Expected: {expected_size}, Actual: {len(frame_data)}")
            return False

        # 转换为 int16 数组
        raw_iq = np.frombuffer(frame_data, dtype=np.int16)

        # 假设每帧有 2048 个数据点，且每帧是 32 行，每行 2048 列
        num_rows = 32
        num_cols = 2048
        total_frames = len(raw_iq) // (num_rows * num_cols)  # 计算帧数

        # 检查帧数是否正确
        #print(f"计算到帧数: {total_frames}, 预计每帧大小: {num_rows * num_cols} 字节")

        # 将数据重塑为每帧 32x2048 的 2D 数组
        reshaped_data = raw_iq[:total_frames * num_rows * num_cols].reshape((total_frames, num_rows, num_cols))

        # 加载现有的 .mat 文件，如果文件不存在，则创建一个新文件
        try:
            existing_data = scipy.io.loadmat(filename)
        except FileNotFoundError:
            existing_data = {}

        # 为当前帧生成唯一的变量名（时间戳 + 帧号）
        frame_timestamp = f"frame_{timestamp}"

        # 将当前帧的数据添加到现有数据字典中
        existing_data[frame_timestamp] = reshaped_data[0]

        # 使用 scipy 的 savemat 保存多个帧为独立的工作区（变量）
        scipy.io.savemat(filename, existing_data)
        #print(f"数据成功保存到 {filename}，包含 {len(existing_data)} 帧数据")

        return True
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False

def reorder_frame(frame_bytes: bytes, sample: int, chirp: int, window: np.ndarray | None = None):
        n_ant = 4
        expected_bytes = chirp * n_ant * sample * 2 * 2  # chirp * 天线 * 样点 * (IQ) * int16

        #if len(frame_bytes) != expected_bytes:
            #raise ValueError(f"帧大小不匹配: got {len(frame_bytes)}, expect {expected_bytes}")

        # 读取为 int16，右移4位（除以16）
        arr_iq = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 16.0  # I,Q 交替
        arr_iq = arr_iq.reshape(chirp, n_ant, sample, 2)  # (chirp, ant, sample, IQ)

        I = arr_iq[..., 0]
        Q = arr_iq[..., 1]
        iq = I + 1j * Q  # (chirp, ant, sample)

        # 原始顺序 [TX0RX0, TX1RX0, TX0RX1, TX1RX1] → 虚拟天线 [0, 2, 1, 3]
        virtual_ant_map = [0, 2, 1, 3]
        iq = iq[:, virtual_ant_map, :]          # (chirp, 4, sample)
        iq = np.transpose(iq, (1, 0, 2))        # -> (4, chirp, sample)

        if window is not None:
            if len(window) != sample:
                raise ValueError("window 长度必须等于 sample")
            iq = iq * window[np.newaxis, np.newaxis, :]

        return iq

def process_iq_data(iq_data, sample, chirp):
    """
    对 I/Q 数据进行进一步处理，比如做 FFT 或其他信号处理操作
    """
    # 幅度处理（取复数幅度）
    magnitude = np.abs(iq_data)


    return magnitude


