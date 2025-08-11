import numpy as np
import scipy.io
from datetime import datetime

def save_to_mat(frame_data, filename="raw_data.mat"):
    try:
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        # 添加时间戳到数据字典的键中
        data_with_timestamp = {
            f'frame_{timestamp}': frame_data
        }

        # 使用 scipy 的 savemat 保存为 .mat 文件
        scipy.io.savemat(filename, data_with_timestamp)
        #print(f"数据成功保存到 {filename}，变量名为 frame_{timestamp}")
        return True
    except Exception as e:
        print(f"保存数据时出错: {e}")
        return False

def reorder_frame(frame_bytes: bytes, sample: int, chirp: int, window: np.ndarray | None = None):
        n_ant = 4
        expected_bytes = chirp * n_ant * sample * 2 * 2  # chirp * 天线 * 样点 * (IQ) * int16
        if len(frame_bytes) != expected_bytes:
            raise ValueError(f"帧大小不匹配: got {len(frame_bytes)}, expect {expected_bytes}")

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


