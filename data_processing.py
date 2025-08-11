import numpy as np

def reorder_frame(frame_bytes: bytes, sample: int, chirp: int):
    """
    重新排列接收到的帧数据
    """
    n_ant = 4
    expected_bytes = chirp * n_ant * sample * 2 * 2  # chirp * 天线 * 样点 * (IQ) * int16
    if len(frame_bytes) != expected_bytes:
        raise ValueError(f"帧大小不匹配: got {len(frame_bytes)}, expect {expected_bytes}")

    # 读取为 int16，逻辑做 右移4位（除以16）
    arr_iq = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 16.0  # I,Q 交替
    arr_iq = arr_iq.reshape(chirp, n_ant, sample, 2)  # (chirp, ant, sample, IQ)

    I = arr_iq[..., 0]
    Q = arr_iq[..., 1]
    iq = I + 1j * Q  # (chirp, ant, sample)

    # 虚拟天线映射表：将原始顺序 [TX0RX0, TX1RX0, TX0RX1, TX1RX1] 映射为 [0, 2, 1, 3]
    virtual_ant_map = [0, 2, 1, 3]
    iq = iq[:, virtual_ant_map, :]          # (chirp, 4, sample)
    iq = np.transpose(iq, (1, 0, 2))        # -> (4, chirp, sample)

    return iq

def process_iq_data(iq_data, sample, chirp):
    """
    对 I/Q 数据进行进一步处理，比如做 FFT 或其他信号处理操作
    """
    # 幅度处理（取复数幅度）
    magnitude = np.abs(iq_data)


    return magnitude


