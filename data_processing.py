import numpy as np
import scipy.io
from datetime import datetime

C = 3e8  # 光速，单位 m/s
CenterFrequency = 77  # 中心频率，单位 GHz
wavelength = C / (CenterFrequency * 1e9)  # 波长，单位 m
ADC_SAMPLE_RATE = 7.14  # 采样率，单位 MHz
FM = 3000  # 调频带宽，单位 MHz
CHIRP_T0 = 98  # 微秒
CHIRP_T1 = 14  # 微秒
CHIRP_T2 = 0   # 微秒
CHIRP_PERIOD = CHIRP_T0 + CHIRP_T1 + CHIRP_T2  # Chirp周期，单位微秒

def reorder_frame(frame_bytes: bytes, sample: int, chirp: int, window: np.ndarray | None = None):
    n_ant = 4
    #expected_bytes = chirp * n_ant * sample * 2 * 2  # chirp * 天线 * 样点 * (IQ) * int16
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


def calculate_distance_from_fft(adc_data, n_chirp, n_points):
    """
    使用多种方法从ADC数据中计算距离。
    处理逻辑为：选择第0个虚拟天线的所有Chirp数据，进行平均后做FFT。
    然后使用Macleod和Chirp-Z插值进行峰值细化。
    """
    # 步骤 1: 重排帧数据并选择第0个虚拟天线的数据
    iq = reorder_frame(adc_data, n_points, n_chirp)

    # iq的形状为 (4, n_chirp, n_points)
    # 选取第0个虚拟天线的数据，形状为 (n_chirp, n_points)
    data_for_fft = iq[0, :, :]

    # 对所有chirp的数据进行平均（或求和），以提高信噪比
    # 结果 x_complex 形状为 (n_points,)，是一个一维数组
    x_complex = np.mean(data_for_fft, axis=0)

    # 步骤 2: 对数据进行FFT
    fft_result = np.fft.fft(x_complex)

    # 步骤 3: 计算幅度谱并找到FFT峰值
    fft_sum = np.abs(fft_result)
    valid_points = n_points // 2
    max_index = np.argmax(fft_sum[:valid_points])

    # 计算频率偏移
    f_fft_peak = max_index * ADC_SAMPLE_RATE * 1e6 / n_points

    # 步骤 4: Macleod 插值
    X_km1 = fft_result[max(0, max_index - 1)]
    X_k0 = fft_result[max_index]
    X_kp1 = fft_result[min(valid_points - 1, max_index + 1)]

    mag2_km1 = np.abs(X_km1)**2
    mag2_k0 = np.abs(X_k0)**2
    mag2_kp1 = np.abs(X_kp1)**2

    denom = mag2_km1 - 2 * mag2_k0 + mag2_kp1
    delta = 0.5 * (mag2_km1 - mag2_kp1) / denom if denom != 0 else 0.0
    f_macleod = (max_index + delta) * ADC_SAMPLE_RATE * 1e6 / n_points

    # 步骤 5: 基于Macleod峰值进行CZT插值
    M = 32  # CZT 点数
    B = ADC_SAMPLE_RATE * 1e6 / n_points  # 分析频宽 ≈ 1 bin
    f_start = f_macleod - B / 2
    f_step = B / M

    # 使用 NumPy 的 CZT 函数可能更快，但这里保留了你原有的循环实现
    X_czt = np.zeros(M, dtype=complex)
    for m in range(M):
        sum_czt = 0 + 0j
        for n in range(n_points):
            phase = -2 * np.pi * f_step * m * n / (ADC_SAMPLE_RATE * 1e6)
            phase0 = 2 * np.pi * f_start * n / (ADC_SAMPLE_RATE * 1e6)
            sum_czt += fft_result[n] * np.exp(1j * (phase0 + phase))
        X_czt[m] = sum_czt

    # 步骤 6: 在CZT结果上再次进行Macleod插值
    peak_idx = np.argmax(np.abs(X_czt)) # 找到CZT结果的峰值

    # 增加边界检查，防止索引越界
    if peak_idx > 0 and peak_idx < M - 1:
        mag2_czt_km1 = np.abs(X_czt[peak_idx - 1])**2
        mag2_czt_k0 = np.abs(X_czt[peak_idx])**2
        mag2_czt_kp1 = np.abs(X_czt[peak_idx + 1])**2
        denom2 = mag2_czt_km1 - 2 * mag2_czt_k0 + mag2_czt_kp1
        delta_czt = 0.5 * (mag2_czt_km1 - mag2_czt_kp1) / denom2 if denom2 != 0 else 0.0
        f_czt_macleod = f_start + (peak_idx + delta_czt) * f_step
    else:
        # 如果峰值在边界，不进行Macleod插值
        f_czt_macleod = f_start + peak_idx * f_step

    # 步骤 7: 基于FFT峰值进行CZT插值
    f_start2 = f_fft_peak - B / 2
    X_czt_fftpeak = np.zeros(M, dtype=complex)
    for m in range(M):
        sum_czt_fftpeak = 0 + 0j
        for n in range(n_points):
            phase = -2 * np.pi * f_step * m * n / ADC_SAMPLE_RATE / 1e6
            phase0 = 2 * np.pi * f_start2 * n / ADC_SAMPLE_RATE / 1e6
            sum_czt_fftpeak += fft_result[n] * np.exp(1j * (phase0 + phase))
        X_czt_fftpeak[m] = sum_czt_fftpeak

    peak_idx2 = np.argmax(np.abs(X_czt_fftpeak))
    f_czt_fftpeak = f_start2 + peak_idx2 * f_step

    # 步骤 8: 计算距离
    R_fft = (C * f_fft_peak * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)
    R_macleod = (C * f_macleod * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)
    R_czt_fftpeak = (C * f_czt_fftpeak * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)
    R_czt_macleod = (C * f_czt_macleod * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)

    # 输出结果
    print(f"FFT Distance: {R_fft:.4f} m | Macleod: {R_macleod:.4f} m | CZT@peak: {R_czt_fftpeak:.4f} m | CZT@Macleod: {R_czt_macleod:.4f} m")

    return R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod