import numpy as np
import scipy.io
from datetime import datetime
from scipy.signal import czt, get_window

#============ 雷达参数配置 =================

C = 3e8  # 光速，单位 m/s
CenterFrequency = 77  # 中心频率，单位 GHz
wavelength = C / (CenterFrequency * 1e9)  # 波长，单位 m
ADC_SAMPLE_RATE = 7.14  # 采样率，单位 MHz
FM = 3000  # 调频带宽，单位 MHz
CHIRP_T0 = 98 # 微秒
CHIRP_T1 = 14  # 微秒
CHIRP_T2 = 0   # 微秒
CHIRP_PERIOD = CHIRP_T0 + CHIRP_T1 + CHIRP_T2  # Chirp周期，单位微秒

""""
    数据重组前：
    chirp 0:
      TX0RX0: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX1RX0: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX0RX1: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX1RX1: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
    chirp 1:
      TX0RX0: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX1RX0: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX0RX1: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX1RX1: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
    ...
    chirp (total_blocks-1):
      TX0RX0: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX1RX0: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX0RX1: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]
      TX1RX1: [I0, Q0, I1, Q1, ..., I(block_size-1), Q(block_size-1)]

    数据重组后：
    [虚拟天线0][chirp 0][样点0], [虚拟天线0][chirp 0][样点1], ..., [虚拟天线0][chirp 0][样点block_size-1]
    [虚拟天线0][chirp 1][样点0], [虚拟天线0][chirp 1][样点1], ..., [虚拟天线0][chirp 1][样点block_size-1]
      ...

    [虚拟天线1][chirp 0][样点0], ..., [虚拟天线1][chirp total_blocks-1][样点block_size-1]
      ...

    [虚拟天线3][chirp 0][样点0], ..., [虚拟天线3][chirp total_blocks-1][样点block_size-1]

    天线映射：
    原始顺序 [TX0RX0, TX1RX0, TX0RX1, TX1RX1] 映射为 [0, 2, 1, 3]

    实际天线排布
    --------------------
    |                  |
    |          TX1     |
    |          TX0     |
    |  RX0 RX1         |
    |                  |
    --------------------

    虚拟天线排布
    [ TX1RX0    TX1RX1 ]     -------->        [ 2  3 ]
    [ TX0RX0    TX0RX1 ]     -------->        [ 0  1 ]


    注意：
    原始数据按 IQ 排列（即 I 在前，Q 在后）
    但是文档标注为 QI，但经实测和角度稳定性验证为 IQ

    补充说明：
    用codeblock存储的MAT文件存储的时候是列优先，
    所以在用MATLAB打开mat文件校验数据的时候，其数据格式应该如下：(2048*32) 采样点 256 chirp 32

    用Python存储的mat文件是行优先，
    所以在用Python打开mat文件校验数据的时候，其数据格式应该如下：(32*2048) 采样点 256 chirp 32

    T0R0 Chirp0 I    T0R0 Chirp0 I
    T0R0 Chirp0 Q    T0R0 Chirp0 Q
    T0R0 Chirp0 I           *
    T0R0 Chirp0 Q           *
    T0R0 Chirp0 I    T1R0 Chirp0 I
    T0R0 Chirp0 Q    T1R0 Chirp0 Q     ****
         *                  *
         *                  *
         *                  *
    T0R0 Chirp0 I    T1R0 Chirp0 I
    T0R0 Chirp0 Q    T1R0 Chirp0 Q

"""""
#============ 雷达数据处理 =================

def reorder_frame(frame_bytes: bytes, chirp: int, sample: int,  window: np.ndarray | None = None):
    #expected_bytes = chirp * n_ant * sample * 2 * 2  # chirp * 天线 * 样点 * (IQ) * int16
    #if len(frame_bytes) != expected_bytes:
        #raise ValueError(f"帧大小不匹配: got {len(frame_bytes)}, expect {expected_bytes}")

    n_ant = 4  # 虚拟天线数固定为4
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

def Perform1D_FFT(iq):
    """
    对每个 Chirp 数据执行 1D FFT，保留所有 Chirp。

    输入：
        iq: np.ndarray, 形状为 (n_ant, n_chirp, n_points)

    输出：
        fft1_results: np.ndarray, 形状为 (n_ant, n_chirp, n_points)
    """
    n_ant, n_chirp, n_points = iq.shape
    fft_results = np.zeros((n_ant, n_chirp, n_points), dtype=complex)

    for ant in range(n_ant):
        # 对每个天线上的所有 Chirp 进行 1D FFT
        # axis=-1 表示对最后一个轴（样本点数）做 FFT
        fft_results[ant, :, :] = np.fft.fft(iq[ant, :, :], axis=-1)

    return fft_results

def Perform2D_FFT(fft_results):
    """
    对 1D FFT 结果执行 2D FFT，以获取多普勒信息。

    输入：
        fft1_results: np.ndarray, 形状为 (n_ant, n_chirp, n_points)

    输出：
        fft2d_results: np.ndarray, 形状为 (n_ant, n_chirp, n_points)
    """
    # 对 Chirp 维度（第二个轴）执行 FFT
    # 这将生成一个形状为 (n_ant, n_chirp, n_points) 的数组
    fft2d_intermediate = np.fft.fft(fft_results, axis=1)

    # 对 FFT 结果进行移位，使多普勒零点位于中心
    # 这一步是可选的，但有助于可视化
    fft2d_results = np.fft.fftshift(fft2d_intermediate, axes=1)

    return fft2d_results


def calculate_distance_from_fft2(fft_result_in, n_chirp, n_points):
    """
    使用多种方法从ADC数据中计算距离。
    处理逻辑为：选择第0个虚拟天线的所有Chirp数据，进行平均后做FFT，传入的数据是fft_result_in。
    然后使用Macleod和Chirp-Z插值进行峰值细化。
    """
    #传入的fft_result_in是已经计算好的FFT结果
    #fft_result_in是（ant, chirp, n_points）形状的数组，现在对0号虚拟天线进行处理，求平均
    fft_result = np.mean(fft_result_in[:, :], axis=0)

    # 步骤 3: 计算幅度谱并找到FFT峰值
    fft_sum = np.abs(fft_result)
    valid_points = n_points // 2
    max_index = np.argmax(fft_sum[:valid_points])
    max_index = int(max_index)
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
    # print(f"FFT Distance: {R_fft:.4f} m | Macleod: {R_macleod:.4f} m | \
    #         CZT@peak: {R_czt_fftpeak:.4f} m | CZT@Macleod: {R_czt_macleod:.4f} m")

    return R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod


def calculate_distance_from_fft(fft_result_in, n_chirp, n_points):
    """
    使用多种方法从FFT结果中计算距离。

    参数:
    fft_result_in (np.ndarray): 1D FFT 结果。
    n_chirp (int): Chirp 帧的数量。
    n_points (int): 每个 Chirp 的采样点数。

    返回:
    tuple: 包含四种距离计算结果的元组 (R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod)。
    """
    fft_result = np.mean(fft_result_in[:, :], axis=0)

    # 步骤 3: 计算幅度谱并找到FFT峰值
    fft_sum = np.abs(fft_result)
    valid_points = n_points // 2
    max_index = np.argmax(fft_sum[:valid_points])

    # 计算频率偏移，这是最基础的FFT距离
    f_fft_peak = max_index * ADC_SAMPLE_RATE * 1e6 / n_points

    # 步骤 4: Macleod 插值，对FFT峰值进行二次细化
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
    M = 32
    fs = ADC_SAMPLE_RATE * 1e6
    # 放大分析频宽到2个FFT Bin的宽度，以确保覆盖到峰值
    B = 1.0 * fs / n_points
    f_start = f_macleod - B / 2
    f_end = f_macleod + B / 2
    f_step_czt = (f_end - f_start) / (M-1)
    w = np.exp(-1j * 2 * np.pi * f_step_czt / fs)
    a = np.exp(1j * 2 * np.pi * f_start / fs)
    X_czt = czt(fft_result, M, w, a)

    # 步骤 6: 在CZT结果上再次进行Macleod插值，进一步细化
    peak_idx = np.argmax(np.abs(X_czt))
    if peak_idx > 0 and peak_idx < M - 1:
        mag2_czt_km1 = np.abs(X_czt[peak_idx - 1])**2
        mag2_czt_k0 = np.abs(X_czt[peak_idx])**2
        mag2_czt_kp1 = np.abs(X_czt[peak_idx + 1])**2
        denom2 = mag2_czt_km1 - 2 * mag2_czt_k0 + mag2_czt_kp1
        delta_czt = 0.5 * (mag2_czt_km1 - mag2_czt_kp1) / denom2 if denom2 != 0 else 0.0
        f_czt_macleod = f_start + (peak_idx + delta_czt) * f_step_czt
    else:
        f_czt_macleod = f_start + peak_idx * f_step_czt

    # 步骤 7: 基于FFT峰值进行CZT插值
    f_start2 = f_fft_peak - B / 2
    f_end2 = f_fft_peak + B / 2
    f_step_czt2 = (f_end2 - f_start2) / (M - 1)
    w2 = np.exp(-1j * 2 * np.pi * f_step_czt2 / fs)
    a2 = np.exp(1j * 2 * np.pi * f_start2 / fs)
    X_czt_fftpeak = czt(fft_result, M, w2, a2)

    # 再次进行Macleod插值以细化CZT结果
    peak_idx2 = np.argmax(np.abs(X_czt_fftpeak))
    if peak_idx2 > 0 and peak_idx2 < M - 1:
        mag2_czt_fftpeak_km1 = np.abs(X_czt_fftpeak[peak_idx2 - 1])**2
        mag2_czt_fftpeak_k0 = np.abs(X_czt_fftpeak[peak_idx2])**2
        mag2_czt_fftpeak_kp1 = np.abs(X_czt_fftpeak[peak_idx2 + 1])**2
        denom3 = mag2_czt_fftpeak_km1 - 2 * mag2_czt_fftpeak_k0 + mag2_czt_fftpeak_kp1
        delta_czt2 = 0.5 * (mag2_czt_fftpeak_km1 - mag2_czt_fftpeak_kp1) / denom3 if denom3 != 0 else 0.0
        f_czt_fftpeak = f_start2 + (peak_idx2 + delta_czt2) * f_step_czt2
    else:
        f_czt_fftpeak = f_start2 + peak_idx2 * f_step_czt2

    # 步骤 8: 将频率转换为距离
    R_fft = (C * f_fft_peak * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)
    R_macleod = (C * f_macleod * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)
    R_czt_fftpeak = (C * f_czt_fftpeak * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)
    R_czt_macleod = (C * f_czt_macleod * CHIRP_T0 * 1e-6) / (2.0 * FM * 1e6)

    # 输出结果
    # print(f"FFT Distance: {R_fft:.4f} m | Macleod: {R_macleod:.4f} m | \
    #         CZT@peak: {R_czt_fftpeak:.4f} m | CZT@Macleod: {R_czt_macleod:.4f} m")

    return R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod


#=========距离计算函数，直接利用时域iq数据进行变换=============

# ---- 工具函数 ----
def _parabolic_delta(m1, m0, p1, eps=1e-18):
    denom = (m1 - 2.0*m0 + p1)
    if np.abs(denom) < eps:
        denom = np.sign(denom) * eps if denom != 0 else eps
    return 0.5 * (m1 - p1) / denom

def _build_czt_aw(f_start, B, M, fs):
    # 夹取到 [0, fs/2 - B]
    f_start = float(max(0.0, min(f_start, fs/2 - B)))
    df = B / (M - 1)  # 覆盖完整带宽
    W = np.exp(-1j * 2 * np.pi * (df / fs))
    A = np.exp( 1j * 2 * np.pi * (f_start / fs))
    return A, W, df, f_start

def _coarse_peak_fft(x_td, fs):
    N = x_td.shape[-1]
    X = np.fft.fft(x_td, n=N)
    X_pos = X[: N//2 + 1]
    mag2 = np.abs(X_pos)**2
    kmax = int(np.argmax(mag2))
    if 0 < kmax < (mag2.size - 1):
        delta = _parabolic_delta(mag2[kmax-1], mag2[kmax], mag2[kmax+1])
    else:
        delta = 0.0
    f_bin = fs / N
    f_fft_peak = kmax * f_bin
    f_macleod  = (kmax + delta) * f_bin
    return kmax, f_fft_peak, f_macleod

def calculate_distance_from_iq(
    iq,                     # ndarray, shape (n_ant, n_chirp, n_sample)
    r_bins=3.0,             # CZT 覆盖的原始 FFT bin 数
    M=128,                  # CZT 点数
    use_window='hamming',   # None/'hann'/'hamming'...
    coherent=True,          # True: 沿 chirp 复数相干；False: 选能量最大一条
    antenna_index=0,        # 使用的虚拟天线索引
    sample_slice=None       # (i0, i1) 仅用规则区样点；None=全长
):
    """
    返回: (R_fft, R_fft_macleod, R_czt_only, R_combo, diag)
      - R_fft         : 算法1  纯 FFT 测距
      - R_fft_macleod : 算法2  FFT+Macleod（在FFT谱上做3点二次插值）
      - R_czt_only    : 算法3  CZT测距（对IQ做CZT，取CZT峰bin，不做Macleod）
                         *窗口中心使用 FFT 粗峰，仅用于定位带宽，不参与最终估计*
      - R_combo       : 算法4  FFT+Macleod → CZT（以Macleod粗频为中心）→ Macleod（二次插值）
    """
    fs = float(ADC_SAMPLE_RATE) * 1e6  # Hz
    T_chirp = float(CHIRP_T0) * 1e-6   # s
    B_chirp = float(FM) * 1e6          # Hz

    # ---- 取指定天线 & 规则区 ----
    x = iq[antenna_index]              # (n_chirp, n_sample)
    if sample_slice is not None:
        i0, i1 = sample_slice
        x = x[:, i0:i1]
    n_chirp, n_sample = x.shape

    # ---- 时域聚合 & 加窗 ----
    if coherent:
        x_td = x.mean(axis=0).astype(np.complex128, copy=False)
    else:
        Xc_all = np.fft.fft(x, axis=-1)
        idx = np.argmax(np.max(np.abs(Xc_all)**2, axis=-1))
        x_td = x[idx].astype(np.complex128, copy=False)

    if use_window is not None:
        win = get_window(use_window, x_td.size, fftbins=True).astype(np.float64)
        win = win / np.sqrt((win**2).mean())  # ENBW 归一
        x_td = x_td * win

    # ---- 粗定位 + Macleod 细化（得到 f_fft_peak, f_macleod）----
    kmax, f_fft_peak, f_macleod = _coarse_peak_fft(x_td, fs)

    # ---- CZT参数（带宽 B 统一，以便公平对比）----
    B = float(r_bins) * fs / n_sample

    # ===== 算法3：CZT-only（以 FFT 粗峰为中心；不做Macleod）=====
    f_start_czt_only = f_fft_peak - B/2
    A1, W1, df1, f_start_czt_only = _build_czt_aw(f_start_czt_only, B, M, fs)
    Xc1 = czt(x_td, M, W1, A1)
    pk1 = int(np.argmax(np.abs(Xc1)))
    f_czt_only = f_start_czt_only + pk1 * df1  # 不做三点二次插值

    # ===== 算法4：组合（Macleod 粗频为中心 + CZT + Macleod 二次插值）=====
    f_start_combo = f_macleod - B/2
    A2, W2, df2, f_start_combo = _build_czt_aw(f_start_combo, B, M, fs)
    Xc2 = czt(x_td, M, W2, A2)
    pk2 = int(np.argmax(np.abs(Xc2)))
    if 0 < pk2 < (M - 1):
        m1 = np.abs(Xc2[pk2-1])**2; m0 = np.abs(Xc2[pk2])**2; p1 = np.abs(Xc2[pk2+1])**2
        delta2 = _parabolic_delta(m1, m0, p1)
    else:
        delta2 = 0.0
    f_combo = f_start_combo + (pk2 + delta2) * df2

    # ---- 频率 -> 距离 ----
    fb2R = lambda fb: C * fb * T_chirp / (2.0 * B_chirp)
    R_fft         = fb2R(f_fft_peak)     # 算法1
    R_fft_macleod = fb2R(f_macleod)      # 算法2
    R_czt_only    = fb2R(f_czt_only)     # 算法3
    R_combo       = fb2R(f_combo)        # 算法4

    diag = {
        "antenna_used": int(antenna_index),
        "n_chirp": int(n_chirp),
        "n_sample": int(n_sample),
        "fs_Hz": float(fs),
        "B_czt_Hz": float(B),
        "M": int(M),
        "r_bins": float(r_bins),
        "coherent": bool(coherent),
        "window": use_window if use_window is not None else "none",
        # 粗估
        "kmax": int(kmax),
        "f_fft_peak_Hz": float(f_fft_peak),
        "f_macleod_Hz": float(f_macleod),
        # CZT-only（算法3）
        "f_start_czt_only_Hz": float(f_start_czt_only),
        "df_czt_only_Hz": float(df1),
        "f_czt_only_Hz": float(f_czt_only),
        "pk_czt_only": int(pk1),
        "czt_only_spectrum": Xc1,
        # 组合（算法4）
        "f_start_combo_Hz": float(f_start_combo),
        "df_combo_Hz": float(df2),
        "f_combo_Hz": float(f_combo),
        "pk_combo": int(pk2),
        "delta_combo_bins": float(delta2),
        "czt_combo_spectrum": Xc2,
        "sample_slice": sample_slice if sample_slice else "full"
    }

    return R_fft, R_fft_macleod, R_czt_only, R_combo, diag

#=========角度计算函数，基于2DFFT结果进行角度估计=============

def estimate_az_el_from_fft2d(fft2d_results):
    """
    根据 2D FFT 结果估计 水平角(az) 与 俯仰角(el)
    - 使用全局变量 wavelength
    - 阵列为 2x2 平面阵，虚拟天线排布：
          [2 3]   (y=1)
          [0 1]   (y=0)
           x=0  x=1
    - 阵元间距固定为 λ/2
    - 自动选取能量最强的 (doppler, range) 点

    参数
    ----
    fft2d_results : np.ndarray
        形状 (4, n_chirp, n_range)，4 对应虚拟天线 [0,1,2,3]

    返回
    ----
    az_deg : float
        水平角（°）
    el_deg : float
        俯仰角（°）
    (k_dop, k_rng) : tuple[int, int]
        实际使用的 (doppler_idx, range_idx)
    extra : dict
        调试信息：相位差、空间正弦分量等
    """
    global wavelength
    d_spacing = wavelength / 2.0

    assert fft2d_results.ndim == 3 and fft2d_results.shape[0] == 4

    # 1) 找全局最强点
    power_sum = np.sum(np.abs(fft2d_results)**2, axis=0)  # (n_chirp, n_range)
    k_dop, k_rng = np.unravel_index(np.argmax(power_sum), power_sum.shape)

    # 2) 取该点 4 阵元复值
    v0 = fft2d_results[0, k_dop, k_rng]  # (x=0, y=0)
    v1 = fft2d_results[1, k_dop, k_rng]  # (x=1, y=0)
    v2 = fft2d_results[2, k_dop, k_rng]  # (x=0, y=1)
    v3 = fft2d_results[3, k_dop, k_rng]  # (x=1, y=1)

    # 3) 相位差（相邻阵元，取平均）
    dphi_x1 = np.angle(v1 * np.conj(v0))
    dphi_x2 = np.angle(v3 * np.conj(v2))
    dphi_x  = np.angle(np.mean(np.exp(1j * np.array([dphi_x1, dphi_x2]))))

    dphi_y1 = np.angle(v2 * np.conj(v0))
    dphi_y2 = np.angle(v3 * np.conj(v1))
    dphi_y  = np.angle(np.mean(np.exp(1j * np.array([dphi_y1, dphi_y2]))))

    # 4) 相位差转角度
    coef = wavelength / (2.0 * np.pi * d_spacing)
    s_x = coef * dphi_x  # = sin(az)*cos(el)
    s_y = coef * dphi_y  # = sin(el)

    s_y = float(np.clip(s_y, -0.999999, 0.999999))
    el = np.arcsin(s_y)
    cos_el = np.cos(el)
    if abs(cos_el) < 1e-6:
        cos_el = 1e-6
    ratio = float(np.clip(s_x / cos_el, -0.999999, 0.999999))
    az = np.arcsin(ratio)

    az_deg = np.degrees(az)
    el_deg = np.degrees(el)

    extra = dict(
        dphi_x=float(dphi_x),
        dphi_y=float(dphi_y),
        s_x=float(s_x),
        s_y=float(s_y),
        wavelength=wavelength,
        d_spacing=d_spacing
    )
    return az_deg, el_deg, (int(k_dop), int(k_rng)), extra

###==================== 基于最小二乘法进行IQ校准(2DFFT峰值点) ===================
def amplitude_calibration(zij_vector: np.ndarray):
    """
    使用最小二乘法进行幅度校准，返回校准因子矩阵（包含归一化还原）。

    输入：
        zij_vector: 形状为 (n_ant,) 的复数向量，每个元素代表虚拟通道 (tx, rx) 的响应
    输出：
        alpha_matrix: 形状为 (n_ant, n_ant) 的幅度校准因子矩阵（绝对校准，非相对）
    """
    n_ant = zij_vector.shape[0]
    # 虚拟天线映射（假设你的映射是固定的：4个虚拟通道对应 (tx0,rx0), (tx0,rx1), (tx1,rx0), (tx1,rx1)）
    tx_map = np.array([0, 0, 1, 1])
    rx_map = np.array([0, 1, 0, 1])
    n_tx = len(np.unique(tx_map))  # 实际发射天线数（这里是2）
    n_rx = len(np.unique(rx_map))  # 实际接收天线数（这里是2）

    # 提取观测幅度（每个虚拟通道的幅度）
    y_ij = np.abs(zij_vector)

    # 初始化发射/接收幅度因子（以第0个天线为基准，初始为1）
    alpha_tx = np.ones(n_tx)
    alpha_rx = np.ones(n_rx)

    # 迭代求解最小二乘（固定一方，更新另一方）
    max_iterations = 100
    tol = 1e-6  # 收敛阈值
    for _ in range(max_iterations):
        alpha_tx_old = alpha_tx.copy()
        alpha_rx_old = alpha_rx.copy()

        # 固定发射因子，更新接收因子（每个rx对应的虚拟通道）
        for j in range(n_rx):
            # 找到所有属于第j个接收天线的虚拟通道索引
            rx_mask = (rx_map == j)
            if np.any(rx_mask):
                # 最小二乘：alpha_rx[j] 使得 sum((y_ij - alpha_tx[i] * alpha_rx[j])^2) 最小
                numerator = np.sum(y_ij[rx_mask] * alpha_tx[tx_map[rx_mask]])
                denominator = np.sum(alpha_tx[tx_map[rx_mask]] ** 2)
                if denominator > 1e-9:
                    alpha_rx[j] = numerator / denominator

        # 固定接收因子，更新发射因子（每个tx对应的虚拟通道）
        for i in range(n_tx):
            # 找到所有属于第i个发射天线的虚拟通道索引
            tx_mask = (tx_map == i)
            if np.any(tx_mask):
                # 最小二乘：alpha_tx[i] 使得 sum((y_ij - alpha_tx[i] * alpha_rx[j])^2) 最小
                numerator = np.sum(y_ij[tx_mask] * alpha_rx[rx_map[tx_mask]])
                denominator = np.sum(alpha_rx[rx_map[tx_mask]] ** 2)
                if denominator > 1e-9:
                    alpha_tx[i] = numerator / denominator

        # 检查收敛（参数变化小于阈值则停止）
        if np.max(np.abs(alpha_tx - alpha_tx_old)) < tol and np.max(np.abs(alpha_rx - alpha_rx_old)) < tol:
            break

    # 计算归一化前的“理论基准幅度”（用于还原）
    # 以参考通道（tx0, rx0）的实际幅度为基准，确保校准后不丢失原始量级
    ref_idx = np.where((tx_map == 0) & (rx_map == 0))[0][0]  # 参考虚拟通道索引
    theoretical_ref_amplitude = alpha_tx[0] * alpha_rx[0]  # 校准模型中的基准幅度
    actual_ref_amplitude = y_ij[ref_idx]  # 实际观测的基准幅度
    scale_factor = actual_ref_amplitude / theoretical_ref_amplitude  # 还原比例（消除归一化影响）

    # 应用还原比例，确保校准后的基准通道幅度与实际一致
    alpha_tx *= scale_factor
    alpha_rx *= 1  # 发射/接收因子中只需一个乘比例，避免重复放大（这里选择tx）

    # 构建幅度校准矩阵（实际发射×接收天线的矩阵）
    alpha_matrix = np.outer(alpha_tx, alpha_rx)

    return alpha_matrix

def phase_calibration(
    zij_vector: np.ndarray,
    ref_tx: int = 0,
    ref_rx: int = 0
) -> np.ndarray:
    """
    基于固定天线排布的相位校准（2发2收，虚拟通道映射固定）。
    物理模型：虚拟通道相位 = 发射天线相位 + 接收天线相位。

    输入：
        zij_vector: 形状为 (4,) 的复数向量，4个虚拟通道的峰值响应（顺序：[0,1,2,3]）
        ref_tx: 参考发射天线编号（0或1，默认0）
        ref_rx: 参考接收天线编号（0或1，默认0）
    输出：
        phi_matrix: 形状为 (2, 2) 的相位校准矩阵（发射×接收），元素为相位值（rad）
    """
    # 固定天线映射（根据你的实体与虚拟天线排布）
    # 虚拟通道索引：0→TX0RX0，1→TX0RX1，2→TX1RX0，3→TX1RX1
    tx_map = np.array([0, 0, 1, 1])  # 虚拟通道→发射天线（0:TX0，1:TX1）
    rx_map = np.array([0, 1, 0, 1])  # 虚拟通道→接收天线（0:RX0，1:RX1）

    n_virtual = zij_vector.shape[0]
    if n_virtual != 4:
        raise ValueError("zij_vector必须为4元素向量（对应4个虚拟通道）")

    # 1. 提取实体天线信息（固定2发2收）
    tx_ids = np.unique(tx_map)  # [0,1]
    rx_ids = np.unique(rx_map)  # [0,1]
    n_tx, n_rx = len(tx_ids), len(rx_ids)  # 均为2

    # 2. 相位解缠绕（消除[-π, π]跳变）
    raw_phase = np.angle(zij_vector)
    unwrapped_phase = np.unwrap(raw_phase)

    # 3. 构建线性方程组：phi_tx[t] + phi_rx[r] = 观测相位（以参考天线为基准）
    tx_idx = {t: i for i, t in enumerate(tx_ids)}  # {0:0, 1:1}
    rx_idx = {r: i for i, r in enumerate(rx_ids)}  # {0:0, 1:1}
    ref_tx_idx = tx_idx[ref_tx]
    ref_rx_idx = rx_idx[ref_rx]

    num_unknowns = (n_tx - 1) + (n_rx - 1)  # 2个未知数（非参考天线相位）
    A = np.zeros((n_virtual, num_unknowns))
    b = np.zeros(n_virtual)

    for i in range(n_virtual):
        t = tx_map[i]
        r = rx_map[i]
        t_idx = tx_idx[t]
        r_idx = rx_idx[r]

        # 方程右侧：当前相位 - 参考通道相位
        ref_mask = (tx_map == ref_tx) & (rx_map == ref_rx)
        ref_phase = unwrapped_phase[ref_mask][0] if np.any(ref_mask) else 0
        b[i] = unwrapped_phase[i] - ref_phase

        # 方程左侧：非参考天线相位系数
        if t != ref_tx:
            tx_unknown_idx = t_idx - (1 if t_idx > ref_tx_idx else 0)
            A[i, tx_unknown_idx] = 1.0
        if r != ref_rx:
            rx_unknown_idx = (n_tx - 1) + (r_idx - (1 if r_idx > ref_rx_idx else 0))
            A[i, rx_unknown_idx] = 1.0

    # 4. 最小二乘求解
    x, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < num_unknowns:
        print(f"警告：相位方程组秩不足（有效方程数{rank} < 未知数{num_unknowns}）")

    # 5. 重构发射/接收相位（参考天线相位为0）
    phi_tx = np.zeros(n_tx)
    phi_rx = np.zeros(n_rx)
    tx_unknowns = x[:n_tx - 1]
    rx_unknowns = x[n_tx - 1:]

    tx_unknown_idx = 0
    for t_idx in range(n_tx):
        if t_idx != ref_tx_idx:
            phi_tx[t_idx] = tx_unknowns[tx_unknown_idx]
            tx_unknown_idx += 1

    rx_unknown_idx = 0
    for r_idx in range(n_rx):
        if r_idx != ref_rx_idx:
            phi_rx[r_idx] = rx_unknowns[rx_unknown_idx]
            rx_unknown_idx += 1

    # 6. 构建相位矩阵（发射相位 + 接收相位）
    phi_matrix = np.outer(phi_tx, np.ones(n_rx)) + np.outer(np.ones(n_tx), phi_rx)
    return phi_matrix


def apply_channel_calibration(
    iq_data: np.ndarray,
    alpha_matrix: np.ndarray,
    phi_matrix: np.ndarray,
) -> np.ndarray:
    """
    应用幅度和相位校准（基于固定天线排布，无需传入映射参数）。

    输入：
        iq_data: 形状为 (4, n_chirp, n_points) 的IQ数据（4个虚拟通道）
        alpha_matrix: 形状为 (2, 2) 的幅度校准矩阵（发射×接收）
        phi_matrix: 形状为 (2, 2) 的相位校准矩阵（发射×接收）
    输出：
        calibrated_iq: 校准后的IQ数据（与输入形状一致）
    """
    # 固定虚拟通道→实体天线映射
    tx_map = np.array([0, 0, 1, 1])  # 虚拟通道0-3对应发射天线
    rx_map = np.array([0, 1, 0, 1])  # 虚拟通道0-3对应接收天线

    # 校验输入维度
    if iq_data.shape[0] != 4:
        raise ValueError("iq_data第一维度必须为4（对应4个虚拟通道）")
    if alpha_matrix.shape != (2, 2) or phi_matrix.shape != (2, 2):
        raise ValueError("alpha_matrix和phi_matrix必须为(2,2)矩阵（2发2收）")

    # 找到参考通道的绝对幅度校准因子
    # 假设 tx0, rx0 是参考通道，其在 alpha_matrix 中的索引为 [0, 0]
    ref_alpha = alpha_matrix[0, 0]

    # 将每个通道的绝对校准因子转换为相对校准因子
    # 这样，所有通道都会相对于参考通道进行校准
    relative_alpha_matrix = alpha_matrix / ref_alpha

    # 提取每个虚拟通道的相对校准因子
    alpha_vector = relative_alpha_matrix[tx_map, rx_map] # 4元素向量（每个虚拟通道的幅度因子）
    phi_vector = phi_matrix[tx_map, rx_map] # 4元素向量（每个虚拟通道的幅度因子）

    # 幅度补偿：乘以 1.0 / 相对校准因子
    # 这会使校准后的幅度都与参考通道的幅度一致
    amp_comp = 1.0 / alpha_vector[:, np.newaxis, np.newaxis]
    phase_comp = np.exp(-1j * phi_vector[:, np.newaxis, np.newaxis])

    # 应用校准
    calibrated_iq = iq_data * amp_comp * phase_comp

    return calibrated_iq

###==================== 基于复数通道比值法进行IQ校准 ===================
def complex_channel_calibration(zij_vector: np.ndarray):
    """
    使用复数通道比值法进行校准，并返回一个复数校准因子向量。

    输入：
        zij_vector: np.ndarray, 形状为 (n_ant,) 的复数向量，
                    代表每个通道在目标峰值处的响应。

    输出：
        beta_vector: np.ndarray, 形状为 (n_ant,) 的复数校准因子向量。
    """
    # 将第一个通道作为基准，其复数校准因子为 1
    beta_vector = np.ones_like(zij_vector)

    # 找到基准通道的复数值 z00
    z00 = zij_vector[0]

    # 对于其他通道，计算其相对于基准通道的比值
    # 这个比值就是每个通道的复数校准因子
    for i in range(1, len(zij_vector)):
        beta_vector[i] = zij_vector[i] / z00

    return beta_vector

def apply_complex_calibration(iq_data: np.ndarray, beta_vector: np.ndarray):
    """
    使用复数校准因子对 IQ 数据进行校准。

    输入：
        iq_data: np.ndarray, 形状为 (n_ant, n_chirp, n_points)
        beta_vector: np.ndarray, 形状为 (n_ant,) 的复数校准因子向量。

    输出：
        calibrated_iq: np.ndarray, 校准后的 IQ 数据。
    """
    # 补偿因子是 beta_vector 的倒数
    compensation_vector = 1 / beta_vector

    # 将补偿因子广播到 IQ 数据上进行校准
    # np.newaxis 将一维向量转换为 (n_ant, 1, 1) 的形状，以便广播
    calibrated_iq = iq_data * compensation_vector[:, np.newaxis, np.newaxis]

    return calibrated_iq