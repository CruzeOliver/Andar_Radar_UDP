from typing import Dict, Any, List
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QGraphicsPathItem
from PyQt5.QtCore import QRectF,Qt
import pyqtgraph as pg
from pyqtgraph import ImageView
from pyqtgraph.opengl import GLViewWidget, GLMeshItem, GLAxisItem
import pyqtgraph.opengl as gl
import numpy as np
from PyQt5.QtGui import QPainterPath, QPen, QColor, QTransform, QFont
from collections import deque
from pyqtgraph.opengl.items.GLAxisItem import GLAxisItem
class PgDisplay:
    """
    负责：
    1) 初始化所有 pyqtgraph 视图（ADC 4路，1DFFT 4路，2DFFT 4路）
    2) 提供数据更新接口：update_adc4(), update_fft1d(), update_fft2d()
    使用者（主窗体）只需在构造时把占位 QWidget 传进来。
    """

    def __init__(self,
                 adc_placeholders: Dict[str, QWidget],
                 fft1d_placeholders: Dict[str, QWidget],
                 fft2d_placeholders: Dict[str, QWidget],
                 point_cloud_placeholders: Dict[str, QWidget],
                 constellation_placeholders: Dict[str, QWidget],
                 amp_phase_placeholders: Dict[str, QWidget],
                 waterfall_placeholders: Dict[str, QWidget],
                 frequency_placeholders: Dict[str, QWidget],
                 *,
                 r_max: float = 6.0,         # 最大量程 (距离)
                 fov_deg: float = 180.0,      # 扇形角度（例如120°）
                 theta_center_deg: float = 90 # 半圆朝上
                 ):
        """
        adc_placeholders: {'tx0rx0': QWidget, ...}
        fft1d_placeholders: {'1DFFTtx0rx0': QWidget, ...}
        fft2d_placeholders: {'2DFFTtx0rx0': QWidget, ...}
        """
        pg.setConfigOptions(antialias=True)
        #maxlen=5 # 表示队列最大容量为5。当加入第6个元素时，最旧的元素会自动被移除。
        self._r_buffer = deque(maxlen = 5)
        self._theta_buffer = deque(maxlen = 5)

        self.pg_plot_dict: Dict[str, Dict[str, Any]] = {}  # ADC & 1DFFT 曲线
        self.pg_img_dict: Dict[str, ImageView] = {}        # 2DFFT 图像
        self.pg_cloud_dict: Dict[str, Dict[str, Any]] = {} # Point Cloud 图像
        self.pg_const_dict: Dict[str, Dict[str, Any]] = {} # Constellation Diagram 图像
        self.pg_amp_phase_dict: Dict[str, Dict[str, Any]] = {} # Amp-Phase 图像
        self.pg_waterfall_dict: Dict[str, Dict[str, Any]] = {} # Waterfall 图像
        self.pg_frequency_dict: Dict[str, Dict[str, Any]] = {} # frequency 图像

        self._colormap = self._build_jet_colormap()

        self._init_adc(adc_placeholders)
        self._init_waterfall(waterfall_placeholders)
        self._init_constellation_placeholders(constellation_placeholders)
        self._init_amp_phase(amp_phase_placeholders)
        self._init_fft1d(fft1d_placeholders)
        self._init_frequency(frequency_placeholders)
        self._init_fft2d(fft2d_placeholders)

        self._r_max = float(r_max)
        self._theta_center = np.deg2rad(theta_center_deg)
        self._fov = np.deg2rad(fov_deg)
        self._init_point_cloud_semicircle(point_cloud_placeholders)

        self.frequency_cache = {
            'FFT': deque(maxlen=20),
            'Macleod': deque(maxlen=20),
            'CZT': deque(maxlen=20),
            'Macleod-CZt': deque(maxlen=20),
        }



    # -------------------- Public Update APIs --------------------
    def update_adc4(self, iq: np.ndarray, chirp: int, sample: int):
        """
        iq: shape (4, n_chirp, n_sample) 复数
        只画第 0 条 chirp（与原逻辑保持一致），I 红 Q 蓝
        """
        t = np.arange(sample)
        adc_keys = ['tx0rx0', 'tx0rx1', 'tx1rx0', 'tx1rx1']
        for ant_idx, key in enumerate(adc_keys):
            I = np.real(iq[ant_idx, 0, :])
            Q = np.imag(iq[ant_idx, 0, :])
            h = self.pg_plot_dict.get(key)
            if not h:
                continue
            h['I'].setData(t, I)
            h['Q'].setData(t, Q)
            h['pw'].setXRange(0, sample, padding=0.02)

    def update_waterfall(self, iq: np.ndarray, chirp: int, sample: int, channelstr: str):
        """
        使用新的 IQ 数据更新瀑布图（只更新选择的通道数据）。

        参数：
        - iq: np.ndarray, 形状为 (4, 32, 256)，包含四个天线的 IQ 数据。
        - chirp: 当前的 chirp 索引（1 到 32）。
        - sample: 当前的样本数（0 到 255）。
        - channelstr: str, 选择的天线通道，如 'tx0rx0', 'tx0rx1', 'tx1rx0', 'tx1rx1'。
        """
        # waterfall_keys 用于识别瀑布图
        waterfall_keys = ['Waterfall']

        # 如果 chirp 是从 1 开始计数，需要减去 1
        chirp = chirp - 1

        # 检查 chirp 和 sample 是否在有效范围内
        if chirp < 0 or chirp >= iq.shape[1]:
            raise ValueError(f"Chirp index {chirp + 1} exceeds the available range (1 to {iq.shape[1]}).")

        if sample > iq.shape[2]:
            raise ValueError(f"Sample index {sample} exceeds the available range (0 to {iq.shape[2] - 1}).")

        # 根据选择的通道获取对应的 IQ 数据
        channel_map = {
            'tx0rx0': 0,
            'tx0rx1': 1,
            'tx1rx0': 2,
            'tx1rx1': 3
        }

        # 检查选择的通道是否有效
        if channelstr not in channel_map:
            raise ValueError(f"Invalid channel selected: {channelstr}")

        # 获取选择通道的索引
        channel_idx = channel_map[channelstr]

        # 获取所选通道的 IQ 数据
        iq_data = iq[channel_idx, chirp, :sample]

        # 计算幅度数据
        magnitude = np.abs(iq_data)
        magnitude = np.flip(magnitude)

        for key in waterfall_keys:
            h = self.pg_waterfall_dict.get(key)
            if not h:
                continue

            # 检查 data_cache 是否已初始化，如果未初始化则进行初始化
            if 'data_cache' not in h or h['data_cache'].shape[1] != magnitude.size:
                h['data_cache'] = np.zeros((32, magnitude.size), dtype=np.float32)

            # 滚动数据缓存，保持最新的数据
            # 将数组向下滚动一行，使第一行成为空
            h['data_cache'] = np.roll(h['data_cache'], shift=1, axis=0)

            # 将新的数据插入到数组的第一行
            h['data_cache'][0] = magnitude
            transposed_data = h['data_cache'].T
            h['img'].setImage(transposed_data, autoLevels=True)

            # 显示图像
            h['pw'].show()

    def update_constellations(self,
                          iq: np.ndarray,
                          *,
                          key_map: dict = None,
                          max_points: int = 4000,
                          remove_dc: bool = True,
                          set_ref_circle: bool = True,
                          autorange: bool = True,
                          show_fit: bool = True,
                          nsig: float = 2.0):
        """
        批量更新四路星座图（I/Q 散点）并叠加椭圆拟合。
        - 一律使用 all_samples：将 (n_chirp, n_sample) 展平后全部点绘制（超量自动抽样）。
        - 每个子图：散点 + 参考圆(RMS) + 可选椭圆拟合(2σ，主/次轴+数值文本)。

        参数
        ----
        iq : np.ndarray
            复数 IQ 数据，形状 (4, n_chirp, n_sample)。
        key_map : dict
            占位键名 -> 天线索引映射。默认与你现在的键名一致：
            {'CDtx0rx0':0,'CDtx0rx1':1,'CDtx1rx0':2,'CDtx1rx1':3}
        max_points : int
            点数过大时的等间隔抽样上限，默认 4000。
        remove_dc : bool
            是否减去平均值（去直流偏移），默认 True。
        set_ref_circle : bool
            是否绘制 RMS 参考圆，默认 True。
        autorange : bool
            是否自动设置坐标范围，默认 True。
        show_fit : bool
            是否叠加椭圆拟合（PCA 统计椭圆 + 主/次轴 + 文本），默认 True。
        nsig : float
            拟合椭圆半轴的 σ 倍数，默认 2.0（约覆盖 95% 点，假设近似高斯）。
        """
        assert iq.ndim == 3 and iq.shape[0] == 4, "iq 形状必须是 (4, chirp, sample)"
        if key_map is None:
            key_map = {
                'CDtx0rx0': 0,
                'CDtx0rx1': 1,
                'CDtx1rx0': 2,
                'CDtx1rx1': 3,
            }

        for key, ant_idx in key_map.items():
            if key not in getattr(self, 'pg_const_dict', {}):
                continue
            h = self.pg_const_dict[key]

            # 1) 展平 + 抽样（all_samples 策略）
            z = np.asarray(iq[ant_idx], dtype=np.complex64).ravel()
            if z.size == 0:
                h['scatter'].setData(x=[], y=[])
                h['unit_circle'].setData([], [])
                # 清空拟合层
                h['ellipse'].setData([], [])
                h['major_axis'].setData([], [])
                h['minor_axis'].setData([], [])
                h['metrics_text'].setText("")
                continue

            if remove_dc:
                m = np.nanmean(z)
                if np.isfinite(m):
                    z = z - m

            if z.size > max_points:
                step = max(1, z.size // max_points)
                z = z[::step]

            # 2) 清洗无效值并拆 I/Q
            z_clean = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            I = np.real(z_clean)
            Q = np.imag(z_clean)

            # 3) 散点
            h['scatter'].setData(
                x=I, y=Q,
                pen=None,
                brush=h['scatter'].opts.get('brush'),
                size=h['scatter'].opts.get('size', 3)
            )

            # 4) 参考圆（RMS 半径）
            if set_ref_circle and z_clean.size > 0:
                R = float(np.sqrt(np.mean(I*I + Q*Q)))
                if (not np.isfinite(R)) or R < 1e-9:
                    R = 1.0
                t = np.linspace(0, 2*np.pi, 361, dtype=np.float32)
                h['unit_circle'].setData(R*np.cos(t), R*np.sin(t))
            else:
                h['unit_circle'].setData([], [])

            # 5) 自动坐标范围
            if autorange and z_clean.size > 0:
                r = float(np.nanmax(np.abs(z_clean)))
                if (not np.isfinite(r)) or r < 1e-9:
                    r = 1.0
                pad = 0.1 * r
                try:
                    h['pw'].setRange(
                        xRange=(-r - pad, r + pad),
                        yRange=(-r - pad, r + pad),
                        padding=0.0
                    )
                except Exception:
                    h['pw'].setRange(xRange=(-1.0, 1.0), yRange=(-1.0, 1.0), padding=0.0)

            # 6) 椭圆拟合（PCA on I,Q）
            if show_fit and I.size >= 8:
                cx = float(np.mean(I)); cy = float(np.mean(Q))
                X = np.vstack([I - cx, Q - cy])          # 2×N
                C = np.cov(X)                             # 2×2
                if np.any(~np.isfinite(C)):
                    C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

                vals, vecs = np.linalg.eigh(C)           # λ1≤λ2
                vals = np.clip(vals, 1e-12, None)
                idx_max = int(np.argmax(vals)); idx_min = 1 - idx_max
                lam_major, lam_minor = float(vals[idx_max]), float(vals[idx_min])
                v_major = vecs[:, idx_max]
                v_minor = vecs[:, idx_min]

                a = nsig * np.sqrt(lam_major)            # 长半轴
                b = nsig * np.sqrt(lam_minor)            # 短半轴
                theta = float(np.arctan2(v_major[1], v_major[0]))

                # 椭圆轮廓
                tt = np.linspace(0, 2*np.pi, 361, dtype=np.float32)
                ex = a * np.cos(tt); ey = b * np.sin(tt)
                Rm = np.column_stack([v_major, v_minor])  # 2×2
                E = Rm @ np.vstack([ex, ey])              # 2×361
                exw = E[0, :] + cx; eyw = E[1, :] + cy
                h['ellipse'].setData(exw, eyw)

                # 主/次轴线段
                p1 = np.array([cx, cy]) + a * v_major
                p2 = np.array([cx, cy]) - a * v_major
                q1 = np.array([cx, cy]) + b * v_minor
                q2 = np.array([cx, cy]) - b * v_minor
                h['major_axis'].setData([p1[0], p2[0]], [p1[1], p2[1]])
                h['minor_axis'].setData([q1[0], q2[0]], [q1[1], q2[1]])

                # 文本指标：轴比 & 倾角
                r_ax = float(b / a) if a > 1e-12 else 1.0
                deg  = float(np.degrees(theta))
                if deg > 45:
                    deg -= 90
                elif deg < -45:
                    deg += 90
                text = f"axis_ratio b/a = {r_ax:.3f}\ntilt = {deg:.1f}°"

                # 放右上角（优先），否则放中心
                try:
                    vb = h['pw'].getViewBox()
                    (x0, x1), (y0, y1) = vb.state['viewRange'][0], vb.state['viewRange'][1]
                    tx = x0 + 0.02*(x1 - x0)
                    ty = y1 - 0.06*(y1 - y0)
                    h['metrics_text'].setPos(tx, ty)
                except Exception:
                    h['metrics_text'].setPos(cx, cy)
                h['metrics_text'].setText(text)
            else:
                # 关闭拟合或点太少：清空拟合层
                h['ellipse'].setData([], [])
                h['major_axis'].setData([], [])
                h['minor_axis'].setData([], [])
                h['metrics_text'].setText("")

    def update_amp_phase(self,
                     iq: np.ndarray,
                     *,
                     chirp: int = 0,
                     sample: int | None = None,
                     key_map: dict | None = None,
                     unwrap_phase: bool = True,
                     decimate: int = 1,
                     remove_dc: bool = False,
                     autorange: bool = True):
        """
        iq : np.ndarray
            复数 IQ 数据，形状 (4, n_chirp, n_sample)
        chirp : int
            选择第几个 chirp 来画时序（默认第 0 个）
        sample : int | None
            仅取前 sample 个样点；为 None 则取整条 chirp
        key_map : dict | None
            占位键名 -> 天线索引 的映射。默认按当前工程：
            {'APtx0rx0':0, 'APtx0rx1':1, 'APtx1rx0':2, 'APtx1rx1':3}
        unwrap_phase : bool
            是否对相位做 np.unwrap ，如果np.unwrap = true 相位会呈现一条直线，else相位会在 -π 到 π 之间跳变
        decimate : int
            下采样因子（>=1）。例如 4 表示每 4 点取 1 点
        remove_dc : bool
            是否对 z(t) 去直流（z -= mean(z)）。用于相位更稳的场景
        autorange : bool
            是否自动设置坐标范围
        批量更新四路“幅度/相位时序”图，并相对 0 通道显示对比：
            - 灰色虚线：0 通道的 |z| 与 phase（同一 chirp/窗口）
            - 文本指标：ΔAmp(dB) 与 ΔPhase(°)（采用RMSE）
        """
        assert iq.ndim == 3 and iq.shape[0] == 4, "iq 形状必须是 (4, n_chirp, n_sample)"
        n_chirp, n_sample = iq.shape[1], iq.shape[2]
        if chirp < 0 or chirp >= n_chirp:
            return

        if key_map is None:
            key_map = {
                'APtx0rx0': 0,
                'APtx0rx1': 1,
                'APtx1rx0': 2,
                'APtx1rx1': 3,
            }

        # 统一抽样窗口
        end = n_sample if sample is None else min(sample, n_sample)
        sl = slice(0, end, max(1, decimate))

        # ---------- 准备参考通道（idx=0） ----------
        ref_idx = 0
        z_ref = iq[ref_idx, chirp, :end]
        if remove_dc:
            mref = np.nanmean(z_ref)
            if np.isfinite(mref):
                z_ref = z_ref - mref
        z_ref = np.nan_to_num(z_ref[sl], nan=0.0, posinf=0.0, neginf=0.0)

        amp_ref = np.abs(z_ref).astype(np.float32)
        ph_ref  = np.angle(z_ref).astype(np.float32)
        if unwrap_phase:
            ph_ref = np.unwrap(ph_ref)

        t_ref = np.arange(z_ref.size, dtype=np.int32)
        eps = 1e-12  # 防除零

        # ---------- 遍历每个通道 ----------
        for key, ant_idx in key_map.items():
            h = self.pg_amp_phase_dict.get(key)
            if not h:
                continue

            # 取该通道数据
            z = iq[ant_idx, chirp, :end]
            if z.size == 0:
                h['amp'].setData([], [])
                h['phase'].setData([], [])
                h['amp_ref'].setData([], [])
                h['phase_ref'].setData([], [])
                if 'metrics_text' in h:
                    h['metrics_text'].setText("")
                continue

            if remove_dc:
                m = np.nanmean(z)
                if np.isfinite(m):
                    z = z - m

            z = np.nan_to_num(z[sl], nan=0.0, posinf=0.0, neginf=0.0)
            amp = np.abs(z).astype(np.float32)
            ph  = np.angle(z).astype(np.float32)
            if unwrap_phase:
                ph = np.unwrap(ph)
            # 将相位从弧度转换为角度
            ph_deg = np.degrees(ph)

            t = np.arange(z.size, dtype=np.int32)

            # --- 更新曲线（本通道） ---
            h['amp'].setData(t, amp)
            h['phase'].setData(t, ph_deg)

            # --- 画参考通道（同一窗口）的虚线 ---
            h['amp_ref'].setData(t_ref, amp_ref)
            h['phase_ref'].setData(t_ref, np.degrees(ph_ref))

            # --- 计算与参考通道的差值：ΔAmp(dB) 与 ΔPhase(°) ---
            min_len = min(amp.size, amp_ref.size)
            if min_len >= 8:
                a = amp[:min_len]; ar = amp_ref[:min_len]
                p = ph[:min_len];  pr = ph_ref[:min_len]

                # ΔAmp（dB）：20*log10(|z|/|z_ref|)
                delta_amp_db = 20.0 * np.log10((a + eps) / (ar + eps))
                delta_amp_db_rmse = np.sqrt(np.nanmean(np.square(delta_amp_db)))

                # ΔPhase（度）：(phase - phase_ref)
                delta_phase = np.degrees(p - pr)

                # 如果未展开，将相位差规整到 [-180°, 180°]
                if not unwrap_phase:
                    delta_phase = (delta_phase + 180) % 360 - 180

                # 这里你原有对 [-45, 45] 的处理，根据你的需求保留或删除
                # delta_phase = np.where(delta_phase > 45, delta_phase - 90, delta_phase)
                # delta_phase = np.where(delta_phase < -45, delta_phase + 90, delta_phase)
                delta_phase_deg_rmse = np.sqrt(np.nanmean(np.square(delta_phase)))

                if ant_idx == ref_idx:
                    text = "REF (Ch0)"
                else:
                    text = f"ΔAmp(RMSE) ≈ {delta_amp_db_rmse:.2f} dB\nΔPhase(RMSE) ≈ {delta_phase_deg_rmse:.1f}°"

                try:
                    vb = h['pw_amp'].getViewBox()
                    (x0, x1), (y0, y1) = vb.state['viewRange'][0], vb.state['viewRange'][1]
                    tx = x1 - 0.02*(x1 - x0)
                    ty = y0 + 0.40*(y1 - y0)
                    h['metrics_text'].setPos(tx, ty)
                except Exception:
                    h['metrics_text'].setPos(t[0] if t.size else 0, (np.nanmax(amp) if amp.size else 1.0))
                h['metrics_text'].setText(text)
            else:
                if 'metrics_text' in h:
                    h['metrics_text'].setText("")

            # --- 自动范围 ---
            if autorange:
                # Amp
                amax = float(np.nanmax(amp)) if amp.size else 1.0
                amax = 1.0 if (not np.isfinite(amax) or amax < 1e-6) else amax
                xmax = max(t[-1] if t.size else 1, t_ref[-1] if t_ref.size else 1)
                h['pw_amp'].setXRange(0, max(1, xmax), padding=0.02)
                h['pw_amp'].setYRange(0, amax * 1.05, padding=0.02)

                # Phase
                if ph_deg.size and np.degrees(ph_ref).size:
                    pmin = float(np.nanmin([np.nanmin(ph_deg), np.nanmin(np.degrees(ph_ref))]))
                    pmax = float(np.nanmax([np.nanmax(ph_deg), np.nanmax(np.degrees(ph_ref))]))
                    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax - pmin < 1e-6:
                        pmin, pmax = -180.0, 180.0
                else:
                    pmin, pmax = -180.0, 180.0
                pad = 0.05 * (pmax - pmin)
                h['pw_phase'].setXRange(0, max(1, xmax), padding=0.02)
                h['pw_phase'].setYRange(pmin - pad, pmax + pad, padding=0.02)

    def update_fft1d(self, fft_results_in: np.ndarray, sample: int):
        """
        更新四个天线的 1D FFT 图，并显示峰值的 bin。
        fft_results_in: shape (4, n_chirp, n_points)
        策略：对 chirp 维度做均值，再取幅度
        """
        fft1d_keys = ['1DFFTtx0rx0', '1DFFTtx0rx1', '1DFFTtx1rx0', '1DFFTtx1rx1']
        max_bin = sample // 2  # 正频率部分的 bin 数
        x = np.arange(max_bin)

        for ant_idx, key in enumerate(fft1d_keys):
            h = self.pg_plot_dict.get(key)
            if not h:
                continue

            # 对 chirp 维度做均值处理
            avg_fft = np.mean(fft_results_in[ant_idx, :, :], axis=0)
            mag = np.abs(avg_fft[:max_bin])  # 计算幅度谱
            # 找到峰值所在的 bin
            peak_bin = np.argmax(mag)

            # 动态调整 Y 轴范围
            min_y = 0  # Y轴下限通常为0
            max_y = np.max(mag) * 1.15  # 找到最大值，并增加15%的裕量

            # 确保最大值不为0，避免绘图异常
            if max_y == 0:
                max_y = 1.0  # 如果所有值都为0，则设置一个默认最大值

            # 更新幅度图
            h['MAG'].setData(x, mag)
            h['pw'].setXRange(0, max_bin, padding=0.02)
            h['pw'].setYRange(min_y, max_y, padding=0.02) # 新增：设置Y轴范围

            # 在右上角显示峰值 bin
            peak_bin_text = f"Peak Bin: {peak_bin}"  # 显示峰值 bin
            h['metrics_text'].setText(peak_bin_text)

            # 设置文本位置为右上角
            try:
                # 重新获取更新后的视图范围
                vb = h['pw'].getViewBox()
                (x0, x1), (y0, y1) = vb.state['viewRange'][0], vb.state['viewRange'][1]
                tx = x1 - 0.02 * (x1 - x0)  # 右侧 2% 边距
                ty = y1 - 0.15 * (y1 - y0)  # 上方 15% 边距
                h['metrics_text'].setPos(tx, ty)
            except Exception:
                pass


    def update_frequency(self, iq: np.ndarray, diag: dict):
        """
        更新频率图表，绘制 FFT 峰值附近的频谱，并用不同颜色的线表示四种算法的峰值。

        参数:
            iq : ndarray
                复数 IQ 数据，形状为 (n_ant, n_chirp, n_sample)
            diag : dict
                包含四种算法的峰值频率字典：
                - 'f_fft_peak_Hz'
                - 'f_macleod_Hz'
                - 'f_czt_only_Hz'
                - 'f_combo_Hz'
        """
        # 从 diag 中获取频率峰值，增加默认值处理
        f_fft_peak = diag.get('f_fft_peak_Hz', None)
        f_macleod = diag.get('f_macleod_Hz', None)
        f_czt = diag.get('f_czt_only_Hz', None)
        f_combo = diag.get('f_combo_Hz', None)

        # 确保图表句柄存在 - 支持所有已初始化的图表
        for key, h in self.pg_frequency_dict.items():
            # 获取 PlotWidget 句柄
            pw = h.get('pw')
            if not pw:
                continue

            # 获取所有曲线句柄
            curve_fft = h.get('FFT')
            curve_fft_peak = h.get('FFT-Peak')
            curve_macleod = h.get('Macleod')
            curve_czt = h.get('CZT')
            curve_combo = h.get('Macleod-CZt')

            try:
                # 验证IQ数据形状有效性
                if len(iq.shape) != 3:
                    raise ValueError(f"IQ数据形状无效: {iq.shape}, 期望 (n_ant, n_chirp, n_sample)")

                n_ant, n_chirp, n_sample = iq.shape
                if n_sample <= 0:
                    raise ValueError(f"样本数量无效: {n_sample}")

                # 计算频谱（使用第一个天线和第一个chirp的数据）
                fs = 7.14 * 1e6
                freq_axis = np.fft.fftfreq(n_sample, 1/fs)
                fft_spectrum = np.abs(np.fft.fft(iq[0, 0, :]))

                # 频谱处理：去除异常值并标准化
                valid_mask = np.isfinite(fft_spectrum)
                if not np.any(valid_mask):
                    raise ValueError("频谱数据全部为无效值")

                fft_spectrum = fft_spectrum[valid_mask]
                freq_axis = freq_axis[valid_mask]

                # 限制频谱动态范围，避免极端值
                max_val = np.percentile(fft_spectrum, 99.9)
                fft_spectrum = np.clip(fft_spectrum, 0, max_val)

                # 获取峰值频率点的索引（FFT）
                peak_idx = np.argmax(fft_spectrum)

                # 提取峰值附近的频谱数据（动态调整范围）
                range_bins = 2
                start_idx = max(peak_idx - range_bins, 0)
                end_idx = min(peak_idx + range_bins + 1, len(fft_spectrum))

                # 确保有效范围
                if start_idx >= end_idx:
                    start_idx, end_idx = max(0, peak_idx - 1), min(len(fft_spectrum), peak_idx + 2)

                zoomed_freq_axis = freq_axis[start_idx:end_idx]
                zoomed_fft_spectrum = fft_spectrum[start_idx:end_idx]

                # 更新FFT曲线（显示局部放大频谱）
                if curve_fft:
                    curve_fft.setData(zoomed_freq_axis, zoomed_fft_spectrum)

                # 计算参考幅值（用于垂直线高度）
                magnitude = np.max(zoomed_fft_spectrum) if len(zoomed_fft_spectrum) > 0 else 1.0
                magnitude = float(magnitude) if np.isfinite(magnitude) else 1.0

                # 绘制各算法的垂直线（增加有效性检查）
                algorithms = [
                    (curve_fft_peak, f_fft_peak),
                    (curve_macleod, f_macleod),
                    (curve_czt, f_czt),
                    (curve_combo, f_combo)
                ]

                for curve, freq in algorithms:
                    if curve and freq is not None and np.isfinite(freq):
                        # 确保频率在显示范围内
                        if zoomed_freq_axis.size > 0:
                            freq_min, freq_max = np.min(zoomed_freq_axis), np.max(zoomed_freq_axis)
                            if not (freq_min - 0.1*abs(freq_max) <= freq <= freq_max + 0.1*abs(freq_max)):
                                # 超出范围时不绘制，避免线条过长
                                curve.clear()
                                continue
                        curve.setData([freq, freq], [0, magnitude * 1.05])  # 稍高于频谱峰值
                    elif curve:
                        curve.clear()  # 清除无效数据

                # 动态调整Y轴范围
                if len(zoomed_fft_spectrum) > 0:
                    max_amp = np.max(zoomed_fft_spectrum)
                    padding = 0.1 * max_amp  # 增加更多余量
                    y_min, y_max = 0.0, float(max_amp + padding)
                    # 限制最大范围，防止溢出
                    if y_max > 1e12:
                        y_max = 1e12
                    pw.setYRange(y_min, y_max)

                # 动态调整X轴范围
                if len(zoomed_freq_axis) > 0:
                    x_min, x_max = float(np.min(zoomed_freq_axis)), float(np.max(zoomed_freq_axis))
                    if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                        # 增加左右对称的 padding
                        range_width = x_max - x_min
                        pw.setXRange(
                            x_min - 0.05 * range_width,
                            x_max + 0.05 * range_width
                        )

            except Exception as e:
                print(f"更新频率图表 {key} 时出错: {str(e)}")
                # 出错时清除曲线，避免显示错误数据
                for curve in [curve_fft, curve_macleod, curve_czt, curve_combo]:
                    if curve:
                        curve.clear()



    def update_frequency2(self, diag: dict):
        """
        更新频率图表。

        参数:
            diag : dict
                包含四种算法频率值的字典。
                键包括: 'f_fft_peak_Hz', 'f_macleod_Hz',
                'f_czt_only_Hz', 'f_combo_Hz'。
            key : str
                图表在 pg_frequency_dict 中的键名。
        """
        frequency_keys = ['frequency']
        # 从字典中安全地获取数据，如果键不存在则返回 NaN
        f_fft = diag.get('f_fft_peak_Hz')
        f_macleod = diag.get('f_macleod_Hz')
        f_czt = diag.get('f_czt_only_Hz')
        f_combo = diag.get('f_combo_Hz')

        # 确保图表句柄存在
        h = self.pg_frequency_dict.get('frequency')
        if not h:
            return

        # 获取曲线句柄
        curve_fft = h.get('FFT')
        curve_macleod = h.get('Macleod')
        curve_czt = h.get('CZT')
        curve_combo = h.get('Macleod-CZt')

        # 将新数据添加到缓存
        self.frequency_cache['FFT'].append(f_fft)
        self.frequency_cache['Macleod'].append(f_macleod)
        self.frequency_cache['CZT'].append(f_czt)
        self.frequency_cache['Macleod-CZt'].append(f_combo)

        # 更新绘图
        x_data = list(range(len(self.frequency_cache['FFT'])))

        if curve_fft:
            curve_fft.setData(x_data, list(self.frequency_cache['FFT']))
        if curve_macleod:
            curve_macleod.setData(x_data, list(self.frequency_cache['Macleod']))
        if curve_czt:
            curve_czt.setData(x_data, list(self.frequency_cache['CZT']))
        if curve_combo:
            curve_combo.setData(x_data, list(self.frequency_cache['Macleod-CZt']))

    def update_fft2d(self, fft2d_results: np.ndarray, n_points: int, n_chirp: int):
        """
        fft2d_results: shape (4, n_chirp, n_points)
        显示 log10(|data|)，并将 range 轴截半
        """
        fft2d_keys = ['2DFFTtx0rx0', '2DFFTtx0rx1', '2DFFTtx1rx0', '2DFFTtx1rx1']
        max_range_bin = n_points // 2

        for ant_idx, key in enumerate(fft2d_keys):
            iv = self.pg_img_dict.get(key)
            if not isinstance(iv, ImageView):
                continue

            raw = fft2d_results[ant_idx, :, :]
            display_data = np.log10(np.abs(raw[:, :max_range_bin]) + 1e-12)

            iv.setImage(display_data, autoLevels=True)
            iv.setColorMap(self._colormap)

            # 坐标映射：X -> Doppler, Y -> Range
            doppler_bins, range_bins = display_data.shape
            x_min, x_max = -doppler_bins / 2, doppler_bins / 2
            y_min, y_max = 0, range_bins
            rect = QRectF(x_min, y_min, (x_max - x_min), (y_max - y_min))
            iv.getImageItem().setRect(rect)

            view = iv.getView()
            view.setLabel('bottom', 'Doppler Bin')
            view.setLabel('left', 'Range Bin')
            view.setAspectLocked(False)
            view.invertY(False)
            view.autoRange()

    def update_point_cloud_polar(self, key: str,
                                 r: float, # 现在接受标量 float
                                 theta_deg: float, # 现在接受标量 float
                                 *,
                                 size: float = 5.0,
                                 color='r'):
        """
        r-θ(度) -> 半圆散点
        每次传入一个标量，内部暂存，当数量达到5个时再统一绘制
        """
        if key not in self.pg_cloud_dict:
            return

        h = self.pg_cloud_dict[key]

        # 1. 将新传入的标量数据添加到 deque
        self._r_buffer.append(r)
        self._theta_buffer.append(theta_deg)

        # 2. 如果 deque 未满（即元素少于5个），则直接返回，不进行绘制
        if len(self._r_buffer) < 5:
            return

        # 3. 如果 deque 已满，则将所有元素转换为 NumPy 数组进行绘制
        r_array = np.array(self._r_buffer)
        theta_deg_array = np.array(self._theta_buffer)

        theta_rad = np.deg2rad(theta_deg_array)
        mask = (r_array >= 0) & (r_array <= self._r_max) & \
               (theta_rad >= h['theta_min']) & (theta_rad <= h['theta_max'])

        if not np.any(mask):
            h['scatter'].setData([])# deque 会自动管理大小，无需手动清空
            return

        rv = r_array[mask]
        tv = theta_rad[mask]
        x = rv * np.cos(tv)
        y = rv * np.sin(tv)

        # 4. 用所有缓存的数据点进行绘制
        h['scatter'].setData(x=x, y=y, size=size, brush=color, pen=None)


    # -------------------- Private: Init Helpers --------------------

    def _set_plot_style(self, pw: pg.PlotWidget):
        pw.setBackground('w')
        pw.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        pw.getAxis('left').setPen(pg.mkPen(color='k', width=1))
        pw.getAxis('bottom').setTextPen('k')
        pw.getAxis('left').setTextPen('k')
        pw.showGrid(x=True, y=True, alpha=0.3)

    def _init_adc(self, placeholders: Dict[str, QWidget]):
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)
            pw = pg.PlotWidget()
            self._set_plot_style(pw)
            pw.addLegend(offset=(10, 10))
            pw.setLabel('bottom', 'Sample points')
            pw.setLabel('left', 'Amplitude')
            pw.setTitle(f"ADC {key}", color='k', size='12pt')
            layout.addWidget(pw)

            curve_I = pw.plot(pen=pg.mkPen('r', width=2), name='I')
            curve_Q = pw.plot(pen=pg.mkPen('b', width=2), name='Q')
            self.pg_plot_dict[key] = {'pw': pw, 'I': curve_I, 'Q': curve_Q}

    def _init_waterfall(self, waterfall_placeholders: Dict[str, QWidget]):

        for name, placeholder in waterfall_placeholders.items():
            pw = pg.PlotWidget(parent=placeholder)
            placeholder.setLayout(QVBoxLayout())
            placeholder.layout().addWidget(pw)

            img_item = pg.ImageItem()
            pw.addItem(img_item)

            # 使用自定义的colormap
            img_item.setLookupTable(self._colormap.getLookupTable())

            pw.setLabel('bottom', 'Time (Samples)')
            pw.setLabel('left', 'Chirp Frame')

            self.pg_waterfall_dict[name] = {
                'pw': pw,
                'img': img_item,
                'data_cache': np.empty((0, 0), dtype=np.float32)
            }
            #pw.hide()  # 初始时隐藏，直到数据可用时再显示


    def _init_amp_phase(self, placeholders: Dict[str, QWidget]):
        """
        为每个占位 QWidget 初始化“幅度/相位时序”双图：
        - 上：|z|（Amplitude）
        - 下：unwrap(angle(z))（Phase）
        """
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)

            # --- 上图：Amplitude ---
            pw_amp = pg.PlotWidget()
            self._set_plot_style(pw_amp)
            pw_amp.addLegend(offset=(10, 10))
            pw_amp.setLabel('bottom', 'Sample')
            pw_amp.setLabel('left', 'Amplitude')
            pw_amp.setTitle(f"Amp {key}", color='k', size='12pt')
            curve_amp = pw_amp.plot(pen=pg.mkPen('r', width=2), name='Amplitude')

            # --- 下图：Phase ---
            pw_phase = pg.PlotWidget()
            self._set_plot_style(pw_phase)
            pw_phase.addLegend(offset=(10, 10))
            pw_phase.setLabel('bottom', 'Sample')
            pw_phase.setLabel('left', 'Phase ')
            pw_phase.setTitle(f"Phase {key}", color='k', size='12pt')
            curve_phase = pw_phase.plot(pen=pg.mkPen('b', width=2), name='unwrap(angle)')

            # 布局 & 保存句柄
                    # 布局 & 保存句柄
            layout.addWidget(pw_amp)
            layout.addWidget(pw_phase)

            # 参考通道（0 通道）对比用的虚线
            amp_ref_curve   = pw_amp.plot(pen=pg.mkPen((120, 120, 120), width=1, style=Qt.DashLine),
                                        name='Ref(Ch0) |z|')
            phase_ref_curve = pw_phase.plot(pen=pg.mkPen((120, 120, 120), width=1, style=Qt.DashLine),
                                            name='Ref(Ch0) phase')

            # 文本指标（显示 ΔAmp / ΔPhase）
            metrics_text = pg.TextItem(
                color=(20, 20, 20),
                fill=pg.mkBrush(255, 255, 255, 200),
                anchor=(1, 0)  # 右下角对齐：x=1(右), y=0(下)
            )
            # 将文本加到“上图：Amp”里
            pw_amp.addItem(metrics_text)

            self.pg_amp_phase_dict[key] = {
                'pw_amp': pw_amp, 'amp': curve_amp, 'amp_ref': amp_ref_curve,
                'pw_phase': pw_phase, 'phase': curve_phase, 'phase_ref': phase_ref_curve,
                'metrics_text': metrics_text
            }

    def _init_constellation_placeholders(self, placeholders: Dict[str, QWidget]):
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)
            pw = pg.PlotWidget()
            self._set_plot_style(pw)
            pw.setTitle(f"Constellation {key}", color='k', size='12pt')
            pw.setLabel('bottom', 'I')
            pw.setLabel('left', 'Q')
            pw.setAspectLocked(True)
            pw.setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0.05)

            axis_pen = pg.mkPen((150, 150, 150), width=1, style=Qt.DotLine)
            pw.addItem(pg.InfiniteLine(angle=0, pen=axis_pen))
            pw.addItem(pg.InfiniteLine(angle=90, pen=axis_pen))

            circle_pen = pg.mkPen((255, 0, 0), width=3, style=Qt.DashLine)
            unit_circle = pw.plot([], [], pen=circle_pen, name='ref_circle')

            scatter = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(30, 120, 255, 200),
                size=3, pxMode=True,
                name='const_points'
            )
            pw.addItem(scatter)

            # === 椭圆拟合相关（新增） ===
            ellipse_pen = pg.mkPen((0, 150, 0, 220), width=2)   # 椭圆轮廓：绿
            axis_pen2   = pg.mkPen((0, 120, 0, 160), width=1)   # 主/次轴：淡绿
            text_item   = pg.TextItem(color=(10, 120, 10), fill=pg.mkBrush(255, 255, 255, 180))

            ellipse_curve = pw.plot([], [], pen=ellipse_pen, name='fit_ellipse')
            major_axis = pg.PlotDataItem(pen=axis_pen2)  # 主轴线段
            minor_axis = pg.PlotDataItem(pen=axis_pen2)  # 次轴线段
            pw.addItem(major_axis)
            pw.addItem(minor_axis)
            pw.addItem(text_item)

            layout.addWidget(pw)

            # 保存所有句柄
            self.pg_const_dict[key] = {
                'pw': pw,
                'unit_circle': unit_circle,
                'scatter': scatter,
                'ellipse': ellipse_curve,
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'metrics_text': text_item,
            }

    def _init_fft1d(self, placeholders: Dict[str, QWidget]):
        """
        初始化每个 1D FFT 图表，并添加用于显示峰值的 bin 的 TextItem。
        """
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)
            pw = pg.PlotWidget()
            self._set_plot_style(pw)
            pw.addLegend(offset=(10, 10))
            pw.setLabel('bottom', 'FFT Bin')
            pw.setLabel('left', 'Amplitude')
            pw.setTitle(f"{key}", color='k', size='12pt')

            # --- 添加用于显示 Peak Bin 的 TextItem ---
            metrics_text = pg.TextItem(color=(20, 20, 20),
                                    fill=pg.mkBrush(255, 255, 255, 200),
                                    anchor=(1, 1))  # 右上角对齐
            pw.addItem(metrics_text)

            layout.addWidget(pw)
            curve = pw.plot(pen=pg.mkPen('r', width=2), name='MAG')

            # 保存句柄
            self.pg_plot_dict[key] = {'pw': pw, 'MAG': curve, 'metrics_text': metrics_text}


    def _init_frequency(self, placeholders: Dict[str, QWidget]):
        """
        为每个占位 QWidget 初始化频率图表，
        x 轴为 count（计数），y 轴为频率，不同颜色代表不同的算法。
        """
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)

            # 创建一个 PlotWidget 用于频率图
            pw = pg.PlotWidget()
            self._set_plot_style(pw)  # 设置图表样式
            pw.addLegend(offset=(10, 10))
            pw.setLabel('bottom', 'Frequency (Hz)')  # x 轴为 count
            pw.setLabel('left', 'Amplitude')  # y 轴为频率
            pw.setTitle(f"Frequency", color='k', size='12pt')

            # 将 PlotWidget 添加到容器的布局中
            layout.addWidget(pw)

            # 使用不同的颜色表示不同算法
            curve_algo_0 = pw.plot(pen=pg.mkPen('k', width=2), name='FFT')  # 黑色
            curve_algo_1 = pw.plot(pen=pg.mkPen('r', width=2), name='FFT-Peak')  # 红色
            curve_algo_2 = pw.plot(pen=pg.mkPen('b', width=2), name='Macleod')  # 蓝色
            curve_algo_3 = pw.plot(pen=pg.mkPen('g', width=2), name='CZT')  # 绿色
            curve_algo_4 = pw.plot(pen=pg.mkPen('m', width=2), name='Macleod-CZt')  # 品红色

            # 保存句柄，方便后续更新
            self.pg_frequency_dict[key] = {'pw': pw, 'FFT': curve_algo_0,'FFT-Peak': curve_algo_1, 'Macleod': curve_algo_2,
                                           'CZT': curve_algo_3,'Macleod-CZt': curve_algo_4}

    def _init_fft2d(self, placeholders: Dict[str, QWidget]):
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)
            iv = pg.ImageView(view=pg.PlotItem())
            iv.ui.menuBtn.hide()
            # 这里不使用内置 gradient，用统一 colormap
            layout.addWidget(iv)
            self.pg_img_dict[key] = iv

    def _init_point_cloud_semicircle(self, placeholders: Dict[str, QWidget]):
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)
            pw = pg.PlotWidget()
            self._set_plot_style(pw)
            pw.setTitle(f"2D(Polar) {key}", color='k', size='12pt')
            pw.setAspectLocked(True)

            # 半圆朝上：x ∈ [-r_max, r_max], y ∈ [0, r_max]
            pw.setRange(xRange=(-self._r_max, self._r_max),
                        yRange=(0, self._r_max), padding=0.02)

            # 画网格（同心弧 + 方位射线）
            theta_min = self._theta_center - self._fov/2
            theta_max = self._theta_center + self._fov/2
            for item in self._make_polar_grid(theta_min, theta_max, self._r_max):
                pw.addItem(item)

            scatter = pg.ScatterPlotItem(pen=None, size=5, brush='r')
            pw.addItem(scatter)

            layout.addWidget(pw)
            self.pg_cloud_dict[key] = {
                'pw': pw,
                'scatter': scatter,
                'theta_min': theta_min,
                'theta_max': theta_max
            }

    def _make_polar_grid(self, theta_min: float, theta_max: float, r_max: float):
        items = []

        # —— 同心弧线（半径刻度）——
        n_rings = 6
        radii = np.linspace(r_max/n_rings, r_max, n_rings)
        pen_ring = QPen(QColor(255, 1, 1))
        pen_ring.setStyle(Qt.DashLine)
        pen_ring.setCosmetic(True)

        for r in radii:
            path = QPainterPath()
            thetas = np.linspace(theta_min, theta_max, 200)
            x = r * np.cos(thetas)
            y = r * np.sin(thetas)
            path.moveTo(x[0], y[0])
            for i in range(1, len(x)):
                path.lineTo(x[i], y[i])
            item = QGraphicsPathItem(path)
            item.setPen(pen_ring)
            items.append(item)

        # —— 方位射线（角度刻度）——
        # 生成 0° 到 180° 的角度，用于绘制射线
        theta_min_deg = np.rad2deg(theta_min)
        theta_max_deg = np.rad2deg(theta_max)
        thetas_ray_deg = np.arange(theta_min_deg, theta_max_deg + 1, 10)
        thetas_ray_rad = np.deg2rad(thetas_ray_deg)

        pen_ray = QPen(QColor(1, 1, 255))
        pen_ray.setStyle(Qt.DotLine)
        pen_ray.setCosmetic(True)

        for th_deg, th_rad in zip(thetas_ray_deg, thetas_ray_rad):
            # 绘制射线
            path = QPainterPath()
            path.moveTo(0, 0)
            path.lineTo(r_max * np.cos(th_rad), r_max * np.sin(th_rad))
            item = QGraphicsPathItem(path)
            item.setPen(pen_ray)
            items.append(item)

            # 绘制角度标签
            # 标签的角度从 0° 到 180° 映射到 90° 到 -90° （逆时针旋转）
            label_deg = 90 - th_deg

            # 将文本放置在最外圈，距离中心点 r_max 的 1.1 倍处
            text_x = r_max * 1.1 * np.cos(th_rad)
            text_y = r_max * 1.1 * np.sin(th_rad)

            text_item = pg.TextItem(text=f"{label_deg:.0f}°", color=(0, 0, 0))
            # 旋转文本以适应射线方向
            text_item.setTransform(QTransform().rotate(th_deg))
            text_item.setPos(text_x, text_y)
            items.append(text_item)
        return items

    def _build_jet_colormap(self) -> pg.ColorMap:
        # 色表（0-255的RGB）
        pos = np.linspace(0.0, 1.0, 7)
        colors = [
            (0, 0, 131), (0, 0, 255), (0, 255, 255),
            (255, 255, 0), (255, 0, 0), (128, 0, 0), (0, 0, 0)
        ]
        return pg.ColorMap(pos, colors)

    def reset(self):
        """重置所有图表数据，清空显示内容并重置状态"""
        # 重置 ADC 和 1DFFT 曲线
        for h in self.pg_plot_dict.values():
            if 'I' in h:
                h['I'].clear()
            if 'Q' in h:
                h['Q'].clear()
            if 'MAG' in h:
                h['MAG'].clear()

        # 重置瀑布图数据缓存
        for h in self.pg_waterfall_dict.values():
            h['data_cache'] = np.empty((0, 0), dtype=np.float32)
            h['img'].clear()

        # 重置 2DFFT 图像
        for iv in self.pg_img_dict.values():
            iv.clear()
            # 重置颜色映射
            iv.setColorMap(self._colormap)

        # 重置点云数据及缓冲区
        self._r_buffer.clear()
        self._theta_buffer.clear()
        for h in self.pg_cloud_dict.values():
            h['scatter'].clear()

        # 重置星座图（包括拟合元素）
        for h in self.pg_const_dict.values():
            h['scatter'].clear()
            h['unit_circle'].clear()
            h['ellipse'].clear()
            h['major_axis'].clear()
            h['minor_axis'].clear()
            h['metrics_text'].setText("")
            # 重置坐标范围
            h['pw'].setRange(xRange=(-1, 1), yRange=(-1, 1), padding=0.05)

        # 重置幅度/相位图（包括参考曲线和文本）
        for h in self.pg_amp_phase_dict.values():
            h['amp'].clear()
            h['phase'].clear()
            h['amp_ref'].clear()
            h['phase_ref'].clear()
            h['metrics_text'].setText("")
            # 重置坐标范围
            h['pw_amp'].setRange(xRange=(0, 1), yRange=(0, 1), padding=0.02)
            h['pw_phase'].setRange(xRange=(0, 1), yRange=(-np.pi, np.pi), padding=0.02)



#============================ Stable 3D Waterfall ====================
    #显示效果不佳，暂时弃用
    '''
    def init_3d_waterfall(self, waterfall_placeholders: Dict[str, QWidget]):
        """
        初始化3D瀑布图（稳定版）：
        1. 完全移除GLColorMap和setHeightMask依赖
        2. 用自定义颜色数组实现幅值着色
        3. 确保Z轴数据形状严格匹配(len(x), len(y))

        参数:
        - waterfall_placeholders: 瀑布图占位符字典（key=标识，value=QWidget）
        """
        for key, placeholder in waterfall_placeholders.items():
            glview = gl.GLViewWidget()
            layout = QVBoxLayout(placeholder)
            layout.addWidget(glview)
            layout.setContentsMargins(0, 0, 0, 0)  # 清除边距避免视图裁剪

            # 1. 创建X-Y网格（对应sample-chirp平面，z=0基准面）
            grid = gl.GLGridItem(
                size=pg.Vector(256, 32, 1),  # 网格大小：x=256(sample)，y=32(chirp)
                color=(0.8, 0.8, 0.8, 0.5)   # 浅灰色半透明网格（不影响数据查看）
            )
            grid.setSpacing(x=32, y=4, z=1)  # 网格间距：x每32个sample，y每4个chirp
            grid.translate(0, 0, 0)
            glview.addItem(grid)

            # 2. 创建3D坐标轴（x=sample，y=chirp，z=幅值）
            axis = gl.GLAxisItem()
            init_max_mag = self.get_max_magnitude()
            axis.setSize(x=256, y=32, z=init_max_mag * 0.8)  # 轴长匹配数据范围
            axis.translate(-10, -2, 0)  # 偏移避免与数据重叠
            glview.addItem(axis)

            # 3. 初始化表面图（核心：用color数组实现着色，无版本依赖）
            # 初始Z数据：(256,32) = (len(x), len(y))，确保形状匹配
            init_z = np.zeros((256, 32))
            # 初始颜色数组：与Z数据同形状，默认蓝色（低幅值）
            init_color = np.full((256, 32, 4), (0.0, 0.0, 1.0, 1.0), dtype=np.float32)

            surface_plot = gl.GLSurfacePlotItem(
                x=np.arange(256),          # x轴：sample索引(0-255)
                y=np.arange(32),           # y轴：chirp索引(0-31)
                z=init_z,                  # Z轴：初始幅值（全0）
                colors=init_color,         # 颜色数组（与Z同形状，(H,W,4)）
                computeNormals=False       # 关闭法向量计算，提升性能（无视觉影响）
            )
            glview.addItem(surface_plot)

            # 存储视图对象（包含初始幅值范围，用于后续颜色计算）
            self.pg_waterfall_dict[key] = {
                'glview': glview,
                'surface': surface_plot,
                'axis': axis,
                'init_max_mag': init_max_mag  # 初始最大幅值（用于颜色归一化）
            }

            # 设置默认视角（确保能同时看到x/y/z轴，避免视角遮挡）
            glview.setCameraPosition(
                distance=500,    # 视角距离：确保完整显示数据
                elevation=30,    # 仰角30°：看到x-y平面和z轴高度
                azimuth=-45,     # 方位角-45°：同时看到x轴（右）和y轴（前）
                pos=pg.Vector(128, 16, 0)  # 视角中心：数据中心点（避免偏移）
            )

    def update_3d_waterfall(self, iq: np.ndarray, channelstr: str):
        """
        更新3D瀑布图数据（稳定版）：
        1. 计算幅值并转置，确保Z轴形状正确
        2. 用自定义函数生成颜色数组（低幅值→蓝，高幅值→红）
        3. 动态调整坐标轴z轴长度，适配当前幅值范围

        参数：
        - iq: np.ndarray, 形状(4, 32, 256)（通道, chirp, sample）
        - channelstr: str, 天线通道（'tx0rx0'/'tx0rx1'/'tx1rx0'/'tx1rx1'）
        """
        # 1. 通道映射验证
        channel_map = {'tx0rx0': 0, 'tx0rx1': 1, 'tx1rx0': 2, 'tx1rx1': 3}
        if channelstr not in channel_map:
            raise ValueError(f"无效通道：{channelstr}，可选通道：{list(channel_map.keys())}")

        # 2. 检查瀑布图是否初始化
        waterfall_obj = self.pg_waterfall_dict.get('Waterfall')
        if not waterfall_obj:
            raise RuntimeError("请先调用init_3d_waterfall初始化3D瀑布图")

        # 3. 验证IQ数据维度（必须为(4,32,256)）
        if iq.shape != (4, 32, 256):
            raise ValueError(f"IQ数据维度错误：期望(4,32,256)，实际{iq.shape}")

        # 4. 提取指定通道数据并计算幅值
        channel_idx = channel_map[channelstr]
        iq_data = iq[channel_idx, :, :]  # 原始形状：(32,256) = (chirp, sample)
        magnitude_data = np.abs(iq_data) + 1e-6  # 计算幅值，加1e-6避免0值异常

        # 5. 转置幅值数据：确保Z轴形状=(256,32) = (len(x), len(y))
        z_data = magnitude_data.T  # 转置后形状：(256,32)，完全匹配GLSurfacePlotItem要求

        # 6. 生成颜色数组（核心：无版本依赖，手动映射幅值到颜色）
        # 步骤1：归一化幅值到0-1范围（让颜色过渡均匀）
        current_min_mag = np.min(z_data)
        current_max_mag = np.max(z_data)
        # 处理极端情况（所有幅值相同），避免除零错误
        if current_max_mag - current_min_mag < 1e-8:
            norm_mag = np.zeros_like(z_data)
        else:
            norm_mag = (z_data - current_min_mag) / (current_max_mag - current_min_mag)

        # 步骤2：颜色映射（低幅值→蓝(0,0,1)，中幅值→紫(0.5,0,0.5)，高幅值→红(1,0,0)）
        # 红色通道：随幅值增大从0→1
        red = norm_mag
        # 蓝色通道：随幅值增大从1→0
        blue = 1 - norm_mag
        # 绿色通道：固定为0（避免颜色混杂）
        green = np.zeros_like(norm_mag)
        # 透明度：固定为1（不透明，确保可见）
        alpha = np.ones_like(norm_mag)

        # 合并为颜色数组（形状：(256,32,4)，与Z数据匹配）
        color_array = np.stack([red, green, blue, alpha], axis=-1).astype(np.float32)

        # 7. 更新表面图数据（Z轴+颜色数组）
        waterfall_obj['surface'].setData(
            x=np.arange(256),  # 保持x轴不变（sample索引）
            y=np.arange(32),   # 保持y轴不变（chirp索引）
            z=z_data,          # 更新Z轴幅值数据
            colors=color_array # 更新颜色数组（与幅值同步）
        )

        # 8. 动态调整z轴长度（适配当前幅值范围，避免轴过长/过短）
        z_axis_length = (current_max_mag - current_min_mag) * 0.8  # 取幅值范围的80%
        waterfall_obj['axis'].setSize(
            x=256,  # x轴长度固定（256个sample）
            y=32,   # y轴长度固定（32个chirp）
            z=z_axis_length if z_axis_length > 1e-8 else waterfall_obj['init_max_mag'] * 0.8
        )
        # 调整z轴起点到当前最小幅值（避免轴悬空）
        waterfall_obj['axis'].translate(-10, -2, current_min_mag)

    def get_max_magnitude(self):
        """
        辅助函数：返回IQ数据的最大幅值估计（必须根据实际数据校准！）
        方法：运行一次程序后，查看update中的current_max_mag打印值，填入此处
        """
        # 示例：若实际最大幅值为5.2，则改为return 5.2
        # 初始可先设为1.0，后续根据实际数据调整
        return 1.0
        '''