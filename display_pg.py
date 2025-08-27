from typing import Dict, Any, List
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QGraphicsPathItem
from PyQt5.QtCore import QRectF,Qt
import pyqtgraph as pg
from pyqtgraph import ImageView
import numpy as np
from PyQt5.QtGui import QPainterPath, QPen, QColor


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
        self._r_buffer = []
        self._theta_buffer = []

        self.pg_plot_dict: Dict[str, Dict[str, Any]] = {}  # ADC & 1DFFT 曲线
        self.pg_img_dict: Dict[str, ImageView] = {}        # 2DFFT 图像
        self.pg_cloud_dict: Dict[str, Dict[str, Any]] = {} # Point Cloud 图像
        self._colormap = self._build_jet_colormap()
        self._r_max = float(r_max)
        self._theta_center = np.deg2rad(theta_center_deg)
        self._fov = np.deg2rad(fov_deg)
        self._init_point_cloud_semicircle(point_cloud_placeholders)

        self._init_adc(adc_placeholders)
        self._init_fft1d(fft1d_placeholders)
        self._init_fft2d(fft2d_placeholders)


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

    def update_fft1d(self, fft_results_in: np.ndarray, sample: int):
        """
        fft_results_in: shape (4, n_chirp, n_points)
        策略：对 chirp 维度做均值，再取幅度
        """
        fft1d_keys = ['1DFFTtx0rx0', '1DFFTtx0rx1', '1DFFTtx1rx0', '1DFFTtx1rx1']
        max_bin = sample // 2
        x = np.arange(max_bin)
        for ant_idx, key in enumerate(fft1d_keys):
            h = self.pg_plot_dict.get(key)
            if not h:
                continue
            avg_fft = np.mean(fft_results_in[ant_idx, :, :], axis=0)
            mag = np.abs(avg_fft[:max_bin])
            h['MAG'].setData(x, mag)
            h['pw'].setXRange(0, max_bin, padding=0.02)

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

    def update_point_cloud_polar2(self, key: str,
                                 r: np.ndarray,
                                 theta_deg: np.ndarray,
                                 *,
                                 size: float = 5.0,
                                 color='r'):
        """ r-θ(度) -> 半圆散点 """
        if key not in self.pg_cloud_dict:
            return

        h = self.pg_cloud_dict[key]
        theta = np.deg2rad(theta_deg)
        mask = (r >= 0) & (r <= self._r_max) & \
               (theta >= h['theta_min']) & (theta <= h['theta_max'])
        if not np.any(mask):
            h['scatter'].setData([])
            return

        rv = r[mask]
        tv = theta[mask]
        x = rv * np.cos(tv)
        y = rv * np.sin(tv)
        # 大点量建议：h['scatter'].setData(x=x, y=y, size=size, brush=color, pen=None)
        spots = [{'pos': (x[i], y[i])} for i in range(len(x))]
        h['scatter'].setData(spots, size=size, brush=color)

    def update_point_cloud_polar(self, key: str,
                                 r: float, # 修改：现在接受标量 float
                                 theta_deg: float, # 修改：现在接受标量 float
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

        # 1. 将新传入的标量数据添加到缓冲区
        self._r_buffer.append(r)
        self._theta_buffer.append(theta_deg)

        # 2. 如果缓冲区未满，则直接返回，不进行绘制
        if len(self._r_buffer) < 5:
             return

        # 3. 如果缓冲区已满（达到5个），则进行绘制
        # 将缓冲区列表转换为 NumPy 数组
        r_array = np.array(self._r_buffer)
        theta_deg_array = np.array(self._theta_buffer)

        theta_rad = np.deg2rad(theta_deg_array)
        mask = (r_array >= 0) & (r_array <= self._r_max) & \
               (theta_rad >= h['theta_min']) & (theta_rad <= h['theta_max'])

        if not np.any(mask):
            h['scatter'].setData([])
            # 清空缓冲区
            self._r_buffer.clear()
            self._theta_buffer.clear()
            return

        rv = r_array[mask]
        tv = theta_rad[mask]
        x = rv * np.cos(tv)
        y = rv * np.sin(tv)

        # 4. 用所有缓存的数据点进行绘制
        h['scatter'].setData(x=x, y=y, size=size, brush=color, pen=None)

        # 5. 绘制完成后，清空缓冲区
        self._r_buffer.clear()
        self._theta_buffer.clear()
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

    def _init_fft1d(self, placeholders: Dict[str, QWidget]):
        for key, container in placeholders.items():
            layout = QVBoxLayout(container)
            pw = pg.PlotWidget()
            self._set_plot_style(pw)
            pw.addLegend(offset=(10, 10))
            pw.setLabel('bottom', 'FFT Bin')
            pw.setLabel('left', 'Amplitude')
            pw.setTitle(f"{key}", color='k', size='12pt')
            layout.addWidget(pw)

            curve = pw.plot(pen=pg.mkPen('r', width=2), name='MAG')
            self.pg_plot_dict[key] = {'pw': pw, 'MAG': curve}

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
        pen_ring.setCosmetic(True)         # 线宽不随缩放变化

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
        n_rays = 5
        thetas_ray = np.linspace(theta_min, theta_max, n_rays)
        pen_ray = QPen(QColor(180, 180, 180))
        pen_ray.setStyle(Qt.DotLine)
        pen_ray.setCosmetic(True)

        for th in thetas_ray:
            path = QPainterPath()
            path.moveTo(0, 0)
            path.lineTo(r_max * np.cos(th), r_max * np.sin(th))
            item = QGraphicsPathItem(path)
            item.setPen(pen_ray)
            items.append(item)

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
        # 清空曲线与图像
        for h in self.pg_plot_dict.values():
            if 'I' in h: h['I'].clear()
            if 'Q' in h: h['Q'].clear()
            if 'MAG' in h: h['MAG'].clear()
        for iv in self.pg_img_dict.values():
            iv.clear()
        for h in self.pg_cloud_dict.values():
            h['scatter'].clear()
