from UI.Ui_Radar_UDP import Ui_MainWindow
import sys, socket, threading
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QFileDialog, QMessageBox, QSizePolicy
from PyQt5.QtGui import QPixmap, QIcon
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from plot_utils import *
from data_processing import *
import scipy.io
from udp_handler import *

# ================== Qt 信号总线 ==================
class Bus(QObject):
    log         = pyqtSignal(str)     # log日志重定向
    frame_ready = pyqtSignal(bytes, int, int, int)# frame, sample_point, chirp_num, txrx


# ================== 接收线程（Python threading + socket） ==================
class UdpRxThread(threading.Thread):
    def __init__(self, ip: str, port: int, bus: Bus):
        super().__init__(daemon=True)
        self.ip, self.port = ip, port
        self.bus = bus
        self._stop_evt = threading.Event()
        self._sock = None
        self._asm  = DataAssembler(bus)

    def run(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.ip, self.port))      # 0.0.0.0:8888
            self._sock.settimeout(0.5)                 # 短超时便于退出
            self.bus.log.emit(f"[OK] 监听 {self.ip}:{self.port} ...")
        except Exception as e:
            self.bus.log.emit(f"[ERR] 绑定失败: {e!r}")
            return

        while not self._stop_evt.is_set():
            try:
                data, (sip, sport) = self._sock.recvfrom(PKT_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            if sip != PEER_IP:
                continue

            res = self._asm.process(data)
            if res is not None:
                frame, sample, chirp, txrx = res
                self.bus.frame_ready.emit(frame, sample, chirp, txrx)

        try:
            if self._sock:
                self._sock.close()
        finally:
            self._sock = None
            self.bus.log.emit("[OK] 接收线程已退出")

    def stop(self):
        self._stop_evt.set()
        # 唤醒一次阻塞的 recvfrom，加速退出
        try:
            tmp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tmp.sendto(b"", ("127.0.0.1", self.port))
            tmp.close()
        except Exception:
            pass

# ================== 主窗口 ==================
class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Radar UDP Interface")
        self.setMinimumSize(1800, 1400)

        self.pushButton_Disconnect.setEnabled(False)
        options = ["CPP", "Python"]
        self.comboBox_MatFrom.addItems(options)
        self.comboBox_MatFrom.currentIndex = 0  # 默认选择第一个选项

        self.fft_results_1D = None
        self.fft_result_2D = None
        self.frame_all_data = None
        self.frame_data_list = []
        self.rx_thread = None
        self.tx_sock   = None

        self.last_display_time = time.time()# 记录最后显示的时间
        self.display_interval = 1.0 # 1秒间隔

        # 存放绘图 canvas 和 axes 对象
        self.canvas_dict = {}
        self.ax_dict = {}
        self.setup_mpl_widgets()

        self.bus = Bus()
        self.bus.log.connect(self._log)
        self.bus.frame_ready.connect(self.on_frame_ready)

    def setup_mpl_widgets(self):
        """
        创建 Matplotlib 图形并将其关联到指定的 Qt widgets。
        """
        self.setWindowIcon(QIcon('Radar_UDP_icon2.png'))  # 设置窗口图标
        pixmap = QPixmap(r'CJLU_logo.png')
        if pixmap.isNull():
            QMessageBox.warning(self, "图像加载失败", "无法加载图像，请检查文件路径是否正确。")
        else:
            self.CJLU_logo_label.setPixmap(pixmap)
            self.CJLU_logo_label.setScaledContents(True)

        # 原始 ADC4 窗口
        adc4_keys = ['tx0rx0', 'tx0rx1', 'tx1rx0', 'tx1rx1']

        # 1D FFT 四个窗口
        fft_keys = ['1DFFTtx0rx0', '1DFFTtx0rx1', '1DFFTtx1rx0', '1DFFTtx1rx1']

        # 2D FFT 窗口
        fft2d_keys = ['2DFFTtx0rx0', '2DFFTtx0rx1', '2DFFTtx1rx0', '2DFFTtx1rx1']

        # 所有布局对应 QWidget
        self.layout_dict = {}
        for key in adc4_keys:
            self.layout_dict[key] = QVBoxLayout(getattr(self, f'widget_{key}'))

        for key in fft_keys:
            self.layout_dict[key] = QVBoxLayout(getattr(self, f'widget_{key}'))

        for key in fft2d_keys:
            self.layout_dict[key] = QVBoxLayout(getattr(self, f'widget_{key}'))

        # 循环初始化所有图形
        for key, layout in self.layout_dict.items():
            fig, ax = plt.subplots(figsize=(5, 3))  # 初始大小
            canvas = FigureCanvas(fig)

            # 自适应 widget 尺寸
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.updateGeometry()

            layout.addWidget(canvas)
            self.canvas_dict[key] = canvas
            self.ax_dict[key] = ax

            # 根据 key 初始化不同样式
            if key in fft_keys:
                init_1DFFT_plot(ax)
            elif key in fft2d_keys:
                init_2DFFT_plot(ax)
            else:
                init_ADC4_plot(ax)

    def _log(self, s: str):
        # 重定向日志到 textEdit_log
        try:
            self.textEdit_log.append(s)
        except Exception:
            print(s)

    # ---- 连接：开接收 + 备发送 ----
    def UDP_connect(self):
        self.UDP_disconnect()  # 防止重复
        self.rx_thread = UdpRxThread(LISTEN_IP, LISTEN_PORT, self.bus)
        self.rx_thread.start()
        try:
            self.tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.bus.log.emit(f"[OK] 发送目标 {PEER_IP}:{PEER_PORT}")
            self.pushButton_Connect.setEnabled(False)
            self.pushButton_Disconnect.setEnabled(True)
        except Exception as e:
            self.bus.log.emit(f"[ERR] 创建发送 socket 失败: {e!r}")
            self.tx_sock = None

    # ---- 断开：停线程 + 关socket ----
    def UDP_disconnect(self):
        if self.rx_thread:
            self.rx_thread.stop()
            self.rx_thread.join(timeout=2.0)
            self.rx_thread = None
        if self.tx_sock:
            try: self.tx_sock.close()
            except Exception: pass
            self.tx_sock = None
        self.bus.log.emit("[OK] 已断开")
        self.pushButton_Connect.setEnabled(True)
        self.pushButton_Disconnect.setEnabled(False)

    # ---- 整帧到达回调函数 ----
    def on_frame_ready(self, frame: bytes, sample: int, chirp: int, txrx: int):
        """
        数据格式正确,接收到一帧数据后回调函数
        """
         # 保存到 .mat 文件
        if self.checkBox_IsSave.isChecked():
            if not self.save_to_mat(frame,sample,chirp,"raw_data.mat"):
                #self.bus.log.emit("[OK] 原始数据已保存到 raw_data.mat")
                self.bus.log.emit("[ERR] 保存原始数据失败")

        current_time = time.time()
        iq = reorder_frame(frame, chirp, sample)
        self.fft_results_1D = Perform1D_FFT(iq)
        self.fft_result_2D = Perform2D_FFT(self.fft_results_1D)
        # 判断是否满足显示间隔
        if current_time - self.last_display_time > self.display_interval:
            self.DisplayADC4Waveform(iq, chirp, sample)
            self.Display1DFFT(self.fft_results_1D, sample)
            self.Display2DFFT(self.fft_result_2D, sample, chirp)
            self.last_display_time = current_time
        else:
            pass
        #R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod = calculate_distance_from_fft2(self.fft_results_1D[0], chirp, sample)
        #self.bus.log.emit(f"距离计算结果：FFT={R_fft:.2f} m, Macleod={R_macleod:.2f} m, CZT FFT Peak={R_czt_fftpeak:.2f} m, CZT Macleod={R_czt_macleod:.2f} m")

# ================== Plot图像部分内容 ==================
    def DisplayADC4Waveform(self, iq, chirp: int, sample: int):
        """
        接收到一帧数据后，显示 ADC4 波形
        """
        # 绘制时域 I/Q 波形
        t = np.arange(sample)
        for ant_idx, (key, ax) in enumerate(list(self.ax_dict.items())[:4]):
            I = np.real(iq[ant_idx, 0, :])  # 实部
            Q = np.imag(iq[ant_idx, 0, :])  # 虚部
            ax.clear()  # 清除之前的图像

            # 绘制 I 和 Q 波形
            ax.plot(t, I, label='I', color='r')  # 红色绘制 I
            ax.plot(t, Q, label='Q', color='b')  # 蓝色绘制 Q
            #init_ADC4_plot(ax)

            # 更新图形
            ax.set_title(f"ADC {key}")
            ax.set_xlabel("Sample points")
            ax.set_ylabel("Amplitude")
            ax.legend(loc='best')
            ax.grid(True)
            self.canvas_dict[key].draw()  # 更新画布

    def Display1DFFT(self, fft_results_in, sample: int):
        """
        显示四路虚拟天线的 1D FFT 结果。

        fft_results_in: np.ndarray, 形状为 (n_ant, n_chirp, n_points)
        sample: int, FFT 点数
        """
        fft_keys = ['1DFFTtx0rx0', '1DFFTtx0rx1', '1DFFTtx1rx0', '1DFFTtx1rx1']

        # FFT 结果的有效点数
        max_bin = sample // 2

        # 对每个天线的数据进行循环
        for ant_idx, key in enumerate(fft_keys):
            fft_ax = self.ax_dict[key]
            fft_canvas = self.canvas_dict[key]

            fft_ax.clear()

            # 从三维数组中选择一个天线的数据，并对所有 Chirp 的 FFT 结果求平均
            # 结果是一个形状为 (n_points,) 的一维数组
            avg_fft_result = np.mean(fft_results_in[ant_idx, :, :], axis=0)

            # 仅绘制 FFT 结果的前半部分
            fft_ax.plot(np.abs(avg_fft_result[:max_bin]), color='r')

            fft_ax.set_title(f"{key}")
            fft_ax.set_xlabel("FFT Bin")
            fft_ax.set_ylabel("Amplitude")
            fft_ax.grid(True)

            fft_canvas.draw()

    def Display2DFFT(self, fft2d_results, n_points: int, n_chirp: int):
        """
        显示四路虚拟天线的 2D FFT（距离-多普勒）结果。

        fft2d_results: np.ndarray, 形状为 (n_ant, n_chirp, n_points)
        n_points: int, 距离维度点数
        n_chirp: int, 多普勒维度点数
        """

        fft2d_keys = ['2DFFTtx0rx0', '2DFFTtx0rx1', '2DFFTtx1rx0', '2DFFTtx1rx1']

        max_range_bin = n_points // 2

        # 确保Perform2D_FFT 的数组形状是 (n_ant, n_points, n_chirp)，
        fft2d_results_T = np.transpose(fft2d_results, (0, 2, 1))

        for ant_idx, key in enumerate(fft2d_keys):
            fft_ax = self.ax_dict[key]
            fft_canvas = self.canvas_dict[key]

            # 你的颜色条处理代码，保持不变
            if hasattr(fft_ax, 'cbar'):
                try:
                    fft_ax.cbar.remove()
                except ValueError:
                    pass
                del fft_ax.cbar

            fft_ax.clear()

            # 获取要显示的数据，形状为 (n_points, n_chirp)
            fft_data = fft2d_results_T[ant_idx, :, :]
            display_data = np.abs(fft_data[:max_range_bin, :])

            # 确保数据没有负数或零，以避免 log 警告
            display_data[display_data <= 1e-6] = 1e-6

            im = fft_ax.imshow(
                np.log10(display_data),
                aspect='auto',
                cmap='jet',
                origin='lower',
                interpolation='none'
            )

            fft_ax.set_title(f"{key}")
            fft_ax.set_ylabel("Range Bin")
            fft_ax.grid(False)

            # 修正多普勒轴的刻度和标签
            doppler_bins = np.linspace(-n_chirp / 2, n_chirp / 2, 5)
            fft_ax.set_xticks(np.linspace(0, n_chirp - 1, 5))
            fft_ax.set_xticklabels([f"{int(bin)}" for bin in doppler_bins])
            fft_ax.set_xlabel("Doppler Bin")

            # 你的颜色条创建代码，保持不变
            cbar = fft_canvas.figure.colorbar(im, ax=fft_ax)
            cbar.set_label("Amplitude (dB)")
            fft_ax.cbar = cbar

            fft_canvas.draw()

# ================== 文件读取部分内容 ==================
    def save_to_mat(self,frame_data, sample_number, chirp_number, filename="raw_data.mat"):
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

    def ReadFile(self):
        """
        打开文件对话框，选择 .mat 文件并读取数据
        """
        file_dialog = QFileDialog(self, "Open MAT File")
        file_dialog.setNameFilter("MAT files (*.mat)")
        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.read_mat_file(file_path)

    def read_mat_file(self, filename):
        """
        读取 MAT 文件中的数据
        """
        try:
            data = scipy.io.loadmat(filename) # 读取 .mat 文件
            self.frame_all_data = data

            self.bus.log.emit(f"读取文件：{filename}")# 打印文件中包含的变量
            #print(f"文件中包含的变量：{list(data.keys())}")

            # 获取所有包含帧数据的变量（以 "frame" 开头的变量名）
            self.frame_data_list = [key for key in data.keys() if key.startswith('frame')]
            self.current_index = 0  # 初始化为第一帧
            # 获取第一帧的数据
            frame_data = self.frame_all_data[self.frame_data_list[self.current_index]]
            self.show_matrix(frame_data)  # 显示第一帧
        except Exception as e:
            print(f"读取文件时出错: {e}")
            QMessageBox.warning(self, "读取失败", f"读取文件失败：{e}")

    def show_matrix(self, frame_data):
        """
        显示当前帧的数据
        """
        #print(f"显示当前帧数据：{frame_data}")
        #print(f"帧数据形状：{frame_data.shape}")
        self.bus.log.emit(f"{self.frame_data_list[self.current_index]} 数据已加载")
        selected_label = self.comboBox_MatFrom.currentText()
        if selected_label == "CPP":  # C++ 数据
            frame_data = frame_data.T  # 转置数据，确保行优先
            sample = frame_data.shape[0] // 8  # 4 虚拟天线，每个天线 2 个通道（I/Q）
            chirp = frame_data.shape[1]
            frame_data_flat = frame_data.flatten()
        elif selected_label == "Python":  # Python 数据
            sample = frame_data.shape[1] // 8  # 4 虚拟天线，每个天线 2 个通道（I/Q）
            chirp = frame_data.shape[0]
            frame_data_flat = frame_data.flatten()
        iq = reorder_frame(frame_data_flat, int(chirp), int(sample))
        self.DisplayADC4Waveform(iq, chirp, sample)
        self.fft_results_1D = Perform1D_FFT(iq)
        self.fft_result_2D = Perform2D_FFT(self.fft_results_1D)
        self.Display1DFFT(self.fft_results_1D, sample)
        self.Display2DFFT(self.fft_result_2D, sample, chirp)
        R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod = calculate_distance_from_fft2(self.fft_results_1D[0], chirp, sample)
        self.bus.log.emit(f"距离计算结果：FFT={R_fft:.4f} m, Macleod={R_macleod:.4f} m, CZT FFT Peak={R_czt_fftpeak:.4f} m, CZT Macleod={R_czt_macleod:.4f} m")

    def ShowNextFrame(self):
        if self.current_index < len(self.frame_data_list) - 1:
            self.current_index += 1
            self.show_matrix(self.frame_all_data[self.frame_data_list[self.current_index]])
        else:
            QMessageBox.information(self, "没有更多数据", "已到达文件末尾！")

    def CloseFile(self):
        self.frame_all_data = None
        self.frame_data_list = []  # 清空数据
        self.current_index = 0  # 重置索引
        self.textEdit_log.clear()  # 清空日志
        for key, ax in self.ax_dict.items():
            ax.clear()
            ax.figure.canvas.draw()  # 重新绘制图形
        self.bus.log.emit("已关闭文件，清空数据")

    def closeEvent(self, e):
        self.UDP_disconnect()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyMainForm()
    win.show()
    sys.exit(app.exec_())
