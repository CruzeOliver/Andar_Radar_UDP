from UI.Ui_Radar_UDP import Ui_MainWindow
import sys, socket, threading
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal, QRectF, Qt
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox,  QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtGui import QPixmap, QIcon
import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView
from data_processing import *
import scipy.io
import warnings
from udp_handler import *
from display_pg import PgDisplay

# 加入DPI缩放，可以让GUI，在不同分辨率显示器之间跨越 ，不变形
QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 启用 DPI 缩放
QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)      # 启用高 DPI 图标和图像

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
        self.setWindowIcon(QIcon('Radar_UDP_icon.png'))
        pixmap = QPixmap(r'CJLU_logo.png')
        if pixmap.isNull():
            QMessageBox.warning(self, "图像加载失败", "无法加载图像，请检查文件路径是否正确。")
        else:
            self.CJLU_logo_label.setPixmap(pixmap)
            self.CJLU_logo_label.setScaledContents(True)
        self.resize(1800, 1400)

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
        self.save_filename = None
        self.current_index = 0
        self.generate_unique_filename()

        self.last_display_time = time.time()# 记录最后显示的时间
        self.display_interval = 0.8

        adc4_keys  = ['tx0rx0', 'tx0rx1', 'tx1rx0', 'tx1rx1']
        fft1d_keys = ['1DFFTtx0rx0', '1DFFTtx0rx1', '1DFFTtx1rx0', '1DFFTtx1rx1']
        fft2d_keys = ['2DFFTtx0rx0', '2DFFTtx0rx1', '2DFFTtx1rx0', '2DFFTtx1rx1']
        point_cloud_keys = ['PointCloud']
        ConstellationDiagram_keys = ['CDtx0rx0', 'CDtx0rx1', 'CDtx1rx0', 'CDtx1rx1']
        amp_phase_keys = ['APtx0rx0', 'APtx0rx1', 'APtx1rx0', 'APtx1rx1']

        adc_placeholders = {k: getattr(self, f'widget_{k}') for k in adc4_keys}
        fft1d_placeholders = {k: getattr(self, f'widget_{k}') for k in fft1d_keys}
        fft2d_placeholders = {k: getattr(self, f'widget_{k}') for k in fft2d_keys}
        point_cloud_placeholders = {k: getattr(self, f'widget_{k}') for k in point_cloud_keys}
        constellation_placeholders = {k: getattr(self, f'widget_{k}') for k in ConstellationDiagram_keys}
        amp_phase_placeholders = {k: getattr(self, f'widget_{k}') for k in amp_phase_keys}

        #GUI显示界面绑定实例化
        self.display = PgDisplay(
            adc_placeholders   = adc_placeholders,
            fft1d_placeholders = fft1d_placeholders,
            fft2d_placeholders = fft2d_placeholders,
            point_cloud_placeholders = point_cloud_placeholders,
            constellation_placeholders = constellation_placeholders,
            amp_phase_placeholders = amp_phase_placeholders
        )

        self.bus = Bus()
        self.bus.log.connect(self._log)
        self.bus.frame_ready.connect(self.on_frame_ready)

        self.tableWidget_distance.setColumnCount(6)
        header_labels = ['index','Angel','FFT', 'Macleod', 'CZT FFT Peak', 'CZT Macleod']
        self.tableWidget_distance.setHorizontalHeaderLabels(header_labels)
        self.tableWidget_distance.setEditTriggers(QTableWidget.NoEditTriggers)
        # QHeaderView.Stretch 模式会使所有列等宽拉伸，填充可用空间。
        self.tableWidget_distance.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget_distance.verticalHeader().setVisible(False)

    def generate_unique_filename(self):
        """生成一个唯一的 .mat 文件名并保存为实例属性"""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.save_filename = f"{timestamp}_raw_data_py.mat"

    # ---- 重定向日志到 textEdit_log ----
    def _log(self, s: str):
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
        if self.checkBox_IsSave.isChecked():
            self.bus.log.emit("[OK] 已启用原始数据保存功能")

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
        if self.checkBox_IsSave.isChecked():
            self.bus.log.emit(f"[OK] 原始数据保存至{self.save_filename}，请在文件夹中查看")

    # ---- 整帧到达回调函数 ----
    def on_frame_ready(self, frame: bytes, sample: int, chirp: int, txrx: int):
        """
        数据格式正确,接收到一帧数据后回调函数
        """
         # 保存到 .mat 文件
        if self.checkBox_IsSave.isChecked():
            if not self.save_to_mat(frame,sample,chirp,self.save_filename):
                #self.bus.log.emit("[OK] 原始数据已保存到 raw_data.mat")
                self.bus.log.emit("[ERR] 保存原始数据失败")

        current_time = time.time()
        iq = reorder_frame(frame, chirp, sample)
        self.fft_results_1D = Perform1D_FFT(iq)
        self.fft_result_2D = Perform2D_FFT(self.fft_results_1D)

        #根据2dfft结果 进行不同Channel的IQ校准
        peak_idx = np.unravel_index(np.argmax(np.abs(self.fft_result_2D[0])), self.fft_result_2D[0].shape)
        zij_vector = self.fft_result_2D[:, peak_idx[0], peak_idx[1]]
        beta_vector = complex_channel_calibration(zij_vector)
        if self.checkBox_complex_calibration.isChecked():
            iq = apply_complex_calibration(iq, beta_vector)

        # 判断是否满足显示间隔
        if current_time - self.last_display_time > self.display_interval:
            self.display.update_adc4(iq, chirp, sample)
            if self.checkBox_APcoherence.isChecked():
                self.display.update_constellations(iq, remove_dc=True, max_points=3000, show_fit=True)
                self.display.update_amp_phase(iq, chirp=0, decimate=1, unwrap_phase=False)
            if self.checkBox_1dfft.isChecked():
                self.display.update_fft1d(self.fft_results_1D, sample)
            if self.checkBox_2dfft.isChecked():
                self.display.update_fft2d(self.fft_result_2D, sample, chirp)
            self.last_display_time = current_time
        else:
            pass
        R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod = calculate_distance_from_fft2(self.fft_results_1D[0], chirp, sample)
        az, el, idx, info = estimate_az_el_from_fft2d(self.fft_result_2D)
        self.display.update_point_cloud_polar("PointCloud", R_macleod, 90.0-az, size=10.0, color='g')

        # 更新表格显示距离、角度计算结果
        row_data = [f"{self.current_index}",f"{az:.4f}",f"{R_fft:.4f} m",
                    f"{R_macleod:.4f} m",f"{R_czt_fftpeak:.4f} m",f"{R_czt_macleod:.4f} m"]
        row_count = self.tableWidget_distance.rowCount()
        self.tableWidget_distance.insertRow(row_count)
        for i, value in enumerate(row_data):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)# 设置单元格居中对齐
            self.tableWidget_distance.setItem(row_count, i, item)
        self.tableWidget_distance.scrollToBottom()# 滚动到底部
        self.current_index += 1


# ================== 文件读取部分内容 ==================
    def save_to_mat(self,frame_data, sample_number, chirp_number, filename= None):
        try:
            # 获取当前时间戳，确保每一帧有唯一的变量名
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)#消除对于“__header__”的警告
                try:
                    from scipy.io.matlab.mio import MatWriteWarning
                    warnings.simplefilter("ignore", category=MatWriteWarning)
                except ImportError:
                    pass
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
        self.fft_results_1D = Perform1D_FFT(iq)
        self.fft_result_2D  = Perform2D_FFT(self.fft_results_1D)

        peak_idx = np.unravel_index(np.argmax(np.abs(self.fft_result_2D[0])), self.fft_result_2D[0].shape)
        zij_vector = self.fft_result_2D[:, peak_idx[0], peak_idx[1]]
        # alpha_matrix = amplitude_calibration(zij_vector)
        # phi_matrix = phase_calibration(zij_vector)
        # if self.checkBox_2dfft.isChecked():
        #     iq = apply_calibration(iq, alpha_matrix, phi_matrix)
        beta_vector = complex_channel_calibration(zij_vector)
        if self.checkBox_complex_calibration.isChecked():
            iq = apply_complex_calibration(iq, beta_vector)

        self.display.update_adc4(iq, chirp, sample)
        self.display.update_constellations(iq, remove_dc=True, max_points=3000, show_fit=True)
        self.display.update_amp_phase(iq, chirp=0, decimate=1, unwrap_phase=False)


        if self.checkBox_1dfft.isChecked():
            self.display.update_fft1d(self.fft_results_1D, sample)
        if self.checkBox_2dfft.isChecked():
            self.display.update_fft2d(self.fft_result_2D, sample, chirp)

        R_fft, R_macleod, R_czt_fftpeak, R_czt_macleod = calculate_distance_from_fft2(self.fft_results_1D[0], chirp, sample)
        az, el, idx, info = estimate_az_el_from_fft2d(self.fft_result_2D)
        self.display.update_point_cloud_polar("PointCloud", R_macleod, 90.0-az, size=10.0, color='g')

        # 更新表格显示距离、角度计算结果
        row_data = [f"{self.current_index}",f"{az:.4f}",f"{R_fft:.4f} m",
                    f"{R_macleod:.4f} m",f"{R_czt_fftpeak:.4f} m",f"{R_czt_macleod:.4f} m"]
        row_count = self.tableWidget_distance.rowCount()
        self.tableWidget_distance.insertRow(row_count)
        for i, value in enumerate(row_data):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)# 设置单元格居中对齐
            self.tableWidget_distance.setItem(row_count, i, item)
        self.tableWidget_distance.scrollToBottom()# 滚动到底部

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
        self.tableWidget_distance.clearContents()  # 清空表格内容
        self.tableWidget_distance.setRowCount(0)
        self.display.reset()
        self.bus.log.emit("已关闭文件，清空数据")

    def closeEvent(self, e):
        self.UDP_disconnect()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyMainForm()
    win.show()
    sys.exit(app.exec_())
