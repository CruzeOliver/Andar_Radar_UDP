from UI.Ui_Radar_UDP import Ui_MainWindow
import sys, socket, threading, struct
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from plot_utils import init_ADC4_plot

# ================== 协议/网络参数 ==================
LISTEN_IP   = "0.0.0.0"        # 监听所有网卡
LISTEN_PORT = 8888             # 本地接收端口
PEER_IP     = "192.168.1.55"   # 雷达设备IP
PEER_PORT   = 6666             # 若需主动发送，发往的端口
PKT_SIZE    = 1024             # 每个UDP包固定 1024B
MAGIC       = b"\x44\x33\x22\x11"  # 魔数头

# ================== 合理范围/安全上限 ==================
MAX_SAMPLES = 8192
MAX_CHIRPS  = 8192
MAX_FRAME_BYTES = 64 * 1024 * 1024

# ================== Qt 信号总线 ==================
class Bus(QObject):
    log         = pyqtSignal(str)     # log日志重定向
    frame_ready = pyqtSignal(bytes, int, int, int)# frame, sample_point, chirp_num, txrx

# ================== 帧装配状态机 ==================
def _valid_cfg(sample_point: int, chirp_num: int) -> bool:
    return 1 <= sample_point <= MAX_SAMPLES and 1 <= chirp_num <= MAX_CHIRPS

class AsmState:
    # 状态
    awaiting_cfg: bool = True   # True=等待配置包；False=收集数据包
    # 配置
    sample_number: int = 0
    chirp_number:  int = 0
    tx_rx_type:    int = 1      # 缺省按1处理
    # 拼帧
    total_pkts:    int = 0
    frame_buf:     bytearray = bytearray()
    pkg_cnt:       int = 0

class DataAssembler:
    """
    状态机：
      awaiting_cfg=True  时，只接受配置包（magic 开头），解析 sample/chirp(/txrx) 并计算帧大小、分包数；
      awaiting_cfg=False 时，连续收 1024B 数据包，攒满一帧后返回 bytes，并回到 awaiting_cfg=True。
    """
    def __init__(self, bus: Bus):
        self.bus = bus
        self.s   = AsmState()

    def process(self, datagram: bytes) -> bytes | None:
        """输入一个 1024B 数据报；若完成一帧则返回 bytes，否则返回 None。"""
        if len(datagram) != PKT_SIZE:
            return None  #非 1024B 包忽略

        if self.s.awaiting_cfg:
            if not datagram.startswith(MAGIC):
                # 等配置，非magic包一律忽略
                return None
            parsed = self._parse_config(datagram)
            if not parsed:
                # 解析失败或不在合理范围，继续等待下一个配置包
                return None

            sample_point, chirp_num, txrx = parsed
            self.s.sample_number = sample_point
            self.s.chirp_number  = chirp_num
            self.s.tx_rx_type    = txrx

            # 计算一帧字节数
            if txrx == 4:
                # 4 虚拟天线 * chirp * sample * (I/Q各int16=4字节)
                total_bytes = 4 * chirp_num * sample_point * 4
            else:
                # txrx==1 或默认：每样本4字节
                total_bytes = chirp_num * sample_point * 4

            if total_bytes <= 0 or total_bytes > MAX_FRAME_BYTES:
                self.bus.log.emit(f"[CFG] total_bytes={total_bytes} 超限，丢弃配置")
                return None

            try:
                self.s.frame_buf = bytearray(total_bytes)
            except MemoryError:
                self.bus.log.emit(f"[CFG] 申请内存失败 total_bytes={total_bytes}")
                return None

            self.s.total_pkts = (total_bytes + PKT_SIZE - 1) // PKT_SIZE
            self.s.pkg_cnt = 0
            self.s.awaiting_cfg = False

            self.bus.log.emit(f"[CFG] sample={sample_point} chirp={chirp_num} txrx={txrx} "
                              f"bytes={total_bytes} pkts={self.s.total_pkts}")
            return None

        else:
            # 收集数据包
            offset = self.s.pkg_cnt * PKT_SIZE
            if offset >= len(self.s.frame_buf):
                self.bus.log.emit("Error: Buffer overflow detected!")
                self._reset_to_wait_cfg()
                return None

            n = min(PKT_SIZE, len(self.s.frame_buf) - offset)
            self.s.frame_buf[offset:offset + n] = datagram[:n]
            self.s.pkg_cnt += 1

            if self.s.pkg_cnt >= self.s.total_pkts:
                # 帧完成
                frame = bytes(self.s.frame_buf)
                self._reset_to_wait_cfg()
                return (frame, self.s.sample_number, self.s.chirp_number, self.s.tx_rx_type)

            return None

    def _parse_config(self, buf: bytes):
        """
        固定布局（已从运行日志锁定）：
        magic(0..3) = 44 33 22 11
        [4..11]     = 8字节
        [12..15]    = samplePoint (int32, LE)
        [16..19]    = chirpNum    (int32, LE)
        [20..23]    = TxRxType    (int32, LE)
        """
        try:
            sample_point, chirp_num, txrx = struct.unpack_from("<iii", buf, 12)
        except struct.error:
            self.bus.log.emit("[CFG] 解析异常（长度不足）")
            return None

        # 合理范围校验
        if not (1 <= sample_point <= MAX_SAMPLES and 1 <= chirp_num <= MAX_CHIRPS):
            self.bus.log.emit(f"[CFG] 越界: sample={sample_point} chirp={chirp_num}")
            return None

        # TxRxType 非法就兜底为 1
        if txrx not in (1, 4):
            self.bus.log.emit(f"[CFG] txrx={txrx} 非法，按 1 处理")
            txrx = 1

        return (sample_point, chirp_num, txrx)

    def _reset_to_wait_cfg(self):
        self.s.awaiting_cfg = True
        self.s.total_pkts = 0
        self.s.pkg_cnt = 0
        self.s.frame_buf = bytearray()

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

        self.rx_thread = None
        self.tx_sock   = None

        # 存放绘图 canvas 和 axes 对象
        self.canvas_dict = {}
        self.ax_dict = {}

        # 定义布局，将每个 widget 与 matplotlib 图形关联
        self.layout_dict = {
            'tx0rx0': QVBoxLayout(self.widget_tx0rx0),
            'tx0rx1': QVBoxLayout(self.widget_tx0rx1),
            'tx1rx0': QVBoxLayout(self.widget_tx1rx0),
            'tx1rx1': QVBoxLayout(self.widget_tx1rx1)
        }

        # 初始化 Matplotlib 图形
        for key, layout in self.layout_dict.items():
            fig, ax = plt.subplots(figsize=(0, 0))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            self.canvas_dict[key] = canvas
            self.ax_dict[key] = ax
            init_ADC4_plot(ax)

        self.bus = Bus()
        self.bus.log.connect(self._log)
        self.bus.frame_ready.connect(self.on_frame_ready)

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

    def reorder_frame(self, frame_bytes: bytes, sample: int, chirp: int, window: np.ndarray | None = None):
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


    # ---- 整帧到达回调：在这里做解析/计算/存档/绘图 ----
    def on_frame_ready(self, frame: bytes, sample: int, chirp: int, txrx: int):
        """
        接收到一帧数据后，绘制 I/Q 波形
        """
        if txrx == 4:
            iq = self.reorder_frame(frame, sample, chirp)  # (4, chirp, sample)

        # 获取时域 I/Q 波形
        t = np.arange(sample)

        for ant_idx, (key, ax) in enumerate(self.ax_dict.items()):
            I = np.real(iq[ant_idx, 0, :])  # 实部
            Q = np.imag(iq[ant_idx, 0, :])  # 虚部

            ax.clear()  # 清除之前的图像

            # 绘制 I 和 Q 波形
            ax.plot(t, I, label='I', color='r')  # 红色绘制 I
            ax.plot(t, Q, label='Q', color='b')  # 蓝色绘制 Q

            # 重新绘制坐标轴和标题
            self.init_plot(ax)

            # 更新图形
            ax.legend(loc='best')
            self.canvas_dict[key].draw()  # 更新画布

    def closeEvent(self, e):
        self.UDP_disconnect()
        super().closeEvent(e)

# ================== 入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyMainForm()
    win.show()
    sys.exit(app.exec_())
