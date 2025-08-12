from dataclasses import dataclass
import struct
from Andar_udp import Bus
# ================== 网络和接收线程 ==================

# 网络协议/参数
LISTEN_IP = "0.0.0.0"        # 监听所有网卡
LISTEN_PORT = 8888           # 本地接收端口
PEER_IP = "192.168.1.55"     # 雷达设备IP
PEER_PORT = 6666             # 若需主动发送，发往的端口
PKT_SIZE = 1024              # 每个UDP包固定 1024B
MAGIC = b"\x44\x33\x22\x11"  # 魔数头

MAX_SAMPLES = 8192
MAX_CHIRPS = 8192
MAX_FRAME_BYTES = 64 * 1024 * 1024

# ================== 帧装配状态机 ==================
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
                total_bytes = 4 * chirp_num * sample_point * 2 *2
            else:
                # txrx==1 或默认：每样本4字节
                total_bytes = chirp_num * sample_point * 2 *2

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

            # self.bus.log.emit(f"[CFG] sample={sample_point} chirp={chirp_num} txrx={txrx} "
            #                   f"bytes={total_bytes} pkts={self.s.total_pkts}")
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
        [12..15]    = chirpNum (int32, LE)
        [16..19]    = samplePoint    (int32, LE)
        [20..23]    = TxRxType    (int32, LE)
        """
        try:
            chirp_num,sample_point,  txrx = struct.unpack_from("<iii", buf, 12)
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

