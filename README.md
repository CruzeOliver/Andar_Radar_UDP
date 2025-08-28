

# 项目名称：Andar_GUi_UDP

## 简介



这个项目是一个基于 Python 的桌面应用程序，用于实时采集（通过UDP进行数据传输）、处理和可视化雷达数据。它利用 `PyQt` 构建用户界面，并使用 `PyQtGraph` 进行高效的图形显示。主要功能包括：

- 实时显示四路虚拟天线的 ADC 原始数据。
- 实时计算并显示 1D FFT 结果。
- 实时计算并显示 2D FFT（距离-多普勒）结果。
- 实时计算并显示2D 点云。
- 支持将实时数据保存到 `.mat` 文件中。

本程序特别针对雷达信号处理进行了优化，利用 `PyQtGraph` 的高性能绘图能力，确保在处理高帧率数据时界面的流畅性。



## 项目结构



项目的核心文件可能包括：

- `Andar_udp.py`：主程序文件，包含 GUI 界面和数据处理逻辑。
- `Radar_UDP.ui`：Qt Designer 设计的界面文件。
- `data_processing.py`：主要是包含对于雷达IQ数据进行重组、1DFFT、2DFFT等处理
- `udp_handler.py`：UDP信号传输等相关工作
- `raw_data.mat`：程序保存的雷达原始数据文件。



## 如何运行

### 1. 依赖库

在运行项目之前，请确保你已经安装了所有必要的 Python 库。你可以使用 `pip` 来安装它们：

Bash

```
pip install PyQt5 pyqtgraph numpy scipy
```

- **`PyQt5`**：用于构建图形用户界面。
- **`pyqtgraph`**：用于高性能的科学绘图。
- **`numpy`**：用于处理数值数组。
- **`scipy`**：用于加载和保存 `.mat` 文件。



### 2. 运行程序

安装完依赖后，直接运行主程序文件即可：

```
python Andar_udp.py
```



### 3. 数据保存

程序支持将实时数据保存到 `.mat` 文件。在每次启动程序时，会自动生成一个以当前时间戳命名的 `.mat` 文件，例如 `2025_08_20_14_30_00_raw_data_py.mat`，所有处理的帧数据都会被追加到这个文件中。



## 主要功能

### 传输协议：UDP
包长：1024byte

数据类型：大端序

First Frame（1024byte）：

|    4byte 	  |     4byte   	  |    4byte  	  |    4byte  	 |    4byte    	|   4byte	   |

FirstNumber     SecondNumber         FrameID            ChirpNum     Sample_POINT      TXRXTYPE

其余补零（1000byte）

FirstNumber : 0x11223344

SecondNumber : 0x44332211

FrameID : 发送完ADC采样一次的所有数据自增加一（最大0xFFFFFFFF）

ChirpNum : 下位机配置信息（例：64）

Sample_POINT : 下位机配置信息（例：128）

TXRXTYPE : 下位机配置信息（例：1）

仅有三种模式：TX1RX1（1）、TX1RX2（2）、TX2RX2（4）

Other Frame（1024byte）：纯ADC数据

### 实时 ADC 数据 （tabe page：ADC-T2R2）

显示 I/Q 通道的原始数据，用于观察信号质量和噪声。

### 1D FFT（距离）（tabe page：1D-FFT-T2R2）

显示 1D FFT 结果，可以帮助你找到距离上的峰值，即目标的距离信息。

### 2D FFT（距离-多普勒）（tabe page：2D-FFT-T2R2）

显示距离-多普勒热力图，X 轴为多普勒（速度），Y 轴为距离。这可以帮助你同时分析目标的速度和距离，并且多普勒零点（静止目标）位于图表中心。

### 2D 点云显示（tabe page：Point Cloud）

根据计算得到的角度、距离。在极坐标图中进行显示。每次显示5个点（每一frame一个点），根据deque，先进先出。



## 贡献

如果你发现任何问题或有改进建议，欢迎提交 Issue 或 Pull Request。