import matplotlib.pyplot as plt

def init_ADC4_plot(ax):
    """
    初始化每个图形的样式：背景颜色，坐标轴范围，标题等
    """
    ax.set_facecolor('white')  # 设置背景色为白色
   # ax.set_xlabel('Sample Points', fontsize=8)  # 设置 X 轴标签和字体大小
    #ax.set_ylabel('Amplitude', fontsize=8)  # 设置 Y 轴标签和字体大小
    ax.set_title('ADC Waveform', fontsize=10)  # 设置标题字体大小
    ax.grid(True)  # 显示网格
    ax.set_xlim(-100, 100)  # 设置 X 轴范围，适应数据范围
    ax.set_ylim(-100, 100)  # 设置 Y 轴范围，适应数据范围