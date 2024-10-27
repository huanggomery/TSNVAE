# 全局配置

# 禁止直接使用该类
class __GlobalConfig:
    def __init__(self):
        self.device = "cuda"              # 训练设备
        self.visual_size = 224            # 视觉图像尺寸
        self.tactile_size = 64            # 触觉图像尺寸
        self.position_accuracy = 0.0001   # 机器人的定位精度 m
        self.grasp_error = 0.0015         # 抓取误差 m
        self.data_root = "/data"          # 数据根目录，组织结构见 load_data.py


GlobalConfig = __GlobalConfig()