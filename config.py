# 全局配置

# 禁止直接使用该类
class __GlobalConfig:
    def __init__(self):
        self.device = "cuda"              # 训练设备
        self.lr = 1e-4                    # 学习率
        self.latent_dim = 5               # 隐藏层的长度
        self.z_dim = 128                  # 触觉向量的长度
        self.delta_time = 0.5             # 控制和采样的间隔时间，s
        self.visual_size = 224            # 视觉图像尺寸
        self.tactile_size = 120            # 触觉图像尺寸
        self.position_accuracy = 0.1      # 机器人的定位精度 mm
        self.grasp_error = 2              # 抓取误差 mm
        self.data_root = "/data"          # 数据根目录，组织结构见 load_data.py
        self.save_root = "/save"          # 训练参数保存路径


GlobalConfig = __GlobalConfig()

v_encoder_param = {
    "output_dim": GlobalConfig.latent_dim,
}

v_decoder_param = {
    "input_dim": GlobalConfig.latent_dim,
    "output_dim": 3,
}

t_encoder_param = {
    "output_dim": GlobalConfig.z_dim,
}

t_decoder_param = {
    "input_dim": GlobalConfig.z_dim,
    "output_dim": 3,
}

velocity_param = {
    "latent_dim": GlobalConfig.latent_dim,
    "delta_time": GlobalConfig.delta_time,
    "device": GlobalConfig.device,
    "use_data_efficiency": True,
}

target_param = {
    "input_dim": GlobalConfig.z_dim,
    "output_dim": GlobalConfig.latent_dim,
}
