# 可视化隐藏层，参考Fig.3

import torch
import numpy as np

from models.model import TsNewtonianVAE
from models.load_data import NVAEDataset

import config
from config import GlobalConfig


latent_x_all = np.zeros((0, GlobalConfig.latent_dim))
latent_target_all = np.zeros((0, GlobalConfig.latent_dim))

def eval(model, mode = "train"):
    dataset = NVAEDataset(mode, GlobalConfig.device)
    for I, I_z, _ in dataset:
        # I : torch.Tensor (steps, C, H, W)
        # I_z : torch.Tensor (C, H, W)
        x = model.get_latent(I).cpu().numpy()  # (steps, DIM)
        x_g = model.get_target_latent(I_z).unsqueeze(0).cpu().numpy()  # (1, DIM)
        
        latent_x_all = np.concatenate((latent_x_all, x), axis=0)
        latent_target_all = np.concatenate((latent_target_all, x_g), axis=0)

    # TODO: 加载机械臂保存的位姿参数

def draw():
    pass
    # TODO: 尚未开发


if __name__ == "__main__":
    model = TsNewtonianVAE(
        config.v_encoder_param,
        config.v_decoder_param,
        config.t_encoder_param,
        config.t_decoder_param,
        config.target_param,
        GlobalConfig.delta_time,
        GlobalConfig.device
    )

    eval(model, "train")
    draw()
