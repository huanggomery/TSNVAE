# 可视化隐藏层，参考Fig.3

import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.model import TsNewtonianVAE
from models.load_data import NVAEDataset

import config
from config import GlobalConfig


def eval(model, mode = "train"):
    latent_x_all = np.zeros((0, GlobalConfig.latent_dim))
    latent_target_all = np.zeros((0, GlobalConfig.latent_dim))

    dataset = NVAEDataset(mode, GlobalConfig.device)
    for I, I_z, _ in dataset:
        # I : torch.Tensor (steps, C, H, W)
        # I_z : torch.Tensor (C, H, W)
        x = model.get_latent(I).cpu().numpy()  # (steps, DIM)
        x_g = model.get_target_latent(I_z).unsqueeze(0).cpu().numpy()  # (1, DIM)
        
        latent_x_all = np.concatenate((latent_x_all, x), axis=0)
        latent_target_all = np.concatenate((latent_target_all, x_g), axis=0)

    # 加载机械臂保存的位姿参数
    positions = np.zeros((0, GlobalConfig.latent_dim))
    for i in range(len(dataset)):
        filename = workspace_path + GlobalConfig.data_root + "/" + mode + "/{}".format(i+1) + "/pos.npy"
        position = np.load(filename)
        position = position[:, [0,1,3,4,5]] - np.array([407.83, -106.0, -180.0, 0, 0])
        positions = np.concatenate((positions, position), axis=0)

    return latent_x_all, latent_target_all, positions

def draw(x, x_target, pos):
    plt.figure(0)
    plt.title("latent scatter")
    plt.scatter(x[:, 0], x[:, 1], s=1, c=[0,0,1])
    plt.scatter(x_target[:, 0], x_target[:, 1], s=5, c=[1,0,0])
    plt.axis("equal")

    plt.figure(1)
    plt.subplot(1,5,1)
    plt.title("latent x - position x")
    plt.scatter(x[:, 0], pos[:, 0], s=1)

    plt.subplot(1,5,2)
    plt.title("latent y - position y")
    plt.scatter(x[:, 1], pos[:, 1], s=1)

    plt.subplot(1,5,3)
    plt.title("latent rx - position rx")
    plt.scatter(x[:, 2], pos[:, 2], s=1)

    plt.subplot(1,5,4)
    plt.title("latent ry - position ry")
    plt.scatter(x[:, 3], pos[:, 3], s=1)

    plt.subplot(1,5,5)
    plt.title("latent rz - position rz")
    plt.scatter(x[:, 4], pos[:, 4], s=1)

    plt.show()


if __name__ == "__main__":
    model = TsNewtonianVAE(
        config.v_encoder_param,
        config.v_decoder_param,
        config.t_encoder_param,
        config.t_decoder_param,
        config.velocity_param,
        config.target_param,
        GlobalConfig.delta_time,
        GlobalConfig.device
    )
    model.load("."+GlobalConfig.save_root, "model.pth")
    # model.load_part("v_encoder", "."+GlobalConfig.save_root+"/v_encoder.pth")
    # model.load_part("t_encoder", "."+GlobalConfig.save_root+"/t_encoder.pth")
    # model.load_part("target_model", "."+GlobalConfig.save_root+"/target.pth")
    model.eval()

    x, x_target, pos = eval(model, "test")
    draw(x, x_target, pos)
