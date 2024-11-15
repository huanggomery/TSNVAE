import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.distributions import TactileEncoder, Tac3dEncoder, TargetModel
from pred_model.t_encoder.data import MyDataset
from config import GlobalConfig


def eval(encoder, target):
    pos_ori = np.zeros((0, GlobalConfig.latent_dim))
    pos_pred = np.zeros((0, GlobalConfig.latent_dim))

    dataset = MyDataset(mode="test", device=GlobalConfig.device)
    with torch.no_grad():
        for img, pos in dataset:
            img = img.unsqueeze(0)
            pos = pos.unsqueeze(0).cpu().numpy()
            z = encoder(img)["loc"]
            pos1 = target(z)["loc"].cpu().numpy()
            pos_ori = np.concatenate((pos_ori, pos), axis=0)
            pos_pred = np.concatenate((pos_pred, pos1), axis=0)

    return pos_ori, pos_pred

def draw(pos_ori, pos_pred):
    plt.figure(0)
    plt.scatter(pos_ori[:, 0], pos_pred[:, 0], s=1)
    plt.figure(1)
    plt.scatter(pos_ori[:, 1], pos_pred[:, 1], s=1)
    plt.show()

if __name__ == "__main__":
    encoder = Tac3dEncoder(GlobalConfig.z_dim).to(GlobalConfig.device)
    target = TargetModel(GlobalConfig.z_dim, 2).to(GlobalConfig.device)
    encoder.eval()
    target.eval()
    encoder.load_state_dict(torch.load(
        workspace_path+GlobalConfig.save_root+"/t_encoder.pth",
        map_location=torch.device(GlobalConfig.device)
    ))
    target.load_state_dict(torch.load(
        workspace_path+GlobalConfig.save_root+"/target.pth",
        map_location=torch.device(GlobalConfig.device)
    ))

    pos_ori, pos_pred = eval(encoder, target)
    draw(pos_ori, pos_pred)
