'''
加载训练数据
数据组织格式：
data
|--train
|  |--1
|  |  |--action.npy
|  |  |--I_z.jpg
|  |  |--0.jpg
|  |  |--1.jpg
|  |  |--...
|  |--2
|  |  |--action.npy
|  |  |--I_z.jpg
|  |  |--0.jpg
|  |  |--1.jpg
|  |  |--...
|  |--...
|--test
|  |--1
|  |  |--action.npy
|  |  |--I_z.jpg
|  |  |--0.jpg
|  |  |--1.jpg
|  |  |--...
|  |--...

Copyright 2024, HuangChen. All rights reserved.
Author: HuangChen (huangchen_123@stu.hit.edu.cn)

'''

import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2

from config import GlobalConfig


# 图片处理尺寸
VISUAL_SIZE = GlobalConfig.visual_size
TACTILE_SIZE = GlobalConfig.tactile_size

# 数据根目录
DATA_ROOT = workspace_path + GlobalConfig.data_root

class NVAEDataset(Dataset):
    def __init__(self, mode: str = "train", device: str = "cpu"):
        super().__init__()

        self.device = device
        self.data = []  # 保存所有轨迹的数据

        if mode == "train":
            self.load_data(DATA_ROOT + "/train")
        elif mode == "test":
            self.load_data(DATA_ROOT + "/test")
        else:
            raise Exception("模式只能是train或者test")

    def load_data(self, data_path):
        with os.scandir(data_path) as entries:
            traj_dirs = [entry.name for entry in entries if entry.is_dir()]

        for traj in traj_dirs:
            path = data_path + "/" + traj
            traj_dict = dict()  # 空字典，保存该轨迹下的所有数据

            # 加载动作
            action = np.load(path + "/action.npy")
            action_tensor = torch.from_numpy(action).to(device=self.device, dtype=torch.float32)
            traj_dict["u"] = action_tensor

            # 加载插入状态的触觉图像
            img = cv2.imread(path + "/I_z.jpg")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_np = cv2.resize(img, (TACTILE_SIZE, TACTILE_SIZE)).astype(np.float32)
            img_torch = torch.from_numpy(img_np).permute(2,0,1).to(device=self.device)
            img_torch /= 255 # 归一化
            traj_dict["I_z"] = img_torch

            # 加载该轨迹的所有视觉图像
            traj_dict["I"] = []
            step = 0
            while 1:
                img_name = path + "/" + str(step) + ".jpg"
                if not os.path.exists(img_name):
                    break
                img = cv2.imread(img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_np = cv2.resize(img, (VISUAL_SIZE, VISUAL_SIZE)).astype(np.float32)
                img_torch = torch.from_numpy(img_np).permute(2,0,1).to(device=self.device)
                img_torch /= 255 # 归一化
                traj_dict["I"].append(img_torch)
                step += 1

            traj_dict["I"] = torch.stack(traj_dict["I"])
            self.data.append(traj_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]["I"], self.data[index]["I_z"], self.data[index]["u"]


def get_loader(mode: str = "train", device: str = "cpu"):
    data_set = NVAEDataset(mode=mode, device=device)
    return DataLoader(dataset=data_set, batch_size=8, shuffle=True)

if __name__ == "__main__":
    dataloader = get_loader("train")
    for I, I_z, u in dataloader:
        # 把step维度放前面
        input_var_dict = {"I": I.permute(1,0,2,3,4), "I_z": I_z, "u": u.permute(1,0,2)}
        # model.train(input_var_dict)