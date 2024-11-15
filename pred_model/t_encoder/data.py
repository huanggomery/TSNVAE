import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

from config import GlobalConfig

# 图片处理尺寸
VISUAL_SIZE = GlobalConfig.visual_size
TACTILE_SIZE = GlobalConfig.tactile_size

# 数据根目录
DATA_ROOT = workspace_path + GlobalConfig.data_root

class MyDataset(Dataset):
    def __init__(self, mode: str = "train", device: str = "cpu"):
        super().__init__()

        data_path = DATA_ROOT + "/train"

        self.positions = np.zeros((0, GlobalConfig.latent_dim))
        self.imgs = []

        with os.scandir(data_path) as entries:
            traj_dirs = [entry.name for entry in entries if entry.is_dir()]

        for traj in traj_dirs:
            path = data_path + "/" + traj

            position = np.load(path + "/pos.npy")
            position = position[:, :2] - np.array([410.18, -104.76])
            position = position[0].reshape(1,GlobalConfig.latent_dim)
            self.positions = np.concatenate((self.positions, position))

            # img_name = path + "/I_z.jpg"
            # img = cv2.imread(img_name)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img_np = cv2.resize(img, (TACTILE_SIZE, TACTILE_SIZE)).astype(np.float32)
            # img_torch = torch.from_numpy(img_np).permute(2,0,1).to(device=device)
            # img_torch /= 255 # 归一化
            # self.imgs.append(img_torch)

            tactile = np.load(path + "/tactile.npy").reshape(20,20,6).astype(np.float32)
            tactile_torch = torch.from_numpy(tactile).permute(2,0,1).to(device=device)
            self.imgs.append(tactile_torch)

        self.positions = torch.from_numpy(self.positions).to(device=device, dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index], self.positions[index]


if __name__ == "__main__":
    dataset = MyDataset("train", "cpu")
    for _, pos in dataset:
        print(pos*1000)