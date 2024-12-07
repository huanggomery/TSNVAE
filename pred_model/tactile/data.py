import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import GlobalConfig

DATA_PATH = "data/gelsight"
DATA_SIZE = 800

class MyDataset(Dataset):
    def __init__(self, mode: str = "train", device: str = "cpu"):
        super().__init__()

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120, 120)),
        ])

        self.tactile = []
        for i in range(DATA_SIZE):
            tactile = Image.open(DATA_PATH + "/img_left_{}.jpg".format(i+1))
            tactile = trans(tactile).to(device=device)
            self.tactile.append(tactile)

        position = np.load(DATA_PATH + "/pos.npy")
        position = position[:, [0,1,3,4,5]]
        self.position_torch = torch.from_numpy(position).to(device=device, dtype=torch.float32)

        train_size = int(DATA_SIZE * 0.9)
        if mode == "train":
            self.tactile = self.tactile[:train_size]
            self.position_torch = self.position_torch[:train_size]
        else:
            self.tactile = self.tactile[train_size:]
            self.position_torch = self.position_torch[train_size:]
            
    def __len__(self):
        return len(self.tactile)

    def __getitem__(self, index):
        return self.tactile[index], self.position_torch[index]
