import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from config import GlobalConfig

# 图片处理尺寸
VISUAL_SIZE = GlobalConfig.visual_size
TACTILE_SIZE = GlobalConfig.tactile_size

# 数据根目录
DATA_ROOT = workspace_path + GlobalConfig.data_root

class MyDataset(Dataset):
    def __init__(self, mode: str = "train", device: str = "cpu"):
        super().__init__()

        self.device = device
        data_path = DATA_ROOT + "/" + mode

        self.positions = np.zeros((0, GlobalConfig.latent_dim))
        self.imgs = []

        with os.scandir(data_path) as entries:
            traj_dirs = [entry.name for entry in entries if entry.is_dir()]

        for traj in traj_dirs:
            path = data_path + "/" + traj

            position = np.load(path + "/pos.npy")
            position = position[:, [0,1,3,4,5]] - np.array([407.83, -106.0, -180.0, 0, 0])
            self.positions = np.concatenate((self.positions, position))

            step = 0
            while 1:
                img_name = path + "/" + str(step) + ".jpg"
                if not os.path.exists(img_name):
                    break

                img = Image.open(img_name)
                self.imgs.append(img)

                step += 1

        self.positions = torch.from_numpy(self.positions).to(device=device, dtype=torch.float32)

        self.transform = transforms.Compose([
            # 将PIL图像转换为Tensor
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomCrop(VISUAL_SIZE),
            # 随机调整亮度、对比度、饱和度和色调
            transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.1),
            # 高斯模糊
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
            # 随机擦除
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.15), ratio=(0.3, 3.3), value=0, inplace=False),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_torch = self.transform(self.imgs[index]).to(device=self.device)
        return img_torch, self.positions[index]
