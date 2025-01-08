import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# 图片处理尺寸
VISUAL_SIZE = 240
TACTILE_SIZE = 120

# 数据根目录
DATA_ROOT = "data/cropped"

DATA_SIZE = 220
TRAIN_RATIO = 0.9
SEQUENCE = 8


class VTTDataset(Dataset):
    def __init__(self, mode: str, device: str = "cpu"):
        super().__init__()

        self.device = device
        self.imgs = []
        self.tactile = []
        self.label = []

        self.img_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((VISUAL_SIZE, VISUAL_SIZE)),
        ])

        self.tactile_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((TACTILE_SIZE, TACTILE_SIZE)),
        ])

        if mode == "train":
            self.load_data(1, int(DATA_SIZE * TRAIN_RATIO))
        elif mode == "test":
            self.load_data(int(DATA_SIZE * TRAIN_RATIO), DATA_SIZE+1)
        else:
            raise ValueError("mode is incorrect")

    def load_data(self, start, end):
        for i in range(start, end):
            imgs, tactile = [], []
            dir = DATA_ROOT + "/{}".format(i)
            for s in range(SEQUENCE):
                I = Image.open(dir + "/{}.jpg".format(s))
                T = Image.open(dir + "/tac_left_{}.jpg".format(s))
                I = self.img_trans(I)
                T = self.tactile_trans(T)
                imgs.append(I)
                tactile.append(T)

            imgs = torch.stack(imgs, dim=0)
            tactile = torch.stack(tactile, dim=0)
            imgs = imgs.to(dtype=torch.float32, device=self.device)
            tactile = tactile.to(dtype=torch.float32, device=self.device)

            label = np.load(dir + "/label.npy")
            label = torch.from_numpy(label).to(dtype=torch.float32, device=self.device)

            self.imgs.append(imgs)
            self.tactile.append(tactile)
            self.label.append(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.imgs[index], self.tactile[index], self.label[index]


def get_loader(mode: str, device: str = "cpu"):
    dataset = VTTDataset(mode, device)
    return DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    dataloader = get_loader("test", "cuda")
    for I, T, label in dataloader:
        print(I.shape)
        print(T.shape)
        print(label.shape)
        break