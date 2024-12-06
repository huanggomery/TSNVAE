import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from models.distributions import Tac3dEncoder, TargetModel, TactileEncoder
from torchvision import transforms
from PIL import Image

from config import GlobalConfig

workspace_path = "./"

# class TactileEncoder(nn.Module):
#     def __init__(self):
#         super(TactileEncoder, self).__init__()
#         # 定义第一个卷积层，输入通道为3（RGB图像），输出通道为32
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         # 定义第二个卷积层，输入通道为32，输出通道为64
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         # 定义第三个卷积层，输入通道为64，输出通道为128
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
#         # 定义全连接层，将特征图展平后连接到128维向量
#         self.fc = nn.Linear(128 * 15 * 15, 128)  # 假设特征图大小为8x8

#     def forward(self, x):
#         # 应用第一个卷积层和ReLU激活函数
#         x = F.relu(self.conv1(x))
#         # 应用最大池化
#         x = F.max_pool2d(x, 2, 2)
        
#         # 应用第二个卷积层和ReLU激活函数
#         x = F.relu(self.conv2(x))
#         # 应用最大池化
#         x = F.max_pool2d(x, 2, 2)
        
#         # 应用第三个卷积层和ReLU激活函数
#         x = F.relu(self.conv3(x))
#         # 应用最大池化
#         x = F.max_pool2d(x, 2, 2)
        
#         # 展平特征图
#         x = x.view(-1, 128 * 15 * 15)  # 假设特征图大小为8x8
#         # 应用全连接层
#         x = self.fc(x)
        
#         return x


class MyDataset(Dataset):
    def __init__(self, mode: str = "train", device: str = "cpu"):
        super().__init__()

        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((120, 120)),
        ])

        self.tactile = []
        for i in range(800):
            tactile = Image.open("data/gelsight/img_left_{}.jpg".format(i+1))
            tactile = trans(tactile).to(device=device)
            self.tactile.append(tactile)

        position = np.load("data/gelsight/pos.npy")
        position = position[:, [0,1,3,4,5]]
        self.position_torch = torch.from_numpy(position).to(device=device, dtype=torch.float32)

        if mode == "train":
            self.tactile = self.tactile[:700]
            self.position_torch = self.position_torch[:700]
        else:
            self.tactile = self.tactile[700:]
            self.position_torch = self.position_torch[700:]
            
    def __len__(self):
        return len(self.tactile)

    def __getitem__(self, index):
        return self.tactile[index], self.position_torch[index]


loss_fn = torch.nn.MSELoss().to(GlobalConfig.device)

def train(encoder, target, epochs=100):
    models = nn.ModuleList([encoder, target])
    params = models.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    dataset = MyDataset(device=GlobalConfig.device)
    dataloader = DataLoader(dataset, 32, shuffle=True)

    for i in range(epochs):
        total_loss = 0

        for img, pos in dataloader:
            z = encoder(img)["loc"]
            pos1 = target(z)["loc"]
            loss = loss_fn(pos, pos1)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: {} Train loss: {:.6f}".format(i+1, total_loss))

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
    plt.subplot(1,5,1)
    plt.scatter(pos_ori[:, 0], pos_pred[:, 0], s=1)
    plt.subplot(1,5,2)
    plt.scatter(pos_ori[:, 1], pos_pred[:, 1], s=1)
    plt.subplot(1,5,3)
    plt.scatter(pos_ori[:, 2], pos_pred[:, 2], s=1)
    plt.subplot(1,5,4)
    plt.scatter(pos_ori[:, 3], pos_pred[:, 3], s=1)
    plt.subplot(1,5,5)
    plt.scatter(pos_ori[:, 4], pos_pred[:, 4], s=1)
    plt.show()


if __name__ == "__main__":
    encoder = TactileEncoder(output_dim=128).to(GlobalConfig.device)
    target = TargetModel(GlobalConfig.z_dim, GlobalConfig.latent_dim).to(GlobalConfig.device)
    # encoder.load_state_dict(torch.load(
    #     workspace_path+GlobalConfig.save_root+"/t_encoder.pth",
    #     map_location=torch.device(GlobalConfig.device)
    # ))
    # target.load_state_dict(torch.load(
    #     workspace_path+GlobalConfig.save_root+"/target.pth",
    #     map_location=torch.device(GlobalConfig.device)
    # ))

    train(encoder, target, 100)
    torch.save(encoder.state_dict(), workspace_path+GlobalConfig.save_root+"/t_encoder.pth")
    torch.save(target.state_dict(), workspace_path+GlobalConfig.save_root+"/target.pth")

    encoder.eval()
    target.eval()
    pos_ori, pos_pred = eval(encoder, target)
    draw(pos_ori, pos_pred)
    