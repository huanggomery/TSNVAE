# NVAE用到的模型，encoder, decoder, transition, velocity

import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

import torch
from torch import nn
import torchvision
import numpy as np

from pixyz import distributions as dist
from pixyz.utils import epsilon

from config import GlobalConfig


sigma = GlobalConfig.position_accuracy   # 机器人的重复定位精度


# 利用ResNet18将视觉图片编码为向量，I_t -> x_t
class VisualEncoder(dist.Normal):
    # output_dim：特征向量的维度
    def __init__(self, output_dim: int):
        super().__init__(var=["x_t"], cond_var=["I_t"], name="q")

        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.loc = nn.Sequential(
            nn.Linear(1000, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_dim)
        )

        self.scale = nn.Sequential(
            nn.Linear(1000, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_dim),
            nn.Softplus()
        )

    # 输入为 (B, C, W, H) 的图片，例如 (32, 3, 224, 224)
    def forward(self, I_t: torch.Tensor) -> dict:
        feature = self.encoder(I_t)

        loc = self.loc(feature)
        scale = self.scale(feature) + epsilon()

        return {"loc": loc, "scale": scale}


# 利用CNN将触觉图片编码为向量，I_z -> z
# TODO: 硬编码，只能处理 64*64的图像
class TactileEncoder(dist.Normal):
    # output_dim：特征向量的维度
    def __init__(self, output_dim: int):
        super().__init__(var=["z"], cond_var=["I_z"])

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
        )

        self.loc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

        self.scale = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
            nn.Softplus()
        )

    # 输入为 (B, C, W, H) 的图片，例如 (32, 3, 64, 64)
    def forward(self, I_z: torch.Tensor) -> dict:
        feature = self.encoder(I_z)
        B, C, W, H = feature.shape
        feature = feature.reshape((B, C*W*H))

        loc = self.loc(feature)
        scale = self.scale(feature) + epsilon()

        return {"loc": loc, "scale": scale}


# 根据触觉编码向量，推断目标x_g，z -> x_g
class TargetModel(dist.Normal):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(var=["x_t"], cond_var=["z"])

        self.loc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_dim)
        )

        self.scale = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, output_dim),
            nn.Softplus()
        )

    # 输入为 (B, INPUT_DIM) 的向量
    def forward(self, z: torch.Tensor) -> dict:
        loc = self.loc(z)
        scale = self.scale(z)

        return {"loc": loc, "scale": scale}


# 将视觉向量恢复成视觉图片， x_t -> I_t
# TODO: 硬编码，只能恢复成 224*224的图片
class VisualDecoder(dist.Normal):
    # input_dim：特征向量的维度，output_dim：图片的Channel数
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(var=["I_t"], cond_var=["x_t"])

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 512*7*7),
        )

        hiddens = [512, 256, 128, 64, 32]
        modules = []
        for i in range(len(hiddens)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i], hiddens[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hiddens[i+1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(32, output_dim, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)

    # 输入为(B, INPUT_DIM) 的向量
    def forward(self, x_t: torch.Tensor) -> dict:
        x_t = self.fc(x_t)
        x_t = x_t.view(-1, 512, 7, 7)  # 重塑为(批量大小, 512, 7, 7)
        loc = self.decoder(x_t)

        return {"loc": loc, "scale": 0.01}


# 将触觉向量恢复成触觉图片， z -> I_z
class TactileDecoder(dist.Normal):
    # input_dim：特征向量的维度，output_dim：图片的Channel数
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(var=["I_z"], cond_var=["z"])

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256*2*2),
            nn.LeakyReLU(),
        )

        hiddens = [256, 128, 64, 32, 16]
        modules = []
        for i in range(len(hiddens)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hiddens[i], hiddens[i+1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hiddens[i+1]),
                    nn.ReLU()
                )
            )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(16, output_dim, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*modules)

    # 输入为(B, INPUT_DIM) 的向量
    def forward(self, z: torch.Tensor) -> dict:
        z = self.fc(z)
        z = z.view(-1, 256, 2, 2)  # 重塑为(批量大小, 256, 2, 2)
        loc = self.decoder(z)

        return {"loc": loc, "scale": 0.01}


# 运动学状态转移方程
class Transition(dist.Normal):
    #   p(x_t | x_{t-1}, u_{t-1}; v_t) = N(x_t | x_{t-1} + ∆t·v_t, σ^2)
    def __init__(self, delta_time: float):
        super().__init__(var=["x_t"], cond_var=["x_t0", "v_t"])

        self.delta_time = delta_time

    def forward(self, x_t0: torch.Tensor, v_t: torch.Tensor):
        # 论文中将 v_{t+1} 简化为了 u_t，所以此v_t1可能就是u_t
        x_t = x_t0 + self.delta_time * v_t

        return {"loc": x_t, "scale": sigma}


# 计算下一个时刻的速度
class Velocity(dist.Deterministic):
    def __init__(self):
        super().__init__(var=["v_t"], cond_var=["x_t0", "v_t0", "u_t0"], name="f")

    def forward(self, x_t0: torch.Tensor, v_t0: torch.Tensor, u_t0: torch.Tensor) -> dict:
        # 论文中将 v_{t+1} 简化为了 u_t，参见公式(6)
        return {"v_t": u_t0}

if __name__ == "__main__":
    encoder = VisualEncoder(6)
    decoder = VisualDecoder(6, 3)

    I_t = torch.zeros((32, 3, 224, 224))
    x_t = encoder(I_t)
    I_t_1 = decoder(x_t['loc'])

    print(I_t.shape)
    print(x_t['loc'].shape)
    print(I_t_1['loc'].shape)


    # encoder = TactileEncoder(128)
    # decoder = TactileDecoder(128, 3)
    # I_z = torch.zeros((32, 3, 64, 64))
    # z = encoder(I_z)["loc"]
    # print(z.shape)
    # I_z1 = decoder(z)["loc"]
    # print(I_z1.shape)