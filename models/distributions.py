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
# TODO: 硬编码，只能处理 120*120的图像
class TactileEncoder(dist.Normal):
    # output_dim：特征向量的维度
    def __init__(self, output_dim: int):
        super().__init__(var=["z"], cond_var=["I_z"])

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.loc = nn.Linear(128*15*15, output_dim)
        self.scale = nn.Sequential(
            nn.Linear(128*15*15, output_dim),
            nn.Softplus(),
        )

    # 输入为 (B, C, W, H) 的图片，例如 (32, 3, 120, 120)
    def forward(self, I_z: torch.Tensor) -> dict:
        feature = self.encoder(I_z)
        B, C, W, H = feature.shape
        feature = feature.reshape((B, C*W*H))

        loc = self.loc(feature)
        scale = self.scale(feature) + epsilon()

        return {"loc": loc, "scale": scale}

# 利用CNN将触觉图片编码为向量，I_z -> z
# TODO: 硬编码，只能处理 6*20*20 的图像
class Tac3dEncoder(dist.Normal):
    # output_dim：特征向量的维度
    def __init__(self, output_dim: int):
        super().__init__(var=["z"], cond_var=["I_z"])

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.loc = nn.Sequential(
            nn.Linear(6400, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
        )

        self.scale = nn.Sequential(
            nn.Linear(6400, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim),
            nn.Softplus()
        )

    def forward(self, I_z: torch.Tensor) -> dict:
        feature = self.encoder(I_z)
        feature = torch.flatten(feature, 1)

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

        return {"loc": loc, "scale": 0.1}


# 将触觉向量恢复成触觉图片， z -> I_z
class TactileDecoder(dist.Normal):
    # input_dim：特征向量的维度，output_dim：图片的Channel数
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(var=["I_z"], cond_var=["z"])

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim*15*15),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_dim, kernel_size=2, stride=2),
        )

    # 输入为(B, INPUT_DIM) 的向量
    def forward(self, z: torch.Tensor) -> dict:
        z = self.fc(z)
        z = z.view(-1, 128, 15, 15)  # 重塑为(批量大小, 128, 2, 2)
        loc = self.decoder(z)

        return {"loc": loc, "scale": 0.01}

class Tac3dDecoder(dist.Normal):
    # input_dim：特征向量的维度，output_dim：图片的Channel数
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(var=["I_z"], cond_var=["z"])

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64*10*10),
            nn.LeakyReLU(),
        )

        hiddens = [64, 32, 16]
        modules = []
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        )
        modules.append(
            nn.ConvTranspose2d(16, output_dim, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(*modules)

    # 输入为(B, INPUT_DIM) 的向量
    def forward(self, z: torch.Tensor) -> dict:
        z = self.fc(z)
        z = z.view(-1, 64, 10, 10)  # 重塑为(批量大小, 256, 2, 2)
        loc = self.decoder(z)

        return {"loc": loc, "scale": 0.1}

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
    def __init__(self, latent_dim: int, delta_time: float, device: str, use_data_efficiency: bool):
        super().__init__(var=["v_t"], cond_var=["x_t0", "v_t0", "u_t0"], name="f")

        self.delta_time = delta_time
        self.use_data_efficiency = use_data_efficiency

        if not self.use_data_efficiency:

            self.coefficient_ABC = nn.Sequential(
                nn.Linear(latent_dim*3, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim*3),
            )

        else:
            self.A = torch.zeros((1, latent_dim, latent_dim)).to(device)
            self.B = torch.zeros((1, latent_dim, latent_dim)).to(device)
            self.C = torch.diag_embed(torch.ones(1, latent_dim)).to(device)

    def forward(self, x_t0: torch.Tensor, v_t0: torch.Tensor, u_t0: torch.Tensor) -> dict:
        # 论文中将 v_{t+1} 简化为了 u_t，参见公式(6)

        combined_vector = torch.cat([x_t0, v_t0, u_t0], dim=1)

        # For data efficiency
        if self.use_data_efficiency:
            A = self.A
            B = self.B
            C = self.C
        else:
            _A, _B, _C = torch.chunk(self.coefficient_ABC(combined_vector), 3, dim=-1)
            A = torch.diag_embed(_A)
            B = torch.diag_embed(-torch.exp(_B))
            C = torch.diag_embed(torch.exp(_C))

        # Dynamics inspired by Newton's motion equation
        v_t = v_t0 + self.delta_time * (torch.einsum("ijk,ik->ik", A, x_t0) + torch.einsum(
            "ijk,ik->ik", B, v_t0) + torch.einsum("ijk,ik->ik", C, u_t0))

        # return {"v_t": v_t}
        return {"v_t": u_t0}


if __name__ == "__main__":
    pass