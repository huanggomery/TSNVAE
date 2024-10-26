# NVAE用到的模型，encoder, decoder, transition, velocity

import torch
from torch import nn
import torchvision
import numpy as np

from pixyz import distributions as dist
from pixyz.utils import epsilon


sigma = 0.0001   # 机器人的重复定位精度为 0.1mm


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
    
    def forward(self, z: torch.Tensor) -> dict:
        loc = self.loc(z)
        scale = self.scale(z)

        return {"loc": loc, "scale": scale}


# 将视觉向量恢复成视觉图片， x_t -> I_t
class VisualDecoder(dist.Normal):
    # input_dim：特征向量的维度，output_dim：图片的Channel数
    def __init__(self, input_dim: int, output_dim: int, img_size: int, device: str):
        super().__init__(var=["I_t"], cond_var=["x_t"])

        self.loc = nn.Sequential(
            nn.Conv2d(input_dim+2, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_dim, 3, stride=1, padding=1),
            nn.Tanh()
        )

        self.image_size = img_size
        a = np.linspace(-1, 1, self.image_size)
        b = np.linspace(-1, 1, self.image_size)
        x, y = np.meshgrid(a, b)
        x = x.reshape(self.image_size, self.image_size, 1)
        y = y.reshape(self.image_size, self.image_size, 1)
        self.xy = np.concatenate((x, y), axis=-1)

        self.input_dim = input_dim
        self.device = device

    # 输入为(B, INPUT_DIM) 的向量
    def forward(self, x_t: torch.Tensor) -> dict:
        batchsize = len(x_t)
        xy_tiled = torch.from_numpy(
            np.tile(self.xy, (batchsize, 1, 1, 1)).astype(np.float32)).to(self.device)

        z_tiled = torch.repeat_interleave(
            x_t, self.image_size*self.image_size, dim=0)
        z_tiled = z_tiled.view(batchsize, self.image_size, self.image_size, self.input_dim)

        z_and_xy = torch.cat((z_tiled, xy_tiled), dim=3)
        z_and_xy = z_and_xy.permute(0, 3, 2, 1)

        loc = self.loc(z_and_xy)/2.

        return {"loc": loc, "scale": 0.01}


# 将触觉向量恢复成触觉图片， z -> I_z
class TactileDecoder(dist.Normal):
    # input_dim：特征向量的维度，output_dim：图片的Channel数
    def __init__(self, input_dim: int, output_dim: int, img_size: int, device: str):
        super().__init__(var=["I_z"], cond_var=["z"])

        self.loc = nn.Sequential(
            nn.Conv2d(input_dim+2, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_dim, 3, stride=1, padding=1),
            nn.Tanh()
        )

        self.image_size = img_size
        a = np.linspace(-1, 1, self.image_size)
        b = np.linspace(-1, 1, self.image_size)
        x, y = np.meshgrid(a, b)
        x = x.reshape(self.image_size, self.image_size, 1)
        y = y.reshape(self.image_size, self.image_size, 1)
        self.xy = np.concatenate((x, y), axis=-1)

        self.input_dim = input_dim
        self.device = device

    # 输入为(B, INPUT_DIM) 的向量
    def forward(self, z: torch.Tensor) -> dict:
        batchsize = len(z)
        xy_tiled = torch.from_numpy(
            np.tile(self.xy, (batchsize, 1, 1, 1)).astype(np.float32)).to(self.device)

        z_tiled = torch.repeat_interleave(
            z, self.image_size*self.image_size, dim=0)
        z_tiled = z_tiled.view(batchsize, self.image_size, self.image_size, self.input_dim)

        z_and_xy = torch.cat((z_tiled, xy_tiled), dim=3)
        z_and_xy = z_and_xy.permute(0, 3, 2, 1)

        loc = self.loc(z_and_xy)/2.

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
    # encoder = VisualEncoder(6)
    # decoder = VisualDecoder(6, 3, 224, "cpu")

    # I_t = torch.zeros((32, 3, 224, 224))
    # x_t = encoder(I_t)
    # I_t_1 = decoder(x_t['loc'])

    # print(I_t.shape)
    # print(x_t['loc'].shape)
    # print(I_t_1['loc'].shape)

    # encoder = TactileEncoder(6)
    # I_z = torch.zeros((32, 3, 224, 224))
    # z = encoder(I_z)
    # print(z["loc"].shape)

    trans = Transition(0.01)
    d = trans.sample({"x_t0": torch.zeros((1, 2)), "v_t": torch.zeros((1, 1))})
    print(d)