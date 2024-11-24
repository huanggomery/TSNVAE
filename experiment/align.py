# 测试对准但不插入

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time

from models.model import TsNewtonianVAE
import arm

import config
from config import GlobalConfig

device = torch.device("cpu")
delta_time = 0.1
p = 0.5   # 比例控制
velocity_limit = np.array([30, 30, 10, 10, 10])


def align_loop(arm: arm.Arm, model: TsNewtonianVAE):
    while True:
        print("输入0 退出， 其他继续")
        choice = input()
        if choice == '0':
            break

        while True:
            img_bgr = arm.get_image()
            img_rgb = img_bgr[:, :, ::-1].copy()
            img_torch = torch.from_numpy(img_rgb).to(dtype=torch.float32, device=device).permute(2,0,1) / 255   # (3, 480, 640) 范围 [0,1]

            tactile = arm.get_tactile().reshape(20,20,6).astype(np.float32)
            tactile_torch = torch.from_numpy(tactile).permute(2,0,1).to(device=device)  # (3, 20, 20)

            pose = model.get_latent(img_torch).cpu().numpy()
            target_pose = model.get_target_latent(tactile_torch).cpu().numpy()

            action = (target_pose - pose) * p
            action = np.clip(action, -velocity_limit, velocity_limit)
            print(action)

            if action.max() < 0.3:
                break

            action = np.insert(action, 2, 0)
            arm.arm.vc_set_cartesian_velocity(action, is_radian=False, duration=delta_time)
            time.sleep(delta_time)
        
        print("已到达目标！")
        print()

if __name__ == "__main__":
    my_arm = arm.Arm(arm.ip, delta_time)
    model = TsNewtonianVAE(
        config.v_encoder_param,
        config.v_decoder_param,
        config.t_encoder_param,
        config.t_decoder_param,
        config.velocity_param,
        config.target_param,
        GlobalConfig.delta_time,
        GlobalConfig.device
    )
    model.load("."+GlobalConfig.save_root, "model.pth")
    model.eval()

    align_loop(my_arm, model)
