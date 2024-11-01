# 检验视觉图片重建效果

import os
import sys

current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

import torch
from PIL import Image
import cv2
import numpy as np

from models.model import TsNewtonianVAE
from models.load_data import NVAEDataset

import config
from config import GlobalConfig

save_root = current_file_path + "/v_recon"
i = 0

# 显示用cv2从jpg读取并处理后的图片
# img_torch: [C, W, H]
def show_img_torch(imgs_torch):
    for i in range(len(imgs_torch)):
        img_torch = imgs_torch[i]
        img_torch *= 255
        img = img_torch.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("pic{}".format(i), img)
    cv2.waitKey(0)

def save_pic(img_origin, img_recon):
    global i
    for step in range(img_origin.shape[0]):
        img0 = img_origin[step] * 255
        img0 = img0.permute(1,2,0).to(torch.uint8)
        img0 = img0.cpu().numpy()
        img = Image.fromarray(img0)
        img.save("{}/origin/{}.jpg".format(save_root, i))

        img1 = img_recon[step] * 255
        img1 = img1.permute(1,2,0).to(torch.uint8)
        img1 = img1.cpu().numpy()
        img = Image.fromarray(img1)
        img.save("{}/recon/{}.jpg".format(save_root, i))

        i += 1

def recon(model):
    dataset = NVAEDataset("train", GlobalConfig.device)

    for I, _, _ in dataset:
        # I : torch.Tensor (steps, C, H, W)
        x = model.get_latent(I)    # (steps, DIM)
        I_pred = model.v_decoder(x)["loc"]   # (steps, C, H, W)

        # save_pic(I, I_pred)
        for step in range(I.shape[0]):
            show_img_torch([I[step], I_pred[step]])


if __name__ == "__main__":
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

    if not os.path.exists(save_root + "/origin"):
        os.makedirs(save_root + "/origin")
    if not os.path.exists(save_root + "/recon"):
        os.makedirs(save_root + "/recon")

    recon(model)
