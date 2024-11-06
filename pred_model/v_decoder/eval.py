import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

import torch
import cv2
import numpy as np

from models.distributions import VisualDecoder
from config import GlobalConfig
from pred_model.data import MyDataset


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

# 从训练数据中重建图像
def eval(decoder):
    dataset = MyDataset(device=GlobalConfig.device)
    with torch.no_grad():
        for img, pos in dataset:
            pos = pos.unsqueeze(0)
            img1 = decoder(pos)["loc"].squeeze()
            show_img_torch([img, img1])

# 直接根据位置重建图像，并显示移动情况
def step_recon(decoder):
    step = 0.001
    for i in range(10):
        pos = torch.tensor([[0, -step*i]], dtype=torch.float32, device=GlobalConfig.device)
        img = decoder(pos)["loc"].squeeze()
        show_img_torch([img])


if __name__ == "__main__":
    decoder = VisualDecoder(GlobalConfig.latent_dim, 3).to(GlobalConfig.device)
    decoder.load_state_dict(torch.load(
        workspace_path+GlobalConfig.save_root+"/v_decoder.pth",
        map_location=torch.device(GlobalConfig.device)
    ))

    step_recon(decoder)
    # eval(decoder)