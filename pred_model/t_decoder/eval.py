import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

import torch
import cv2
import numpy as np

from models.distributions import TactileEncoder, TactileDecoder, Tac3dEncoder, Tac3dDecoder
from config import GlobalConfig
from pred_model.t_decoder.data import MyDataset


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
def eval(encoder, decoder, mode="train"):
    dataset = MyDataset(device=GlobalConfig.device, mode=mode)
    with torch.no_grad():
        for img, _ in dataset:
            img = img.unsqueeze(0)
            z = encoder(img)["loc"]
            img1 = decoder(z)["loc"]
            show_img_torch([img.squeeze()[:3], img1.squeeze()[:3]])


if __name__ == "__main__":
    encoder = Tac3dEncoder(GlobalConfig.z_dim).to(GlobalConfig.device)
    decoder = Tac3dDecoder(GlobalConfig.z_dim, 6).to(GlobalConfig.device)
    encoder.load_state_dict(torch.load(
        workspace_path+GlobalConfig.save_root+"/t_encoder.pth",
        map_location=torch.device(GlobalConfig.device)
    ))
    decoder.load_state_dict(torch.load(
        workspace_path+GlobalConfig.save_root+"/t_decoder.pth",
        map_location=torch.device(GlobalConfig.device)
    ))
    encoder.eval()
    decoder.eval()
    eval(encoder, decoder, "train")
