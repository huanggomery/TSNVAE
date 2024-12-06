import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.distributions import TactileEncoder, TactileDecoder, Tac3dEncoder, Tac3dDecoder
from config import GlobalConfig
from pred_model.t_data import MyDataset


loss_fn = torch.nn.MSELoss().to(GlobalConfig.device)

def train(encoder, decoder, epochs=100):
    params = decoder.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-5)
    
    dataset = MyDataset(device=GlobalConfig.device)
    dataloader = DataLoader(dataset, 32, shuffle=True)

    for i in range(epochs):
        total_loss = 0

        for img, _ in dataloader:
            with torch.no_grad():
                z = encoder(img)["loc"]
            img1 = decoder(z)["loc"]
            loss = loss_fn(img, img1)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch: {} Train loss: {:.6f}".format(i+1, total_loss))

if __name__ == "__main__":
    encoder = TactileEncoder(GlobalConfig.z_dim).to(GlobalConfig.device)
    decoder = TactileDecoder(GlobalConfig.z_dim, 3).to(GlobalConfig.device)
    encoder.load_state_dict(torch.load(
        workspace_path+GlobalConfig.save_root+"/t_encoder.pth",
        map_location=torch.device(GlobalConfig.device)
    ))
    encoder.eval()
    # decoder.load_state_dict(torch.load(
    #     workspace_path+GlobalConfig.save_root+"/t_decoder.pth",
    #     map_location=torch.device(GlobalConfig.device)
    # ))

    train(encoder, decoder, 20000)
    torch.save(decoder.state_dict(), workspace_path+GlobalConfig.save_root+"/t_decoder.pth")
