import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, "../.."))
sys.path.append(workspace_path)

import torch
from torch.utils.data import DataLoader

from models.distributions import VisualEncoder
from config import GlobalConfig
from pred_model.v_data import MyDataset


loss_fn = torch.nn.MSELoss().to(GlobalConfig.device)

def eval(encoder):
    dataset = MyDataset(mode="test", device=GlobalConfig.device)
    dataloader = DataLoader(dataset, 1, shuffle=True)

    total_loss = 0

    with torch.no_grad():
        for img, pos in dataloader:
            pos1 = encoder(img)["loc"]
            loss = loss_fn(pos, pos1)
            total_loss += loss

    return total_loss

def train(encoder, epochs=100):
    params = encoder.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-3)
    
    dataset = MyDataset(device=GlobalConfig.device)
    dataloader = DataLoader(dataset, 32, shuffle=True)

    min_eval_loss = None

    for i in range(epochs):
        total_loss = 0

        for img, pos in dataloader:
            pos1 = encoder(img)["loc"]
            loss = loss_fn(pos, pos1)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print("Epoch: {} Train loss: {:.6f}".format(i+1, total_loss))

        if (i+1) % 10 == 0:
            eval_loss = eval(encoder)
            print("Epoch: {} Eval loss: {:.6f}".format(i+1, eval_loss))

            if min_eval_loss == None or eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                torch.save(encoder.state_dict(), workspace_path+GlobalConfig.save_root+"/v_encoder.pth")

if __name__ == "__main__":
    encoder = VisualEncoder(GlobalConfig.latent_dim).to(GlobalConfig.device)
    # encoder.load_state_dict(torch.load(
    #     workspace_path+GlobalConfig.save_root+"/v_encoder.pth",
    #     map_location=torch.device(GlobalConfig.device)
    # ))

    train(encoder, 100)
