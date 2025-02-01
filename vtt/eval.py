import numpy as np
import torch
import matplotlib.pyplot as plt

from model import VTT
from load_data import get_loader

device = "cuda"
loss_fn = torch.nn.MSELoss().to(device)


def eval(vtt: VTT, visual: bool = True, tactile: bool = True):
    dataloader = get_loader("test", device)

    ground_truth = np.zeros((0, 2))
    predict = np.zeros((0, 2))

    total_loss = 0
    with torch.no_grad():
        for I, T, label in dataloader:
            if not visual:
                I = torch.zeros_like(I).to(dtype=torch.float32, device=device)
            if not tactile:
                T = torch.zeros_like(T).to(dtype=torch.float32, device=device)
            y, _, attn_list = vtt(I, T, return_attention=True)
            loss = loss_fn(label, y)
            total_loss += loss.item()

            ground_truth = np.concat((ground_truth, label.cpu().numpy()), axis=0)
            predict = np.concat((predict, y.cpu().numpy()), axis=0)

    return total_loss / len(dataloader.dataset), ground_truth, predict


def draw(ground_truth, predict):
    plt.figure(0)
    plt.scatter(ground_truth[:, 0], predict[:, 0], s=10)
    plt.plot([np.min(ground_truth[:, 0]), np.max(ground_truth[:, 0])],
             [np.min(ground_truth[:, 0]), np.max(ground_truth[:, 0])],
             color='red')
    plt.figure(1)
    plt.scatter(ground_truth[:, 1], predict[:, 1], s=10)
    plt.plot([np.min(ground_truth[:, 1]), np.max(ground_truth[:, 1])],
             [np.min(ground_truth[:, 1]), np.max(ground_truth[:, 1])],
             color='red')
    plt.show()


if __name__ == "__main__":
    vtt = VTT(img_size=[240], img_patch_size=80, tactile_size=[120], tactile_patch_size=60,
              sequence=8, in_chans=3, embed_dim=512).to(device=device)
    vtt.load_state_dict(torch.load(
        "save/vtt.pth",
        map_location=torch.device(device)
    ))

    _, ground_truth, predict = eval(vtt)
    err = ground_truth - predict
    print('最大误差：', np.max(np.abs(err), axis=0))
    draw(ground_truth, predict)
