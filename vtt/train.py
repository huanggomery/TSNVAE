import numpy as np
import torch

from model import VTT
from load_data import get_loader
from eval import eval

device = "cuda"
loss_fn = torch.nn.MSELoss().to(device)


def train(vtt: VTT, epoch: int, visual: bool = True, tactile: bool = True):
    if visual ==False and tactile == False:
        raise Exception("visual and tactile can't be False at the same time")

    params = vtt.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-4)

    dataloader = get_loader("train", device)

    min_eval_loss = None
    train_loss, eval_loss = [], []

    for i in range(epoch):
        total_loss = 0
        vtt.train()

        for I, T, label in dataloader:
            if not visual:
                I = torch.zeros_like(I).to(dtype=torch.float32, device=device)
            if not tactile:
                T = torch.zeros_like(T).to(dtype=torch.float32, device=device)
            y, _ = vtt(I, T)
            loss = loss_fn(label, y)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        total_loss /= len(dataloader.dataset)
        train_loss.append(total_loss)
        np.save("save/train_loss.npy", np.array(train_loss))
        print("Epoch: {} Train loss: {:.6f}".format(i+1, total_loss))

        if (i+1) % 10 == 0:
            vtt.eval()
            loss, _, _ = eval(vtt)
            eval_loss.append(loss)
            np.save("save/eval_loss.npy", np.array(eval_loss))
            print("Epoch: {} Eval loss: {:.6f}".format(i+1, loss))

            if min_eval_loss == None or loss < min_eval_loss:
                min_eval_loss = loss
                if visual and tactile:
                    torch.save(vtt.state_dict(), "save/vtt.pth")
                elif visual == False:
                    torch.save(vtt.state_dict(), "save/vtt_without_visual.pth")
                else:
                    torch.save(vtt.state_dict(), "save/vtt_without_tactile.pth")


if __name__ == "__main__":
    vtt = VTT(img_size=[240], img_patch_size=80, tactile_size=[120], tactile_patch_size=60,
              sequence=8, in_chans=3, embed_dim=512).to(device=device)
    # vtt.load_state_dict(torch.load(
    #     "save/vtt.pth",
    #     map_location=torch.device(device)
    # ))

    train(vtt, 100, visual=True, tactile=True)
