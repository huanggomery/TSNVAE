# 训练TSNVAE

from models.model import TsNewtonianVAE
from models.load_data import get_loader

import config
from config import GlobalConfig

device = GlobalConfig.device


def train(model, epochs = 500):
    dataloader = get_loader(mode="train", device=device)

    for i in range(epochs):
        total_loss = 0
        for I, I_z, u in dataloader:
            input_var_dict = {"I": I.permute(1,0,2,3,4), "I_z": I_z, "u": u.permute(1,0,2)}
            loss = model.train(input_var_dict)
        loss /= len(dataloader.dataset)

        print("Epoch: {} Train loss: {:.4f}".format(i+1, loss))


if __name__ == "__main__":
    model = TsNewtonianVAE(
        config.v_encoder_param,
        config.v_decoder_param,
        config.t_encoder_param,
        config.t_decoder_param,
        config.target_param,
        GlobalConfig.delta_time,
        GlobalConfig.device
    )

    train(model, 500)

    model.save("."+GlobalConfig.save_root, "model.pth")