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
            total_loss += loss
        total_loss /= len(dataloader.dataset)

        print("Epoch: {} Train loss: {:.4f}".format(i+1, total_loss))


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
    # model.load_part("v_encoder", "."+GlobalConfig.save_root+"/v_encoder.pth")
    # model.load_part("v_decoder", "."+GlobalConfig.save_root+"/v_decoder.pth")
    # model.load_part("t_encoder", "."+GlobalConfig.save_root+"/t_encoder.pth")
    # model.load_part("t_decoder", "."+GlobalConfig.save_root+"/t_decoder.pth")
    # model.load_part("target_model", "."+GlobalConfig.save_root+"/target.pth")
    train(model, 500)

    model.save("."+GlobalConfig.save_root, "model.pth")