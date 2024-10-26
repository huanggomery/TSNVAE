import os

from pixyz.models import Model
from pixyz.losses import Parameter, LogProb, KullbackLeibler as KL, Expectation as E
from pixyz import distributions as dist

import torch
from torch import nn

from distributions import VisualEncoder, TactileEncoder, VisualDecoder, TactileDecoder, Velocity, Transition, TargetModel

sigma_g = 0.0015   # 抓取误差为 1.5mm

class TsNewtonianVAE(Model):
    def __init__(self,
                 v_encoder_param: dict,
                 v_decoder_param: dict,
                 t_encoder_param: dict,
                 t_decoder_param: dict,
                 velocity_param: dict,
                 target_param: dict,
                 delta_time: float,
                 device: str = "cpu"
                 ):
        self.v_encoder = VisualEncoder(**v_encoder_param).to(device)
        self.t_encoder = TactileEncoder(**t_encoder_param).to(device)
        self.v_decoder = VisualDecoder(**v_decoder_param).to(device)
        self.t_decoder = TactileDecoder(**t_decoder_param).to(device)
        self.transition = Transition(delta_time=delta_time).to(device)
        self.velocity = Velocity(**velocity_param).to(device)
        self.target_model = TargetModel(**target_param).to(device)
        self.norm_g = dist.Normal(loc=0, scale=sigma_g, var=["x_t"])

        self.distributions = nn.ModuleList(
            [self.v_encoder, self.v_decoder, self.t_encoder, self.t_decoder, self.transition, self.target_model]
        )
        params = self.distributions.parameters()
        self.optimizer = torch.optim.Adam(params, lr=3e-4)

        # 损失函数
        self.v_recon_loss = -E(self.transition, LogProb(self.v_decoder)).mean()
        self.v_KL_loss = KL(self.v_encoder, self.transition).mean()
        self.t_recon_loss = -E(self.t_encoder, LogProb(self.t_decoder)).mean()
        self.vt_recon_loss = -E(self.target_model, LogProb(self.v_decoder)).mean()
        self.vt_KL_loss = KL(self.v_encoder, self.target_model).mean()
        self.add_KL_loss = KL(self.v_encoder, self.norm_g) + KL(self.target_model, self.norm_g).mean()

        self.delta_time = delta_time

    # 根据输入的序列计算总损失，基于公式(4)(5)
    def calculate_loss(self, input_var_dict: dict):
        I = input_var_dict["I"]
        I_z = input_var_dict["I_z"]
        z = self.t_encoder.sample({"I_z"}, reparam=True)["z"]
        u = input_var_dict["u"]
        beta = input_var_dict["beta"]

        total_loss = 0.

        T, B, C = u.shape

        x_t0 = self.v_encoder.sample({"I_t": I[0]}, reparam=True)["x_t"]

        for step in range(1, T-1):
            x_t = self.v_encoder.sample({"I_t": I[step]}, reparam=True)["x_t"]
            v_t = (x_t - x_t0) / self.delta_time
            v_t1 = self.velocity(x_t, v_t, u[step])["v_t"]

            # 计算损失
            v_recon_loss, _ = self.v_recon_loss({"x_t0": x_t, "v_t": v_t1, "I_t": I[step+1]})
            v_KL_loss, _ = self.v_KL_loss({"I_t": I[step+1], "x_t0": x_t, "v_t": v_t1})
            t_recon_loss, _ = self.t_recon_loss({"I_z": I_z})
            vt_recon_loss, _ = self.vt_recon_loss({"z": z, "I_t": I[0]})
            vt_KL_loss, _ = self.vt_KL_loss({"I_t": I[0], "z": z})
            add_KL_loss, _ = self.add_KL_loss({"I_t": I[0], "z": z})

            total_loss += (v_recon_loss + v_KL_loss + t_recon_loss + vt_recon_loss + vt_KL_loss + add_KL_loss)

            x_t0 = x_t
        
        return total_loss/T

    # 输入序列数据，计算损失，训练模型
    def train(self, train_x_dict: dict):
        self.distributions.train()

        loss = self.calculate_loss(train_x_dict)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path, filename):
        os.makedirs(path, exist_ok=True)

        torch.save({
            'distributions': self.distributions.to("cpu").state_dict(),
        }, f"{path}/{filename}")

        self.distributions.to(self.device)

    def load(self, path, filename):
        self.distributions.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions'])

if __name__ == "__main__":
    encoder = TactileEncoder(6)
    decoder = TactileDecoder(6, 3, 64, device="cpu")
    t_recon_loss = E(encoder, LogProb(decoder))
    I_z = torch.zeros((32, 3, 64, 64))
    loss = t_recon_loss({"I_z": I_z})
    print(loss)