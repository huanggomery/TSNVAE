import os
import sys
current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

from pixyz.models import Model
from pixyz.losses import Parameter, LogProb, KullbackLeibler as KL, Expectation as E
from pixyz import distributions as dist

import torch
from torch import nn

from models.distributions import VisualEncoder, TactileEncoder, VisualDecoder, TactileDecoder, Velocity, Transition, TargetModel, Tac3dEncoder, Tac3dDecoder

from config import GlobalConfig

sigma_g = GlobalConfig.grasp_error   # 抓取误差

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
        self.device = device
        self.v_encoder = VisualEncoder(**v_encoder_param).to(device)
        # self.t_encoder = TactileEncoder(**t_encoder_param).to(device)
        self.t_encoder = Tac3dEncoder(**t_encoder_param).to(device)
        self.v_decoder = VisualDecoder(**v_decoder_param).to(device)
        # self.t_decoder = TactileDecoder(**t_decoder_param).to(device)
        self.t_decoder = Tac3dDecoder(**t_decoder_param).to(device)
        self.transition = Transition(delta_time=delta_time).to(device)
        self.velocity = Velocity(**velocity_param).to(device)
        self.target_model = TargetModel(**target_param).to(device)
        self.norm_g = dist.Normal(loc=0, scale=sigma_g, var=["x_t"]).to(device)

        self.distributions = nn.ModuleList(
            [self.v_encoder, self.v_decoder, self.t_encoder, self.t_decoder, self.transition, self.velocity, self.target_model]
        )
        params = self.distributions.parameters()
        self.optimizer = torch.optim.Adam(params, lr=GlobalConfig.lr)

        # 损失函数
        '''
        v_recon_loss: 训练图片编码和解码
        v_KL_loss: 训练运动学模型
        t_recon_loss: 训练触觉的编码和解码
        vt_recon_loss: 训练触觉target模型
        vt_KL_loss: 训练视觉编码和target模型的预测结果保持一致
        add_KL_loss: 训练视觉编码、target模型与先验概率保持一致
        '''
        self.v_recon_loss = -E(self.transition, LogProb(self.v_decoder)).mean()
        self.v_KL_loss = 1 * KL(self.v_encoder, self.transition).mean()
        self.t_recon_loss = -E(self.t_encoder, LogProb(self.t_decoder)).mean()
        self.vt_recon_loss = -E(self.target_model, LogProb(self.v_decoder)).mean()
        self.vt_KL_loss = 1 * KL(self.v_encoder, self.target_model).mean()
        self.add_KL_loss = 1 * (KL(self.v_encoder, self.norm_g).mean() + KL(self.target_model, self.norm_g).mean())

        self.delta_time = delta_time

    # 根据输入的序列计算总损失，基于公式(4)(5)
    def calculate_loss(self, input_var_dict: dict):
        I = input_var_dict["I"]
        I_z = input_var_dict["I_z"]
        z = self.t_encoder.sample({"I_z": I_z}, reparam=True)["z"]
        u = input_var_dict["u"]

        total_loss = 0.

        t_recon_loss, _ = self.t_recon_loss({"I_z": I_z})
        vt_recon_loss, _ = self.vt_recon_loss({"z": z, "I_t": I[0]})
        vt_KL_loss, _ = self.vt_KL_loss({"I_t": I[0], "z": z})
        add_KL_loss, _ = self.add_KL_loss({"I_t": I[0], "z": z})

        T, B, C = u.shape

        x_t0 = self.v_encoder.sample({"I_t": I[0]}, reparam=True)["x_t"]

        for step in range(1, T):
            x_t = self.v_encoder.sample({"I_t": I[step]}, reparam=True)["x_t"]
            v_t = (x_t - x_t0) / self.delta_time
            v_t1 = self.velocity(x_t, v_t, u[step])["v_t"]

            # 计算损失
            v_recon_loss, _ = self.v_recon_loss({"x_t0": x_t, "v_t": v_t1, "I_t": I[step+1]})
            v_KL_loss, _ = self.v_KL_loss({"I_t": I[step+1], "x_t0": x_t, "v_t": v_t1})

            total_loss += (v_recon_loss + v_KL_loss)

            x_t0 = x_t

        total_loss += (t_recon_loss + vt_recon_loss + vt_KL_loss + add_KL_loss) * T
        
        return total_loss/T

    # 输入序列数据，计算损失，训练模型
    def train(self, train_x_dict: dict):
        self.distributions.train()

        loss = self.calculate_loss(train_x_dict)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # 输入视觉图像I_t，计算隐藏层x_t
    # 输入: torch.Tensor (B, C, H, W) 或 (C, H, W)
    # 输出: torch.Tensor (B, X_DIM) 或 (X_DIM)
    def get_latent(self, I_t: torch.Tensor):
        is_batch = I_t.dim() == 4
        if not is_batch:
            I_t = I_t.unsqueeze(0)
        with torch.no_grad():
            x_t = self.v_encoder(I_t)["loc"]

        if not is_batch:
            x_t = x_t.squeeze()
        
        return x_t

    # 输入触觉图像I_z，计算隐藏层的目标x_g
    # 输入: torch.Tensor (B, C, H, W) 或 (C, H, W)
    # 输出: torch.Tensor (B, X_DIM) 或 (X_DIM)
    def get_target_latent(self, I_z: torch.Tensor):
        is_batch = I_z.dim() == 4
        if not is_batch:
            I_z = I_z.unsqueeze(0)
        with torch.no_grad():
            z = self.t_encoder(I_z)["loc"]
            x_g = self.target_model(z)["loc"]

        if not is_batch:
            x_g = x_g.squeeze()
        
        return x_g
    
    # 评估模式
    def eval(self):
        self.v_encoder.eval()
        self.v_decoder.eval()
        self.t_encoder.eval()
        self.t_decoder.eval()
        self.target_model.eval()
        self.transition.eval()
        self.velocity.eval()

    def save(self, path, filename):
        os.makedirs(path, exist_ok=True)

        torch.save({
            'distributions': self.distributions.to("cpu").state_dict(),
        }, f"{path}/{filename}")

        self.distributions.to(self.device)

    def load(self, path, filename):
        self.distributions.load_state_dict(torch.load(
            f"{path}/{filename}", map_location=torch.device('cpu'))['distributions'])

    def load_part(self, model_name, path):
        if model_name == "v_encoder":
            self.v_encoder.load_state_dict(torch.load(path))
        elif model_name == "v_decoder":
            self.v_decoder.load_state_dict(torch.load(path))
        elif model_name == "t_encoder":
            self.t_encoder.load_state_dict(torch.load(path))
        elif model_name == "t_decoder":
            self.t_decoder.load_state_dict(torch.load(path))
        elif model_name == "target_model":
            self.target_model.load_state_dict(torch.load(path))
        else:
            raise Exception("no such model: {}".format(model_name))

if __name__ == "__main__":
    pass
