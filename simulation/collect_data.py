# 采集仿真数据

import argparse
import os
import sys
from random import gauss
import numpy as np
from PIL import Image

current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

from config import GlobalConfig
from simulation.peg_in_hole import PegInHole

# 固定的零件位置
hole_position = np.array([0.5, 0, 0])

# 保存的目录
save_dir = workspace_path + GlobalConfig.data_root

# 随机运动相关参数
xy_std = 0.003            # 随机运动标准差
restoring_force = 0.2     # 恢复力系数
max_action_limit = 0.01  # 最大动作限制


def save_visual(env, step, dir):
    img, _ = env.get_image()
    img = Image.fromarray(img[:,:,:3])
    img.save(dir + "/{}.jpg".format(step))

def save_tactile(env, dir):
    img = env.get_tactile()
    img = Image.fromarray(img[0])  # 只保存一个手指的
    img.save(dir + "/I_z.jpg")

# 返回 numpy 向量 (2,)
def get_position(env):
    ee_link = env.robot.get_link_state_by_name(env.robot.end_effector_name)
    pos = np.array([ee_link[0][0], ee_link[0][1]])
    return pos

# 随机运动，返回 numpy 向量 (2,)
def random_action(env, mean_pos):
    # 高斯分布的随机扰动
    pos_noise = np.array([gauss(0, xy_std), gauss(0, xy_std)])

    # 恢复
    cur_pos = get_position(env)
    pos_restore = -restoring_force * (cur_pos - mean_pos)

    action = pos_noise + pos_restore
    action = np.clip(action, -np.array([max_action_limit, max_action_limit]), np.array([max_action_limit, max_action_limit]))

    return action

# 采集一条轨迹的数据
def collect_trajectory(env, traj_id, saved):
    if saved:
        # 保存动作及状态信息
        actions = np.zeros((0, 2))
        tcp_positions = np.zeros((0, 2))
        # 创建文件夹
        dir = save_dir + "/" + str(traj_id)
        if not os.path.exists(dir):
            os.makedirs(dir)

    env.reset()
    env.robot.control_gripper("open")

    # 随机的抓取位置误差
    pos_err = gauss(0, 0.0015)
    if pos_err < -0.003:
        pos_err = -0.003
    if pos_err > 0.003:
        pos_err = 0.003

    # 抓起物体并提升
    env.robot.step([0,0,0],[0,0,np.pi/2], is_delta=True)
    env.robot.step([hole_position[0], hole_position[1]+pos_err, 0.055], is_delta=False)
    env.robot.control_gripper("close")
    env.robot.step([0,0,0.05])

    mean_pos = get_position(env)

    # 记录此时的I_g, I_z, TCP位置
    if saved:
        save_visual(env, 0, dir)
        save_tactile(env, dir)
        tcp_positions = np.concatenate((tcp_positions, (mean_pos-hole_position).reshape(1,2)), axis=0)

    for step in range(1, 21):
        a = random_action(env, mean_pos)
        env.robot.step([a[0], a[1], 0])

        if saved:
            # t时刻的动作
            actions = np.concatenate((actions, a.reshape((1,2)) / 0.5), axis=0)
            # t+1时刻的视觉和位置
            save_visual(env, step, dir)
            cur_pos = get_position(env)
            tcp_positions = np.concatenate((tcp_positions, (cur_pos-hole_position).reshape((1,2))), axis=0)

    # 保存动作和位置
    if saved:
        np.save(dir + "/action.npy", actions)
        np.save(dir + "/tcp.npy", tcp_positions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--saved", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    save_dir = save_dir + "/" + args.mode

    env = PegInHole(args.render)

    for i in range(args.epochs):
        collect_trajectory(env, i+1, args.saved)
