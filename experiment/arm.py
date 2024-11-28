import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from random import gauss, uniform
import pyrealsense2 as rs
import numpy as np


from xarm.wrapper import XArmAPI
from PyTac3D import Sensor

ip = '192.168.1.118'
SN = ''
home_pos = [300, -100.0, 45.00, -180.0, 0.0, 0.0]
usb_pos = [407.83, -106., 60]
velocity_limit = 5  # 随机运动时的速度限制，单位 mm/s
rotation_limit = 5  # 随机运动时的旋转限制，单位 °/s

def Tac3DRecvCallback(frame, param):
    global SN
    SN = frame['SN']
    pass

class Arm:
    def __init__(self, ip, delta_time) -> None:
        # 初始化机械臂
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)

        # 初始化手爪
        self.arm.set_gripper_enable(enable=True)
        self.arm.set_gripper_mode(0)

        # 初始化相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # 初始化触觉传感器
        global SN
        self.tacSensor = Sensor(recvCallback=Tac3DRecvCallback, port=9988)
        self.tacSensor.waitForFrame()
        time.sleep(1)
        self.tacSensor.calibrate(SN)
        time.sleep(1)

        self.delta_time = delta_time

    def go_home(self):
        self.arm.set_mode(0)
        self.loose()
        self.arm.set_position(x=home_pos[0], y=home_pos[1], z=home_pos[2], roll=home_pos[3], pitch=home_pos[4], yaw=home_pos[5],radius=False, wait=True, speed=50)

    def loose(self):
        self.arm.set_gripper_position(800, wait=True)
    
    def grasp(self):
        self.arm.set_gripper_position(520, wait=True)

    def catch_random(self):
        # 随机的抓取位置误差
        self.x_err = gauss(0, 1.5)
        self.x_err = np.clip(self.x_err, -4, 4)
        self.pitch_err = gauss(0, 4)
        self.pitch_err = np.clip(self.pitch_err, -10, 10)

        # 移动到指定位置
        self.arm.set_position(x=usb_pos[0]+self.x_err,
                              y=usb_pos[1],
                              z=usb_pos[2],
                              roll=home_pos[3],
                              pitch=self.pitch_err,
                              yaw=home_pos[5],
                              wait=True, relative=False, speed=30)

        # 抓住物体
        self.grasp()
        # 提起高度随机
        z = uniform(15, 22)
        self.arm.set_position(z=z, wait=True, relative=True, speed=8)

    # 返回动作 [dx, dy] [drx, dry, drz]
    def step_random(self):
        # 均匀分布的随机运动
        pos_noise = np.array([uniform(-velocity_limit, velocity_limit), uniform(-velocity_limit, velocity_limit)])
        rot_noise = np.array([uniform(-rotation_limit, rotation_limit), uniform(-rotation_limit, rotation_limit), uniform(-rotation_limit, rotation_limit)])

        # 恢复
        restore_factor = 0.1
        cur_position = self.get_pos()
        cur_pos = cur_position[:2]
        pos_restore = -restore_factor * (cur_pos - np.array([usb_pos[0]+self.x_err, usb_pos[1]])) / self.delta_time
        cur_rot = cur_position[3:]
        if cur_rot[0] > 0:
            cur_rot[0] = -180 - (180-cur_rot[0])
        rot_restore = -restore_factor * (cur_rot - np.array([home_pos[3], self.pitch_err, home_pos[5]])) / self.delta_time

        pos_action = pos_noise + pos_restore
        rot_action = rot_noise + rot_restore

        # 安全保障
        if np.max(pos_action) > velocity_limit*2 or np.max(rot_action) > rotation_limit*2:
            raise Exception("超过最大运动限制！")

        self.arm.vc_set_cartesian_velocity([pos_action[0], pos_action[1], 0, rot_action[0],rot_action[1],rot_action[2]], is_radian=False, duration=self.delta_time)
        time.sleep(self.delta_time+0.1)

        return pos_action, rot_action

    # 返回 [x, y, z, roll, pitch, yaw]
    def get_pos(self):
        pos = np.array(self.arm.get_position()[1])
        if pos[3] > 0:
            pos[3] -= 360
        return pos
    
    # 返回 [angle-1, ..., angle-7]
    def get_joint_states(self):
        states = np.array(self.arm.get_joint_states(is_radian=False)[1][0])
        return states

    # 返回 (400, 6)
    def get_tactile(self):
        frame = self.tacSensor.getFrame()
        pos = frame['3D_Positions']
        displace = frame['3D_Displacements']
        tactile = np.concatenate((pos,displace), axis=1)
        return tactile

    # 返回 (480, 640, 3) 范围[0, 255]
    def get_image(self):
        # 等待并获取一帧数据
        frames = self.pipeline.wait_for_frames()
        # 获取 RGB 图像帧
        color_frame = frames.get_color_frame()
        # 将 RGB 图像数据转换为 NumPy 数组
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
