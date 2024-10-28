'''
加载和控制 UR5机器人,末端WSG50二指手抓,DIGIT触觉传感器

Copyright 2024, HuangChen. All rights reserved.
Author: HuangChen (huangchen_123@stu.hit.edu.cn)

'''

import os
import sys

current_file_path = os.path.dirname(__file__)  # 当前文件所在文件夹路径
workspace_path = os.path.abspath(os.path.join(current_file_path, ".."))
sys.path.append(workspace_path)

import pybullet as p
import pybulletX as px
import tacto
import time
import functools
import numpy as np

from config import GlobalConfig

# 机器人urdf文件路径
urdf_path = workspace_path + "/asset/robot/ur5_wsg50_digit.urdf"

# 机器人初始位姿
base_position = [0.0, 0.0, 0.0]
init_state = {
    "end_effector" : {
        "position" : [0.40, 0, 0.1],
        "euler" : [3.1416, 0.0, 0.0],
    },
    "gripper_width" : 0.11
}

# DIGIT传感器参数
width = GlobalConfig.tactile_size
height = GlobalConfig.tactile_size
visualize_gui = True


class UR5(px.Robot):
    end_effector_name = "tcp_joint"
    gripper_names = [
        "base_joint_gripper_left",
        "base_joint_gripper_right",
    ]
    digit_joint_names = ["joint_finger_tip_left", "joint_finger_tip_right"]

    def __init__(self, render):
        super().__init__(urdf_path=urdf_path, base_position=base_position, use_fixed_base=True)
        self.render = render
        self.end_effector_index = self.get_joint_index_by_name(self.end_effector_name)

        # 初始时的位姿
        ee_link = self.get_link_state_by_name(self.end_effector_name)
        self.ee_link_pos = list(ee_link[0])
        self.ee_link_ori = list(p.getEulerFromQuaternion(ee_link[1]))

        # 计算各关节初始转角，并初始化
        self.zero_pose = self._states_to_joint_position(init_state["end_effector"]["position"],
                                                        init_state["end_effector"]["euler"])
        # 初始时手爪张开
        self.zero_pose[6] = -0.05
        self.zero_pose[7] = 0.05

        # 运动到初始位置
        self.reset()
        self.gripper_status = "open"

        # DIGIT传感器
        self.digits = tacto.Sensor(width=width, height=height, visualize_gui=visualize_gui)
        self.digits.add_camera(self.id, self.digit_links)

        # 记录当前TCP位姿
        ee_link = self.get_link_state_by_name(self.end_effector_name)
        self.ee_link_pos = list(ee_link[0])
        self.ee_link_ori = list(p.getEulerFromQuaternion(ee_link[1]))

    # action_pos: [dx, dy, dz]
    # action_ori: [drx, dry, drz]
    # gripper_status: "open" or "close"
    def step(self, action_pos, action_ori = None, max_joint_velocity = None, is_delta = True):
        if is_delta:
            target_pos = [
                self.ee_link_pos[0] + action_pos[0],
                self.ee_link_pos[1] + action_pos[1],
                self.ee_link_pos[2] + action_pos[2],
            ]
        else:
            target_pos = action_pos

        if action_ori is None:
            target_ori = self.ee_link_ori
        else:
            if is_delta:
                target_ori = [
                    self.ee_link_ori[0] + action_ori[0],
                    self.ee_link_ori[1] + action_ori[1],
                    self.ee_link_ori[2] + action_ori[2],
                ]
            else:
                target_ori = action_ori

        while True:
            target_joint_positions = self._states_to_joint_position(target_pos, target_ori)
            if max_joint_velocity is not None:
                # 获取当前关节状态
                joint_states = p.getJointStates(self.id, self.free_joint_indices)
                joint_positions = np.array([state[0] for state in joint_states])
                # 如果速度超过限制，则缩放关节速度
                joint_position_error = target_joint_positions - joint_positions
                max_error = np.max(np.abs(joint_position_error))
                scale = 1.0 if max_error <= max_joint_velocity else max_joint_velocity / max_error

                # 计算下一步的关节位置
                next_joint_positions = []
                for i in range(len(self.free_joint_indices)):
                    joint_velocity = scale * np.sign(joint_position_error[i]) * abs(joint_position_error[i])
                    next_position = joint_positions[i] + joint_velocity
                    next_joint_positions.append(next_position)
                # 驱动关节
                self._drive_joints(next_joint_positions)

            else:
                self._drive_joints(target_joint_positions)

            # 判断是否到达
            ee_link = self.get_link_state_by_name(self.end_effector_name)
            cur_pos = list(ee_link[0])
            cur_ori = list(p.getEulerFromQuaternion(ee_link[1]))
            delta_pos = [target_pos[0]-cur_pos[0],
                         target_pos[1]-cur_pos[1],
                         target_pos[2]-cur_pos[2]]
            delta_ori = [target_ori[0]-cur_ori[0],
                         target_ori[1]-cur_ori[1],
                         target_ori[2]-cur_ori[2]]
            delta_ori = [dx - 2*np.pi if dx > np.pi else dx + 2*np.pi if dx < -np.pi else dx for dx in delta_ori]
            delta_pos, delta_ori = np.array(delta_pos), np.array(delta_ori)

            new_distance_pos = np.linalg.norm(delta_pos)
            new_distance_ori = np.linalg.norm(delta_ori)

            if new_distance_pos < 0.0001 and new_distance_ori < 0.01:
                break


        # 更新TCP位姿
        self.ee_link_pos = target_pos
        self.ee_link_ori = target_ori

    def control_gripper(self, status):
        if status not in ["open", "close"]:
            raise Exception("status must be 'open' or 'close'")

        self.gripper_status = status

        # 驱动手爪
        if self.gripper_status == "close":
            gripper_target_pos = [-0.003, 0.003]
        else:
            gripper_target_pos = [-0.05, 0.05]
        for i in range(2):
            joint_index = self.gripper_joint_ids[i]
            joint_pos = gripper_target_pos[i]
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=self.free_joint_indices[joint_index],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=60,
                maxVelocity=0.1,
            )

        for _ in range(40):
            p.stepSimulation()
            if self.render:
                time.sleep(1.0/240)

    # 根据关节位置，驱动机械臂
    def _drive_joints(self, joint_positions):
        # 驱动机械臂
        for joint_index, joint_angle in zip(self.free_joint_indices, joint_positions):
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angle,
                # maxVelocity=0.05
            )

        # 驱动手爪
        gripper_velocity = [-0.2,0.2]
        if self.gripper_status == "close":
            gripper_target_pos = [-0.003, 0.003]
        else:
            gripper_target_pos = [-0.05, 0.05]
        for i in range(2):
            joint_index = self.gripper_joint_ids[i]
            joint_pos = gripper_target_pos[i]
            v = gripper_velocity[i]
            p.setJointMotorControl2(
                bodyUniqueId=self.id,
                jointIndex=self.free_joint_indices[joint_index],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=60,
                maxVelocity=0.1,
            )
        p.stepSimulation()
        if self.render:
            time.sleep(1.0/240)

    # 根据末端执行器的世界位姿，用运动学逆解计算各个关节的转角
    # target_pos: [x, y, z]
    # target_ori: [rx, ry, rz]
    def _states_to_joint_position(self, target_pos, target_ori):
        target_ori = p.getQuaternionFromEuler(target_ori)
        joint_positions = np.array(
            p.calculateInverseKinematics(
                bodyUniqueId = self.id,
                endEffectorLinkIndex = self.end_effector_index,
                targetPosition = target_pos,
                targetOrientation = target_ori,
            )
        )
        return joint_positions
    
    @property
    def digit_links(self):
        return [self.get_joint_index_by_name(name) for name in self.digit_joint_names]

    @property
    @functools.lru_cache(maxsize=None)
    def gripper_joint_ids(self):
        return [
            self.free_joint_indices.index(self.get_joint_index_by_name(name))
            for name in self.gripper_names
        ]
    