'''
轴孔装配环境

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
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from ur5 import UR5
from config import GlobalConfig


# 相机设置
img_width = GlobalConfig.visual_size
img_height = GlobalConfig.visual_size
nearVal=0.01
farVal=2

# 零件相关
hole_position = (0.5, 0, 0)

def getCameraUpVector(cameraEyePosition, cameraTargetPosition):
    directionVec = [cameraEyePosition[0]-cameraTargetPosition[0],
                    cameraEyePosition[1]-cameraTargetPosition[1],
                    cameraEyePosition[2]-cameraTargetPosition[2]]
    cameraUpVector = [-directionVec[0]*directionVec[2],
                      -directionVec[1]*directionVec[2],
                      directionVec[0]*directionVec[0]+directionVec[1]*directionVec[1]]
    return cameraUpVector


# PyBullet 仿真环境
class PegInHole:
    def __init__(self, is_render=False) -> None:
        self.is_render = is_render
        self.__init()

    # 用于初始化环境，加载机器人
    def __init(self):
        if self.is_render:
            client = p.connect(p.GUI)
        else:
            client = p.connect(p.DIRECT)

        # Use config to set pybullet simulation parameters
        cfg = {"gravity": {"gravX": 0, "gravY": 0, "gravZ": -9.81},}
        p.setParameters(cfg, client)

        # Load the classic plane and table
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", basePosition=[0, 0, -0.63], physicsClientId=client)
        p.loadURDF("table/table.urdf",basePosition=[0.5, 0, -0.63])

        # 加载机器人
        self.robot = UR5(self.is_render)

        # 加载零件
        self.peg = self.loadPart(
            urdf_path=workspace_path + "/asset/parts/quadrangular_40x60/urdf/quadrangular_40x60.urdf",
            postion=hole_position + np.array([0,0,0.03]),
            fixed=False
        )
        self.hole = self.loadPart(
            urdf_path=workspace_path + "/asset/parts/quadrangular_hole_2mm_40x40/urdf/quadrangular_hole_2mm_40x40.urdf",
            postion=hole_position + np.array([0,0,0.015]),
            fixed=True
        )
        p.changeVisualShape(self.peg, -1, rgbaColor=[1,0,0,1])
        p.changeVisualShape(self.hole, -1, rgbaColor=[0,1,0,1])
        p.changeDynamics(self.peg, -1, lateralFriction=0.8, spinningFriction=0.5)
        p.changeDynamics(self.hole, -1, lateralFriction=0.05, spinningFriction=0.2)
        p.changeDynamics(self.robot.id, 10, lateralFriction=0.8, spinningFriction=0.5)
        p.changeDynamics(self.robot.id, 13, lateralFriction=0.8, spinningFriction=0.5)

        # 其他无关零件
        hole = p.loadURDF(
            fileName=workspace_path + "/asset/parts/quadrangular_hole_2mm_40x40/urdf/quadrangular_hole_2mm_40x40.urdf",
            basePosition=hole_position + np.array([0.103, 0, -0.005]),
            useFixedBase=True
        )
        p.changeVisualShape(hole, -1, rgbaColor=[0,1,0,1])

        # # 设置初始观察视角
        # p.resetDebugVisualizerCamera(cameraDistance=0.3,
        #                              cameraYaw=0,
        #                              cameraPitch=-40,
        #                              cameraTargetPosition=[0.55, -0.35, 0.2])

        # 相机设置
        self.projectionMatrix = p.computeProjectionMatrixFOV(fov=70, aspect=1.0, nearVal=nearVal, farVal=farVal)

    def reset(self):
        # 复位机器人
        self.robot.reset()
        # 重新记录当前TCP位姿
        ee_link = self.robot.get_link_state_by_name(self.robot.end_effector_name)
        self.robot.ee_link_pos = list(ee_link[0])
        self.robot.ee_link_ori = list(p.getEulerFromQuaternion(ee_link[1]))
        self.gripper_status = "open"

        # 复位轴和孔
        p.resetBasePositionAndOrientation(self.peg, (hole_position[0], hole_position[1], 0.03), p.getQuaternionFromEuler([0,0,0]))


    def loadPart(self, urdf_path, postion, orientation=[0,0,0], fixed = False):
        obj = px.Body(urdf_path=urdf_path, base_position=postion, base_orientation=p.getQuaternionFromEuler(orientation),use_fixed_base=fixed)
        self.robot.digits.add_body(obj)
        return obj.id

    def stepSimulation(self):
        p.stepSimulation()
        self.get_image()
        color, depth = self.robot.digits.render()
        if self.is_render:
            self.robot.digits.updateGUI(color, depth)

    # 视觉返回一个H*W*4的numpy矩阵，表示rgba图片
    def get_image(self):
        ee_link = self.robot.get_link_state_by_name(self.robot.end_effector_name)
        ee_link_pos = list(ee_link[0])
        ee_link_ori = list(p.getEulerFromQuaternion(ee_link[1]))
        pos = np.array(ee_link_pos)
        ori = np.array(ee_link_ori)
        rx = R.from_euler('x', -ori[0], degrees=False).as_matrix()
        ry = R.from_euler('y', -ori[1], degrees=False).as_matrix()
        rz = R.from_euler('z', -ori[2], degrees=False).as_matrix()
        R_mat = np.dot(np.dot(rx, ry), rz)
        px = np.array([[0,1,0]])
        px = np.dot(px, R_mat)[0]
        py = np.array([[1,0,0]])
        py = np.dot(py, R_mat)[0]
        pz = np.array([[0,0,1]])
        pz = np.dot(pz, R_mat)[0]

        # 相机位置
        cameraEyePosition = pos - 0.07*pz + 0.035*py
        cameraTargetPosition = cameraEyePosition + 0.2*pz
        viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, py)

        _, _, rgb, depth, _ = p.getCameraImage(img_width, img_height, viewMatrix, self.projectionMatrix)
        return rgb, depth

    # 触觉返回一个list，有2个元素，每个元素都是H*W*3的numpy矩阵，表示触觉图片
    def get_tactile(self):
        color, _ = self.robot.digits.render()
        return color


if __name__ == "__main__":
    env = PegInHole(True)

    while 1:
        env.reset()
        env.robot.control_gripper("open")
        env.robot.step([hole_position[0], hole_position[1], 0.055], is_delta=False)
        env.robot.control_gripper("close")
        env.robot.step([0,0,0.05])
        time.sleep(2)
        