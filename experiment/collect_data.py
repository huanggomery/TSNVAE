import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from random import gauss, uniform

import arm


delta_time = 0.5
l = 48.5    # 位置补偿值
usb_pos0 = [340, -100.]  # USB插排的中心位置，笛卡尔坐标

# 在初始位置获取USB
def get_usb(arm):
    # 固定初始位置
    arm.loose()
    arm.arm.set_position(x=423.54, y=-373.11, z=70, roll=-180, pitch=0, yaw=0,
                         speed=70, is_radian=False, relative=False, wait=True)

    # 随机的抓取位置误差
    x_err = gauss(0, 1.5)
    x_err = np.clip(x_err, -4, 4)
    pitch_err = gauss(0, 4)
    pitch_err = np.clip(pitch_err, -10, 10)
    arm.arm.set_position(x=x_err, pitch=pitch_err, is_radian=False, relative=True, wait=True)

    z_catch_range = [43, 48]  # 抓取时的高度范围
    z_catch = uniform(z_catch_range[0], z_catch_range[1])
    arm.arm.set_position(z=z_catch, wait=True)

    arm.grasp()

    # 提升
    arm.arm.set_position(z=20, wait=True, relative=True)

    return np.array([x_err, 0, 0, 0, pitch_err, 0])

# 移动到随机位置
def move_random(arm):

    xy_limit = 100   # 插排放置位置的范围，单位 mm
    rot_limit = 15   # 插排置角度的范围，单位 °

    x_random = uniform(-xy_limit, xy_limit)
    y_random = uniform(-xy_limit, xy_limit)
    rz_random = uniform(-rot_limit, rot_limit)

    arm.arm.set_position(x=usb_pos0[0]+x_random,
                         y=usb_pos0[1]+y_random,
                         z=88,
                         yaw=rz_random,
                         speed=70,
                         is_radian=False,
                         relative=False,
                         wait=True)

# 返回动作 [dx, dy] [drx, dry, drz]
def step_random(arm, pos_err):
    # 均匀分布的随机运动
    velocity_limit = 5  # 随机运动时的速度限制，单位 mm/s
    rotation_limit = 5  # 随机运动时的旋转限制，单位 °/s
    pos_noise = np.array([uniform(-velocity_limit, velocity_limit),
                          uniform(-velocity_limit, velocity_limit),
                          uniform(-rotation_limit, rotation_limit),
                          uniform(-rotation_limit, rotation_limit),
                          uniform(-rotation_limit, rotation_limit)])

    # 恢复
    restore_factor = 0.3
    pos_restore = -restore_factor * pos_err / delta_time

    pos_action = pos_noise + pos_restore

    # 安全保障
    if np.max(pos_action[:2]) > velocity_limit * 2 or np.max(pos_action[2:]) > rotation_limit * 2:
        raise Exception("超过最大运动限制！")

    arm.arm.set_tool_position(x=pos_action[0]*delta_time,
                              y=pos_action[1]*delta_time,
                              roll=pos_action[2]*delta_time,
                              pitch=pos_action[3]*delta_time,
                              yaw=pos_action[4]*delta_time,
                              speed=5,
                              is_radian=False,
                              wait=True)
    return pos_action

def collect_loop(arm, mode="train"):
    arm.go_home()

    epoch = 1
    while True:
        pos_all = np.zeros((0, 6))
        joint_states = np.zeros((0, 7))
        action_all = np.zeros((0, 5))

        print("输入0 退出， 其他继续")
        choice = input()
        if choice == '0':
            break

        if not os.path.exists("data/{}/{}".format(mode, epoch)):
            os.mkdir("data/{}/{}".format(mode, epoch))

        # 获取USB
        pos_cur = get_usb(arm)

        # 移动到随机位置
        move_random(arm)

        print("输入0 重新采集该轨迹， 其他继续")
        choice = input()
        if choice == '0':
            arm.loose()
            continue

        # 记录视觉、触觉、位姿
        tactile = arm.get_tactile()
        np.save("data/{}/{}/tactile.npy".format(mode, epoch), tactile)
        img = arm.get_image()
        cv2.imwrite("data/{}/{}/0.jpg".format(mode, epoch), img)
        pos_all = np.concatenate((pos_all, pos_cur[np.newaxis, :]), axis=0)
        # joint = arm.get_joint_states()
        # joint_states = np.concatenate((joint_states, joint[np.newaxis, :]), axis=0)

        pos_err = np.array([0,0,0,0,0], dtype=np.float32)

        # arm.arm.set_mode(5)
        # arm.arm.set_state(state=0)

        for i in range(20):
            action = step_random(arm, pos_err)
            action_all = np.concatenate((action_all, action[np.newaxis, :]), axis=0)

            pos_err += action * delta_time
            pos_cur += np.insert(action, 2, 0) * delta_time

            img = arm.get_image()
            cv2.imwrite("data/{}/{}/{}.jpg".format(mode, epoch, i+1), img)
            pos_all = np.concatenate((pos_all, pos_cur[np.newaxis, :]), axis=0)
            # joint = arm.get_joint_states()
            # joint_states = np.concatenate((joint_states, joint[np.newaxis, :]), axis=0)

        np.save("data/{}/{}/pos.npy".format(mode, epoch), pos_all)
        np.save("data/{}/{}/joint.npy".format(mode, epoch), joint_states)
        np.save("data/{}/{}/action.npy".format(mode, epoch), action_all)

        # arm.arm.set_mode(0)
        # arm.arm.set_state(state=0)

        arm.loose()
        print("已采集轨迹数：{}".format(epoch))
        epoch += 1


if __name__ == "__main__":
    my_arm = arm.Arm(arm.ip, 0.5)

    collect_loop(my_arm, "train")
    my_arm.go_home()
