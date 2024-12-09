import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np

import arm

usb_pos = arm.usb_pos


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

        arm.catch_random()

        # 记录视觉、触觉、位姿
        tac1, tac2 = arm.get_tactile()
        cv2.imwrite("data/{}/{}/I_z_left.jpg".format(mode, epoch), tac1)
        cv2.imwrite("data/{}/{}/I_z_right.jpg".format(mode, epoch), tac2)
        img = arm.get_image()
        cv2.imwrite("data/{}/{}/0.jpg".format(mode, epoch), img)
        pos_cur = arm.get_pos() - np.array([usb_pos[0], usb_pos[1], usb_pos[2], -180.0, 0.0, 0.0])
        pos_all = np.concatenate((pos_all, pos_cur[np.newaxis, :]), axis=0)
        # joint = arm.get_joint_states()
        # joint_states = np.concatenate((joint_states, joint[np.newaxis, :]), axis=0)

        arm.arm.set_mode(5)
        arm.arm.set_state(state=0)

        for i in range(20):
            pos_action, rot_action = arm.step_random()
            action = np.concatenate([pos_action, rot_action])
            action_all = np.concatenate((action_all, action[np.newaxis, :]), axis=0)

            img = arm.get_image()
            cv2.imwrite("data/{}/{}/{}.jpg".format(mode, epoch, i+1), img)
            pos_cur = arm.get_pos() - np.array([usb_pos[0], usb_pos[1], usb_pos[2], -180.0, 0.0, 0.0])
            pos_all = np.concatenate((pos_all, pos_cur[np.newaxis, :]), axis=0)
            # joint = arm.get_joint_states()
            # joint_states = np.concatenate((joint_states, joint[np.newaxis, :]), axis=0)

        np.save("data/{}/{}/pos.npy".format(mode, epoch), pos_all)
        np.save("data/{}/{}/joint.npy".format(mode, epoch), joint_states)
        np.save("data/{}/{}/action.npy".format(mode, epoch), action_all)

        arm.arm.set_mode(0)
        arm.arm.set_state(state=0)

        arm.loose()
        print("已采集轨迹数：{}".format(epoch))
        epoch += 1


if __name__ == "__main__":
    my_arm = arm.Arm(arm.ip, 0.5)

    collect_loop(my_arm, "train")
    my_arm.go_home()
