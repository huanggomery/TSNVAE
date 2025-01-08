import cv2
import numpy as np
import json

JSON_PATH = "data/jsons/val"
DEST_PATH = "data/labels/val"
W, H = 640, 480

for i in range(101, 121):
    filename = JSON_PATH + "/{}.json".format(i)
    savename = DEST_PATH + "/{}.txt".format(i)
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)

    shapes = data.get('shapes')

    lines = []

    for index, shape in enumerate(shapes):
        label = shape.get('label')
        points = shape.get('points')
        p1 = (points[0][0], points[0][1])
        p2 = (points[1][0], points[1][1])

        x_center = ((p1[0] + p2[0]) / 2) / W
        width = (p2[0] - p1[0]) / W
        y_center = ((p1[1] + p2[1]) / 2) / H
        height = (p2[1] - p1[1]) / H

        if label == "hole":
            cls = 0
        elif label == "usb":
            cls = 1
        else:
            raise ValueError("label {} is not available".format(label))

        if index > 0:
            lines.append('\n')
        lines.append("{} {} {} {} {}".format(cls, x_center, y_center, width, height))

    with open(savename, 'w') as file:
        file.writelines(lines)
