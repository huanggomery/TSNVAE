from ultralytics import YOLO

imgs = []
for i in range(20):
    imgs.append("datasets/images/val/{}.jpg".format(101+i))
model = YOLO("runs/detect/train/weights/best.pt")
model.predict(imgs, save=True, imgsz=640)