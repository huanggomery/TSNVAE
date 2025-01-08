from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load a pretrained model
results = model.train(data="datasets/data.yaml", epochs=100, imgsz=640)