from ultralytics import YOLO

model = YOLO("yolov8x.pt")
print(model.info(verbose=True))
print(model.model.args)
