from ultralytics import YOLO

model = YOLO("models/sample.pt")
model.export(format="onnx", imgsz=640, simplify=True, dynamic=False)