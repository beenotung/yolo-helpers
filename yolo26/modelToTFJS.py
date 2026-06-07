from ultralytics import YOLO

# 1. 加载 YOLO26 模型
model = YOLO('models/yolo26n.pt')

# 2. 导出为 TensorFlow.js 格式
# nms=True/False 取决于你是否希望在导出时包含 NMS 逻辑
# YOLO26本身支持 NMS-free，保持默认或显式指定即可
model.export(format='tfjs', imgsz=640)