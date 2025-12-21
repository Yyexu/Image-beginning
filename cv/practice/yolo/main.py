from ultralytics import YOLO


def train_model():
    # 加载模型
    model = YOLO('yolo11n.yaml')

    # 开始训练，确保这行代码在 __main__ 保护下
    results = model.train(data='coco8.yaml', epochs=5)
    # 验证
def val_model():
    model = YOLO("yolo11n.pt")  # load an official model
    metrics = model.val(data='coco8.yaml')  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list containing mAP50-95 for each category

def predict_model():
    model = YOLO("yolo11n-seg.pt")

    results = model("https://ultralytics.com/images/bus.jpg")  # results list

    # View results
    for r in results:
        print(r.masks) # print the Boxes object containing the detection bounding boxes

def track_model():
    model = YOLO("yolo11n.pt")
    results = model.track(source="0", conf=0.3, iou=0.5, show=True)

if __name__ == '__main__':
    # 在 Windows 上使用多进程训练，必须写在这里面
    track_model()