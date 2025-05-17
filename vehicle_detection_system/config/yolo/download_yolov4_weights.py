import urllib.request
import os

def download_yolov4_weights(dest_path):
    url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    print(f"Downloading YOLOv4 weights from {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded to {dest_path}")

if __name__ == "__main__":
    weights_path = os.path.join(os.path.dirname(__file__), "yolov4.weights")
    download_yolov4_weights(weights_path)
