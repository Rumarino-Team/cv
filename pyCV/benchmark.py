# This python file should test with a small video replay the performance of the algorithms in terms
# of latency.

from unittest import mock
import cv2
import numpy as np
import time
import tracemalloc
import sys
from typing import Callable
from pyCV.detection_core import YOLOModelManager, transform_to_global, calculate_point_3d

def log_performance(func: Callable, *args ):
    tracemalloc.start()
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    final_time = end - start
    return final_time, current, peak
class MockVideo:
    def __init__(self,height: int , width: int, channels: int,frames: int, seed: int):
        self.height : int = height
        self.width: int = width
        self.channels: int  = channels
        self.frames: int = frames
        self.frame_counter: int = 0
        np.random.seed(seed)

    def read(self)-> tuple[bool, np.ndarray] :
        self.frame_counter += 1
        image = np.random.randn(self.height, self.width, self.channels)
        ret = True
        if self.frame_counter > self.frames:
            ret = False
        return (ret , image)



def benchmark_yolo(yolo_model :str, frame_limit: int, video_path: str | None = None ,mock_data: bool  = False) -> list[float]:
    
    yolo_manager = YOLOModelManager(yolo_model)
    assert video_path and mock_data,  "If video path enable then mock data should be false."
    if mock_data:
        cap = MockVideo(1080, 720, 3, 500, 0)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file")
            exit()

    result = []
    current_frame = 0
    while True:
        current_frame += 1
        ret, frame = cap.read()
        if not ret or current_frame > frame_limit :
            break    
        bench_result = log_performance(yolo_manager.detect, frame) 
        result.append(bench_result)
    return result


def benchmark_transform_to_global():
    pass



if __name__ == "__main__":
    video_path = "no video"
    yolo_path  = "./yolo11n.pt"
    benchmark_yolo(video_path= video_path, yolo_model=yolo_path,frame_limit=40, mock_data=True)

    
    
