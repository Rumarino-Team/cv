# This python file should test with a small video replay the performance of the algorithms in terms
# of latency.

import cv2
import numpy as np
import time
import tracemalloc
from typing import Callable
from .custom_types import DepthImage, Detection
from .detection_core import YOLOModelManager,  calculate_point_3d
import random

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

class MockDetection:
    def __init__(self, cls_number: int, detection_number: int ):
        self.cls_range = range(0,cls_number)

    def generate_detection(self,cls:bool, bbox : bool,  point3d: bool, bbox3d: bool):
        #TODO



class OpenCVManager:
    def __init__(self):
        self.cameras = []
        count = 0
        try:
            while count < 10:
                self.cameras.append(cv2.VideoCapture(count))
                count +=1
        except:
            print(f"A total of {count} cameras")
    def run_cameras(self, visualization_function: Callable| None = None, *visualization_args):
        for idx, camera in enumerate(self.cameras):
            if not camera.isOpened():
                print(f"error opening camera {idx}")
        while True:
            frames = [camera.read()[1] for camera in self.cameras]
            #TODO: We should check if the frame was not read
            for idx in range(self.cameras):
                if visualization_function:
                    visualization_function(*visualization_args);
                cv2.imshow(f"Camera{idx}", frames[idx]);
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        for camera in self.cameras:
            camera.release()
        cv2.destroyAllWindows()


def log_performance(func: Callable, *args ):
    tracemalloc.start()
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    final_time = end - start
    return final_time, current, peak


def benchmark_yolo(yolo_model :str, frame_limit: int, video_path: str | None = None ,mock_data: bool  = False) -> list[float]:
    
    yolo_manager = YOLOModelManager(yolo_model)
    assert video_path and mock_data,  "If video path enable then mock data should be false."
    if mock_data:
        cap = MockVideo(1080, 720, 3, frame_limit, 0)
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


def benchmark_calculate_point_3d(mock_data: bool, detections: list[Detection], depth_frame: DepthImage, camera_intrinsic: list[float], frame_limit: int) :
    """
    This function expect a Detection object that have been filled with the ultralytics
    """
    if mock_data:
        detection_generator = MockDetection(10, 20)
        mock_video = MockVideo(720,480, 1,300,0)
        detections = detection_generator.generate_detection()
    while True:
        current_frame += 1
        ret, depth_frame = cap.read()
        camera_intrinsic = [random.random() for _ in range(4)]
        if not ret or current_frame > frame_limit :
            break    
        bench_result = log_performance(calculate_point_3d, detections, depth_frame, camera_intrinsic) 
        result.append(bench_result)


def camera_map_object_visualization():
    #TODO: 
    pass


if __name__ == "__main__":
    video_path = "no video"
    yolo_path  = "./yolo11n.pt"
    benchmark_yolo(video_path= video_path, yolo_model=yolo_path,frame_limit=40, mock_data=True)

    
    
