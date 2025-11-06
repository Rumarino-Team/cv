from typing import Callable
import numpy as np
from ultralytics import YOLO
from .custom_types import (
    BoundingBox3D,
    CameraIntrinsics,
    DepthImage,
    Detection,
    MapState,
    MapObject,
    Rotation3D,
    Point3D,
)
import torch
import math
import os
import sys
from collections import deque

# Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
third_party_path = os.path.join(project_root, "third_party")

if third_party_path not in sys.path:
    sys.path.insert(0, third_party_path)

depth_anything_available = os.path.exists(
    os.path.join(third_party_path, "depth_anything_v2")
)

# This is optional because its not the only way we can  calculate Depth Images. So I dont want to force people to install it if they are not using it.
if depth_anything_available:
    from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

    class DepthAnythingManager:
        def __init__(self, model_path: str):
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.model_config = {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }
            self.model = DepthAnythingV2(**self.model_config)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.to(self.device).eval()

        def detect(self, image: np.ndarray) -> DepthImage:
            return self.model.infer_image(image)


class YOLOModelManager:
    def __init__(self, model_path: str):
        self.model: YOLO = YOLO(model_path)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run YOLO object detection on an image.
        :param image: Input image as numpy array
        :return: List of Detection objects
        """
        result_list = []
        results = self.model(image)
        for result in results:
            if hasattr(result, "boxes"):
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    conf = float(box.conf.cpu().numpy()[0])
                    cls_w = int(box.cls.cpu().numpy()[0])
                    result_list.append(Detection(x1, y1, x2, y2, cls_w, conf, 0, None))
        return result_list


def calculate_point_3d(
    detections: list[Detection],
    depth_image: DepthImage,
    camera_intrinsic: CameraIntrinsics,
):
    """
    Calculate 3D points for detections using depth information.
    :param detections: List of Detection objects to update
    :param depth_image: Depth image as numpy array
    :param camera_intrinsic: Camera intrinsic parameters (fx, fy, cx, cy)
    """
    for detection in detections:
        x_min, y_min, x_max, y_max = (
            detection._x1,
            detection._y1,
            detection._x2,
            detection._y2,
        )
        if depth_image is not None:
            x_min_int = int(x_min)
            x_max_int = int(x_max)
            y_min_int = int(y_min)
            y_max_int = int(y_max)

            # Extract the depth values within the bounding box
            bbox_depth = depth_image[y_min_int:y_max_int, x_min_int:x_max_int]
            if bbox_depth.size > 0:
                median_depth = float(np.nanmedian(bbox_depth))  # type: ignore
                if not np.isnan(median_depth):
                    # Clamp depth to reasonable range (0.3m to 10m for indoor scenes)
                    # This prevents extreme values from depth estimation errors
                    z = np.clip(median_depth, 0.3, 10.0)
                    
                    fx, fy, cx, cy = camera_intrinsic
                    x_center = (x_min + x_max) / 2.0
                    y_center = (y_min + y_max) / 2.0
                    x = (x_center - cx) * z / fx
                    y = (y_center - cy) * z / fy

                    detection._point = Point3D(x=x, y=y, z=z)
                    detection._distance = z
                else:
                    detection._point = Point3D(x=0, y=0, z=0)
                    detection._distance = 0
            else:
                detection._point = Point3D(x=0, y=0, z=0)
                detection._distance = 0


def quaternion_to_transform_matrix(rotation: Rotation3D) -> np.ndarray:
    """
    Convert quaternion rotation to 4x4 transformation matrix.
    :param rotation: Rotation3D object with quaternion components
    :return: 4x4 transformation matrix
    """
    w, x, y, z = rotation.w, rotation.x, rotation.y, rotation.z
    rotation_matrix = np.array(
        [
            [
                1 - 2 * y**2 - 2 * z**2,
                2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w,
            ],
            [
                2 * x * y + 2 * z * w,
                1 - 2 * x**2 - 2 * z**2,
                2 * y * z - 2 * x * w,
            ],
            [
                2 * x * z - 2 * y * w,
                2 * y * z + 2 * x * w,
                1 - 2 * x**2 - 2 * y**2,
            ],
        ]
    )

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    return transform_matrix


def transform_to_global(
    detections: list[Detection],
    imu_point: Point3D,
    imu_rotation: Rotation3D,
):
    """
    Transform detection points from camera frame to global frame.
    Camera frame: X=right, Y=down, Z=forward (OpenCV convention)
    World frame: X=forward, Y=left, Z=up (ORB-SLAM convention)
    
    :param detections: List of Detection objects to transform
    :param imu_point: IMU position in global frame
    :param imu_rotation: IMU orientation as quaternion
    """
    transform_matrix = quaternion_to_transform_matrix(imu_rotation)
    transform_matrix[0:3, 3] = [imu_point.x, imu_point.y, imu_point.z]

    for detection in detections:
        if detection._point is not None:
            # Convert from camera frame (X=right, Y=down, Z=forward) 
            # to a frame compatible with ORB-SLAM world frame
            # Camera Z (forward) -> World X
            # Camera -X (left) -> World Y  
            # Camera -Y (up) -> World Z
            cam_x, cam_y, cam_z = detection._point.x, detection._point.y, detection._point.z
            
            # Apply coordinate transformation
            point_camera_adjusted = np.array([cam_z, -cam_x, -cam_y, 1.0])
            
            # Transform to world frame
            point_global = np.dot(transform_matrix, point_camera_adjusted)
            detection._point = Point3D(
                x=point_global[0], y=point_global[1], z=point_global[2]
            )


def map_3d_bounding_box(
    detections: list[Detection], 
    box_map: dict[str, tuple[float, float, float]],
    class_names: list[str]
):
    for detection in detections:
        assert detection._point is not None, "Detection doesn't have a filled 3D point attribute."
        
        cls_name = class_names[detection._cls] if detection._cls < len(class_names) else f"class_{detection._cls}"
        
        if cls_name not in box_map:
            continue
        
        # Create 3D bounding box with the dimensions from box_map
        # Use identity quaternion (no rotation): x=0, y=0, z=0, w=1
        bbox3d = BoundingBox3D(
            Rotation3D(0, 0, 0, 1), box_map[cls_name][0], box_map[cls_name][1], box_map[cls_name][2]
        )
        detection._bbox_3d = bbox3d


def calculate_distance(p1: Point3D, p2: Point3D) -> float:
    return math.dist((p1.x, p1.y, p1.z), (p2.x, p2.y, p2.z))

def calculate_median_detection(history: list[Detection], key_func: Callable):
    sorted_detection = sorted(history, key=key_func)
    mid = len(sorted_detection) //2
    return sorted_detection[mid]

def map_objects(
    state: MapState,
    class_name: list[str],
    box_map: dict[str, tuple[float, float, float]],
    detections: list[Detection],
    camera_intrinsic: CameraIntrinsics,
    depth_image: DepthImage,
    imu_point: Point3D,
    imu_rotation: Rotation3D,
    history_buffer: dict[int, deque[Detection]],
    expected_frequencies: dict[str, int] | None = None
):
    """
    Map detected objects to a global 3D map with frequency-based eviction.
    
    :param expected_frequencies: Dictionary specifying expected count per class (e.g., {'buoy': 3, 'gate': 1}).
                                 If None, no eviction is performed.
    """
    calculate_point_3d(detections, depth_image, camera_intrinsic)
    transform_to_global(detections, imu_point, imu_rotation)
    map_3d_bounding_box(detections, box_map, class_name)

    # First-time detections
    for detection in detections:
        if detection._cls < 0 or detection._cls >= len(class_name):
            continue
        cls_name = class_name[detection._cls]
        if state.obj_frequencies.get(cls_name, 0) == 0:
            new_object = MapObject(
                track_id=state.id_counter,
                cls=detection._cls,
                conf=detection._conf,
                point=detection._point,
                bbox_3d=detection._bbox_3d,
            )
            state.objects.append(new_object)
            state.obj_frequencies[cls_name] = 1
            state.id_counter += 1

    # Now match/update existing objects
    # Use a larger threshold to handle depth estimation noise and static camera scenarios
    new_object_threshold = 100.0  # Increased from 0.5 to 10.0 meters
    eviction_size = 5
    matched_detections = set()  # Track which detections have been matched

    for obj in state.objects:
        best_object_match_distance = math.inf
        best_detection = None
        best_detection_idx = None

        for det_idx, detection in enumerate(detections):
            if detection._cls < 0 or detection._cls >= len(class_name):
                continue
            
            # Skip if this detection was already matched to another object
            if det_idx in matched_detections:
                continue

            # Only consider detections of the same class
            if obj.cls != detection._cls:
                continue

            # Skip detections with invalid 3D points
            if detection._point is None or obj.point is None:
                continue

            object_detection_distance = calculate_distance(obj.point, detection._point)

            # Find the closest detection to this object
            if object_detection_distance < best_object_match_distance:
                best_object_match_distance = object_detection_distance
                best_detection = detection
                best_detection_idx = det_idx

        # If we found a close match, update the object
        if best_detection is not None and best_object_match_distance < new_object_threshold:
            if len(history_buffer[obj.track_id]) == eviction_size:
                history_buffer[obj.track_id].popleft()
            history_buffer[obj.track_id].append(best_detection)
            #Smothing Techniques
            median_map_object = calculate_median_detection(history_buffer[obj.track_id], lambda x: x._distance )
            obj.conf = median_map_object._conf
            obj.point = median_map_object._point
            obj.bbox_3d = median_map_object._bbox_3d
            obj.update_count += 1  # Increment update count
            matched_detections.add(best_detection_idx)

    # Add unmatched detections as new objects
    for det_idx, detection in enumerate(detections):
        if det_idx not in matched_detections:
            if detection._cls < 0 or detection._cls >= len(class_name):
                continue
            
            cls_name = class_name[detection._cls]
            new_object = MapObject(
                track_id=state.id_counter,
                cls=detection._cls,
                conf=detection._conf,
                point=detection._point,
                bbox_3d=detection._bbox_3d,
            )
            state.objects.append(new_object)
            state.obj_frequencies[cls_name] = state.obj_frequencies.get(cls_name, 0) + 1
            state.id_counter += 1

    # Eviction logic: Keep only the most frequently updated objects per class
    if expected_frequencies is not None:
        # Group objects by class
        objects_by_class: dict[int, list[MapObject]] = {}
        for obj in state.objects:
            if obj.cls not in objects_by_class:
                objects_by_class[obj.cls] = []
            objects_by_class[obj.cls].append(obj)
        
        # For each class, keep only the expected number of most frequently updated objects
        objects_to_keep = []
        for cls_id, objs in objects_by_class.items():
            if cls_id < 0 or cls_id >= len(class_name):
                # Keep all objects of unknown classes
                objects_to_keep.extend(objs)
                continue
            
            cls_name = class_name[cls_id]
            expected_count = expected_frequencies.get(cls_name, len(objs))
            
            if len(objs) <= expected_count:
                # Keep all objects if we haven't exceeded the expected count
                objects_to_keep.extend(objs)
            else:
                # Sort by update_count (descending) and keep the top N
                sorted_objs = sorted(objs, key=lambda x: x.update_count, reverse=True)
                kept = sorted_objs[:expected_count]
                evicted = sorted_objs[expected_count:]
                
                objects_to_keep.extend(kept)
                
                # Clean up history buffers for evicted objects
                for evicted_obj in evicted:
                    if evicted_obj.track_id in history_buffer:
                        del history_buffer[evicted_obj.track_id]
                
                # Update frequency count
                state.obj_frequencies[cls_name] = expected_count
        
        # Update the objects list
        state.objects = objects_to_keep

    # Note: We don't need to add map objects to history_buffer here 
    # as they are already being managed in the matching logic above






