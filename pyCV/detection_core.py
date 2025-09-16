#!/usr/bin/env python3
from sympy import rotations
import numpy as np
from ultralytics import YOLO
from .custom_types import BoundingBox3D, CameraIntrinsics, DepthImage, Detection, Rotation3D, Point3D
import cv2
import torch
import glob

third_parties = glob.glob("auv/third_party/*")
submodules_names = [module_path.split("/")[2] for module_path in third_parties]
# This is optional because its not the only way we can  calculate Depth Images. So I dont want to force people to install it if they are not using it.
if "depth_anything_v2" in submodules_names:
    from ..third_party.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
    class DepthAnythingManager:
        def __init__(self, model_path: str):
            self.device ='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            self.model_config ={'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
            self.model = DepthAnythingV2(**model_configs)
            self.model.load_state_dict(torch.load(model_path), map_location="cpu")
            self.model.to(self.device).eval()

        def detect(self, image: np.ndarray) -> DepthImage:
            return self.model.infer_image(image).numpy()





def ros_img_to_cv2(msg, encoding="bgr8") -> np.ndarray:
    """
    Convert a ROS sensor_msgs/Image message to an OpenCV numpy array.
    :param msg: ROS Image message
    :param encoding: Desired encoding ("bgr8", "mono8", "mono16", "32FC1")
    :return: OpenCV numpy array
    """
    if encoding not in ["bgr8", "mono8", "mono16", "32FC1"]:
        raise ValueError(f"Unsupported encoding: {encoding}")

    dtype_map = {
        "bgr8": np.uint8,
        "mono8": np.uint8,
        "mono16": np.uint16,
        "32FC1": np.float32,
    }

    dtype = dtype_map[encoding]

    # Calculate expected array size based on encoding and dimensions
    channels = 3 if encoding == "bgr8" else 1
    expected_size = msg.height * msg.width * channels
    actual_size = len(msg.data)

    if actual_size != expected_size:
        # Try to infer correct dimensions based on actual data size
        if encoding == "bgr8" and actual_size % 3 == 0:
            total_pixels = actual_size // 3
            # Try to determine if width is correct but height is wrong
            if total_pixels % msg.width == 0:
                corrected_height = total_pixels // msg.width
                img_array = np.frombuffer(msg.data, dtype=dtype).reshape(
                    (corrected_height, msg.width, 3)
                )
                return img_array

    # Convert the byte data to a NumPy array
    img_array = np.frombuffer(msg.data, dtype=dtype)

    try:
        # Reshape based on image dimensions and encoding
        if encoding == "bgr8":
            # Use step value if available for proper alignment
            if msg.step > 0 and msg.step >= msg.width * 3:
                img_array = img_array.reshape((msg.height, msg.width, 3))
            else:
                img_array = np.reshape(img_array, (msg.height, msg.width, 3))
        else:
            # Single-channel
            img_array = np.reshape(img_array, (msg.height, msg.width))

        return img_array

    except Exception as e:
        raise ValueError(
            f"Reshape failed: {e}. Image info: height={msg.height}, width={msg.width}, step={msg.step}, encoding={msg.encoding}"
        )



class YOLOModelManager:
    def __init__(self, model_path: str ):
        self.model : YOLO  = YOLO(model_path)
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
                    result_list.append(
                        Detection(x1, y1, x2, y2, cls_w, conf, 0, None)
                    )
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
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        )
        if depth_image is not None:
            x_min_int = int(x_min)
            x_max_int = int(x_max)
            y_min_int = int(y_min)
            y_max_int = int(y_max)

            # Extract the depth values within the bounding box
            bbox_depth = depth_image[y_min_int:y_max_int, x_min_int:x_max_int]
            if bbox_depth.size > 0:
                mean_depth = float(np.nanmean(bbox_depth))  # type: ignore
                if not np.isnan(mean_depth):
                    fx, fy, cx, cy = camera_intrinsic
                    z = mean_depth
                    detection.depth = z

                    x_center = (x_min + x_max) / 2.0
                    y_center = (y_min + y_max) / 2.0
                    x = (x_center - cx) * z / fx
                    y = (y_center - cy) * z / fy

                    detection.point = Point3D(x=x, y=y, z=z)
                else:
                    detection.point = Point3D(x=0, y=0, z=0)
                    detection.depth = 0
            else:
                detection.point = Point3D(x=0, y=0, z=0)
                detection.depth = 0


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
    :param detections: List of Detection objects to transform
    :param imu_point: IMU position in global frame
    :param imu_rotation: IMU orientation as quaternion
    """
    transform_matrix = quaternion_to_transform_matrix(imu_rotation)
    transform_matrix[0:3, 3] = [imu_point.x, imu_point.y, imu_point.z]

    for detection in detections:
        if detection.point is not None:
            point_homogeneous = np.array(
                [detection.point.x, detection.point.y, detection.point.z, 1.0]
            )
            point_global = np.dot(transform_matrix, point_homogeneous)
            detection.point = Point3D(
                x=point_global[0], y=point_global[1], z=point_global[2]
            )

def map_3d_bounding_box(detections: list[Detection], box_map: dict[str, tuple[float , float , float]]):
    """
    This function fills the bbox3d parameter of the detection object.
    It will raise an error if we didnt previously calculated the 3d point.
    """
    for detection in detections:
        assert not detection.point, "Detection didnt had filled the 3d point attribute."
        cls = str(detection.cls)
        bbox3d = BoundingBox3D(Rotation3D(0,0,0,0), box_map[cls][0], box_map[cls][1], box_map[cls][2])
        detection.bbox_3d = bbox3d
        

