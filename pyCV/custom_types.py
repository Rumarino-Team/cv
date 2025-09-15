from dataclasses import dataclass
import numpy as np

CameraIntrinsics = tuple[float, float, float, float]
DepthImage = np.ndarray
@dataclass
class Point3D:
    x: float
    y: float
    z: float


@dataclass
class Rotation3D:
    x: float
    y: float
    z: float
    w: float

@dataclass
class BoundingBox3D:
    rotation: Rotation3D
    width: float
    height: float
    length: float
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: float
    depth: float = 0
    point: Point3D | None = None
    bbox_3d: BoundingBox3D| None  = None

