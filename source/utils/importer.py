"""
Library open3d is a bit finicky with imports.
One can either import from cuda (for GPU-enables machines) or cpu (for others).
By checking the config, based on the "device" attribute, we can decide which one to import.
For use simply import from utils.importer, i.e. "from utils.importer import Pointcloud".
"""

from utils.recursive_config import Config

_conf = Config()
if _conf["device"] == "cuda":
    from open3d.cuda.pybind.geometry import (
        AxisAlignedBoundingBox,
        PointCloud,
        TriangleMesh,
    )
    from open3d.cuda.pybind.utility import Vector3dVector
else:
    from open3d.cpu.pybind.geometry import (
        AxisAlignedBoundingBox,
        PointCloud,
        TriangleMesh,
    )
    from open3d.cpu.pybind.utility import Vector3dVector
