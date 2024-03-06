from __future__ import annotations

from collections import namedtuple

BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
Detection = namedtuple("Detection", ["name", "conf", "bbox"])
Match = namedtuple("Match", ["drawer", "handle"])
