from enum import Enum

import numpy as np


class EdgeType(Enum):
    flat = 0
    man = 1
    woman = -1


class EdgeColor(Enum):
    flat = (0, 255, 0)
    man = (0, 0, 255)
    woman = (255, 0, 0)


class Edge:
    def __init__(
        self,
        edge_id: int,
        edge_type: str,
        edge_color: tuple[int, int, int],
        edge_contour: np.ndarray,
    ):
        self.edge_id = edge_id
        self.edge_type = edge_type
        self.edge_color = edge_color
        self.edge_contour = edge_contour


class Piece:
    def __init__(
        self,
        piece_id: int,
        rgb_img: np.ndarray,
        contour: np.ndarray,
        edge_list: list[Edge],
    ):
        self.piece_id = piece_id
        self.rgb_img = rgb_img
        self.contour = contour
        self.edge_list = edge_list


class PipelineProcessImgParams:
    def __init__(
        self,
        radius: float = 0.6,
        shift_x: float = 1.1,
        shift_y: float = 0.86,
        thresh: int = 110,
        min_area=20,
        max_area=10000,
        max_value=255,
        min_perimeter=100,
        max_perimeter=1000,
        prominence=10500,
        flat_threshold=20,
        diff_length=0.01,
    ):
        self.crop_radius = radius
        self.crop_shift_x = shift_x
        self.crop_shift_y = shift_y
        self.th_thresh = thresh
        self.th_maxvalue = max_value
        self.filter_contour_min_area = min_area
        self.filter_contour_max_area = max_area
        self.filter_contour_min_perimeter = min_perimeter
        self.filter_contour_max_perimeter = max_perimeter
        self.edges_peaks_prominence = prominence
        self.edge_type = EdgeType
        self.edge_color = EdgeColor
        self.flat_threshold = flat_threshold
        self.max_diff_length = diff_length


class LoopingList(list):
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)
