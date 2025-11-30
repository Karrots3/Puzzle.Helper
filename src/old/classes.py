from enum import Enum

import numpy as np

DEBUG = False

MIN_AREA_PIECE = 100
MAX_AREA_PIECE = 7000

MIN_PERIMETER_PIECE = 20
MAX_PERIMETER_PIECE = 8000

MIN_LEN_EDGE = 500
SAMPLE_POINTS = 1000
MAX_SCORE_MSE_DISTANCE = 20


class EdgeType(Enum):
    FLAT = 0
    MALE = 1
    FEMALE = -1


class EdgeColorType(Enum):
    FLAT = "green"
    MALE = "blue"
    FEMALE = "red"


class Edge:
    def __init__(self, edge_id, contour, norm_contour, length, approx, edge_type):
        self.edge_id = edge_id
        self.contour = contour
        self.normalized_contour = norm_contour
        self.straight_length = length
        self.approx = approx
        self.edge_type = edge_type
        self.edge_color_type = EdgeColorType[edge_type.name].value


class Piece:
    def __init__(
        self,
        piece_id: int,
        bw_thresh_fixed: np.ndarray,
        bw_thresh_otsu: np.ndarray,
        img_tresh: np.ndarray,
        contour: np.ndarray,
        area: float,
        perimeter: float,
        bounding_rect: tuple[int, int, int, int],
        min_area_rect: tuple[int, int, int, int],
        box_points: np.ndarray,
        hull: np.ndarray,
        solidity: float,
        aspect_ratio: float,
        extent: float,
        edge_list: list[Edge],
    ):
        self.piece_id = piece_id
        self.bw_thresh_fixed = bw_thresh_fixed
        self.bw_thresh_otsu = bw_thresh_otsu
        self.img_tresh = img_tresh
        self.contour = contour
        self.area = area
        self.perimeter = perimeter
        self.bounding_rect = bounding_rect
        self.min_area_rect = min_area_rect
        self.box_points = box_points
        self.hull = hull
        self.solidity = solidity
        self.aspect_ratio = aspect_ratio
        self.extent = extent
        self.edge_list = edge_list

    def assert_piece(self):
        assert self.img_org is not None, "Image is not loaded"
        assert self.bw_thresh_fixed is not None, "Threshold image is not loaded"
        assert self.bw_thresh_otsu is not None, "Threshold image is not loaded"
        assert self.contour is not None, "Contour is not loaded"
        assert self.area is not None, "Area is not loaded"
        assert self.perimeter is not None, "Perimeter is not loaded"
        assert self.bounding_rect is not None, "Bounding rect is not loaded"
        assert self.min_area_rect is not None, "Min area rect is not loaded"
        assert self.box_points is not None, "Box points are not loaded"
        assert self.hull is not None, "Hull is not loaded"
        assert self.solidity is not None, "Solidity is not loaded"
        assert self.aspect_ratio is not None, "Aspect ratio is not loaded"
        assert self.extent is not None, "Extent is not loaded"
        assert len(self.edge_list) > 0, "Edge list is not loaded"


class LoopingList(list):
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)
