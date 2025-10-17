MIN_AREA_PIECE = 100
MAX_AREA_PIECE = 7000
MIN_PERIMETER_PIECE = 20
MAX_PERIMETER_PIECE = 8000

MIN_LEN_EDGE = 500
SAMPLE_POINTS = 1000
MAX_SCORE_MSE_DISTANCE = 20

FLAT_EDGE = 0
MALE_EDGE = 1
FEMALE_EDGE = -1
COLORS_GENDER = {FLAT_EDGE: "red", MALE_EDGE: "green", FEMALE_EDGE: "blue"}


class Edge:
    def __init__(self, edge_id, contour, norm_contour, length, approx, edge_type):
        self.edge_id = edge_id
        self.contour = contour
        self.normalized_contour = norm_contour
        self.straight_length = length
        self.approx = approx
        self.edge_type = edge_type


class Piece:
    def __init__(
        self,
        piece_id,
        img_thresh,
        contour,
        area,
        perimeter,
        bounding_rect,
        min_area_rect,
        box_points,
        hull,
        solidity,
        aspect_ratio,
        extent,
        edge_list: list[Edge] = [],
    ):
        self.piece_id = piece_id
        self.img_thresh = img_thresh
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
        self.edge_list: list[Edge] = edge_list


class LoopingList(list):
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)
