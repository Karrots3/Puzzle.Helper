# %% Imports
import pickle


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


with open("./results/list_pieces.pkl", "rb") as f:
    pieces = pickle.load(f)


with open("./results/single_piece.pkl", "wb") as f:
    pickle.dump(list(pieces.values())[0], f)
