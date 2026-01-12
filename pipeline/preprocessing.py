"""Image preprocessing pipeline for puzzle pieces."""

from itertools import combinations
from math import sqrt
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

from pipeline.config import DEFAULT_MAX_DIFF_LENGTH_RATIO
from pipeline.models import (
    ContourError,
    Edge,
    EdgeColor,
    EdgeType,
    LoopingList,
    Piece,
)


class PipelineProcessImgParams:
    """Parameters for image processing pipeline."""
    
    def __init__(
        self,
        radius: float = 0.70,
        shift_x: float = 1.05,
        shift_y: float = 0.86,
        thresh: int = 110,
        min_area=200,
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


def _crop_img(
    img: np.ndarray, params: PipelineProcessImgParams, plot: bool = False
) -> np.ndarray:
    """Crop image to center region."""
    radius = params.crop_radius
    shift_x = params.crop_shift_x
    shift_y = params.crop_shift_y
    
    _row, _col = img.shape[:2]
    _center_row, _center_col = _row // 2, _col // 2
    _center_row, _center_col = int(_center_row * shift_y), int(_center_col * shift_x)
    _radius_row, _radius_col = int(_center_row * radius), int(_center_col * radius)
    _radius = min(_radius_row, _radius_col)
    cropped_img = img[
        _center_row - _radius : _center_row + _radius,
        _center_col - _radius : _center_col + _radius,
        :,
    ]
    
    return cropped_img


def _threshold_img(
    img: np.ndarray, params: PipelineProcessImgParams, plot: bool = False
) -> np.ndarray:
    """Apply thresholding to image."""
    thresh = params.th_thresh
    maxvalue = params.th_maxvalue
    
    _, img_otsu = cv.threshold(img, thresh, maxvalue, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    return img_otsu


def _get_contour(
    bw_img: np.ndarray, params: PipelineProcessImgParams, plot: bool = False
) -> np.ndarray | None:
    """Extract contour from binary image."""
    min_area = params.filter_contour_min_area
    
    contours, _ = cv.findContours(bw_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        
        if area > min_area and area != hull_area:
            filtered_contours.append(contour)
    
    # Calculate areas for error reporting
    contour_areas = [cv.contourArea(contour) for contour in filtered_contours]
    
    # Raise exception if we don't have exactly one contour
    if len(filtered_contours) == 0:
        raise ContourError(
            message="No contours found after filtering",
            num_contours=0,
            contour_areas=[],
        )
    elif len(filtered_contours) > 1:
        raise ContourError(
            message=f"Found {len(filtered_contours)} contours, expected exactly 1",
            num_contours=len(filtered_contours),
            contour_areas=contour_areas,
        )
    
    contour = filtered_contours[0]
    return contour


def _get_corner_indices(
    bw_img: np.ndarray,
    contour: np.ndarray,
    params: PipelineProcessImgParams,
    plot: bool = False,
) -> LoopingList:
    """Find corner indices in contour."""
    prominence = params.edges_peaks_prominence
    
    width, height = bw_img.shape[:2]
    
    (_circle_x, _circle_y), _ = cv.minEnclosingCircle(contour)
    centered_contour = contour - np.array([_circle_x, _circle_y])
    
    # Ensure peaks are not at start or end of the distances array
    distances = np.sum(centered_contour**2, axis=2)[:, 0]
    distance_offset = np.argmin(distances)
    shifted_distances = np.concatenate(
        [distances[distance_offset:], distances[:distance_offset]]
    )
    
    peak_indices = [
        (distance_idx + distance_offset) % len(distances)
        for distance_idx in find_peaks(shifted_distances, prominence=prominence)[0]
    ]
    peak_indices.sort()
    
    # Filter corners by rectangle geometry
    def _compute_rectangle_error(indices):
        peaks_coordinates = LoopingList(
            np.take(contour, sorted(list(indices)), axis=0)[:, 0, :]
        )
        
        lengths = [
            sqrt(np.sum((peaks_coordinates[i0] - peaks_coordinates[i1]) ** 2))
            for i0, i1 in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]
        ]
        
        def f_error(a, b):
            return abs(b - a) / (a + b)
        
        return sum(
            [
                f_error(lengths[i], lengths[j])
                for i, j in [(0, 2), (1, 3), (4, 5), (0, 1)]
            ]
        )
    
    # Sort rectangles combinations
    rectangles: list[tuple[float, list[int]]] = []
    for indices in combinations(peak_indices, 4):
        error = _compute_rectangle_error(indices)
        rectangles.append((error, indices))
    
    # Select best rectangle
    error, indices = sorted(rectangles)[0]
    corner_indices = LoopingList(indices)
    
    return corner_indices


def _get_edges(
    bw_img: np.ndarray,
    contour: np.ndarray,
    corner_indices: list[int],
    params: PipelineProcessImgParams,
    plot: bool = False,
) -> list[Edge]:
    """Extract and classify edges from contour."""
    height, width = bw_img.shape[:2]
    contour_center = np.mean(contour[:, 0, :], axis=0)
    edge_type = params.edge_type
    edge_color = params.edge_color
    flat_threshold = params.flat_threshold
    
    # Extract edges between corners
    corner_indices.sort()
    _pki = corner_indices
    edges = [
        np.concatenate(
            [
                contour[_pki[-1] :, :, :],  # from pki[-1] to end
                contour[: _pki[0], :, :],  # from 0 to pki[0]
            ],
            axis=0,
        ),
        contour[_pki[0] : _pki[1], :, :],
        contour[_pki[1] : _pki[2], :, :],
        contour[_pki[2] : _pki[3], :, :],
    ]
    
    def classify_edge(edge, piece_center, flat_thresh):
        """Classify edge as flat, man, or woman."""
        pts = edge[:, 0, :]
        p0, p1 = pts[0], pts[-1]
        
        # Vector of the edge baseline
        edge_vec = p1 - p0
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-6:
            return edge_type.flat
        
        # Unit normal to the edge
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        
        # Signed distance of each point to baseline
        distances = []
        for p in pts:
            v = p - p0
            signed_dist = np.dot(v, normal)
            distances.append(signed_dist)
        distances = np.array(distances)
        
        max_dev = np.max(np.abs(distances))
        
        # Flat check
        if max_dev < flat_thresh:
            return edge_type.flat
        
        # Determine sign of deviation at the most extreme point
        idx_max = np.argmax(np.abs(distances))
        extreme_sign = np.sign(distances[idx_max])
        
        # Decide male/female based on whether extreme point is closer/farther than center
        vec_center = piece_center - p0
        center_side = np.sign(np.dot(vec_center, normal))
        
        if extreme_sign == center_side:
            return edge_type.woman
        else:
            return edge_type.man
    
    edges_type = [classify_edge(edge, contour_center, flat_threshold) for edge in edges]
    edges_color = [edge_color[et.name].value for et in edges_type]
    
    return [
        Edge(
            edge_id=i,
            edge_type=edges_type[i],
            edge_color=edges_color[i],
            edge_contour=edges[i],
        )
        for i in range(4)
    ]


def process_image(
    path: Path | str,
    params: PipelineProcessImgParams | None = None,
    plot: bool = False,
) -> Piece:
    """
    Process a single image to extract puzzle piece information.
    
    Args:
        path: Path to image file
        params: Processing parameters (uses defaults if None)
        plot: Whether to plot intermediate results
        
    Returns:
        Piece object with extracted information
    """
    if params is None:
        params = PipelineProcessImgParams()
    
    bgr_img = cv.imread(str(path))
    if bgr_img is None:
        raise ValueError(f"Could not load image from {path}")
    
    bgr_img = _crop_img(bgr_img, params, plot)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    bw_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    
    img = _threshold_img(bw_img, params, plot)
    contour = _get_contour(img, params, plot)
    corner_indices = _get_corner_indices(bw_img, contour, params, plot)
    edges = _get_edges(bw_img, contour, corner_indices, params, plot)
    
    # Extract piece_id from filename if possible
    piece_id = 0
    if isinstance(path, (Path, str)):
        try:
            piece_id = int(Path(path).stem)
        except ValueError:
            pass
    
    piece = Piece(piece_id=piece_id, rgb_img=rgb_img, contour=contour, edge_list=edges)
    return piece
