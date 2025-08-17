import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.common import Piece, LoopingList, Edge
from scipy.signal import find_peaks
import itertools
import math

## Compute rectangle
def _compute_rectangle_error(piece, indices):
    # get coordinates of corners
    corners = LoopingList(
        np.take(piece.contour, sorted(list(indices)), axis=0)[:, 0, :]
    )
    # compute the side lengths and diagonal lengths
    lengths = [
        math.sqrt(np.sum((corners[i0] - corners[i1]) ** 2))
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

# Extract edges
def _sub_contour(c, idx0, idx1):
    if idx1 > idx0:
        return c[idx0:idx1]
    else:
        return np.concatenate([c[idx0:], c[:idx1]])

def _transform_point(point, transform):
    matrix, translate = transform
    return (cv2.transform(np.array([[point]]), matrix) + translate)[0, 0]

def _transform_contour(contour, transform):
    matrix, translate = transform
    return cv2.transform(contour, matrix) + translate

def _get_transform(center, x, y, degrees):
    matrix = cv2.getRotationMatrix2D(center, degrees, 1)
    translate = (x, y) - center
    return (matrix, translate)

def _get_contour_transform(contour, idx, x, y, degrees):
    return _get_transform(contour[idx][0], x, y, degrees)


def set_edges(piece: Piece, find_peak_prominence=500, plot=False) -> Piece:
    (cx, cy), cr = cv2.minEnclosingCircle(piece.int_contour)
    centered_contour = piece.int_contour - np.array([cx, cy])

    # ensure peaks are not at start or end of the distances array
    distances = np.sum(centered_contour**2, axis=2)[:, 0]
    distance_offset = np.argmin(distances)
    shifted_distances = np.concatenate(
        [distances[distance_offset:], distances[:distance_offset]]
    )

    # find peak distances
    peak_indices = [
        (distance_idx + distance_offset) % len(distances)
        for distance_idx in find_peaks(
            shifted_distances, prominence=find_peak_prominence
        )[0]
    ]
    peak_indices.sort()

    piece.update(
        center=np.array([cx, cy]),
        peak_indices=LoopingList(peak_indices),
    )

    # if less than 4 peaks, remove piece
    if len(peak_indices) < 4:
        print("Removing piece", piece.name)
        plt.figure()
        plt.title(f"piece {piece.name} area {piece.area}")
        plt.plot(piece.contour[:, :, 0], -piece.contour[:, :, 1])
        plt.axis('equal')
        plt.show()
        return Piece(
            name="remove",
        )

    if plot:
        plt.figure()
        plt.title(f"piece {piece.name}")
        plt.plot(piece.contour[:, :, 0], -piece.contour[:, :, 1])
        plt.plot(
            piece.contour[peak_indices, :, 0],
            -piece.contour[peak_indices, :, 1],
            marker='o',
            ls = '',
            c='red',
        )
        peak_coords = piece.contour[peak_indices, :, 0:2]
        plt.axis('equal')
        plt.show()


    # form a good rectangle with peak indices
    rectangles = []  # list of (score, [indices])
    for indices in itertools.combinations(peak_indices, 4):
        error = _compute_rectangle_error(piece, indices)
        rectangles.append((error, indices))

    error, indices = sorted(rectangles)[0]
    
    piece.update(
        rectangle_error=error,
        corner_indices=LoopingList(indices)
    )

    if plot:
        plt.figure()
        plt.title(piece.name)
        plt.plot(piece.contour[:, :, 0], -piece.contour[:, :, 1])
        plt.plot(
            piece.contour[piece.corner_indices, :, 0],
            -piece.contour[piece.corner_indices, :, 1],
            marker='o',
            ls = '',
            c='blue',
        )
        plt.axis('equal')
        plt.show()

    
    edges = LoopingList()
    for quarter in range(4):
        idx0 = piece.corner_indices[quarter]
        idx1 = piece.corner_indices[quarter + 1]
        p0 = piece.contour[idx0][0]
        p1 = piece.contour[idx1][0]
        # normalize the contour: first point at (0, 0), last point at (X, 0)
        dx, dy = p1 - p0
        straight_length = math.sqrt(dx**2 + dy**2)
        angle_degrees = math.degrees(math.atan2(dy, dx))

        transform = _get_contour_transform(piece.contour, idx0, 0, 0, angle_degrees)
        normalized_piece_contour = _transform_contour(piece.contour, transform)
        normalized_edge_contour = _sub_contour(normalized_piece_contour, idx0, idx1 + 1)
        normalized_piece_center = _transform_point(piece.center, transform)

        # compute the sign of the edge
        heights = normalized_edge_contour[:, 0, 1]
        if np.max(np.abs(heights)) > 10:
            sign = 1 if np.max(heights) > -np.min(heights) else -1
        else:
            sign = 0

        # rotate male contours by 180° for easy match with female contours
        if sign == 1:
            angle_degrees += 180
            transform = _get_contour_transform(piece.contour, idx1, 0, 0, angle_degrees)
            normalized_piece_contour = _transform_contour(piece.contour, transform)
            normalized_piece_center = _transform_point(piece.center, transform)

        # Calculate integral area of the edge
        # The integral area is the area between the edge curve and the x-axis (straight line from start to end)
        edge_points = normalized_edge_contour[:, 0, :]  # Extract (x, y) coordinates
        x_coords = edge_points[:, 0]
        y_coords = edge_points[:, 1]

        # Calculate area using the trapezoidal rule
        # Area = integral of y with respect to x
        integral_area = np.trapezoid(y_coords, x_coords)

        # Also calculate the signed area (positive above x-axis, negative below)
        signed_area = integral_area

        # Calculate the absolute area (total area regardless of sign)
        absolute_area = np.trapezoid(np.abs(y_coords), x_coords)

        edge = Edge(
            id=f"{piece.name}_{quarter}",
            idx0=idx0,
            idx1=idx1,
            normalized_piece_contour=normalized_piece_contour,
            normalized_piece_center=normalized_piece_center,
            angle_degrees=angle_degrees,
            sign=sign,
            straight_length=straight_length,
            absolute_area=absolute_area,
            integral_area=integral_area,
        )
        edges.append(edge)
    for idx, edge in enumerate(edges):
        edge.update(prev=edges[idx - 1], next=edges[idx + 1])

    print("Finished getting edges for piece", piece.name)

    return piece