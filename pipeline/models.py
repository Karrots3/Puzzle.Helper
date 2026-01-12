"""Data models for puzzle pieces and edges."""

from enum import Enum
from typing import Optional

import numpy as np


class EdgeType(Enum):
    """Edge type enumeration."""
    flat = 0
    man = 1
    woman = -1


class EdgeColor(Enum):
    """Edge color enumeration for visualization."""
    flat = (0, 255, 0)
    man = (0, 0, 255)
    woman = (255, 0, 0)


class Edge:
    """Represents a single edge of a puzzle piece."""
    
    def __init__(
        self,
        edge_id: int,
        edge_type: EdgeType,
        edge_color: tuple[int, int, int],
        edge_contour: np.ndarray,
    ):
        self.edge_id = edge_id
        self.edge_type = edge_type
        self.edge_color = edge_color
        self.edge_contour = edge_contour
        self._normalized_contour: Optional[np.ndarray] = None
        self._straight_length: Optional[float] = None
        self._embedding: Optional[np.ndarray] = None
    
    @property
    def straight_length(self) -> float:
        """Compute and cache straight-line length of edge."""
        if self._straight_length is None:
            p0 = self.edge_contour[0, 0, :]
            p1 = self.edge_contour[-1, 0, :]
            self._straight_length = float(np.linalg.norm(p1 - p0))
        return self._straight_length
    
    def get_embedding(self) -> np.ndarray:
        """Compute and cache embedding (distance from baseline)."""
        if self._embedding is None:
            len_contour = self.edge_contour.shape[0]
            line = np.linspace(
                self.edge_contour[0], self.edge_contour[-1], len_contour
            )
            self._embedding = np.array([
                np.linalg.norm(self.edge_contour[i] - line[i])
                for i in range(len_contour)
            ])
        return self._embedding
    
    @property
    def normalized_contour(self) -> np.ndarray:
        """Compute and cache normalized contour if not already computed."""
        if self._normalized_contour is None:
            self._normalized_contour = self._compute_normalized_contour()
        return self._normalized_contour
    
    def _compute_normalized_contour(self) -> np.ndarray:
        """
        Normalize the edge contour so that:
        - First point is at (0, 0)
        - Last point is at (X, 0) where X is the straight-line distance
        - The edge is rotated to be horizontal
        """
        import cv2
        import math
        
        contour = self.edge_contour.astype(np.float32)
        p0 = contour[0, 0, :]
        p1 = contour[-1, 0, :]
        
        dx, dy = p1 - p0
        straight_length = math.sqrt(dx**2 + dy**2)
        
        if straight_length < 1e-6:
            # Edge is degenerate, return as-is
            return contour
        
        # Calculate rotation angle to make edge horizontal
        angle_degrees = math.degrees(math.atan2(dy, dx))
        
        # Get center for rotation
        center = np.mean(contour[:, 0, :], axis=0)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle_degrees, 1.0)
        
        # Apply rotation
        normalized = cv2.transform(contour, rotation_matrix)
        
        # Translate so first point is at origin
        translation = -normalized[0, 0, :]
        normalized[:, 0, :] += translation
        
        return normalized


class Piece:
    """Represents a single puzzle piece."""
    
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


class Match:
    """Represents a match between two puzzle pieces."""
    
    def __init__(
        self,
        piece_id_1: int,
        piece_id_2: int,
        score: float | None,
        shift_1: int,
        shift_2: int,
    ):
        self.piece_id_1 = piece_id_1
        self.piece_id_2 = piece_id_2
        self.score = score
        self.shift_1 = shift_1
        self.shift_2 = shift_2
    
    def __le__(self, other):
        if self.score is None:
            return False
        if other.score is None:
            return True
        return self.score <= other.score
    
    def __ge__(self, other):
        if self.score is None:
            return False
        if other.score is None:
            return True
        return self.score >= other.score
    
    def __repr__(self):
        return (
            f"Match(piece_id_1={self.piece_id_1}, piece_id_2={self.piece_id_2}, "
            f"score={self.score}, shift_1={self.shift_1}, shift_2={self.shift_2})"
        )


class LoopingList(list):
    """List that wraps around when indexed."""
    
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)


class ContourError(Exception):
    """Exception raised when contour detection fails or finds unexpected number of contours."""
    
    def __init__(
        self,
        message: str,
        num_contours: int,
        contour_areas: list[float],
        image_path: str = "",
    ):
        self.message = message
        self.num_contours = num_contours
        self.contour_areas = contour_areas
        self.image_path = image_path
        super().__init__(self.message)
