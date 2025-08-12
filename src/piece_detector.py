"""
Puzzle piece detection and feature extraction module.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from skimage import measure, morphology
import logging

from image_processor import ImageProcessor


class PieceDetector:
    """
    Detects and analyzes individual puzzle pieces in images.
    """
    
    def __init__(self):
        """Initialize the piece detector."""
        self.image_processor = ImageProcessor()
        self.logger = logging.getLogger(__name__)
    
    def detect_pieces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect individual puzzle pieces in the image.
        
        Args:
            image: Input image containing puzzle pieces
            
        Returns:
            List of detected piece images
        """
        self.logger.debug("Detecting puzzle pieces")
        
        # Preprocess the image
        processed = self.image_processor.preprocess(image)
        
        # Extract edges
        edges = self.image_processor.extract_edges(processed)
        
        # Find contours
        contours = self.image_processor.find_contours(edges)
        
        # Filter and extract pieces
        pieces = self._extract_pieces_from_contours(processed, contours)
        
        self.logger.info(f"Detected {len(pieces)} puzzle pieces")
        return pieces
    
    def _extract_pieces_from_contours(self, image: np.ndarray, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract puzzle pieces from detected contours.
        
        Args:
            image: Original image
            contours: List of detected contours
            
        Returns:
            List of extracted piece images
        """
        pieces = []
        min_area = 500  # Minimum area for a valid piece
        max_area = 50000  # Maximum area for a valid piece
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # Extract piece
            piece = image[y1:y2, x1:x2]
            
            # Check if piece is valid (not too small)
            if piece.shape[0] > 30 and piece.shape[1] > 30:
                pieces.append(piece)
        
        return pieces
    
    def extract_features(self, piece: np.ndarray) -> Dict:
        """
        Extract comprehensive features from a puzzle piece.
        
        Args:
            piece: Puzzle piece image
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Basic geometric features
        features['geometry'] = self._extract_geometric_features(piece)
        
        # Edge features
        features['edges'] = self._extract_edge_features(piece)
        
        # Color features
        features['colors'] = self.image_processor.analyze_colors(piece)
        
        # Pattern features
        features['patterns'] = self.image_processor.analyze_patterns(piece)
        
        # Corner features
        features['corners'] = self._extract_corner_features(piece)
        
        return features
    
    def _extract_geometric_features(self, piece: np.ndarray) -> Dict:
        """
        Extract geometric features from the piece.
        
        Args:
            piece: Puzzle piece image
            
        Returns:
            Dictionary of geometric features
        """
        height, width = piece.shape[:2]
        
        # Convert to grayscale for analysis
        if len(piece.shape) == 3:
            gray = cv2.cvtColor(piece, cv2.COLOR_RGB2GRAY)
        else:
            gray = piece
        
        # Find contours
        edges = self.image_processor.extract_edges(piece)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'area': 0,
                'perimeter': 0,
                'aspect_ratio': 0,
                'circularity': 0,
                'compactness': 0
            }
        
        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate geometric features
        aspect_ratio = width / height if height > 0 else 0
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'compactness': compactness,
            'width': width,
            'height': height
        }
    
    def _extract_edge_features(self, piece: np.ndarray) -> Dict:
        """
        Extract edge-related features from the piece.
        
        Args:
            piece: Puzzle piece image
            
        Returns:
            Dictionary of edge features
        """
        # Extract edges
        edges = self.image_processor.extract_edges(piece)
        
        # Convert to grayscale if needed
        if len(piece.shape) == 3:
            gray = cv2.cvtColor(piece, cv2.COLOR_RGB2GRAY)
        else:
            gray = piece
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Edge statistics
        edge_stats = {
            'edge_density': edge_density,
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude),
            'gradient_max': np.max(gradient_magnitude)
        }
        
        # Analyze edge directions
        edge_directions = self._analyze_edge_directions(grad_x, grad_y)
        edge_stats.update(edge_directions)
        
        return edge_stats
    
    def _analyze_edge_directions(self, grad_x: np.ndarray, grad_y: np.ndarray) -> Dict:
        """
        Analyze edge directions in the piece.
        
        Args:
            grad_x: X-direction gradients
            grad_y: Y-direction gradients
            
        Returns:
            Dictionary of edge direction features
        """
        # Calculate gradient direction
        direction = np.arctan2(grad_y, grad_x)
        
        # Convert to degrees
        direction_deg = np.degrees(direction)
        
        # Create direction histogram
        hist, bins = np.histogram(direction_deg, bins=8, range=(-180, 180))
        hist = hist / np.sum(hist)  # Normalize
        
        # Find dominant directions
        dominant_directions = np.argsort(hist)[-3:]  # Top 3 directions
        
        return {
            'direction_histogram': hist.tolist(),
            'dominant_directions': dominant_directions.tolist(),
            'direction_entropy': -np.sum(hist * np.log2(hist + 1e-10))
        }
    
    def _extract_corner_features(self, piece: np.ndarray) -> Dict:
        """
        Extract corner-related features from the piece.
        
        Args:
            piece: Puzzle piece image
            
        Returns:
            Dictionary of corner features
        """
        # Convert to grayscale
        if len(piece.shape) == 3:
            gray = cv2.cvtColor(piece, cv2.COLOR_RGB2GRAY)
        else:
            gray = piece
        
        # Detect corners using Harris corner detection
        corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        
        # Find corner points
        corner_points = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
        
        corner_features = {
            'corner_count': len(corner_points) if corner_points is not None else 0,
            'corner_density': np.sum(corners > 0.01 * corners.max()) / corners.size,
            'corner_strength_mean': np.mean(corners) if np.any(corners > 0) else 0,
            'corner_strength_max': np.max(corners) if np.any(corners > 0) else 0
        }
        
        return corner_features
    
    def analyze_piece_shape(self, piece: np.ndarray) -> Dict:
        """
        Analyze the shape characteristics of a puzzle piece.
        
        Args:
            piece: Puzzle piece image
            
        Returns:
            Dictionary containing shape analysis
        """
        # Convert to grayscale
        if len(piece.shape) == 3:
            gray = cv2.cvtColor(piece, cv2.COLOR_RGB2GRAY)
        else:
            gray = piece
        
        # Threshold to get binary mask
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'No contours found'}
        
        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Analyze shape
        shape_analysis = {
            'area': cv2.contourArea(contour),
            'perimeter': cv2.arcLength(contour, True),
            'convex_hull_area': cv2.contourArea(cv2.convexHull(contour)),
            'solidity': cv2.contourArea(contour) / cv2.contourArea(cv2.convexHull(contour)) if cv2.contourArea(cv2.convexHull(contour)) > 0 else 0
        }
        
        # Analyze piece edges (for puzzle-specific features)
        edge_analysis = self._analyze_piece_edges(contour)
        shape_analysis.update(edge_analysis)
        
        return shape_analysis
    
    def _analyze_piece_edges(self, contour: np.ndarray) -> Dict:
        """
        Analyze the edges of a puzzle piece for interlocking features.
        
        Args:
            contour: Piece contour
            
        Returns:
            Dictionary of edge analysis features
        """
        # Approximate the contour to simplify it
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Analyze edge segments
        edge_segments = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            
            # Calculate segment length and angle
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
            
            edge_segments.append({
                'length': length,
                'angle': angle,
                'start_point': p1,
                'end_point': p2
            })
        
        # Calculate edge statistics
        lengths = [seg['length'] for seg in edge_segments]
        angles = [seg['angle'] for seg in edge_segments]
        
        edge_analysis = {
            'num_edge_segments': len(edge_segments),
            'avg_segment_length': np.mean(lengths) if lengths else 0,
            'std_segment_length': np.std(lengths) if lengths else 0,
            'avg_segment_angle': np.mean(angles) if angles else 0,
            'std_segment_angle': np.std(angles) if angles else 0
        }
        
        return edge_analysis
    
    def normalize_piece(self, piece: np.ndarray, target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Normalize a puzzle piece to a standard size and format.
        
        Args:
            piece: Input piece image
            target_size: Target size (width, height)
            
        Returns:
            Normalized piece image
        """
        return self.image_processor.normalize_piece(piece, target_size)
