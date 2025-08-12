"""
Image processing utilities for puzzle piece analysis.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
from skimage import filters, morphology, measure
from skimage.color import rgb2gray, rgb2hsv
import logging


class ImageProcessor:
    """
    Handles image preprocessing and enhancement for puzzle piece analysis.
    """
    
    def __init__(self):
        """Initialize the image processor."""
        self.logger = logging.getLogger(__name__)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for puzzle piece detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB conversion (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize if too large
        max_size = 1024
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Apply noise reduction
        image = self._denoise(image)
        
        # Enhance contrast
        image = self._enhance_contrast(image)
        
        return image
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the image."""
        # Bilateral filter for edge-preserving smoothing
        if len(image.shape) == 3:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def extract_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Extract edges from the image using Canny edge detection.
        
        Args:
            image: Input image
            
        Returns:
            Edge map as binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        return edges
    
    def find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in the image.
        
        Args:
            image: Input image (preferably binary/edge image)
            
        Returns:
            List of contours
        """
        # Convert to binary if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def analyze_colors(self, image: np.ndarray) -> Dict:
        """
        Analyze color characteristics of the image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing color analysis results
        """
        if len(image.shape) != 3:
            return {'error': 'Image must be color'}
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        color_stats = {
            'mean_rgb': np.mean(image, axis=(0, 1)),
            'std_rgb': np.std(image, axis=(0, 1)),
            'mean_hsv': np.mean(hsv, axis=(0, 1)),
            'std_hsv': np.std(hsv, axis=(0, 1)),
            'dominant_colors': self._find_dominant_colors(image)
        }
        
        return color_stats
    
    def _find_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[Tuple]:
        """
        Find dominant colors in the image using k-means clustering.
        
        Args:
            image: Input image
            n_colors: Number of dominant colors to find
            
        Returns:
            List of (color, percentage) tuples
        """
        # Reshape image for clustering
        pixels = image.reshape(-1, 3)
        
        # Convert to float32
        pixels = np.float32(pixels)
        
        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Calculate percentages
        total_pixels = len(pixels)
        dominant_colors = []
        
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_pixels) * 100
            color = centers[label].astype(int)
            dominant_colors.append((tuple(color), percentage))
        
        # Sort by percentage
        dominant_colors.sort(key=lambda x: x[1], reverse=True)
        
        return dominant_colors
    
    def analyze_patterns(self, image: np.ndarray) -> Dict:
        """
        Analyze patterns and textures in the image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing pattern analysis results
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate texture features
        texture_features = {
            'entropy': self._calculate_entropy(gray),
            'energy': self._calculate_energy(gray),
            'contrast': self._calculate_contrast(gray),
            'homogeneity': self._calculate_homogeneity(gray)
        }
        
        return texture_features
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def _calculate_energy(self, image: np.ndarray) -> float:
        """Calculate image energy."""
        return np.sum(image.astype(float) ** 2) / (image.shape[0] * image.shape[1])
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        return np.std(image.astype(float))
    
    def _calculate_homogeneity(self, image: np.ndarray) -> float:
        """Calculate image homogeneity."""
        # Simple homogeneity measure based on local variance
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(image.astype(float), -1, kernel)
        local_variance = cv2.filter2D((image.astype(float) - local_mean) ** 2, -1, kernel)
        homogeneity = 1 / (1 + np.mean(local_variance))
        return homogeneity
    
    def segment_pieces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Segment individual puzzle pieces from the image.
        
        Args:
            image: Input image containing multiple puzzle pieces
            
        Returns:
            List of individual piece images
        """
        # Extract edges
        edges = self.extract_edges(image)
        
        # Find contours
        contours = self.find_contours(edges)
        
        # Filter contours by area
        min_area = 1000  # Minimum area for a puzzle piece
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        pieces = []
        for contour in valid_contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract piece with some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            piece = image[y1:y2, x1:x2]
            pieces.append(piece)
        
        return pieces
    
    def normalize_piece(self, piece: np.ndarray, target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Normalize a puzzle piece to a standard size.
        
        Args:
            piece: Input piece image
            target_size: Target size (width, height)
            
        Returns:
            Normalized piece image
        """
        # Resize to target size
        normalized = cv2.resize(piece, target_size)
        
        # Normalize pixel values to [0, 1]
        if normalized.dtype != np.float32:
            normalized = normalized.astype(np.float32) / 255.0
        
        return normalized