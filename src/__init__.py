"""
Puzzle Solver Package

An intelligent Python application that analyzes photos of puzzle pieces
and finds the correct tiles to put together.
"""

__version__ = "0.1.0"
__author__ = "Puzzle Solver Team"

from .puzzle_solver import PuzzleSolver
from .image_processor import ImageProcessor
from .piece_detector import PieceDetector
from .matcher import PieceMatcher
from .visualizer import SolutionVisualizer

__all__ = [
    "PuzzleSolver",
    "ImageProcessor", 
    "PieceDetector",
    "PieceMatcher",
    "SolutionVisualizer",
]
