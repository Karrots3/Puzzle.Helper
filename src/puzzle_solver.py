"""
Main puzzle solver module that coordinates the entire puzzle-solving process.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

from image_processor import ImageProcessor
from piece_detector import PieceDetector
from matcher import PieceMatcher
from visualizer import SolutionVisualizer


class PuzzleSolver:
    """
    Main class for solving puzzles by analyzing photos of puzzle pieces.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the puzzle solver with optional configuration.
        
        Args:
            config: Configuration dictionary with solver parameters
        """
        self.config = config or self._get_default_config()
        self.image_processor = ImageProcessor()
        self.piece_detector = PieceDetector()
        self.matcher = PieceMatcher()
        self.visualizer = SolutionVisualizer()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict:
        """Get default configuration parameters."""
        return {
            'min_piece_size': 50,
            'edge_detection_threshold': 0.1,
            'matching_threshold': 0.8,
            'max_iterations': 1000,
            'debug_mode': False
        }
    
    def load_pieces(self, input_path: str) -> List[np.ndarray]:
        """
        Load puzzle pieces from images in the specified directory.
        
        Args:
            input_path: Path to directory containing puzzle piece images
            
        Returns:
            List of processed puzzle piece images
        """
        self.logger.info(f"Loading puzzle pieces from: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        pieces = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for filename in os.listdir(input_path):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                file_path = os.path.join(input_path, filename)
                try:
                    image = cv2.imread(file_path)
                    if image is not None:
                        processed_image = self.image_processor.preprocess(image)
                        pieces.append(processed_image)
                        self.logger.debug(f"Loaded piece: {filename}")
                except Exception as e:
                    self.logger.warning(f"Failed to load {filename}: {e}")
        
        self.logger.info(f"Successfully loaded {len(pieces)} puzzle pieces")
        return pieces
    
    def solve(self, pieces: List[np.ndarray]) -> Dict:
        """
        Solve the puzzle by finding the correct arrangement of pieces.
        
        Args:
            pieces: List of puzzle piece images
            
        Returns:
            Dictionary containing the solution and metadata
        """
        self.logger.info("Starting puzzle solving process")
        
        if len(pieces) < 2:
            raise ValueError("At least 2 puzzle pieces are required")
        
        # Step 1: Detect and extract individual pieces
        detected_pieces = []
        for i, piece in enumerate(pieces):
            detected = self.piece_detector.detect_pieces(piece)
            for j, detected_piece in enumerate(detected):
                detected_pieces.append({
                    'id': f"{i}_{j}",
                    'image': detected_piece,
                    'features': self.piece_detector.extract_features(detected_piece)
                })
        
        self.logger.info(f"Detected {len(detected_pieces)} individual pieces")
        
        # Step 2: Analyze piece edges and find matches
        matches = self.matcher.find_matches(detected_pieces)
        
        # Step 3: Build solution grid
        solution = self._build_solution(detected_pieces, matches)
        
        self.logger.info("Puzzle solving completed")
        return {
            'pieces': detected_pieces,
            'matches': matches,
            'solution': solution,
            'metadata': {
                'total_pieces': len(detected_pieces),
                'total_matches': len(matches),
                'grid_size': solution.get('grid_size', (0, 0))
            }
        }
    
    def _build_solution(self, pieces: List[Dict], matches: List[Dict]) -> Dict:
        """
        Build the final solution grid from pieces and matches.
        
        Args:
            pieces: List of detected pieces with features
            matches: List of piece matches
            
        Returns:
            Solution dictionary with grid layout
        """
        # Simple greedy algorithm to build the solution
        # In a real implementation, this would be more sophisticated
        
        placed_pieces = set()
        grid = {}
        
        # Start with the first piece at origin
        if pieces:
            first_piece = pieces[0]
            grid[(0, 0)] = first_piece['id']
            placed_pieces.add(first_piece['id'])
        
        # Iteratively place connected pieces
        for match in matches:
            piece1_id = match['piece1_id']
            piece2_id = match['piece2_id']
            
            if piece1_id in placed_pieces and piece2_id not in placed_pieces:
                # Find position for piece2 relative to piece1
                pos1 = self._find_piece_position(grid, piece1_id)
                if pos1:
                    pos2 = self._calculate_relative_position(pos1, match['direction'])
                    grid[pos2] = piece2_id
                    placed_pieces.add(piece2_id)
            
            elif piece2_id in placed_pieces and piece1_id not in placed_pieces:
                # Find position for piece1 relative to piece2
                pos2 = self._find_piece_position(grid, piece2_id)
                if pos2:
                    pos1 = self._calculate_relative_position(pos2, self._invert_direction(match['direction']))
                    grid[pos1] = piece1_id
                    placed_pieces.add(piece1_id)
        
        # Calculate grid bounds
        if grid:
            min_x = min(pos[0] for pos in grid.keys())
            max_x = max(pos[0] for pos in grid.keys())
            min_y = min(pos[1] for pos in grid.keys())
            max_y = max(pos[1] for pos in grid.keys())
            grid_size = (max_x - min_x + 1, max_y - min_y + 1)
        else:
            grid_size = (0, 0)
        
        return {
            'grid': grid,
            'grid_size': grid_size,
            'placed_pieces': len(placed_pieces)
        }
    
    def _find_piece_position(self, grid: Dict, piece_id: str) -> Optional[Tuple[int, int]]:
        """Find the position of a piece in the grid."""
        for pos, pid in grid.items():
            if pid == piece_id:
                return pos
        return None
    
    def _calculate_relative_position(self, pos: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Calculate the position of a piece relative to another based on direction."""
        x, y = pos
        if direction == 'right':
            return (x + 1, y)
        elif direction == 'left':
            return (x - 1, y)
        elif direction == 'up':
            return (x, y - 1)
        elif direction == 'down':
            return (x, y + 1)
        return pos
    
    def _invert_direction(self, direction: str) -> str:
        """Invert a direction."""
        direction_map = {
            'right': 'left',
            'left': 'right',
            'up': 'down',
            'down': 'up'
        }
        return direction_map.get(direction, direction)
    
    def visualize_solution(self, solution: Dict, output_path: Optional[str] = None) -> None:
        """
        Visualize the puzzle solution.
        
        Args:
            solution: Solution dictionary from solve() method
            output_path: Optional path to save the visualization
        """
        self.visualizer.visualize(solution, output_path)
    
    def save_solution(self, solution: Dict, output_path: str) -> None:
        """
        Save the solution to a file.
        
        Args:
            solution: Solution dictionary from solve() method
            output_path: Path to save the solution
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_solution = self._make_serializable(solution)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_solution, f, indent=2)
        
        self.logger.info(f"Solution saved to: {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
