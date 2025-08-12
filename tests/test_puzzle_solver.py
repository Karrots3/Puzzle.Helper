"""
Unit tests for the PuzzleSolver class.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from puzzle_solver import PuzzleSolver


class TestPuzzleSolver(unittest.TestCase):
    """Test cases for the PuzzleSolver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.solver = PuzzleSolver()
        
        # Create mock pieces for testing
        self.mock_pieces = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ]
    
    def test_initialization(self):
        """Test solver initialization."""
        self.assertIsNotNone(self.solver)
        self.assertIsNotNone(self.solver.config)
        self.assertIsNotNone(self.solver.image_processor)
        self.assertIsNotNone(self.solver.piece_detector)
        self.assertIsNotNone(self.solver.matcher)
        self.assertIsNotNone(self.solver.visualizer)
    
    def test_default_config(self):
        """Test default configuration."""
        config = self.solver._get_default_config()
        expected_keys = ['min_piece_size', 'edge_detection_threshold', 
                        'matching_threshold', 'max_iterations', 'debug_mode']
        
        for key in expected_keys:
            self.assertIn(key, config)
    
    def test_custom_config(self):
        """Test solver with custom configuration."""
        custom_config = {
            'min_piece_size': 100,
            'matching_threshold': 0.9
        }
        solver = PuzzleSolver(config=custom_config)
        
        self.assertEqual(solver.config['min_piece_size'], 100)
        self.assertEqual(solver.config['matching_threshold'], 0.9)
        # Default values should still be present
        self.assertIn('edge_detection_threshold', solver.config)
    
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('cv2.imread')
    def test_load_pieces_success(self, mock_imread, mock_listdir, mock_exists):
        """Test successful loading of puzzle pieces."""
        mock_exists.return_value = True
        mock_listdir.return_value = ['piece1.jpg', 'piece2.png', 'piece3.bmp']
        mock_imread.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        pieces = self.solver.load_pieces('/fake/path')
        
        self.assertEqual(len(pieces), 3)
        self.assertTrue(all(isinstance(p, np.ndarray) for p in pieces))
    
    @patch('os.path.exists')
    def test_load_pieces_path_not_exists(self, mock_exists):
        """Test loading pieces from non-existent path."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            self.solver.load_pieces('/fake/path')
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_load_pieces_no_images(self, mock_listdir, mock_exists):
        """Test loading pieces when no image files are present."""
        mock_exists.return_value = True
        mock_listdir.return_value = ['file1.txt', 'file2.doc']
        
        pieces = self.solver.load_pieces('/fake/path')
        
        self.assertEqual(len(pieces), 0)
    
    def test_solve_insufficient_pieces(self):
        """Test solving with insufficient pieces."""
        with self.assertRaises(ValueError):
            self.solver.solve([self.mock_pieces[0]])
    
    @patch.object(PuzzleSolver, 'piece_detector')
    @patch.object(PuzzleSolver, 'matcher')
    def test_solve_success(self, mock_matcher, mock_detector):
        """Test successful puzzle solving."""
        # Mock piece detection
        mock_detector.detect_pieces.return_value = [self.mock_pieces[0]]
        mock_detector.extract_features.return_value = {'test': 'features'}
        
        # Mock matching
        mock_matcher.find_matches.return_value = [
            {
                'piece1_id': '0_0',
                'piece2_id': '1_0',
                'compatibility': {'score': 0.8},
                'direction': 'right'
            }
        ]
        
        solution = self.solver.solve(self.mock_pieces[:2])
        
        self.assertIsInstance(solution, dict)
        self.assertIn('pieces', solution)
        self.assertIn('matches', solution)
        self.assertIn('solution', solution)
        self.assertIn('metadata', solution)
    
    def test_build_solution_empty_pieces(self):
        """Test building solution with empty pieces list."""
        solution = self.solver._build_solution([], [])
        
        self.assertEqual(solution['grid_size'], (0, 0))
        self.assertEqual(solution['placed_pieces'], 0)
    
    def test_build_solution_with_matches(self):
        """Test building solution with matches."""
        pieces = [
            {'id': '0_0', 'image': self.mock_pieces[0]},
            {'id': '1_0', 'image': self.mock_pieces[1]}
        ]
        
        matches = [
            {
                'piece1_id': '0_0',
                'piece2_id': '1_0',
                'direction': 'right'
            }
        ]
        
        solution = self.solver._build_solution(pieces, matches)
        
        self.assertIn('grid', solution)
        self.assertIn('grid_size', solution)
        self.assertIn('placed_pieces', solution)
    
    def test_find_piece_position(self):
        """Test finding piece position in grid."""
        grid = {(0, 0): 'piece1', (1, 0): 'piece2'}
        
        pos = self.solver._find_piece_position(grid, 'piece1')
        self.assertEqual(pos, (0, 0))
        
        pos = self.solver._find_piece_position(grid, 'piece2')
        self.assertEqual(pos, (1, 0))
        
        pos = self.solver._find_piece_position(grid, 'nonexistent')
        self.assertIsNone(pos)
    
    def test_calculate_relative_position(self):
        """Test calculating relative positions."""
        base_pos = (5, 5)
        
        # Test all directions
        self.assertEqual(self.solver._calculate_relative_position(base_pos, 'right'), (6, 5))
        self.assertEqual(self.solver._calculate_relative_position(base_pos, 'left'), (4, 5))
        self.assertEqual(self.solver._calculate_relative_position(base_pos, 'up'), (5, 4))
        self.assertEqual(self.solver._calculate_relative_position(base_pos, 'down'), (5, 6))
        
        # Test unknown direction
        self.assertEqual(self.solver._calculate_relative_position(base_pos, 'unknown'), (5, 5))
    
    def test_invert_direction(self):
        """Test direction inversion."""
        self.assertEqual(self.solver._invert_direction('right'), 'left')
        self.assertEqual(self.solver._invert_direction('left'), 'right')
        self.assertEqual(self.solver._invert_direction('up'), 'down')
        self.assertEqual(self.solver._invert_direction('down'), 'up')
        self.assertEqual(self.solver._invert_direction('unknown'), 'unknown')
    
    def test_make_serializable(self):
        """Test making objects serializable."""
        # Test numpy array
        arr = np.array([1, 2, 3])
        serialized = self.solver._make_serializable(arr)
        self.assertIsInstance(serialized, list)
        
        # Test dictionary with numpy arrays
        data = {'array': arr, 'number': 42}
        serialized = self.solver._make_serializable(data)
        self.assertIsInstance(serialized['array'], list)
        self.assertEqual(serialized['number'], 42)
        
        # Test list with numpy arrays
        data_list = [arr, 42]
        serialized = self.solver._make_serializable(data_list)
        self.assertIsInstance(serialized[0], list)
        self.assertEqual(serialized[1], 42)
        
        # Test regular object
        regular_obj = "test"
        serialized = self.solver._make_serializable(regular_obj)
        self.assertEqual(serialized, "test")
    
    def test_save_solution(self):
        """Test saving solution to file."""
        solution = {
            'pieces': [],
            'matches': [],
            'solution': {'grid': {}, 'grid_size': (0, 0)},
            'metadata': {'total_pieces': 0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.solver.save_solution(solution, temp_path)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(temp_path))
            
            import json
            with open(temp_path, 'r') as f:
                loaded_solution = json.load(f)
            
            self.assertIn('pieces', loaded_solution)
            self.assertIn('matches', loaded_solution)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
