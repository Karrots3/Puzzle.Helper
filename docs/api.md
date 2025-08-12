# Puzzle Solver API Documentation

## Overview

The Puzzle Solver is a comprehensive Python application that uses computer vision and machine learning techniques to automatically solve puzzles by analyzing photos of puzzle pieces.

## Main Classes

### PuzzleSolver

The main class that orchestrates the entire puzzle solving process.

#### Constructor

```python
PuzzleSolver(config: Optional[Dict] = None)
```

**Parameters:**
- `config`: Optional configuration dictionary with solver parameters

#### Methods

##### `load_pieces(input_path: str) -> List[Dict]`

Load puzzle pieces from a directory of images.

**Parameters:**
- `input_path`: Path to directory containing puzzle piece images

**Returns:**
- List of piece dictionaries with metadata and image data

##### `solve(pieces: Optional[List[Dict]] = None) -> Dict`

Solve the puzzle by finding the correct arrangement of pieces.

**Parameters:**
- `pieces`: Optional list of pieces. If None, uses previously loaded pieces.

**Returns:**
- Dictionary containing the solution with piece positions and connections

##### `visualize_solution(solution: Optional[Dict] = None, output_path: Optional[str] = None) -> None`

Visualize the puzzle solution.

**Parameters:**
- `solution`: Optional solution to visualize. If None, uses current solution.
- `output_path`: Optional path to save visualization

##### `save_solution(output_path: str) -> None`

Save the puzzle solution to file.

**Parameters:**
- `output_path`: Path to save the solution

##### `get_statistics() -> Dict`

Get statistics about the puzzle solving process.

**Returns:**
- Dictionary containing various statistics

### ImageProcessor

Handles all image processing operations for puzzle piece analysis.

#### Methods

##### `preprocess(image: np.ndarray) -> np.ndarray`

Preprocess an image for puzzle piece analysis.

**Parameters:**
- `image`: Input image as numpy array

**Returns:**
- Preprocessed image as numpy array

##### `analyze_patterns(image: np.ndarray) -> Dict`

Analyze patterns in the image.

**Parameters:**
- `image`: Input image as numpy array

**Returns:**
- Dictionary containing pattern analysis results

##### `analyze_colors(image: np.ndarray) -> Dict`

Analyze color characteristics of the image.

**Parameters:**
- `image`: Input image as numpy array

**Returns:**
- Dictionary containing color analysis results

### PieceDetector

Detects and analyzes puzzle pieces in images.

#### Methods

##### `detect_piece(image: np.ndarray) -> Dict`

Detect a single puzzle piece in the image.

**Parameters:**
- `image`: Input image as numpy array

**Returns:**
- Dictionary containing piece detection results

##### `extract_edges(image: np.ndarray) -> Dict`

Extract edge information from the image.

**Parameters:**
- `image`: Input image as numpy array

**Returns:**
- Dictionary containing edge analysis results

##### `detect_corners(image: np.ndarray) -> Dict`

Detect corners in the image.

**Parameters:**
- `image`: Input image as numpy array

**Returns:**
- Dictionary containing corner detection results

### PieceMatcher

Matches puzzle pieces based on their features and characteristics.

#### Methods

##### `find_matches(pieces: List[Dict]) -> List[Tuple]`

Find matches between puzzle pieces.

**Parameters:**
- `pieces`: List of analyzed puzzle pieces

**Returns:**
- List of tuples containing (piece1_idx, piece2_idx, compatibility_score)

##### `find_optimal_arrangement(pieces: List[Dict], matches: List[Tuple]) -> Dict`

Find optimal arrangement of pieces using the matches.

**Parameters:**
- `pieces`: List of puzzle pieces
- `matches`: List of piece matches

**Returns:**
- Dictionary containing optimal arrangement

### SolutionVisualizer

Visualizes puzzle solutions and analysis results.

#### Methods

##### `visualize(solution: Dict, output_path: Optional[str] = None) -> None`

Visualize the complete puzzle solution.

**Parameters:**
- `solution`: Dictionary containing the puzzle solution
- `output_path`: Optional path to save the visualization

##### `visualize_piece_matches(pieces: List[Dict], matches: List[Tuple], output_path: Optional[str] = None) -> None`

Visualize piece matches with side-by-side comparisons.

**Parameters:**
- `pieces`: List of puzzle pieces
- `matches`: List of piece matches
- `output_path`: Optional path to save the visualization

##### `create_animation(solution: Dict, output_path: str) -> None`

Create an animation showing the puzzle assembly process.

**Parameters:**
- `solution`: Dictionary containing the puzzle solution
- `output_path`: Path to save the animation

##### `save_solution_image(solution: Dict, output_path: str) -> None`

Save the puzzle solution as a single image.

**Parameters:**
- `solution`: Dictionary containing the puzzle solution
- `output_path`: Path to save the image

##### `generate_report(solution: Dict, output_path: str) -> None`

Generate a comprehensive report with visualizations.

**Parameters:**
- `solution`: Dictionary containing the puzzle solution
- `output_path`: Path to save the report

## Configuration

The solver can be configured using a dictionary with the following parameters:

```python
config = {
    "min_piece_size": 50,
    "max_piece_size": 500,
    "edge_detection_threshold": 0.1,
    "pattern_matching_threshold": 0.8,
    "corner_detection_sensitivity": 0.05,
    "max_iterations": 1000,
    "visualization_enabled": True,
    "similarity_threshold": 0.7,
    "compatibility_threshold": 0.6,
    "max_matches_per_piece": 4,
    "feature_weight_patterns": 0.3,
    "feature_weight_colors": 0.3,
    "feature_weight_edges": 0.2,
    "feature_weight_corners": 0.2,
    "use_geometric_constraints": True,
    "geometric_tolerance": 0.1
}
```

## Data Structures

### Piece Dictionary

Each puzzle piece is represented as a dictionary with the following structure:

```python
piece = {
    'filename': 'piece_001.jpg',
    'file_path': '/path/to/piece_001.jpg',
    'original_image': np.ndarray,  # Original image data
    'processed_image': np.ndarray,  # Preprocessed image data
    'area': 1000,  # Piece area in pixels
    'perimeter': 120,  # Piece perimeter in pixels
    'bounding_box': (x, y, w, h),  # Bounding rectangle
    'center': (cx, cy),  # Center point
    'contour': np.ndarray,  # Piece contour
    'mask': np.ndarray,  # Binary mask
    'piece_image': np.ndarray,  # Extracted piece image
    'shape_characteristics': {
        'circularity': 0.8,
        'aspect_ratio': 1.0,
        'extent': 0.9,
        'solidity': 0.95,
        'num_corners': 4
    },
    'edges': {...},  # Edge analysis results
    'corners': {...},  # Corner detection results
    'patterns': {...},  # Pattern analysis results
    'colors': {...},  # Color analysis results
    'features': np.ndarray  # Combined feature vector
}
```

### Solution Dictionary

The puzzle solution is represented as a dictionary with the following structure:

```python
solution = {
    'pieces': List[Dict],  # List of analyzed pieces
    'matches': List[Tuple],  # List of (piece1_idx, piece2_idx, compatibility)
    'adjacency_graph': Dict,  # Graph representation of connections
    'arrangement': List[Tuple],  # List of (piece_idx, position)
    'positions': Dict,  # Dictionary mapping piece_idx to (x, y) position
    'grid_size': (width, height)  # Size of the puzzle grid
}
```

## Error Handling

The API uses standard Python exceptions for error handling:

- `FileNotFoundError`: When input path doesn't exist
- `ValueError`: When invalid data is provided
- `RuntimeError`: When processing fails

## Examples

### Basic Usage

```python
from puzzle_solver import PuzzleSolver

# Initialize solver
solver = PuzzleSolver()

# Load pieces
pieces = solver.load_pieces("path/to/puzzle/pieces/")

# Solve puzzle
solution = solver.solve(pieces)

# Visualize result
solver.visualize_solution(solution, "output.png")

# Get statistics
stats = solver.get_statistics()
print(f"Solved puzzle with {stats['total_pieces']} pieces")
```

### Advanced Usage with Custom Configuration

```python
from puzzle_solver import PuzzleSolver

# Custom configuration
config = {
    "min_piece_size": 100,
    "max_piece_size": 1000,
    "compatibility_threshold": 0.7,
    "feature_weight_patterns": 0.4,
    "feature_weight_colors": 0.3,
    "feature_weight_edges": 0.2,
    "feature_weight_corners": 0.1
}

# Initialize solver with custom config
solver = PuzzleSolver(config)

# Load and solve
pieces = solver.load_pieces("puzzle_pieces/")
solution = solver.solve(pieces)

# Generate comprehensive report
solver.visualizer.generate_report(solution, "puzzle_report.html")
```
