# Puzzle Solver

An intelligent Python application that analyzes photos of puzzle pieces and finds the correct tiles to put together.

## Features

- **Image Processing**: Automatically detects and extracts puzzle pieces from photos
- **Computer Vision**: Uses advanced CV techniques to analyze piece shapes and patterns
- **Pattern Matching**: Identifies matching edges and corners between pieces
- **Visualization**: Provides visual feedback showing how pieces fit together
- **Multiple Puzzle Types**: Supports various puzzle formats and piece shapes
- **Piece Saving**: Save detected puzzle pieces as individual image files for analysis
- **Detection Only Mode**: Detect and save pieces without solving the puzzle

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Scikit-image
- TensorFlow (for advanced pattern recognition)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PuzzleSolver
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from puzzle_solver import PuzzleSolver

# Initialize the solver
solver = PuzzleSolver()

# Load puzzle pieces from images
pieces = solver.load_pieces("path/to/puzzle/pieces/")

# Solve the puzzle
solution = solver.solve(pieces)

# Visualize the result
solver.visualize_solution(solution)

# Save detected pieces as individual images
solver.detect_and_save_pieces("path/to/input", "path/to/output", prefix="piece")
```

### Saving Detected Puzzle Pieces

The puzzle solver can save individual detected pieces as separate image files for analysis or further processing:

```python
from puzzle_solver import PuzzleSolver

solver = PuzzleSolver()

# Method 1: Detect and save pieces only
result = solver.detect_and_save_pieces(
    input_path="path/to/puzzle/images",
    output_dir="path/to/save/pieces",
    prefix="detected_piece"
)

# Method 2: Solve puzzle and save pieces during solving
solution = solver.solve(
    pieces=pieces,
    save_detected_pieces=True,
    pieces_output_dir="path/to/save/pieces"
)

# Method 3: Save individual pieces with metadata
solver.piece_detector.save_piece_with_metadata(
    piece=piece_image,
    output_path="path/to/save/piece.png",
    metadata={"piece_id": "001", "source": "image1"}
)
```

The saved pieces are useful for:
- Manual verification of detection accuracy
- Training data for machine learning models
- Debugging detection algorithms
- Creating datasets for research

### Command Line Interface

```bash
# Basic puzzle solving
python main.py --input path/to/pieces --output path/to/solution

# Save detected puzzle pieces as individual images
python main.py --input path/to/pieces --save-detected-pieces --pieces-output-dir path/to/save/pieces

# Only detect and save pieces without solving
python main.py --input path/to/pieces --detect-only --pieces-output-dir path/to/save/pieces

# Full solution with piece saving and visualization
python main.py --input path/to/pieces --output path/to/solution --save-detected-pieces --visualize --save-solution
```

## Project Structure

```
PuzzleSolver/
├── src/
│   ├── __init__.py
│   ├── puzzle_solver.py      # Main solver class
│   ├── image_processor.py    # Image processing utilities
│   ├── piece_detector.py     # Puzzle piece detection
│   ├── matcher.py           # Piece matching algorithms
│   └── visualizer.py        # Solution visualization
├── tests/
│   ├── __init__.py
│   ├── test_puzzle_solver.py
│   ├── test_image_processor.py
│   └── test_matcher.py
├── data/
│   ├── sample_pieces/       # Sample puzzle pieces
│   └── results/            # Output solutions
│       └── detected_pieces/ # Saved individual puzzle pieces
├── docs/
│   ├── api.md
│   └── algorithms.md
├── requirements.txt
├── setup.py
├── main.py
├── example_save_pieces.py   # Example script for piece saving
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.


