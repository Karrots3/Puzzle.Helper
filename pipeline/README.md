# Puzzle Piece Matching Pipeline

A clean, organized pipeline for processing puzzle piece images and finding matches between pieces.

## Structure

- `models.py`: Data models (Edge, Piece, Match, EdgeType, etc.)
- `config.py`: Configuration constants
- `preprocessing.py`: Image preprocessing pipeline (extracts pieces from images)
- `matching.py`: Matching logic (finds compatible edges between pieces)
- `main.py`: Main orchestration script

## Usage

### Basic Usage

Run the complete pipeline:

```python
from pipeline.main import main

main()
```

This will:
1. Process all `.JPG` images in `./data/`
2. Extract puzzle pieces and save to `./data/pieces/`
3. Find matches between all pairs of pieces
4. Save matches to `./data/matches/`

### Processing a Single Image

```python
from pathlib import Path
from pipeline.preprocessing import process_image, PipelineProcessImgParams

params = PipelineProcessImgParams()
piece = process_image("data/0001.JPG", params)
print(f"Piece {piece.piece_id} has {len(piece.edge_list)} edges")
```

### Matching Two Pieces

```python
from pathlib import Path
import pickle
from pipeline.matching import find_match, PipelineParamsMatching

# Load pieces
with open("data/pieces/0001.pkl", "rb") as f:
    piece1 = pickle.load(f)
with open("data/pieces/0002.pkl", "rb") as f:
    piece2 = pickle.load(f)

# Find match
params = PipelineParamsMatching()
match = find_match(piece1, piece2, params)
print(match)
```

## Image Preprocessing

The preprocessing pipeline:
1. **Crops** the image to center region (configurable radius and shift)
2. **Thresholds** using Otsu's method
3. **Extracts** the main contour
4. **Finds** corner indices using peak detection
5. **Classifies** edges as flat, man, or woman
6. **Normalizes** edge contours for matching

## Matching Algorithm

The matching algorithm:
1. Compares edges of opposite types (man vs woman)
2. Filters by length compatibility
3. Creates embeddings (distance from baseline)
4. Uses sliding window to find best alignment
5. Returns match score and shift information

## Configuration

Edit `config.py` to adjust:
- `MIN_LEN_EDGE`: Minimum edge length
- `SAMPLE_POINTS`: Number of points for sampling
- `MAX_SCORE_MSE_DISTANCE`: Maximum MSE for valid matches
- `DEFAULT_MAX_DIFF_LENGTH_RATIO`: Maximum length difference ratio

## Error Handling

Errors during processing are logged to `./data/processing_errors.jsonl` in JSONL format.
