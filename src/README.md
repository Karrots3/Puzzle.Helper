# Puzzle Piece Detection Pipeline

This README explains the step-by-step process of detecting puzzle pieces and their edges from images.

## Overview

The pipeline consists of two main stages:
1. **a_detector_piece.py** - Detects individual puzzle pieces from images
2. **b_detector_edges.py** - Detects and classifies the edges of each piece

---

## Stage 1: Piece Detection (`a_detector_piece.py`)

This script processes images to detect and extract individual puzzle pieces.

### Step-by-Step Process

#### 1. **Image Loading**
- Reads images from the input folder (default: `data/puzzle/`)
- Supports `.JPG` format
- Processes images in parallel using multiprocessing

#### 2. **Image Preprocessing**
- **Grayscale Conversion**: Converts the RGB image to grayscale
- **Center Trimming**: Extracts a central region of the image (60% radius from center) to focus on the puzzle piece
- **Thresholding**: Applies two thresholding methods:
  - **Fixed Threshold**: Uses a fixed threshold value of 110
  - **Otsu Threshold**: Uses Otsu's automatic threshold selection
- Both methods create binary images (black/white) to separate the piece from the background

#### 3. **Contour Extraction**
- Finds all contours in both thresholded images using `cv2.findContours`
- Contours represent the boundaries of detected shapes

#### 4. **Contour to Piece Conversion**
For each valid contour, the script:
- Calculates geometric properties:
  - **Area**: Total area enclosed by the contour
  - **Perimeter**: Length of the contour boundary
  - **Bounding Rectangle**: Minimum rectangle that contains the contour
  - **Minimum Area Rectangle**: Rotated rectangle with minimum area
  - **Convex Hull**: Smallest convex polygon containing the contour
- Computes derived metrics:
  - **Solidity**: Ratio of contour area to convex hull area
  - **Aspect Ratio**: Width to height ratio of bounding rectangle
  - **Extent**: Ratio of contour area to bounding rectangle area

#### 5. **Piece Filtering**
Filters out invalid contours based on:
- **Aspect Ratio**: Excludes square pieces (aspect ratio ≠ 1)
- **Perimeter**: Keeps pieces with perimeter between 100-8000 pixels
- **Area**: Validates area constraints (20-10000 pixels)

#### 6. **Piece Selection**
- Selects the best piece from filtered results:
  - Prefers Otsu method if available
  - Falls back to fixed threshold method
  - Uses the first valid piece if both methods produce results

#### 7. **Output**
- Creates a `Piece` object with all detected properties
- Saves visualization pipeline images showing:
  - Original image
  - Grayscale trimmed image
  - Thresholded images (fixed and Otsu)
  - Detected contours overlaid on images
- Saves all pieces to a pickle file: `results/01_pieces/dict_pieces.pkl`

---

## Stage 2: Edge Detection (`b_detector_edges.py`)

This script processes detected pieces to identify and classify their edges.

### Step-by-Step Process

#### 1. **Contour Preparation**
- Loads pieces from the previous stage
- Extracts the contour of each piece
- Calculates the center point of the piece using `cv2.minEnclosingCircle`

#### 2. **Corner Detection via Peak Finding**
- **Centering**: Translates the contour so its center is at the origin
- **Distance Calculation**: Computes the distance from center to each contour point
- **Peak Detection**: Uses `scipy.signal.find_peaks` to find local maxima in distances
  - These peaks correspond to corner points (furthest from center)
  - Uses prominence threshold (default: 10500) to filter noise
- **Peak Visualization**: Marks all detected peaks on the image

#### 3. **Rectangle Geometry Validation**
- **Combination Testing**: Tests all combinations of 4 peaks to find the best rectangle
- **Error Calculation**: For each 4-corner combination:
  - Computes side lengths and diagonal lengths
  - Calculates rectangle error based on:
    - Opposite sides should be equal
    - Diagonals should be equal
    - Adjacent sides should form right angles
- **Best Rectangle Selection**: Chooses the combination with minimum error

#### 4. **Edge Extraction**
- Divides the contour into 4 segments based on the 4 corner points
- Each segment represents one edge of the puzzle piece
- Handles edge cases where segments wrap around the contour array

#### 5. **Edge Classification**
Each edge is classified into one of three types:

- **FLAT**: Edge is straight (max deviation < threshold, default: 20 pixels)
- **MALE**: Edge protrudes outward (bulge away from piece center)
- **FEMALE**: Edge curves inward (indentation toward piece center)

**Classification Algorithm**:
1. Creates a baseline from first to last point of the edge
2. Calculates signed distance of each point to the baseline
3. Finds maximum deviation from baseline
4. Determines if extreme point is on the same side as piece center:
   - Same side → FEMALE (indentation)
   - Opposite side → MALE (protrusion)

#### 6. **Edge Normalization**
- **Rotation**: Rotates each edge so the baseline is horizontal (angle = 0°)
- **Translation**: Translates so the first point is at origin (0, 0)
- **Length Calculation**: Computes straight-line distance between edge endpoints
- **Validation**: Verifies normalization preserves edge length (within 1% tolerance)

#### 7. **Edge Validation**
- Checks that all edges have reasonable lengths (minimum: 400 pixels)
- Validates edge length consistency (shortest edge should be at least 2/3 of longest edge)
- Prints warnings for pieces with inconsistent edge lengths

#### 8. **Output**
- Creates `Edge` objects for each of the 4 edges with:
  - Original contour
  - Normalized contour
  - Straight length
  - Angle (in degrees)
  - Edge type (FLAT/MALE/FEMALE)
- Saves visualizations:
  - **Contour visualization**: All edges overlaid on piece image with color coding:
    - Green: FLAT edges
    - Blue: MALE edges
    - Red: FEMALE edges
  - **Normalized edges**: All edges normalized and plotted together
  - **Pipeline images**: Complete processing pipeline visualization
- Updates `Piece` objects with `edge_list` attribute
- Saves updated pieces to: `results/02_pieces_with_edges.pkl`

---

## Output Structure

```
results/
├── 01_pieces/
│   ├── dict_pieces.pkl          # Dictionary of Piece objects
│   └── img/                      # Pipeline visualization images
├── 02_pieces_with_edges.pkl     # Pieces with detected edges
├── 02_contour/                   # Edge contour visualizations
├── 02_edges/                     # Normalized edge visualizations
├── 02_pipelines/                 # Complete processing pipelines
└── 02_rectangle/                 # Rectangle detection visualizations
```

## Usage

### Run Piece Detection
```bash
python src/a_detector_piece.py --data-folder data/puzzle --output-folder results/01_pieces
```

### Run Edge Detection
```bash
python src/b_detector_edges.py
```

## Key Parameters

### Piece Detection
- **Threshold value**: 110 (fixed method)
- **Min perimeter**: 100 pixels
- **Max perimeter**: 8000 pixels
- **Min area**: 20 pixels
- **Max area**: 10000 pixels

### Edge Detection
- **Peak prominence**: 10500 (for corner detection)
- **Flat threshold**: 20 pixels (max deviation for flat edges)
- **Min edge length**: 400 pixels
- **Length consistency**: Shortest edge ≥ 2/3 × longest edge

## Dependencies

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- SciPy
- Standard library: `pickle`, `multiprocessing`, `pathlib`

