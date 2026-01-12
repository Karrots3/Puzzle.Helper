# Pipeline Migration Summary

## What Was Done

I've created a clean, organized pipeline structure in the `pipeline/` folder that:

1. **Extracts the working preprocessing logic** from `nb_pieces.py` (marimo notebook) into regular Python modules
2. **Extracts the working matching logic** from `match.py` (marimo notebook) into regular Python modules
3. **Fixes the Edge class** to support `normalized_contour` computation (lazy property)
4. **Adds missing constants** (MIN_LEN_EDGE, SAMPLE_POINTS, MAX_SCORE_MSE_DISTANCE)
5. **Creates a proper main script** that orchestrates the complete pipeline

## New Structure

```
pipeline/
├── __init__.py          # Package initialization
├── config.py            # Configuration constants
├── models.py            # Data models (Edge, Piece, Match, etc.)
├── preprocessing.py     # Image preprocessing pipeline
├── matching.py          # Matching algorithm
├── main.py              # Main orchestration script
└── README.md            # Documentation
```

## Key Improvements

1. **No marimo dependencies**: All code is now regular Python, easier to run and maintain
2. **Proper structure**: Clear separation of concerns (models, preprocessing, matching)
3. **Fixed Edge class**: Now computes `normalized_contour` on-demand as a property
4. **Error handling**: Proper exception handling with logging
5. **Multiprocessing**: Maintains parallel processing for efficiency
6. **Documentation**: README with usage examples

## Usage

### Run the complete pipeline:
```bash
python run_pipeline.py
```

Or:
```python
from pipeline.main import main
main()
```

### Process a single image:
```python
from pipeline.preprocessing import process_image, PipelineProcessImgParams

params = PipelineProcessImgParams()
piece = process_image("data/0001.JPG", params)
```

### Match two pieces:
```python
from pipeline.matching import find_match, PipelineParamsMatching
import pickle

with open("data/pieces/0001.pkl", "rb") as f:
    piece1 = pickle.load(f)
with open("data/pieces/0002.pkl", "rb") as f:
    piece2 = pickle.load(f)

params = PipelineParamsMatching()
match = find_match(piece1, piece2, params)
```

## What's Preserved

- ✅ All image preprocessing steps (cropping, thresholding, contour extraction, corner detection, edge classification)
- ✅ All matching algorithm logic (sliding window, embedding computation)
- ✅ All parameters and configuration values
- ✅ Error handling and logging

## What's Fixed

- ✅ Edge class now properly computes `normalized_contour` when needed
- ✅ Missing constants added to `config.py`
- ✅ Code flow is now clean and organized
- ✅ No dependencies on marimo notebooks
- ✅ Proper imports and module structure

## Old Code

The old code in `src/c_matcher.py`, `match.py`, `nb_pieces.py`, etc. is preserved but not used by the new pipeline. You can reference it if needed, but the new `pipeline/` folder contains the clean, working version.
