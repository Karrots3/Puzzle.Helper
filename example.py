#!/usr/bin/env python3
"""
Example script demonstrating how to use the Puzzle Solver.

This script shows how to:
1. Load puzzle pieces from images
2. Solve the puzzle
3. Visualize the results
4. Save the solution
"""

import os
import sys
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from puzzle_solver import PuzzleSolver


def main():
    """Run the example puzzle solver."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Puzzle Solver Example")
    logger.info("=" * 50)
    
    # Check if sample data exists
    sample_dir = "data/sample_pieces"
    if not os.path.exists(sample_dir):
        logger.warning(f"Sample directory '{sample_dir}' not found.")
        logger.info("Please create the directory and add some puzzle piece images.")
        logger.info("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        return
    
    # Check for image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend([f for f in os.listdir(sample_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        logger.warning(f"No image files found in '{sample_dir}'.")
        logger.info("Please add some puzzle piece images to run this example.")
        return
    
    logger.info(f"Found {len(image_files)} image files in sample directory")
    
    # Create output directory
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize the puzzle solver
        logger.info("Initializing puzzle solver...")
        solver = PuzzleSolver()
        
        # Load puzzle pieces
        logger.info("Loading puzzle pieces...")
        pieces = solver.load_pieces(sample_dir)
        
        if len(pieces) < 2:
            logger.warning("Need at least 2 puzzle pieces to solve.")
            return
        
        logger.info(f"Successfully loaded {len(pieces)} puzzle pieces")
        
        # Solve the puzzle
        logger.info("Solving puzzle...")
        solution = solver.solve(pieces)
        
        # Display results
        metadata = solution.get('metadata', {})
        logger.info("Puzzle solving completed!")
        logger.info(f"Total pieces processed: {metadata.get('total_pieces', 0)}")
        logger.info(f"Total matches found: {metadata.get('total_matches', 0)}")
        logger.info(f"Grid size: {metadata.get('grid_size', (0, 0))}")
        
        # Save solution
        solution_path = os.path.join(output_dir, "example_solution.json")
        solver.save_solution(solution, solution_path)
        logger.info(f"Solution saved to: {solution_path}")
        
        # Create visualization
        viz_path = os.path.join(output_dir, "example_visualization.png")
        solver.visualize_solution(solution, viz_path)
        logger.info(f"Visualization saved to: {viz_path}")
        
        # Save solution image
        img_path = os.path.join(output_dir, "example_solution_image.png")
        solver.visualizer.save_solution_image(solution, img_path)
        logger.info(f"Solution image saved to: {img_path}")
        
        logger.info("Example completed successfully!")
        logger.info(f"Check the '{output_dir}' directory for results.")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


def create_sample_data():
    """Create sample puzzle piece images for testing."""
    import numpy as np
    import cv2
    
    logger = logging.getLogger(__name__)
    
    # Create sample directory
    sample_dir = "data/sample_pieces"
    os.makedirs(sample_dir, exist_ok=True)
    
    logger.info("Creating sample puzzle piece images...")
    
    # Create 6 sample pieces (2x3 grid)
    pieces = []
    
    # Piece 1: Top-left (blue)
    piece1 = np.ones((100, 100, 3), dtype=np.uint8) * [255, 0, 0]  # Blue
    cv2.rectangle(piece1, (10, 10), (90, 90), (0, 255, 0), 2)
    pieces.append(piece1)
    
    # Piece 2: Top-center (green)
    piece2 = np.ones((100, 100, 3), dtype=np.uint8) * [0, 255, 0]  # Green
    cv2.circle(piece2, (50, 50), 30, (255, 0, 0), -1)
    pieces.append(piece2)
    
    # Piece 3: Top-right (red)
    piece3 = np.ones((100, 100, 3), dtype=np.uint8) * [0, 0, 255]  # Red
    cv2.line(piece3, (20, 20), (80, 80), (255, 255, 0), 3)
    pieces.append(piece3)
    
    # Piece 4: Bottom-left (yellow)
    piece4 = np.ones((100, 100, 3), dtype=np.uint8) * [0, 255, 255]  # Yellow
    cv2.ellipse(piece4, (50, 50), (30, 20), 0, 0, 180, (255, 0, 255), -1)
    pieces.append(piece4)
    
    # Piece 5: Bottom-center (magenta)
    piece5 = np.ones((100, 100, 3), dtype=np.uint8) * [255, 0, 255]  # Magenta
    cv2.putText(piece5, "5", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    pieces.append(piece5)
    
    # Piece 6: Bottom-right (cyan)
    piece6 = np.ones((100, 100, 3), dtype=np.uint8) * [255, 255, 0]  # Cyan
    cv2.fillPoly(piece6, [np.array([[20, 20], [80, 20], [50, 80]])], (0, 0, 0))
    pieces.append(piece6)
    
    # Save pieces
    for i, piece in enumerate(pieces, 1):
        filename = os.path.join(sample_dir, f"piece_{i:02d}.png")
        cv2.imwrite(filename, piece)
        logger.info(f"Created: {filename}")
    
    logger.info(f"Created {len(pieces)} sample puzzle pieces in '{sample_dir}'")


if __name__ == "__main__":
    # Check if sample data exists, create if not
    sample_dir = "data/sample_pieces"
    if not os.path.exists(sample_dir) or not os.listdir(sample_dir):
        print("No sample data found. Creating sample puzzle pieces...")
        create_sample_data()
        print()
    
    # Run the example
    main()
