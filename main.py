#!/usr/bin/env python3
"""
Main entry point for the Puzzle Solver application.

This script provides a command-line interface for solving puzzles
by analyzing photos of puzzle pieces.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from puzzle_solver import PuzzleSolver


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('puzzle_solver.log')
        ]
    )


def validate_input_path(input_path: str) -> bool:
    """Validate that the input path exists and contains images."""
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        return False
    
    if not os.path.isdir(input_path):
        print(f"Error: Input path '{input_path}' is not a directory.")
        return False
    
    # Check for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(input_path)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    
    if not image_files:
        print(f"Error: No image files found in '{input_path}'.")
        print(f"Supported formats: {', '.join(image_extensions)}")
        return False
    
    print(f"Found {len(image_files)} image files in '{input_path}'")
    return True


def create_output_directory(output_path: str) -> bool:
    """Create output directory if it doesn't exist."""
    try:
        os.makedirs(output_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating output directory '{output_path}': {e}")
        return False


def main():
    """Main function for the puzzle solver application."""
    parser = argparse.ArgumentParser(
        description='Puzzle Solver - Analyze photos of puzzle pieces and find correct arrangements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/sample_pieces --output data/results
  python main.py --input puzzle_photos/ --output solution/ --verbose
  python main.py --input pieces/ --output results/ --save-solution --visualize
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to directory containing puzzle piece images'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='data/results',
        help='Path to output directory for results (default: data/results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--save-solution',
        action='store_true',
        help='Save solution data to JSON file'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate and save visualization plots'
    )
    
    parser.add_argument(
        '--save-image',
        action='store_true',
        help='Save solution as an image file'
    )
    
    parser.add_argument(
        '--animation',
        action='store_true',
        help='Create animation of puzzle assembly process'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Puzzle Solver application")
    
    # Validate input path
    if not validate_input_path(args.input):
        sys.exit(1)
    
    # Create output directory
    if not create_output_directory(args.output):
        sys.exit(1)
    
    try:
        # Load configuration if provided
        config = None
        if args.config:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
        
        # Initialize puzzle solver
        logger.info("Initializing puzzle solver")
        solver = PuzzleSolver(config=config)
        
        # Load puzzle pieces
        logger.info("Loading puzzle pieces")
        pieces = solver.load_pieces(args.input)
        
        if not pieces:
            logger.error("No valid puzzle pieces found")
            sys.exit(1)
        
        logger.info(f"Successfully loaded {len(pieces)} puzzle pieces")
        
        # Solve the puzzle
        logger.info("Solving puzzle")
        solution = solver.solve(pieces)
        
        # Display solution summary
        metadata = solution.get('metadata', {})
        logger.info("Puzzle solving completed!")
        logger.info(f"Total pieces processed: {metadata.get('total_pieces', 0)}")
        logger.info(f"Total matches found: {metadata.get('total_matches', 0)}")
        logger.info(f"Grid size: {metadata.get('grid_size', (0, 0))}")
        
        # Save solution data
        if args.save_solution:
            solution_path = os.path.join(args.output, 'solution.json')
            solver.save_solution(solution, solution_path)
            logger.info(f"Solution saved to: {solution_path}")
        
        # Generate visualizations
        if args.visualize:
            viz_path = os.path.join(args.output, 'solution_visualization.png')
            solver.visualize_solution(solution, viz_path)
            logger.info(f"Visualization saved to: {viz_path}")
        
        # Save solution image
        if args.save_image:
            img_path = os.path.join(args.output, 'solution_image.png')
            solver.visualizer.save_solution_image(solution, img_path)
            logger.info(f"Solution image saved to: {img_path}")
        
        # Create animation
        if args.animation:
            anim_path = os.path.join(args.output, 'solution_animation.gif')
            solver.visualizer.create_solution_animation(solution, anim_path)
            logger.info(f"Animation saved to: {anim_path}")
        
        # Show visualization if requested
        if args.visualize and not args.save_solution:
            solver.visualize_solution(solution)
        
        logger.info("Puzzle solving process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
