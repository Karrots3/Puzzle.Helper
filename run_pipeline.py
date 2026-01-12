#!/usr/bin/env python3
"""
Main entry point for the puzzle piece matching pipeline.

This script runs the complete pipeline:
1. Processes images to extract puzzle pieces
2. Finds matches between all pairs of pieces
"""

from pipeline.main import main

if __name__ == "__main__":
    main()
