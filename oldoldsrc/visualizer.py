"""
Solution visualization module for puzzle solving results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple
import logging


class SolutionVisualizer:
    """
    Visualizes puzzle solutions and provides interactive feedback.
    """
    
    def __init__(self):
        """Initialize the solution visualizer."""
        self.logger = logging.getLogger(__name__)
        plt.style.use('default')
    
    def visualize(self, solution: Dict, output_path: Optional[str] = None) -> None:
        """
        Visualize the puzzle solution.
        
        Args:
            solution: Solution dictionary from solver
            output_path: Optional path to save the visualization
        """
        self.logger.info("Creating puzzle solution visualization")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Puzzle Solution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Solution grid
        self._plot_solution_grid(axes[0, 0], solution)
        
        # Plot 2: Piece connections
        self._plot_piece_connections(axes[0, 1], solution)
        
        # Plot 3: Match quality distribution
        self._plot_match_quality(axes[1, 0], solution)
        
        # Plot 4: Solution statistics
        self._plot_solution_stats(axes[1, 1], solution)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to: {output_path}")
        
        plt.show()
    
    def _plot_solution_grid(self, ax, solution: Dict) -> None:
        """Plot the solution grid showing piece positions."""
        ax.set_title('Solution Grid', fontweight='bold')
        
        grid = solution.get('solution', {}).get('grid', {})
        grid_size = solution.get('solution', {}).get('grid_size', (0, 0))
        
        if not grid:
            ax.text(0.5, 0.5, 'No solution grid available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create grid visualization
        width, height = grid_size
        if width == 0 or height == 0:
            ax.text(0.5, 0.5, 'Empty grid', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create grid matrix
        grid_matrix = np.zeros((height, width), dtype=int)
        piece_positions = {}
        
        for (x, y), piece_id in grid.items():
            # Adjust coordinates to be non-negative
            adj_x = x - min(pos[0] for pos in grid.keys())
            adj_y = y - min(pos[1] for pos in grid.keys())
            
            if 0 <= adj_x < width and 0 <= adj_y < height:
                grid_matrix[adj_y, adj_x] = 1
                piece_positions[(adj_x, adj_y)] = piece_id
        
        # Plot grid
        im = ax.imshow(grid_matrix, cmap='Blues', alpha=0.7)
        
        # Add piece labels
        for (x, y), piece_id in piece_positions.items():
            ax.text(x, y, piece_id, ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='red')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _plot_piece_connections(self, ax, solution: Dict) -> None:
        """Plot piece connections and matches."""
        ax.set_title('Piece Connections', fontweight='bold')
        
        matches = solution.get('matches', [])
        pieces = solution.get('pieces', [])
        
        if not matches:
            ax.text(0.5, 0.5, 'No matches found', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create piece positions (simplified layout)
        piece_positions = {}
        for i, piece in enumerate(pieces):
            # Arrange pieces in a circle
            angle = 2 * np.pi * i / len(pieces)
            x = np.cos(angle)
            y = np.sin(angle)
            piece_positions[piece['id']] = (x, y)
        
        # Plot pieces
        for piece_id, (x, y) in piece_positions.items():
            ax.scatter(x, y, s=100, alpha=0.7, label=piece_id)
            ax.text(x, y, piece_id, ha='center', va='center', fontsize=8)
        
        # Plot connections
        for match in matches[:10]:  # Limit to top 10 matches
            piece1_id = match['piece1_id']
            piece2_id = match['piece2_id']
            
            if piece1_id in piece_positions and piece2_id in piece_positions:
                x1, y1 = piece_positions[piece1_id]
                x2, y2 = piece_positions[piece2_id]
                
                # Line width based on compatibility score
                score = match['compatibility']['score']
                linewidth = 1 + score * 3
                
                ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.6, linewidth=linewidth)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_match_quality(self, ax, solution: Dict) -> None:
        """Plot distribution of match quality scores."""
        ax.set_title('Match Quality Distribution', fontweight='bold')
        
        matches = solution.get('matches', [])
        
        if not matches:
            ax.text(0.5, 0.5, 'No matches to analyze', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Extract compatibility scores
        scores = [match['compatibility']['score'] for match in matches]
        
        # Create histogram
        ax.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(scores):.3f}')
        ax.axvline(np.median(scores), color='green', linestyle='--', 
                  label=f'Median: {np.median(scores):.3f}')
        
        ax.set_xlabel('Compatibility Score')
        ax.set_ylabel('Number of Matches')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_solution_stats(self, ax, solution: Dict) -> None:
        """Plot solution statistics."""
        ax.set_title('Solution Statistics', fontweight='bold')
        
        # Extract statistics
        metadata = solution.get('metadata', {})
        pieces = solution.get('pieces', [])
        matches = solution.get('matches', [])
        
        # Calculate additional statistics
        total_pieces = len(pieces)
        total_matches = len(matches)
        grid_size = metadata.get('grid_size', (0, 0))
        
        if matches:
            avg_compatibility = np.mean([m['compatibility']['score'] for m in matches])
            max_compatibility = np.max([m['compatibility']['score'] for m in matches])
        else:
            avg_compatibility = 0
            max_compatibility = 0
        
        # Create statistics display
        stats_text = f"""
        Total Pieces: {total_pieces}
        Total Matches: {total_matches}
        Grid Size: {grid_size[0]} x {grid_size[1]}
        Avg Compatibility: {avg_compatibility:.3f}
        Max Compatibility: {max_compatibility:.3f}
        """
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def visualize_piece_comparison(self, piece1: Dict, piece2: Dict, 
                                 compatibility: Dict, output_path: Optional[str] = None) -> None:
        """
        Visualize comparison between two pieces.
        
        Args:
            piece1: First piece data
            piece2: Second piece data
            compatibility: Compatibility analysis
            output_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Piece Comparison: {piece1["id"]} vs {piece2["id"]}', 
                    fontsize=16, fontweight='bold')
        
        # Plot piece images
        if 'image' in piece1:
            axes[0, 0].imshow(piece1['image'])
            axes[0, 0].set_title(f'Piece {piece1["id"]}')
            axes[0, 0].axis('off')
        
        if 'image' in piece2:
            axes[0, 1].imshow(piece2['image'])
            axes[0, 1].set_title(f'Piece {piece2["id"]}')
            axes[0, 1].axis('off')
        
        # Plot compatibility scores
        compatibility_scores = {
            'Feature Similarity': compatibility.get('feature_similarity', 0),
            'Edge Compatibility': compatibility.get('edge_compatibility', 0),
            'Color Compatibility': compatibility.get('color_compatibility', 0),
            'Pattern Compatibility': compatibility.get('pattern_compatibility', 0),
            'Overall Score': compatibility.get('score', 0)
        }
        
        categories = list(compatibility_scores.keys())
        scores = list(compatibility_scores.values())
        
        bars = axes[0, 2].bar(categories, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'purple'])
        axes[0, 2].set_title('Compatibility Scores')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # Plot feature comparisons
        self._plot_feature_comparison(axes[1, 0], piece1, piece2, 'geometry')
        self._plot_feature_comparison(axes[1, 1], piece1, piece2, 'colors')
        self._plot_feature_comparison(axes[1, 2], piece1, piece2, 'patterns')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison visualization saved to: {output_path}")
        
        plt.show()
    
    def _plot_feature_comparison(self, ax, piece1: Dict, piece2: Dict, feature_type: str) -> None:
        """Plot comparison of specific features between two pieces."""
        if 'features' not in piece1 or 'features' not in piece2:
            ax.text(0.5, 0.5, f'No {feature_type} features available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        features1 = piece1['features'].get(feature_type, {})
        features2 = piece2['features'].get(feature_type, {})
        
        if not features1 or not features2:
            ax.text(0.5, 0.5, f'No {feature_type} data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get common features
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            ax.text(0.5, 0.5, f'No common {feature_type} features', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plot comparison
        keys = list(common_keys)[:5]  # Limit to 5 features
        values1 = [features1[k] for k in keys]
        values2 = [features2[k] for k in keys]
        
        x = np.arange(len(keys))
        width = 0.35
        
        ax.bar(x - width/2, values1, width, label=f'Piece {piece1["id"]}', alpha=0.7)
        ax.bar(x + width/2, values2, width, label=f'Piece {piece2["id"]}', alpha=0.7)
        
        ax.set_title(f'{feature_type.title()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_solution_animation(self, solution: Dict, output_path: str) -> None:
        """
        Create an animation showing the puzzle assembly process.
        
        Args:
            solution: Solution dictionary
            output_path: Path to save the animation
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            self.logger.warning("matplotlib.animation not available, skipping animation")
            return
        
        self.logger.info("Creating solution animation")
        
        # This is a placeholder for animation creation
        # In a real implementation, this would show pieces being placed one by one
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title('Puzzle Assembly Animation')
        
        # Create a simple animation showing the grid being filled
        grid = solution.get('solution', {}).get('grid', {})
        
        if not grid:
            ax.text(0.5, 0.5, 'No solution to animate', 
                   ha='center', va='center', transform=ax.transAxes)
            plt.savefig(output_path)
            return
        
        # Create animation frames
        def animate(frame):
            ax.clear()
            ax.set_title(f'Puzzle Assembly - Step {frame + 1}')
            
            # Show partial solution based on frame
            partial_grid = {}
            grid_items = list(grid.items())
            num_items = len(grid_items)
            items_to_show = min(frame + 1, num_items)
            
            for i in range(items_to_show):
                pos, piece_id = grid_items[i]
                partial_grid[pos] = piece_id
            
            # Plot partial grid
            self._plot_solution_grid(ax, {'solution': {'grid': partial_grid}})
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(grid), 
                                     interval=500, repeat=False)
        
        # Save animation
        anim.save(output_path, writer='pillow', fps=2)
        self.logger.info(f"Animation saved to: {output_path}")
    
    def save_solution_image(self, solution: Dict, output_path: str) -> None:
        """
        Save a simple image representation of the solution.
        
        Args:
            solution: Solution dictionary
            output_path: Path to save the image
        """
        self.logger.info(f"Saving solution image to: {output_path}")
        
        # Create a simple grid visualization
        grid = solution.get('solution', {}).get('grid', {})
        
        if not grid:
            # Create empty image
            img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            cv2.putText(img, 'No Solution', (150, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            # Calculate grid dimensions
            x_coords = [pos[0] for pos in grid.keys()]
            y_coords = [pos[1] for pos in grid.keys()]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            
            # Create grid image
            cell_size = 50
            img_width = width * cell_size
            img_height = height * cell_size
            
            img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
            
            # Draw grid cells
            for (x, y), piece_id in grid.items():
                adj_x = x - min_x
                adj_y = y - min_y
                
                # Draw cell
                cv2.rectangle(img, 
                            (adj_x * cell_size, adj_y * cell_size),
                            ((adj_x + 1) * cell_size, (adj_y + 1) * cell_size),
                            (0, 0, 0), 2)
                
                # Add piece ID
                cv2.putText(img, piece_id, 
                           (adj_x * cell_size + 5, adj_y * cell_size + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save image
        cv2.imwrite(output_path, img)
        self.logger.info(f"Solution image saved to: {output_path}")
