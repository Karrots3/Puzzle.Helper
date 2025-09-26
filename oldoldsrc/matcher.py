"""
Puzzle piece matching and compatibility analysis module.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import image as skimage
import logging


class PieceMatcher:
    """
    Matches puzzle pieces based on their features and compatibility.
    """
    
    def __init__(self):
        """Initialize the piece matcher."""
        self.logger = logging.getLogger(__name__)
    
    def find_matches(self, pieces: List[Dict]) -> List[Dict]:
        """
        Find compatible matches between puzzle pieces.
        
        Args:
            pieces: List of pieces with features
            
        Returns:
            List of matches with compatibility scores
        """
        self.logger.info(f"Finding matches between {len(pieces)} pieces")
        
        matches = []
        
        # Compare each pair of pieces
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                piece1 = pieces[i]
                piece2 = pieces[j]
                
                # Calculate compatibility score
                compatibility = self._calculate_compatibility(piece1, piece2)
                
                # If compatibility is above threshold, add to matches
                if compatibility['score'] > 0.6:  # Threshold for good match
                    match = {
                        'piece1_id': piece1['id'],
                        'piece2_id': piece2['id'],
                        'compatibility': compatibility,
                        'direction': self._determine_direction(piece1, piece2, compatibility)
                    }
                    matches.append(match)
        
        # Sort matches by compatibility score
        matches.sort(key=lambda x: x['compatibility']['score'], reverse=True)
        
        self.logger.info(f"Found {len(matches)} potential matches")
        return matches
    
    def _calculate_compatibility(self, piece1: Dict, piece2: Dict) -> Dict:
        """
        Calculate compatibility between two puzzle pieces.
        
        Args:
            piece1: First piece with features
            piece2: Second piece with features
            
        Returns:
            Dictionary containing compatibility analysis
        """
        compatibility = {}
        
        # Feature-based similarity
        feature_similarity = self._calculate_feature_similarity(piece1, piece2)
        compatibility['feature_similarity'] = feature_similarity
        
        # Edge compatibility
        edge_compatibility = self._calculate_edge_compatibility(piece1, piece2)
        compatibility['edge_compatibility'] = edge_compatibility
        
        # Color compatibility
        color_compatibility = self._calculate_color_compatibility(piece1, piece2)
        compatibility['color_compatibility'] = color_compatibility
        
        # Pattern compatibility
        pattern_compatibility = self._calculate_pattern_compatibility(piece1, piece2)
        compatibility['pattern_compatibility'] = pattern_compatibility
        
        # Overall compatibility score (weighted average)
        overall_score = (
            feature_similarity * 0.3 +
            edge_compatibility * 0.3 +
            color_compatibility * 0.2 +
            pattern_compatibility * 0.2
        )
        
        compatibility['score'] = overall_score
        
        return compatibility
    
    def _calculate_feature_similarity(self, piece1: Dict, piece2: Dict) -> float:
        """
        Calculate similarity based on extracted features.
        
        Args:
            piece1: First piece with features
            piece2: Second piece with features
            
        Returns:
            Similarity score between 0 and 1
        """
        # Extract feature vectors
        features1 = self._extract_feature_vector(piece1)
        features2 = self._extract_feature_vector(piece2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([features1], [features2])[0][0]
        
        # Ensure similarity is between 0 and 1
        similarity = max(0, min(1, similarity))
        
        return similarity
    
    def _extract_feature_vector(self, piece: Dict) -> np.ndarray:
        """
        Extract a feature vector from a piece for comparison.
        
        Args:
            piece: Piece with features
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Geometric features
        if 'geometry' in piece['features']:
            geom = piece['features']['geometry']
            features.extend([
                geom.get('area', 0),
                geom.get('perimeter', 0),
                geom.get('aspect_ratio', 0),
                geom.get('circularity', 0),
                geom.get('compactness', 0)
            ])
        
        # Edge features
        if 'edges' in piece['features']:
            edges = piece['features']['edges']
            features.extend([
                edges.get('edge_density', 0),
                edges.get('gradient_mean', 0),
                edges.get('gradient_std', 0),
                edges.get('gradient_max', 0),
                edges.get('direction_entropy', 0)
            ])
        
        # Color features
        if 'colors' in piece['features']:
            colors = piece['features']['colors']
            if 'mean_rgb' in colors:
                features.extend(colors['mean_rgb'])
            if 'std_rgb' in colors:
                features.extend(colors['std_rgb'])
        
        # Pattern features
        if 'patterns' in piece['features']:
            patterns = piece['features']['patterns']
            features.extend([
                patterns.get('entropy', 0),
                patterns.get('energy', 0),
                patterns.get('contrast', 0),
                patterns.get('homogeneity', 0)
            ])
        
        # Corner features
        if 'corners' in piece['features']:
            corners = piece['features']['corners']
            features.extend([
                corners.get('corner_count', 0),
                corners.get('corner_density', 0),
                corners.get('corner_strength_mean', 0),
                corners.get('corner_strength_max', 0)
            ])
        
        # Normalize features
        features = np.array(features, dtype=float)
        if len(features) > 0:
            # Remove NaN values
            features = np.nan_to_num(features, nan=0.0)
            # Normalize to [0, 1] range
            if np.max(features) > 0:
                features = features / np.max(features)
        
        return features
    
    def _calculate_edge_compatibility(self, piece1: Dict, piece2: Dict) -> float:
        """
        Calculate edge compatibility between two pieces.
        
        Args:
            piece1: First piece with features
            piece2: Second piece with features
            
        Returns:
            Edge compatibility score between 0 and 1
        """
        if 'edges' not in piece1['features'] or 'edges' not in piece2['features']:
            return 0.0
        
        edges1 = piece1['features']['edges']
        edges2 = piece2['features']['edges']
        
        # Compare edge characteristics
        edge_density_diff = abs(edges1.get('edge_density', 0) - edges2.get('edge_density', 0))
        gradient_mean_diff = abs(edges1.get('gradient_mean', 0) - edges2.get('gradient_mean', 0))
        direction_entropy_diff = abs(edges1.get('direction_entropy', 0) - edges2.get('direction_entropy', 0))
        
        # Normalize differences
        edge_density_similarity = max(0, 1 - edge_density_diff)
        gradient_similarity = max(0, 1 - gradient_mean_diff / 100)  # Normalize by expected range
        direction_similarity = max(0, 1 - direction_entropy_diff / 3)  # Normalize by expected range
        
        # Weighted average
        edge_compatibility = (
            edge_density_similarity * 0.4 +
            gradient_similarity * 0.4 +
            direction_similarity * 0.2
        )
        
        return edge_compatibility
    
    def _calculate_color_compatibility(self, piece1: Dict, piece2: Dict) -> float:
        """
        Calculate color compatibility between two pieces.
        
        Args:
            piece1: First piece with features
            piece2: Second piece with features
            
        Returns:
            Color compatibility score between 0 and 1
        """
        if 'colors' not in piece1['features'] or 'colors' not in piece2['features']:
            return 0.0
        
        colors1 = piece1['features']['colors']
        colors2 = piece2['features']['colors']
        
        # Compare mean RGB values
        if 'mean_rgb' in colors1 and 'mean_rgb' in colors2:
            mean_rgb_diff = np.linalg.norm(
                np.array(colors1['mean_rgb']) - np.array(colors2['mean_rgb'])
            )
            # Normalize by maximum possible difference (255 * sqrt(3))
            rgb_similarity = max(0, 1 - mean_rgb_diff / (255 * np.sqrt(3)))
        else:
            rgb_similarity = 0.0
        
        # Compare dominant colors
        dominant_color_similarity = self._compare_dominant_colors(colors1, colors2)
        
        # Weighted average
        color_compatibility = rgb_similarity * 0.7 + dominant_color_similarity * 0.3
        
        return color_compatibility
    
    def _compare_dominant_colors(self, colors1: Dict, colors2: Dict) -> float:
        """
        Compare dominant colors between two pieces.
        
        Args:
            colors1: Color features of first piece
            colors2: Color features of second piece
            
        Returns:
            Similarity score between 0 and 1
        """
        if 'dominant_colors' not in colors1 or 'dominant_colors' not in colors2:
            return 0.0
        
        dom_colors1 = colors1['dominant_colors']
        dom_colors2 = colors2['dominant_colors']
        
        # Compare top 3 dominant colors
        similarities = []
        for color1, _ in dom_colors1[:3]:
            for color2, _ in dom_colors2[:3]:
                # Calculate color distance
                color_diff = np.linalg.norm(np.array(color1) - np.array(color2))
                similarity = max(0, 1 - color_diff / (255 * np.sqrt(3)))
                similarities.append(similarity)
        
        if similarities:
            return max(similarities)  # Best match
        else:
            return 0.0
    
    def _calculate_pattern_compatibility(self, piece1: Dict, piece2: Dict) -> float:
        """
        Calculate pattern compatibility between two pieces.
        
        Args:
            piece1: First piece with features
            piece2: Second piece with features
            
        Returns:
            Pattern compatibility score between 0 and 1
        """
        if 'patterns' not in piece1['features'] or 'patterns' not in piece2['features']:
            return 0.0
        
        patterns1 = piece1['features']['patterns']
        patterns2 = piece2['features']['patterns']
        
        # Compare pattern characteristics
        entropy_diff = abs(patterns1.get('entropy', 0) - patterns2.get('entropy', 0))
        energy_diff = abs(patterns1.get('energy', 0) - patterns2.get('energy', 0))
        contrast_diff = abs(patterns1.get('contrast', 0) - patterns2.get('contrast', 0))
        homogeneity_diff = abs(patterns1.get('homogeneity', 0) - patterns2.get('homogeneity', 0))
        
        # Normalize differences
        entropy_similarity = max(0, 1 - entropy_diff / 8)  # Normalize by expected range
        energy_similarity = max(0, 1 - energy_diff / 10000)  # Normalize by expected range
        contrast_similarity = max(0, 1 - contrast_diff / 100)  # Normalize by expected range
        homogeneity_similarity = max(0, 1 - homogeneity_diff)
        
        # Weighted average
        pattern_compatibility = (
            entropy_similarity * 0.3 +
            energy_similarity * 0.3 +
            contrast_similarity * 0.2 +
            homogeneity_similarity * 0.2
        )
        
        return pattern_compatibility
    
    def _determine_direction(self, piece1: Dict, piece2: Dict, compatibility: Dict) -> str:
        """
        Determine the relative direction between two pieces.
        
        Args:
            piece1: First piece
            piece2: Second piece
            compatibility: Compatibility analysis
            
        Returns:
            Direction string ('right', 'left', 'up', 'down', 'unknown')
        """
        # This is a simplified direction determination
        # In a real implementation, this would analyze the actual edge shapes
        
        # For now, return a random direction based on piece IDs
        # This is just a placeholder - real implementation would be much more sophisticated
        piece_id_sum = sum(ord(c) for c in piece1['id'] + piece2['id'])
        directions = ['right', 'left', 'up', 'down']
        return directions[piece_id_sum % 4]
    
    def find_best_matches(self, pieces: List[Dict], target_piece_id: str, max_matches: int = 5) -> List[Dict]:
        """
        Find the best matches for a specific piece.
        
        Args:
            pieces: List of all pieces
            target_piece_id: ID of the target piece
            max_matches: Maximum number of matches to return
            
        Returns:
            List of best matches for the target piece
        """
        target_piece = None
        for piece in pieces:
            if piece['id'] == target_piece_id:
                target_piece = piece
                break
        
        if target_piece is None:
            return []
        
        # Calculate compatibility with all other pieces
        matches = []
        for piece in pieces:
            if piece['id'] != target_piece_id:
                compatibility = self._calculate_compatibility(target_piece, piece)
                match = {
                    'piece_id': piece['id'],
                    'compatibility': compatibility,
                    'direction': self._determine_direction(target_piece, piece, compatibility)
                }
                matches.append(match)
        
        # Sort by compatibility score and return top matches
        matches.sort(key=lambda x: x['compatibility']['score'], reverse=True)
        return matches[:max_matches]
    
    def validate_match(self, piece1: Dict, piece2: Dict, direction: str) -> bool:
        """
        Validate if two pieces can actually fit together in the given direction.
        
        Args:
            piece1: First piece
            piece2: Second piece
            direction: Proposed direction of connection
            
        Returns:
            True if the match is valid, False otherwise
        """
        # This is a simplified validation
        # In a real implementation, this would:
        # 1. Analyze the actual edge shapes
        # 2. Check if the edges are complementary
        # 3. Verify the pieces can physically connect
        
        # For now, just check if compatibility is high enough
        compatibility = self._calculate_compatibility(piece1, piece2)
        return compatibility['score'] > 0.7
    
    def get_piece_neighbors(self, pieces: List[Dict], piece_id: str) -> List[str]:
        """
        Get the IDs of pieces that are likely neighbors of the given piece.
        
        Args:
            pieces: List of all pieces
            piece_id: ID of the target piece
            
        Returns:
            List of neighbor piece IDs
        """
        best_matches = self.find_best_matches(pieces, piece_id, max_matches=4)
        return [match['piece_id'] for match in best_matches]
