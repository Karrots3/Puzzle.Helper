import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import argparse

class PuzzleDetector:
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.results_folder = self.data_folder / "results"
        self.results_folder.mkdir(exist_ok=True)
        
    def preprocess_image(self, img_rgb):
        """
        Preprocess image using adaptive thresholding and morphological operations
        """
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Adaptive thresholding (the method you liked)
        img_binary = cv2.adaptiveThreshold(
            img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        
        # Remove small noise first
        img_cleaned = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        
        # Use more aggressive erosion to separate pieces
        kernel_erode = np.ones((3, 3), np.uint8)
        img_eroded = cv2.erode(img_cleaned, kernel_erode, iterations=2)
        
        # Then dilate back to restore piece size
        img_dilated = cv2.dilate(img_eroded, kernel_erode, iterations=1)
        
        return {
            'original': img_rgb,
            'gray': img_gray,
            'blurred': img_blurred,
            'binary': img_binary,
            'cleaned': img_cleaned,
            'eroded': img_eroded,
            'final': img_dilated
        }
    
    def find_contours(self, img_binary):
        """
        Find contours in the binary image
        """
        # Find contours
        contours, hierarchy = cv2.findContours(
            img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area to remove noise
        min_area = 1000  # Adjust based on your puzzle piece size
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                valid_contours.append(contour)
        
        return valid_contours
    
    def find_contours_with_adaptive_threshold(self, img_binary, min_area=1000):
        """
        Find contours with adaptive area thresholding
        """
        contours, hierarchy = cv2.findContours(
            img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                valid_contours.append(contour)
        
        return valid_contours
    
    def preprocess_with_connected_components(self, img_rgb):
        """
        Preprocessing using connected components to separate pieces
        """
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Adaptive thresholding
        img_binary = cv2.adaptiveThreshold(
            img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        img_cleaned = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_cleaned, connectivity=8)
        
        # Create a new image with only the largest components (excluding background)
        img_components = np.zeros_like(img_cleaned)
        
        # Filter components by area (exclude very small and very large ones)
        min_area = 1000
        max_area = img_cleaned.shape[0] * img_cleaned.shape[1] // 4  # Max 25% of image
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                img_components[labels == i] = 255
        
        return {
            'original': img_rgb,
            'gray': img_gray,
            'binary': img_binary,
            'cleaned': img_cleaned,
            'labels': labels,
            'stats': stats,
            'final': img_components
        }
    
    def preprocess_with_watershed(self, img_rgb):
        """
        Alternative preprocessing using watershed segmentation to separate touching pieces
        """
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Adaptive thresholding
        img_binary = cv2.adaptiveThreshold(
            img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        img_cleaned = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
        
        # Find sure background area
        sure_bg = cv2.dilate(img_cleaned, kernel, iterations=3)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(img_cleaned, cv2.DIST_L2, 5)
        
        # Find sure foreground area
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Finding unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(img_rgb, markers)
        
        # Create binary image from watershed result
        img_watershed = np.zeros_like(img_cleaned)
        img_watershed[markers > 1] = 255
        
        return {
            'original': img_rgb,
            'gray': img_gray,
            'binary': img_binary,
            'cleaned': img_cleaned,
            'sure_bg': sure_bg,
            'sure_fg': sure_fg,
            'unknown': unknown,
            'markers': markers,
            'final': img_watershed
        }
    
    def analyze_contour(self, contour):
        """
        Analyze a single contour to extract features
        """
        # Basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int8(box)
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Solidity (area / hull_area)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Extent (area / bounding rectangle area)
        extent = area / (w * h) if w * h > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'bounding_rect': (x, y, w, h),
            'min_area_rect': rect,
            'box_points': box,
            'hull': hull,
            'solidity': solidity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'contour': contour
        }
    
    def detect_puzzle_pieces(self, img_path):
        """
        Main function to detect puzzle pieces in an image
        """
        # Read image
        img_rgb = cv2.imread(str(img_path))
        if img_rgb is None:
            print(f"Could not read image: {img_path}")
            return None
        
        # Resize if too large
        h, w = img_rgb.shape[:2]
        if max(h, w) > 2000:
            scale = 2000 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        
        # Choose preprocessing method
        if hasattr(self, 'method') and self.method != 'auto':
            method_used = self.method
        else:
            # Try all preprocessing methods
            processed_morph = self.preprocess_image(img_rgb)
            processed_watershed = self.preprocess_with_watershed(img_rgb)
            processed_components = self.preprocess_with_connected_components(img_rgb)
            
            # Find contours with all methods
            contours_morph = self.find_contours(processed_morph['final'])
            contours_watershed = self.find_contours(processed_watershed['final'])
            contours_components = self.find_contours(processed_components['final'])
            
            # Choose the method that finds more pieces (but not too many)
            best_count = max(len(contours_morph), len(contours_watershed), len(contours_components))
            if best_count <= 200:
                if len(contours_components) == best_count:
                    method_used = 'connected_components'
                elif len(contours_morph) == best_count:
                    method_used = 'morphological'
                else:
                    method_used = 'watershed'
            else:
                method_used = 'watershed'  # Default to watershed if too many
        
        # Apply the chosen method
        if method_used == 'morphological':
            processed = self.preprocess_image(img_rgb)
        elif method_used == 'connected_components':
            processed = self.preprocess_with_connected_components(img_rgb)
        else:  # watershed
            processed = self.preprocess_with_watershed(img_rgb)
        
        # Find contours with adaptive thresholding
        contours = self.find_contours_with_adaptive_threshold(processed['final'])
        
        # If too few contours found, try with smaller area threshold
        if len(contours) < 5:
            print(f"Only {len(contours)} contours found, trying with smaller area threshold...")
            contours = self.find_contours_with_adaptive_threshold(processed['final'], min_area=500)
            if len(contours) < 5:
                contours = self.find_contours_with_adaptive_threshold(processed['final'], min_area=200)
        
        print(f"Using {method_used} method - found {len(contours)} contours")
        
        # Analyze each contour
        pieces = []
        for i, contour in enumerate(contours):
            analysis = self.analyze_contour(contour)
            analysis['piece_id'] = i
            pieces.append(analysis)
        
        return {
            'image_path': str(img_path),
            'processed_images': processed,
            'contours': contours,
            'pieces': pieces,
            'num_pieces': len(pieces),
            'method_used': method_used
        }
    
    def visualize_results(self, results, save_path=None):
        """
        Visualize the detection results
        """
        if results is None:
            return
        
        processed = results['processed_images']
        contours = results['contours']
        pieces = results['pieces']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Puzzle Piece Detection Results - {Path(results["image_path"]).name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(processed['original'], cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Binary (adaptive threshold)
        axes[0, 1].imshow(processed['binary'], cmap='gray')
        axes[0, 1].set_title('Adaptive Threshold')
        axes[0, 1].axis('off')
        
        # Cleaned (opening)
        axes[0, 2].imshow(processed['cleaned'], cmap='gray')
        axes[0, 2].set_title('Cleaned (Opening)')
        axes[0, 2].axis('off')
        
        # Final processed or eroded
        if 'eroded' in processed:
            axes[0, 3].imshow(processed['eroded'], cmap='gray')
            axes[0, 3].set_title('Eroded (Separate)')
        else:
            axes[0, 3].imshow(processed['final'], cmap='gray')
            axes[0, 3].set_title('Final Processed')
        axes[0, 3].axis('off')
        
        # Contours on original
        img_with_contours = processed['original'].copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
        axes[1, 0].imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f'Contours ({len(contours)} found)')
        axes[1, 0].axis('off')
        
        # Individual pieces
        if pieces:
            # Show first few pieces
            num_to_show = min(3, len(pieces))
            for i in range(num_to_show):
                piece = pieces[i]
                x, y, w, h = piece['bounding_rect']
                
                # Extract piece region
                piece_img = processed['original'][y:y+h, x:x+w]
                
                # Draw bounding rectangle
                cv2.rectangle(piece_img, (0, 0), (w-1, h-1), (0, 255, 0), 2)
                
                axes[1, i+1].imshow(cv2.cvtColor(piece_img, cv2.COLOR_BGR2RGB))
                axes[1, i+1].set_title(f'Piece {i+1}\nArea: {piece["area"]:.0f}')
                axes[1, i+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, results, output_name=None):
        """
        Save detection results to JSON
        """
        if results is None:
            return
        
        # Prepare data for JSON serialization
        json_data = {
            'image_path': results['image_path'],
            'num_pieces': results['num_pieces'],
            'pieces': []
        }
        
        for piece in results['pieces']:
            # Convert numpy arrays to lists for JSON serialization
            piece_data = {
                'piece_id': piece['piece_id'],
                'area': float(piece['area']),
                'perimeter': float(piece['perimeter']),
                'bounding_rect': piece['bounding_rect'],
                'solidity': float(piece['solidity']),
                'aspect_ratio': float(piece['aspect_ratio']),
                'extent': float(piece['extent']),
                'contour_points': piece['contour'].tolist()
            }
            json_data['pieces'].append(piece_data)
        
        # Save to file
        if output_name is None:
            output_name = Path(results['image_path']).stem + '_detection.json'
        
        output_path = self.results_folder / output_name
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def process_all_images(self):
        """
        Process all images in the data folder
        """
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_folder.glob(f'*{ext}'))
            image_files.extend(self.data_folder.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {self.data_folder}")
            return
        
        print(f"Found {len(image_files)} image(s) to process:")
        for img_file in image_files:
            print(f"  - {img_file.name}")
        
        # Process each image
        for img_file in image_files:
            print(f"\nProcessing: {img_file.name}")
            
            # Detect pieces
            results = self.detect_puzzle_pieces(img_file)
            
            if results:
                print(f"  Found {results['num_pieces']} potential puzzle pieces")
                
                # Save results
                output_name = f"{img_file.stem}_detection.json"
                self.save_results(results, output_name)
                
                # Create visualization
                viz_path = self.results_folder / f"{img_file.stem}_visualization.png"
                self.visualize_results(results, str(viz_path))
                
                # Print piece statistics
                if results['pieces']:
                    areas = [p['area'] for p in results['pieces']]
                    print(f"  Area range: {min(areas):.0f} - {max(areas):.0f}")
                    print(f"  Average area: {np.mean(areas):.0f}")

def main():
    parser = argparse.ArgumentParser(description='Detect puzzle pieces in images')
    parser.add_argument('--data-folder', default='data', help='Folder containing images')
    parser.add_argument('--image', help='Specific image to process (optional)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--method', choices=['auto', 'morphological', 'watershed', 'connected_components'], 
                       default='auto', help='Preprocessing method to use')
    
    args = parser.parse_args()
    
    detector = PuzzleDetector(args.data_folder)
    detector.method = args.method
    
    if args.image:
        # Process specific image
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return
        
        results = detector.detect_puzzle_pieces(img_path)
        if results:
            detector.save_results(results)
            if not args.no_viz:
                detector.visualize_results(results)
    else:
        # Process all images
        detector.process_all_images()

if __name__ == "__main__":
    main()
