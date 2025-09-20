import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
import argparse

DEBUG=False
DEBUG=True

class PiecesExtractor:
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.results_folder = self.data_folder / "results"
        self.results_folder.mkdir(exist_ok=True)
    
    def image_trim(self, img: np.ndarray, radius: int = 100) -> np.ndarray:
        """
        Trim image to focus on center region - from main.py
        """
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2

        perc = radius / 100
        sz = int(perc * min(h, w))
        half_sz = sz // 2
        x1 = max(0, center_x - half_sz)
        y1 = max(0, center_y - half_sz)
        x2 = min(w, center_x + half_sz)
        y2 = min(h, center_y + half_sz)

        img = img[y1:y2, x1:x2]
        return img
    
    def preprocess_with_simple_threshold(self, img_rgb):
        """
        Preprocessing using simple thresholding approach from main.py
        This method produces well-defined contours for edge extraction
        """
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Apply image trimming to focus on center region
        img_trimmed = self.image_trim(img_gray, radius=60)
        
        # Simple fixed threshold (like in main.py)
        _, thresh_fixed = cv2.threshold(img_trimmed, 110, 255, cv2.THRESH_BINARY)
        
        # OTSU thresholding (like in main.py)
        _, thresh_otsu = cv2.threshold(img_trimmed, 110, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if DEBUG:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Simple Threshold Processing (from main.py)', fontsize=16, fontweight='bold')
            axes = axes.flatten()

            images = [
                (img_gray, 'Original Grayscale'),
                (img_trimmed, 'Trimmed Image'),
                (thresh_fixed, 'Fixed Threshold (110)'),
                (thresh_otsu, 'OTSU Threshold'),
            ]

            for i, (img, title) in enumerate(images):
                if i < len(axes):
                    axes[i].imshow(img, cmap='gray')
                    axes[i].set_title(title, fontsize=12, fontweight='bold')
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(images), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()

        return {
            'original': img_rgb,
            'gray': img_gray,
            'trimmed': img_trimmed,
            'thresh_fixed': thresh_fixed,
            'thresh_otsu': thresh_otsu,
            'final': thresh_otsu,  # Use OTSU as final result
            'binary': thresh_otsu,  # For compatibility
            'cleaned': thresh_otsu  # For compatibility
        }
        
    def _find_contours_with_dual_threshold(self, processed_images, min_area=1000):
        """
        Find contours using both OTSU and fixed threshold for better edge extraction
        """
        # Get both threshold results
        thresh_otsu = processed_images['thresh_otsu']
        thresh_fixed = processed_images['thresh_fixed']
        
        # Find contours with both methods
        contours_otsu, _ = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_fixed, _ = cv2.findContours(thresh_fixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        print(f"OTSU threshold found {len(contours_otsu)} contours")
        print(f"Fixed threshold found {len(contours_fixed)} contours")
        
        # Filter contours by area
        valid_contours_otsu = []
        valid_contours_fixed = []
        
        for contour in contours_otsu:
            area = cv2.contourArea(contour)
            if area > min_area:
                valid_contours_otsu.append(contour)
                
        for contour in contours_fixed:
            area = cv2.contourArea(contour)
            if area > min_area:
                valid_contours_fixed.append(contour)
        
        print(f"Valid OTSU contours: {len(valid_contours_otsu)}")
        print(f"Valid fixed contours: {len(valid_contours_fixed)}")
        
        # Choose the method that gives better results
        if len(valid_contours_otsu) >= len(valid_contours_fixed):
            return valid_contours_otsu, 'otsu'
        else:
            return valid_contours_fixed, 'fixed'
    
    def analyze_contour(self, contour):
        """
        Analyze a single contour to extract features
        """
        # THROW ERROR NOT IMPLEMENTED
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
        
        ## Resize if too large
        #h, w = img_rgb.shape[:2]
        #if max(h, w) > 2000:
        #    scale = 2000 / max(h, w)
        #    new_w, new_h = int(w * scale), int(h * scale)
        #    img_rgb = cv2.resize(img_rgb, (new_w, new_h))
        
        # Use only simple threshold method
        processed = self.preprocess_with_simple_threshold(img_rgb)
        method_used = 'simple_threshold'
        
        # Use dual threshold approach for better edge extraction
        contours, threshold_method = self._find_contours_with_dual_threshold(processed, min_area=1000)
        print(f"Using {threshold_method} threshold for contour detection")

        # If too few contours found, try with smaller area threshold
        if len(contours) < 5:
            print(f"Only {len(contours)} contours found, trying with smaller area threshold...")
            contours, threshold_method = self._find_contours_with_dual_threshold(processed, min_area=500)
            if len(contours) < 5:
                contours, threshold_method = self._find_contours_with_dual_threshold(processed, min_area=200)
        
        print(f"Using {method_used} method - found {len(contours)} contours")
        
        # Analyze each contour
        pieces = []
        for i, contour in enumerate(contours):
            analysis = self.analyze_contour(contour)
            analysis['piece_id'] = i
            pieces.append(analysis)

        for p in pieces:
            if p['area'] > 500000:
                pieces.remove(p)
        
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
        
        # OTSU Threshold
        axes[0, 1].imshow(processed['thresh_otsu'])
        axes[0, 1].set_title('OTSU Threshold')
        axes[0, 1].axis('off')
        
        # Fixed Threshold
        axes[0, 2].imshow(processed['thresh_fixed'])
        axes[0, 2].set_title('Fixed Threshold (110)')
        axes[0, 2].axis('off')
        
        # Final processed (OTSU)
        axes[0, 3].imshow(processed['final'])
        axes[0, 3].set_title('Final Processed (OTSU)')
        axes[0, 3].axis('off')
        
        # Contours 
        img_with_contours = processed['final'].copy()
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
                piece_img = processed['final'][y:y+h, x:x+w]
                
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

class EdgeExtractor:
    def __init__(self, piece):
        self.piece = piece
    
    def extract_edges(self):
        pass
    

def main():
    parser = argparse.ArgumentParser(description='Detect puzzle pieces in images')
    parser.add_argument('--data-folder', default='data', help='Folder containing images')
    parser.add_argument('--image', help='Specific image to process (optional)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    # Method is now fixed to simple_threshold only
    
    args = parser.parse_args()
    
    detector = PiecesExtractor(args.data_folder)
    
    if args.image:
        # Process specific image
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return
        
        results = detector.detect_puzzle_pieces(img_path)
        if not results:
            print(f"No results found for {img_path}")
            return

        detector.save_results(results)
        if not args.no_viz:
            detector.visualize_results(results)
        

        
        

    else:
        # Process all images
        detector.process_all_images()

if __name__ == "__main__":
    main()
