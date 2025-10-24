#!/usr/bin/env python3
"""
Image Preprocessing Pipeline for Historia Scribe

This module implements the preprocessing workflow for historical documents,
including binarization, deskewing, noise removal, and line segmentation.
Based on the specifications in Section 3.2 of the roadmap.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from skimage import filters, morphology, segmentation

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def binarize_image(
    image: np.ndarray, 
    method: str = "sauvola", 
    window_size: int = 25, 
    k: float = 0.2
) -> np.ndarray:
    """
    Binarize a grayscale image using the specified method.
    
    Args:
        image: Input grayscale image
        method: Binarization method ("otsu", "sauvola", "adaptive")
        window_size: Window size for local thresholding
        k: Parameter for Sauvola's method
        
    Returns:
        Binary image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "sauvola":
        # TODO: Implement Sauvola's method for historical documents
        # This is particularly effective for documents with non-uniform illumination
        binary = filters.threshold_sauvola(gray, window_size=window_size, k=k)
        binary = (gray > binary).astype(np.uint8) * 255
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, window_size, 2
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    
    return binary


def deskew_image(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Deskew an image by detecting and correcting skew angle.
    
    Args:
        image: Input binary image
        max_angle: Maximum angle to correct (degrees)
        
    Returns:
        Deskewed image
    """
    # TODO: Implement deskewing using Hough transform or projection profile analysis
    # This is crucial for historical documents that may be scanned at an angle
    print("Deskewing not yet implemented")
    return image


def remove_noise(image: np.ndarray, method: str = "median", kernel_size: int = 3) -> np.ndarray:
    """
    Remove noise from a binary image.
    
    Args:
        image: Input binary image
        method: Noise removal method ("median", "gaussian", "bilateral")
        kernel_size: Kernel size for filtering
        
    Returns:
        Denoised image
    """
    if method == "median":
        return cv2.medianBlur(image, kernel_size)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    else:
        raise ValueError(f"Unknown noise removal method: {method}")


def segment_lines(
    image: np.ndarray, 
    method: str = "projection", 
    min_line_height: int = 20, 
    max_line_height: int = 200
) -> List[Tuple[int, int, int, int]]:
    """
    Segment a document image into individual text lines.
    
    Args:
        image: Input binary image
        method: Segmentation method ("projection", "contour")
        min_line_height: Minimum height for a valid text line
        max_line_height: Maximum height for a valid text line
        
    Returns:
        List of bounding boxes (x, y, width, height) for each text line
    """
    if method == "projection":
        # TODO: Implement projection profile analysis for line segmentation
        # This method analyzes horizontal projections to find line boundaries
        print("Projection-based line segmentation not yet implemented")
        return []
    elif method == "contour":
        # TODO: Implement contour-based line segmentation
        # This method finds connected components and groups them into lines
        print("Contour-based line segmentation not yet implemented")
        return []
    else:
        raise ValueError(f"Unknown segmentation method: {method}")


def preprocess_pipeline(
    input_path: Path,
    output_dir: Path,
    config: dict
) -> List[Path]:
    """
    Complete preprocessing pipeline for a historical document.
    
    Args:
        input_path: Path to input image
        output_dir: Directory to save processed line images
        config: Configuration dictionary with preprocessing parameters
        
    Returns:
        List of paths to processed line images
    """
    print(f"Processing {input_path}")
    
    # Load image
    image = load_image(input_path)
    
    # Apply preprocessing steps
    binary = binarize_image(image, **config.get("binarization", {}))
    
    if config.get("deskewing", {}).get("enabled", False):
        binary = deskew_image(binary, **config.get("deskewing", {}))
    
    if config.get("noise_removal", {}).get("enabled", False):
        binary = remove_noise(binary, **config.get("noise_removal", {}))
    
    # Segment into lines
    line_bboxes = segment_lines(binary, **config.get("line_segmentation", {}))
    
    # Save individual line images
    output_paths = []
    for i, bbox in enumerate(line_bboxes):
        x, y, w, h = bbox
        line_image = binary[y:y+h, x:x+w]
        
        output_path = output_dir / f"{input_path.stem}_line_{i:04d}.png"
        cv2.imwrite(str(output_path), line_image)
        output_paths.append(output_path)
    
    print(f"Processed {len(output_paths)} lines from {input_path}")
    return output_paths


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Preprocess historical document images"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input image or directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed images (default: data/processed)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yml"),
        help="Path to configuration file (default: configs/config.yml)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Load configuration from YAML file
    config = {
        "binarization": {"method": "sauvola", "window_size": 25, "k": 0.2},
        "deskewing": {"enabled": True, "max_angle": 10.0},
        "noise_removal": {"enabled": True, "method": "median", "kernel_size": 3},
        "line_segmentation": {"method": "projection", "min_line_height": 20, "max_line_height": 200}
    }
    
    # Process single file or directory
    if args.input_path.is_file():
        preprocess_pipeline(args.input_path, args.output_dir, config)
    elif args.input_path.is_dir():
        for image_path in args.input_path.glob("*.png"):
            preprocess_pipeline(image_path, args.output_dir, config)
    else:
        raise ValueError(f"Input path {args.input_path} does not exist")


if __name__ == "__main__":
    main()
