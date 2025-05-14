#!/usr/bin/env python
"""
Debug script to check image dimensions after a sweep run
"""

import os
import sys
import cv2
import numpy as np
import glob
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check image dimensions in sweep output directory")
    parser.add_argument("--dir", type=str, help="Specific sweep output directory to check")
    args = parser.parse_args()
    
    # Find the directory to examine
    if args.dir:
        # Use specified directory
        sweep_dir = args.dir
        if not os.path.exists(sweep_dir):
            print(f"Specified directory does not exist: {sweep_dir}")
            return
    else:
        # Find the most recent sweep output directory
        sweep_dirs = glob.glob("output/sweep_angle*")
        if not sweep_dirs:
            print("No sweep output directories found.")
            return
        
        # Sort by modification time, newest first
        sweep_dir = max(sweep_dirs, key=os.path.getmtime)
    
    print(f"Examining sweep directory: {sweep_dir}")
    
    # List all image files
    image_files = glob.glob(os.path.join(sweep_dir, "*.png"))
    
    print("\nIMAGE DIMENSIONS:")
    print("-" * 60)
    
    for image_path in sorted(image_files):
        try:
            # Load image
            image = cv2.imread(image_path)
            
            # Get filename without directory
            filename = os.path.basename(image_path)
            
            # Print dimensions
            if image is not None:
                print(f"{filename}: {image.shape}")
            else:
                print(f"{filename}: Failed to load")
        except Exception as e:
            print(f"{filename}: Error - {e}")
    
    print("-" * 60)
    
    # Check specifically for dimension mismatch between cropped images
    cropped_images = {}
    deblurred = None
    aligned_reference = None
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        if "cropped_" in filename or "deblurred" in filename or "aligned_reference" in filename:
            image = cv2.imread(image_path)
            cropped_images[filename] = image.shape if image is not None else None
            if "deblurred" in filename:
                deblurred = image
            if "aligned_reference" in filename:
                aligned_reference = image

    print("\nDIMENSION COMPARISON:")
    print("-" * 60)
    
    for name, shape in cropped_images.items():
        print(f"{name}: {shape}")
    
    # Check if deblurred and aligned reference match
    if deblurred is not None and aligned_reference is not None:
        if deblurred.shape == aligned_reference.shape:
            print("\nSUCCESS: Deblurred and aligned reference have matching dimensions!")
            print(f"  - deblurred: {deblurred.shape}")
            print(f"  - aligned_reference: {aligned_reference.shape}")
        else:
            print("\nWARNING: Dimension mismatch between deblurred and aligned reference!")
            print(f"  - deblurred: {deblurred.shape}")
            print(f"  - aligned_reference: {aligned_reference.shape}")
    
    # Check for any mismatches
    if len(set(str(shape) for shape in cropped_images.values() if shape is not None)) > 1:
        print("\nNOTE: Different dimension groups detected in output images")
        
        # Compare cropped unblurred with deblurred
        if deblurred is not None:
            for name, shape in cropped_images.items():
                if "cropped_unblurred" in name and shape != deblurred.shape:
                    print(f"\nDimension difference between:")
                    print(f"  - {name}: {shape}")
                    print(f"  - deblurred: {deblurred.shape}")
                    
    else:
        print("\nAll cropped/deblurred/aligned images have the same dimensions.")

if __name__ == "__main__":
    main() 