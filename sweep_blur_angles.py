#!/usr/bin/env python
"""
Blur Angle Sweep Script for Coded Exposure Photography Tool

This script performs a sweep of blur angles, running the complete blur-deblur pipeline
for each angle and measuring the PSNR of the results. It uses session logs generated 
by the Coded Exposure Photography Tool to extract other parameters.

Usage:
  python sweep_blur_angles.py [options]

Options:
  --log-dir DIR       Path to the logs directory (default: logs)
  --output-dir DIR    Directory to save results (default: sweep_results)
  --angle-start N     Starting angle for sweep (default: 0)
  --angle-end N       Ending angle for sweep (default: 180)
  --angle-step N      Step size between angles (default: 5)
  --session-id ID     Specific session ID to analyze (optional)
"""

import os
import sys
import json
import argparse
import csv
import numpy as np
import datetime
import glob
from typing import Dict, List, Tuple, Optional, Union, Any

# Import modules from the main application
import blur_core
import image_processing
import utils
import parameter_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sweep blur angles and measure PSNR")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Path to the logs directory")
    parser.add_argument("--output-dir", type=str, default="sweep_results",
                        help="Directory to save results")
    parser.add_argument("--angle-start", type=float, default=0,
                        help="Starting angle for sweep")
    parser.add_argument("--angle-end", type=float, default=180,
                        help="Ending angle for sweep")
    parser.add_argument("--angle-step", type=float, default=5,
                        help="Step size between angles")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Specific session ID to analyze")
    
    return parser.parse_args()

def find_session_logs(log_dir: str, session_id: Optional[str] = None) -> List[str]:
    """
    Find all session log files or a specific session log file.
    
    Args:
        log_dir: Directory containing log files
        session_id: Optional specific session ID to find
        
    Returns:
        List of paths to session log files
    """
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' not found")
        return []
    
    # Get all JSON files in the log directory
    log_files = glob.glob(os.path.join(log_dir, "*.json"))
    
    if session_id:
        # Filter to find logs matching the specified session ID
        matching_logs = []
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    if log_data.get("session_id") == session_id:
                        matching_logs.append(log_file)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading log file {log_file}: {e}")
        
        if not matching_logs:
            print(f"No log files found for session ID: {session_id}")
        
        return matching_logs
    else:
        # Return all log files, sorted by modification time (newest first)
        return sorted(log_files, key=os.path.getmtime, reverse=True)

def extract_session_parameters(log_file: str) -> Dict[str, Any]:
    """
    Extract parameters from a session log file.
    
    Args:
        log_file: Path to a session log file
        
    Returns:
        Dictionary containing session parameters
    """
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # Extract the parameters section
        parameters = log_data.get("parameters", {})
        actions = log_data.get("actions", [])
        
        # Reconstruct the crop_points list from individual parameters
        crop_points = []
        for i in range(1, 5):  # Points 1-4
            x_key = f"crop_point{i}_x"
            y_key = f"crop_point{i}_y"
            
            if x_key in parameters and y_key in parameters:
                # Get coordinates
                x = float(parameters[x_key])
                y = float(parameters[y_key])
                crop_points.append([x, y])
        
        # Add the reconstructed crop_points to parameters for easier access
        if len(crop_points) == 4:
            parameters["crop_points_list"] = crop_points
        
        # Check for complete session with all three phases
        phases = set()
        for action in actions:
            if action.get("type") == "parameters_update":
                phase = action.get("data", {}).get("phase")
                if phase:
                    phases.add(phase)
        
        # Create a session info dictionary
        session_info = {
            "session_id": log_data.get("session_id", "unknown"),
            "parameters": parameters,
            "phases": list(phases),
            "is_complete": set(["Image Creation", "Motion Crop", "Deblurring"]).issubset(phases),
            "actions": actions,
            "log_file": log_file
        }
        
        return session_info
    
    except Exception as e:
        print(f"Error extracting parameters from {log_file}: {e}")
        return {
            "session_id": "error",
            "parameters": {},
            "phases": [],
            "is_complete": False,
            "log_file": log_file
        }

def run_full_pipeline(session_params: Dict[str, Any], blur_angle: float) -> Dict[str, Any]:
    """
    Run the full blur-deblur pipeline with a specific blur angle.
    
    Args:
        session_params: Dictionary containing session parameters
        blur_angle: The angle to use for blurring
        
    Returns:
        Dictionary containing results including PSNR
    """
    try:
        params = session_params["parameters"]
        
        # Get paths from parameters
        background_path = params.get("background_image_path")
        foreground_path = params.get("foreground_image_path")
        output_dir = params.get("output_directory", "output")
        
        # Ensure output directory exists
        utils.ensure_dir_exists(output_dir)
        
        # Create sweep subfolder in output directory for these results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_output_dir = os.path.join(output_dir, f"sweep_angle{blur_angle:.1f}_{timestamp}")
        os.makedirs(sweep_output_dir, exist_ok=True)
        
        # Load images
        background_image = image_processing.load_image(background_path)
        foreground_image = image_processing.load_image(foreground_path, with_alpha=True)
        
        # Convert to RGB for consistent processing
        background_rgb = image_processing.convert_bgr_to_rgb(background_image)
        foreground_rgb = image_processing.convert_bgr_to_rgb(foreground_image)
        
        # Get blur parameters
        blur_length = params.get("blur_length", 100)
        code_type = params.get("code_type", "optimal")
        object_position = (params.get("object_position_x", 0), params.get("object_position_y", 0))
        
        # Generate code array
        code = blur_core.generate_code_array(utils.CODE_LENGTH, code_type)
        
        # PHASE 1: Image Creation - Composite and blur
        # CRITICAL FIX: The main app blurs the foreground object only, then composites with position adjustment
        # DO NOT blur the full composite as we were doing before
        
        # Create unblurred composite for reference (used for PSNR calculation) 
        unblurred_composite = image_processing.composite_images(background_rgb, foreground_rgb, object_position)
        
        # Save the unblurred composite image for reference
        unblurred_path = os.path.join(sweep_output_dir, f"unblurred_composite_angle{blur_angle:.1f}.png")
        image_processing.save_image(unblurred_composite, unblurred_path)
        print(f"Saved unblurred composite: {unblurred_path}")
        
        # Apply motion blur to the FOREGROUND ONLY
        blurred_foreground = blur_core.apply_motion_blur(foreground_rgb, code, blur_length, angle=blur_angle)
        
        # Calculate the center of mass offset of the blur
        offset_x, offset_y = blur_core.calculate_blur_offset(code, blur_length, blur_angle)
        print(f"Blur offset calculated: x={offset_x}, y={offset_y}")
        
        # Adjust position to account for blur offset
        adjusted_position = (object_position[0] - offset_x, object_position[1] - offset_y)
        print(f"Original position: {object_position}, Adjusted position: {adjusted_position}")
        
        # Composite the blurred foreground onto the background with ADJUSTED position
        blurred = image_processing.composite_images(background_rgb.copy(), blurred_foreground, adjusted_position)
        
        # Save the blurred image
        blurred_path = os.path.join(sweep_output_dir, f"blurred_composite_angle{blur_angle:.1f}.png")
        image_processing.save_image(blurred, blurred_path)
        print(f"Saved blurred composite: {blurred_path}")
        
        # PHASE 2: Motion Crop
        print(f"DEBUG - All parameters: {list(params.keys())}")
        
        # Use the reconstructed crop points list if available, otherwise extract points individually
        if "crop_points_list" in params:
            crop_points = params["crop_points_list"]
            print(f"Using reconstructed crop points list: {crop_points}")
        else:
            # Extract points individually as a fallback
            crop_points = []
            
            # Check if we have any crop point parameters at all
            has_any_crop_points = False
            for i in range(1, 5):
                if f"crop_point{i}_x" in params or f"crop_point{i}_y" in params:
                    has_any_crop_points = True
                    break
                    
            if has_any_crop_points:
                print("Found some crop point parameters, attempting to extract them")
                for i in range(1, 5):  # Points 1-4
                    x_key = f"crop_point{i}_x"
                    y_key = f"crop_point{i}_y"
                    
                    if x_key in params and y_key in params:
                        # Get coordinates and ensure they are numbers
                        x = float(params[x_key])
                        y = float(params[y_key])
                        crop_points.append([x, y])
                    else:
                        print(f"WARNING: Missing coordinates for point {i}. Keys present: {x_key in params}, {y_key in params}")
            else:
                print("WARNING: No crop point parameters found in the log")
        
        motion_angle = blur_angle  # Use the same angle for consistency
        
        # Print crop points for debugging
        print(f"Crop points from log: {crop_points}")
        
        # If crop points are empty, use default crop (center region)
        if len(crop_points) != 4:
            print("Warning: Did not find all 4 crop points in the log file.")
            # Default to center region with 10% margin
            h, w = blurred.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            crop_points = [
                [margin_x, margin_y],                    # Top-left
                [w - margin_x, margin_y],                # Top-right
                [w - margin_x, h - margin_y],            # Bottom-right
                [margin_x, h - margin_y]                 # Bottom-left
            ]
            print(f"Using default crop points: {crop_points}")
        
        # Convert crop points to integers
        crop_points_int = [[int(x), int(y)] for x, y in crop_points]
        print(f"Crop points (integers): {crop_points_int}")
        
        # Extract all x and y coordinates
        xs = [p[0] for p in crop_points_int]
        ys = [p[1] for p in crop_points_int]
        
        # Get min/max bounds to create the rectangle
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        
        # Create the source rectangle corners in the correct order
        # Order MUST be: top-left, top-right, bottom-right, bottom-left
        src_corners = np.array([
            [min_x, min_y],  # Top-left
            [max_x, min_y],  # Top-right
            [max_x, max_y],  # Bottom-right
            [min_x, max_y]   # Bottom-left
        ], dtype=np.float32)
        
        # The crop points should be used directly without adjustment for offset,
        # since they were saved in the log from the UI after the user placed them
        # on the already-blurred and position-adjusted composite image
        print(f"Source corners: {src_corners}")
        
        # Apply perspective transformation to the blurred image
        cropped_blurred = image_processing.apply_perspective_transform(
            blurred, src_corners, max_x - min_x, max_y - min_y)
        
        # Save the cropped blurred image
        cropped_blurred_path = os.path.join(sweep_output_dir, f"cropped_blurred_angle{blur_angle:.1f}.png")
        image_processing.save_image(cropped_blurred, cropped_blurred_path)
        print(f"Saved cropped blurred image: {cropped_blurred_path}")
        
        # Also crop the unblurred composite for reference - USING THE EXACT SAME CORNERS
        cropped_unblurred = image_processing.apply_perspective_transform(
            unblurred_composite, src_corners, max_x - min_x, max_y - min_y)
        
        # Save the cropped unblurred reference image
        cropped_unblurred_path = os.path.join(sweep_output_dir, f"cropped_unblurred_angle{blur_angle:.1f}.png")
        image_processing.save_image(cropped_unblurred, cropped_unblurred_path)
        print(f"Saved cropped unblurred reference: {cropped_unblurred_path}")
        
        # PHASE 3: Deblurring
        # Get deblurring parameters
        background_type = params.get("background_type", "constant")
        regularization_factor = params.get("regularization_factor", 0.005)
        
        # Apply deblurring
        deblurred = blur_core.deblur_with_background_estimation(
            cropped_blurred, code, blur_length, 
            background_type=background_type,
            regularization_factor=regularization_factor
        )
        
        # Save the deblurred image
        deblurred_path = os.path.join(sweep_output_dir, f"deblurred_angle{blur_angle:.1f}.png")
        image_processing.save_image(deblurred, deblurred_path)
        print(f"Saved deblurred image: {deblurred_path}")
        
        # Add debug prints for image dimensions
        print("\nIMAGE DIMENSIONS (HxWxC):")
        print(f"Original blurred: {blurred.shape}")
        print(f"Original unblurred: {unblurred_composite.shape}")
        print(f"Cropped blurred: {cropped_blurred.shape}")
        print(f"Cropped unblurred: {cropped_unblurred.shape}")
        print(f"Deblurred: {deblurred.shape}")
        
        # Calculate PSNR - First make sure images are the same size
        try:
            # Check dimensions
            if cropped_unblurred.shape != deblurred.shape:
                print(f"Dimension mismatch: Reference={cropped_unblurred.shape}, Deblurred={deblurred.shape}")
                
                # The issue is that deblurring process reduces the width by (blur_length - 1)
                # Instead of padding deblurred image, crop the unblurred reference to match
                h, w = deblurred.shape[:2]
                
                # Center crop the unblurred image to match deblurred dimensions
                h_diff = cropped_unblurred.shape[0] - h
                w_diff = cropped_unblurred.shape[1] - w
                
                top = h_diff // 2
                left = w_diff // 2
                
                # Create a centered crop
                aligned_reference = cropped_unblurred[top:top+h, left:left+w]
                
                # Save the aligned reference image for verification
                aligned_reference_path = os.path.join(sweep_output_dir, f"aligned_reference_angle{blur_angle:.1f}.png")
                image_processing.save_image(aligned_reference, aligned_reference_path)
                print(f"Saved aligned reference with center crop: {aligned_reference_path}")
                print(f"Aligned reference dimensions: {aligned_reference.shape}")
                
                # Now calculate PSNR with properly sized images
                psnr_value = image_processing.calculate_psnr(aligned_reference, deblurred)
            else:
                # If dimensions already match, calculate directly
                psnr_value = image_processing.calculate_psnr(cropped_unblurred, deblurred)
        
        except Exception as e:
            print(f"Error in PSNR calculation: {e}")
            import traceback
            traceback.print_exc()
            psnr_value = None
        
        # Create a result summary file
        summary_path = os.path.join(sweep_output_dir, "result_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Blur Angle: {blur_angle:.1f}°\n")
            f.write(f"PSNR: {psnr_value:.2f} dB\n" if psnr_value is not None else "PSNR: N/A\n")
            f.write(f"Blur Length: {blur_length}\n")
            f.write(f"Code Type: {code_type}\n")
            f.write(f"Background Type: {background_type}\n")
            f.write(f"Regularization Factor: {regularization_factor}\n")
        
        print(f"Saved result summary: {summary_path}")
        
        # Return results, including paths to the generated images
        return {
            "angle": blur_angle,
            "psnr": psnr_value,
            "success": True,
            "output_dir": sweep_output_dir,
            "images": {
                "unblurred": unblurred_path,
                "blurred": blurred_path,
                "cropped_blurred": cropped_blurred_path,
                "cropped_unblurred": cropped_unblurred_path,
                "deblurred": deblurred_path
            }
        }
    
    except Exception as e:
        print(f"Error running pipeline with angle {blur_angle}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "angle": blur_angle,
            "psnr": None,
            "success": False,
            "error": str(e)
        }

def sweep_angles(session_info: Dict[str, Any], 
                angle_start: float, 
                angle_end: float, 
                angle_step: float,
                output_dir: str) -> List[Dict[str, Any]]:
    """
    Perform a sweep of blur angles and measure PSNR.
    
    Args:
        session_info: Session information dictionary
        angle_start: Starting angle for sweep
        angle_end: Ending angle for sweep
        angle_step: Step size between angles
        output_dir: Directory to save results
        
    Returns:
        List of dictionaries containing evaluation results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate angles for sweep
    angles = np.arange(angle_start, angle_end + angle_step/2, angle_step)
    
    # Perform the sweep
    results = []
    for i, angle in enumerate(angles):
        print(f"Processing angle {angle:.1f}° [{i+1}/{len(angles)}]")
        
        # Run the full pipeline with this angle
        result = run_full_pipeline(session_info, angle)
        
        results.append(result)
    
    return results

def save_results_to_csv(results: List[Dict[str, Any]], 
                      session_info: Dict[str, Any], 
                      output_dir: str) -> str:
    """
    Save the sweep results to a CSV file.
    
    Args:
        results: List of dictionaries containing evaluation results
        session_info: Session information dictionary
        output_dir: Directory to save results
        
    Returns:
        Path to the saved CSV file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with session ID and timestamp
    session_id = session_info.get("session_id", "unknown")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"blur_angle_sweep_{session_id}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Write results to CSV
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['angle', 'psnr', 'success', 'output_directory']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'angle': result['angle'],
                'psnr': result['psnr'] if result['psnr'] is not None else 'N/A',
                'success': result['success'],
                'output_directory': result.get('output_dir', 'N/A')
            })
    
    return filepath

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Find session logs
    print(f"Searching for session logs in: {args.log_dir}")
    log_files = find_session_logs(args.log_dir, args.session_id)
    
    if not log_files:
        print("No session logs found. Please run the Coded Exposure Photography Tool first.")
        return
    
    print(f"Found {len(log_files)} session log(s)")
    
    # Process the most recent session log (or the specified one)
    log_file = log_files[0]
    print(f"Using log file: {log_file}")
    
    # Extract parameters from session log
    session_info = extract_session_parameters(log_file)
    
    if not session_info["is_complete"]:
        print("The selected session is not complete. It should include all three phases.")
        return
    
    print(f"Session ID: {session_info['session_id']}")
    print(f"Phases: {', '.join(session_info['phases'])}")
    
    # Create a timestamped output directory for this sweep
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_base_dir = os.path.join(args.output_dir, f"blur_angle_sweep_{timestamp}")
    os.makedirs(sweep_base_dir, exist_ok=True)
    
    print(f"\nWill process blur angles from {args.angle_start}° to {args.angle_end}° in steps of {args.angle_step}°")
    print(f"All results will be saved in: {sweep_base_dir}")
    print(f"Starting sweep...")
    
    # Perform the angle sweep
    results = sweep_angles(
        session_info,
        args.angle_start,
        args.angle_end,
        args.angle_step,
        sweep_base_dir
    )
    
    if not results:
        print("No results generated.")
        return
    
    # Save results to CSV
    csv_path = save_results_to_csv(results, session_info, sweep_base_dir)
    print(f"\nResults saved to: {csv_path}")
    
    # Find the maximum PSNR value
    valid_results = [r for r in results if r.get('psnr') is not None]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['psnr'])
        print(f"Optimal blur angle: {best_result['angle']:.1f}° (PSNR: {best_result['psnr']:.2f} dB)")
        if 'output_dir' in best_result:
            print(f"Best result images are in: {best_result['output_dir']}")
        
        # Summary of all results
        print("\nAll angles and PSNR values:")
        for r in sorted(valid_results, key=lambda x: x['angle']):
            print(f"  Angle: {r['angle']:.1f}° - PSNR: {r['psnr']:.2f} dB")
    else:
        print("No valid PSNR measurements found.")
    
    print("\nSweep completed successfully!")
    print(f"All images and results are saved in: {sweep_base_dir}")
    print(f"You can now use these results for further analysis or to compare different blur angles.")

if __name__ == "__main__":
    main() 