import numpy as np
import cv2
import time
import random
from typing import Tuple, List, Optional

debug_mode = False

def generate_code_array(length: int, method: str = "optimal") -> np.ndarray:
    """Generate a binary code array for the flutter shutter.
    
    Args:
        length: The length of the code
        method: The method to generate the code. Options:
            - "optimal": The near-optimal 52-bit code from the paper
            - "box": Traditional box filter (all ones)
            - "random": Random binary sequence
            - "mura": Modified Uniformly Redundant Array
    
    Returns:
        A binary code array (0s and 1s)
    """
    if method == "optimal":
        # Near-optimal code from the paper (52-bit code)
        optimal_code = "1010000111000001010000110011110111010111001001100111"
        code = np.array([int(bit) for bit in optimal_code])
        # If requested length is different from 52, we'll need to adjust
        if length != 52:
            # For shorter codes, truncate; for longer, repeat
            if length < 52:
                code = code[:length]
            else:
                repeats = length // 52 + 1
                code = np.tile(code, repeats)[:length]
    elif method == "box":
        # Traditional box filter (all ones)
        code = np.ones(length, dtype=int)
    elif method == "random":
        # Random binary sequence with 50% duty cycle
        code = np.zeros(length, dtype=int)
        ones_count = length // 2
        ones_indices = np.random.choice(length, ones_count, replace=False)
        code[ones_indices] = 1
    elif method == "mura":
        # Simple implementation of a MURA-like code
        code = np.zeros(length, dtype=int)
        for i in range(length):
            # Simple pattern
            code[i] = 1 if (i % 2 == 0 or i % 3 == 0) else 0
    else:
        raise ValueError(f"Unknown code method: {method}")
    
    return code

def apply_motion_blur(image: np.ndarray, code: np.ndarray, 
                      blur_length: int, angle: float = 0.0) -> np.ndarray:
    """Apply motion blur to an image using the code array."""
    # Create a copy to avoid modifying the original
    orig_image = image.copy()
    
    # Get dimensions and check for alpha channel
    height, width = orig_image.shape[:2]
    has_alpha = len(orig_image.shape) == 3 and orig_image.shape[2] == 4
    channels = orig_image.shape[2] if len(orig_image.shape) > 2 else 1
    
    # Calculate padding needed based on blur length and angle
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Add padding to ensure no clipping occurs during blur
    pad_x = int(abs(blur_length * dx)) + 5
    # Add vertical padding only if there is vertical motion
    pad_y = 0 if abs(dy) < 1e-6 else int(abs(blur_length * dy)) + 5
    
    # Create padded image
    if has_alpha:
        # For BGRA images
        padded_image = np.zeros((height + 2*pad_y, width + 2*pad_x, channels), dtype=orig_image.dtype)
        padded_image[pad_y:pad_y+height, pad_x:pad_x+width] = orig_image
    else:
        # For BGR images
        padded_image = np.zeros((height + 2*pad_y, width + 2*pad_x, channels), dtype=orig_image.dtype)
        padded_image[pad_y:pad_y+height, pad_x:pad_x+width] = orig_image
    
    # Work with the padded image
    image = padded_image
    height, width = image.shape[:2]
    
    # Normalize code to sum to 1 to maintain brightness
    norm_code = code / np.sum(code)
    
    blurred = np.zeros_like(image, dtype=float)
    
    # For each position in the code
    for i, weight in enumerate(norm_code):
        if weight == 0:
            continue  # Skip when shutter is closed
        
        # Calculate the shift at this point in the exposure
        shift_fraction = i / (len(code) - 1)  # 0 to 1
        # Using the exact same formula as in calculate_blur_offset
        shift_x = int(shift_fraction * (blur_length - 1) * dx)
        shift_y = int(shift_fraction * (blur_length - 1) * dy)
        
        # Create translation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        
        # Set appropriate border value based on channels
        if has_alpha:
            border_value = (0, 0, 0, 0)
        else:
            border_value = (0, 0, 0)
            
        # Apply the shift
        shifted = cv2.warpAffine(image, M, (width, height), 
                               borderMode=cv2.BORDER_CONSTANT, 
                               borderValue=border_value)
        
        # Add weighted contribution to the blurred image
        blurred += weight * shifted
    
    # Convert back to uint8
    result = np.clip(blurred, 0, 255).astype(np.uint8)
    
    # Crop back to original size plus blur margin to include the complete blur
    # Calculate how much of the original padding to keep to show full blur
    keep_pad_x = min(pad_x, int(abs(blur_length * dx)) + 5)
    keep_pad_y = min(pad_y, int(abs(blur_length * dy)) + 5)
    
    start_y = pad_y - keep_pad_y
    start_x = pad_x - keep_pad_x
    end_y = pad_y + height - 2*pad_y + keep_pad_y
    end_x = pad_x + width - 2*pad_x + keep_pad_x
    
    return result[start_y:end_y, start_x:end_x]

def code_to_psf(code: np.ndarray, blur_length: int) -> np.ndarray:
    """
    Spread each 'open-shutter' instant in the binary `code` over the 1-D sensor,
    taking sub-pixel positions into account.

    Parameters
    ----------
    code : (m,) 0/1 array, optimal sequence (m = 52 in the paper)
    blur_length : int, physical blur extent in pixels (k)

    Returns
    -------
    psf : (blur_length,) float array, normalised to sum to 1
    """
    m = len(code)
    
    # Use linear interpolation to resample the code to the blur length
    x_old = np.linspace(0, 1, m)
    x_new = np.linspace(0, 1, blur_length, endpoint=False)  # endpoint=False to avoid identical first/last samples
    
    # For binary code, first convert to float
    code_float = code.astype(float)
    
    # Interpolate the code
    psf = np.interp(x_new, x_old, code_float)
    
    # Normalize the PSF
    psf = psf / np.sum(psf)
    
    return psf

def create_smearing_matrix(code: np.ndarray, blur_length: int, image_size: int) -> np.ndarray:
    """Creates the smearing matrix A as described in the paper.
    
    Args:
        code: The binary code array used for the blur
        blur_length: The length of the motion blur in pixels
        image_size: The size of the image in the motion direction
        
    Returns:
        The smearing matrix A
    """
    # Number of known values (pixels in the original image)
    n = image_size
    
    # Blur size (in pixels)
    k = blur_length
    
    # Total width after blur = original size + blur - 1
    w = n + k - 1
    
    # Create smearing matrix of size (n+k-1) x n
    A = np.zeros((w, n))
    
    # Convert code to proper PSF using physically accurate model
    if len(code) != k:
        psf = code_to_psf(code, k)
    else:
        # If code length matches blur length, just convert to float
        psf = code.astype(float)
        # Normalize to sum to 1
        psf = psf / np.sum(psf)
    
    # Fill the smearing matrix
    # Each column represents the contribution of one pixel in the source image
    for i in range(n):
        # Place the PSF starting at row i
        A[i:i+k, i] = psf
    
    return A

def extend_A_for_constant_bg(A: np.ndarray) -> np.ndarray:
    """Extend the smearing matrix A to handle constant background.
    
    Adds a column of ones to the smearing matrix A, which allows solving
    for one scalar background value per row.
    
    Args:
        A: The smearing matrix of shape (w, n) where w = n + k - 1
        
    Returns:
        Extended matrix of shape (w, n+1) with a column of ones appended
    """
    # Add a column of ones without normalization
    ones = np.ones((A.shape[0], 1), dtype=A.dtype)
    return np.hstack([A, ones])

def deblur_with_background_estimation(blurred_image: np.ndarray, code: np.ndarray, blur_length: int, 
                                     background_type: str = "none", regularization_factor: float = 0.005) -> np.ndarray:
    """Deblur an image with background estimation.
    
    Args:
        blurred_image: The blurred image
        code: The binary code array used for the blur
        blur_length: The length of the motion blur in pixels
        background_type: The type of background. Options:
            - "none": No background (black)
            - "constant": Constant background (estimated per motion line)
            - "textured": Textured background (estimate edges outside blur)
        regularization_factor: Factor to control regularization strength (default: 0.005)
            
    Returns:
        The deblurred image
    """
    if background_type == "none":
        # For "none", assume black background (no background estimation)
        return deblur_image(blurred_image, code, blur_length, regularization_factor=regularization_factor)
    
    elif background_type == "constant":
        # For "constant", estimate one scalar background value per row
        # Process each channel independently
        if len(blurred_image.shape) == 3:
            # Get image dimensions
            height, width, channels = blurred_image.shape
            deblurred = np.zeros((height, width - blur_length + 1, channels), dtype=np.float32)
            
            # Process each channel
            for c in range(channels):
                channel = blurred_image[:, :, c].astype(np.float32)
                deblurred[:, :, c] = deblur_channel_with_constant_bg(channel, code, blur_length, regularization_factor)
        else:
            # Grayscale image
            deblurred = deblur_channel_with_constant_bg(blurred_image.astype(np.float32), code, blur_length, regularization_factor)
        
        # Keep as float32 for caller to handle clipping and conversion
        return deblurred
    
    elif background_type == "textured":
        # Textured background not fully implemented yet
        raise NotImplementedError("Textured background estimation is not implemented yet. Use 'constant' for best results.")
    
    else:
        raise ValueError(f"Unknown background type: {background_type}")

def deblur_channel_with_constant_bg(blurred_channel: np.ndarray, code: np.ndarray, blur_length: int, regularization_factor: float = 0.005) -> np.ndarray:
    """Deblur a single channel with constant background estimation per row.
    
    Args:
        blurred_channel: One channel of the blurred image
        code: The binary code array used for the blur
        blur_length: The length of the motion blur in pixels
        regularization_factor: Factor to control regularization strength (default: 0.005)
        
    Returns:
        The deblurred channel (float32, not clipped or converted to uint8)
    """
    height, width = blurred_channel.shape
    
    # Calculate the size of the unblurred object
    n = width - blur_length + 1
    
    # Sanity check for n
    if n < 5:
        raise ValueError(f"Deblurred width too small ({n}) – check blur_length or crop.")
    
    # Create output array for deblurred result
    deblurred = np.zeros((height, n), dtype=np.float32)
    
    # Calculate padding size to handle edge effects
    pad_size = int(np.ceil(blur_length / 2))
    
    # Pad the input image horizontally
    padded_channel = np.pad(blurred_channel, ((0, 0), (pad_size, pad_size)), mode='reflect')
    
    # Adjust width after padding
    padded_width = padded_channel.shape[1]
    
    # Recalculate n for the padded image
    padded_n = padded_width - blur_length + 1
    
    # Create the smearing matrix for the padded size
    A = create_smearing_matrix(code, blur_length, padded_n)
    
    # Extend A for constant background estimation
    A_extended = extend_A_for_constant_bg(A)
    
    # Compute SVD for the extended matrix ONCE for all rows
    U, s, Vh = np.linalg.svd(A_extended, full_matrices=False)
    
    # Apply Tikhonov regularization
    lambda_squared = (regularization_factor * s[0])**2
    s_inv = s / (s**2 + lambda_squared)
    
    # Construct regularized pseudo-inverse
    A_pinv = (Vh.T * s_inv) @ U.T
    
    # Process each row independently
    for y in range(height):
        # Extract the current row from padded image
        row = padded_channel[y, :]
        
        try:
            # Solve the extended system [foreground, background_value]
            solution = A_pinv @ row
            
            # Extract the foreground part and the background scalar
            foreground = solution[:-1]
            bg_scalar = solution[-1]
            
            # Add the background back to the foreground
            deblurred_row = foreground + bg_scalar
            
            # Crop to remove padding and store in the result
            crop_start = pad_size
            crop_end = crop_start + n
            deblurred[y, :] = deblurred_row[crop_start:crop_end]
        except Exception as e:
            print(f"Error processing row {y}: {str(e)}")
            # Continue to next row on error
    
    return deblurred

def deblur_image(blurred_image: np.ndarray, code: np.ndarray, blur_length: int, method: str = "least_squares", regularization_factor: float = 0.005) -> np.ndarray:
    """Deblur an image using least squares deconvolution.
    
    Args:
        blurred_image: The blurred image
        code: The binary code array used for the blur
        blur_length: The length of the motion blur in pixels
        method: The deblurring method. Currently only "least_squares" is supported.
        regularization_factor: Factor to control regularization strength (default: 0.005)
        
    Returns:
        The deblurred image
    """
    # Process each channel independently
    if len(blurred_image.shape) == 3:
        # Get image dimensions
        height, width, channels = blurred_image.shape
        
        # Calculate the size of the unblurred result
        n = width - blur_length + 1
        
        # Create correctly sized output array
        deblurred = np.zeros((height, n, channels), dtype=np.float32)
        
        # Process each channel
        for c in range(channels):
            channel = blurred_image[:, :, c].astype(np.float32)
            deblurred[:, :, c] = deblur_channel(channel, code, blur_length, method, regularization_factor)
    else:
        # Grayscale image
        height, width = blurred_image.shape
        n = width - blur_length + 1
        deblurred = np.zeros((height, n), dtype=np.float32)
        deblurred = deblur_channel(blurred_image.astype(np.float32), code, blur_length, method, regularization_factor)
    
    # Convert to uint8 for display
    deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
    return deblurred

def deblur_channel(blurred_channel: np.ndarray, code: np.ndarray, blur_length: int, method: str, regularization_factor: float = 0.005) -> np.ndarray:
    """Deblur a single channel using least squares deconvolution.
    
    Args:
        blurred_channel: One channel of the blurred image
        code: The binary code array used for the blur
        blur_length: The length of the motion blur in pixels
        method: The deblurring method
        regularization_factor: Factor to control regularization strength (default: 0.005)
        
    Returns:
        The deblurred channel
    """
    height, width = blurred_channel.shape
    
    # Calculate the size of the unblurred object
    n = width - blur_length + 1
    
    # Sanity check for n
    if n < 5:
        raise ValueError(f"Deblurred width too small ({n}) – check blur_length or crop.")
    
    # Create output array for deblurred result
    deblurred = np.zeros((height, n), dtype=np.float32)
    
    # Calculate padding size to handle edge effects
    pad_size = int(np.ceil(blur_length / 2))
    
    # Pad the input image horizontally
    padded_channel = np.pad(blurred_channel, ((0, 0), (pad_size, pad_size)), mode='reflect')
    
    # Adjust width after padding
    padded_width = padded_channel.shape[1]
    
    # Recalculate n for the padded image
    padded_n = padded_width - blur_length + 1
    
    # Create the smearing matrix for the padded size
    A = create_smearing_matrix(code, blur_length, padded_n)
    
    # Solve using least squares approach (method agnostic)
    if method == "least_squares":
        # Compute SVD once for all rows
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        
        # Apply Tikhonov regularization
        lambda_squared = (regularization_factor * s[0])**2
        s_inv = s / (s**2 + lambda_squared)
        
        # Construct regularized pseudo-inverse
        A_pinv = (Vh.T * s_inv) @ U.T
        
        # Process each row
        for y in range(height):
            # Extract the current row from padded image
            row = padded_channel[y, :]
            
            try:
                # Deblur using regularized pseudoinverse
                deblurred_row = A_pinv @ row
                
                # Crop to remove padding and store in the result
                crop_start = pad_size
                crop_end = crop_start + n
                deblurred[y, :] = deblurred_row[crop_start:crop_end]
            except Exception as e:
                print(f"Error processing row {y}: {str(e)}")
                # Continue to next row on error
    
    return deblurred

def calculate_blur_offset(code: np.ndarray, blur_length: int, angle: float = 0.0) -> tuple:
    """
    Calculate the center of mass offset for a coded blur.
    
    This function determines the effective center of the blur pattern based on
    the distribution of weights in the coded exposure pattern. This is needed
    to properly align blurred and unblurred images.
    
    Args:
        code: The binary code array used for the blur
        blur_length: The length of the motion blur in pixels
        angle: Direction of motion in degrees (0 = horizontal right, 90 = vertical down)
        
    Returns:
        Tuple of (offset_x, offset_y) representing the blur offset
    """
    # Convert angle to radians and get direction vector
    angle_rad = np.deg2rad(angle)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Normalize code to sum to 1
    norm_code = code / np.sum(code)
    
    # 1. Calculate geometric center-of-mass using the exact one-liner from the transcript
    offset_geom = sum(int(i/(len(code)-1)*(blur_length-1)) * w
                      for i, w in enumerate(norm_code) if w > 0)
    
    # 2. Account for the crop offset from apply_motion_blur
    # For Case ②: If we're no longer adding the +5 safety rim when dx=1, 
    # then we shouldn't add +5 to the offset calculation either
    if abs(dx) > 0.999:  # Horizontal motion (θ ≈ 0°)
        offset_crop_x = abs(blur_length * dx)  # Remove the +5 when horizontal
    else:
        offset_crop_x = abs(blur_length * dx) + 5  # Keep +5 for non-horizontal
        
    if abs(dy) < 1e-6:  # No vertical motion
        offset_crop_y = 0
    else:
        offset_crop_y = abs(blur_length * dy) + 5
    
    # Add both offsets and project onto axes
    offset_x = int(round((offset_geom + offset_crop_x) * dx))
    offset_y = int(round((offset_geom + offset_crop_y) * dy))
    
    return (offset_x, offset_y) 