import cv2
import numpy as np
import os
from PIL import Image
from typing import Tuple, List, Optional
import math
from math import log10, sqrt

def display_image_with_aspect_ratio(window_name: str, image: np.ndarray):
    # display image in resizable window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    height, width = image.shape[:2]
    screen_w, screen_h = 1200, 800
    
    scale = min(screen_w / width, screen_h / height) * 0.9
    window_w, window_h = int(width * scale), int(height * scale)
    
    cv2.resizeWindow(window_name, window_w, window_h)
    
    cv2.imshow(window_name, image)
    
    print("press any key to close window")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_psnr(original_image: np.ndarray, processed_image: np.ndarray) -> float:
    # calc psnr between images
    if original_image.shape != processed_image.shape:
        print(f"warning: image dims dont match:")
        print(f"reference: {original_image.shape}, processed: {processed_image.shape}")
        
        raise ValueError("image dims must match. use crop_reference_for_psnr first.")
        
    original_f = original_image.astype(np.float64)
    processed_f = processed_image.astype(np.float64)
    
    mse = np.mean((original_f - processed_f) ** 2)
    
    if mse == 0:
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    
    return psnr

def create_demo_images(output_dir: str = "output"):
    # create demo bg and obj images
    os.makedirs(output_dir, exist_ok=True)
    
    bg_width, bg_height = 800, 600
    background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 200
    
    grid_spacing = 50
    grid_color = (150, 150, 150)
    
    for x in range(0, bg_width, grid_spacing):
        cv2.line(background, (x, 0), (x, bg_height-1), grid_color, 1)
    
    for y in range(0, bg_height, grid_spacing):
        cv2.line(background, (0, y), (bg_width-1, y), grid_color, 1)
    
    bg_path = os.path.join(output_dir, "background.png")
    cv2.imwrite(bg_path, background)
    print(f"created bg: {bg_path}")
    
    obj_width, obj_height = 200, 100
    object_img = np.ones((obj_height, obj_width, 3), dtype=np.uint8) * 255
    
    cv2.rectangle(object_img, (10, 10), (obj_width-10, obj_height-10), (0, 0, 255), -1)
    
    cv2.putText(object_img, "OBJECT", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    obj_path = os.path.join(output_dir, "object.png")
    cv2.imwrite(obj_path, object_img)
    print(f"created obj: {obj_path}")
    
    return bg_path, obj_path

def load_image(image_path: str, with_alpha: bool = False) -> np.ndarray:
    # load image with optional alpha
    if with_alpha:
        cv_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        cv_img = cv2.imread(image_path)
        
    if cv_img is None:
        raise ValueError(f"failed to load img: {image_path}")
    
    return cv_img

def convert_bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    # convert bgr(a) to rgb(a)
    if len(image.shape) == 3 and image.shape[2] == 4:
        b, g, r, a = cv2.split(image)
        return cv2.merge([r, g, b, a])
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB(A) to BGR(A) format."""
    if len(image.shape) == 3 and image.shape[2] == 4:  # Has alpha channel
        # Split channels
        r, g, b, a = cv2.split(image)
        return cv2.merge([b, g, r, a])
    else:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def composite_images(background: np.ndarray, foreground: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
    # composite fg onto bg at position
    result = background.copy()
    
    x, y = position
    h, w = foreground.shape[:2]
    

    if y + h > result.shape[0] or x + w > result.shape[1]:
        print("warning: obj exceeds bg dims, will clip")
        
    end_y = min(y + h, result.shape[0])
    end_x = min(x + w, result.shape[1])
    h_valid = end_y - y
    w_valid = end_x - x
    
    # Place foreground on background with alpha blending if available
    if len(foreground.shape) == 3 and foreground.shape[2] == 4 and h_valid > 0 and w_valid > 0:
        roi = result[y:end_y, x:end_x]
        
        alpha = foreground[:h_valid, :w_valid, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha[:, :, 0]) + \
                          foreground[:h_valid, :w_valid, c] * alpha[:, :, 0]
    elif h_valid > 0 and w_valid > 0:
        result[y:end_y, x:end_x] = foreground[:h_valid, :w_valid]
    
    return result

def apply_perspective_transform(image: np.ndarray, src_points: np.ndarray, 
                              dst_width: int, dst_height: int) -> np.ndarray:
    # apply perspective transform
    dst_points = np.array([
        [0, 0],
        [dst_width, 0],
        [dst_width, dst_height],
        [0, dst_height]
    ], dtype=np.float32)
    
    print(f"src pts: {src_points}")
    print(f"dst pts: {dst_points}")
    print(f"dst dims: {dst_width} x {dst_height}")
    
    # Calculate transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    transformed = cv2.warpPerspective(image, M, (dst_width, dst_height))
    
    return transformed

def resize_image_for_display(image: np.ndarray, max_width: int, max_height: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    
    # Calculate scale factor
    scale_w = max_width / width if width > max_width else 1.0
    scale_h = max_height / height if height > max_height else 1.0
    scale = min(scale_w, scale_h)  # Use the smaller scale to ensure entire image fits
    
    # Only scale down, never scale up
    if scale >= 1.0:
        return image, 1.0
        
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image
    if len(image.shape) == 3 and image.shape[2] == 4:  # With alpha
        # Use PIL for RGBA images
        pil_img = Image.fromarray(image)
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        resized = np.array(resized_img)
    else:
        # Use cv2 for RGB images
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    return resized, scale

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to disk, handling RGB/BGR conversion."""
    # Make sure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # OpenCV expects BGR format for saving
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Check if the image is in RGB format
        # Note: This is a heuristic - we assume input is RGB if coming from our UI pipeline
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # If it has alpha or is already BGR, save as is
        image_to_save = image
    
    # Save the image
    cv2.imwrite(output_path, image_to_save)

def crop_reference_for_psnr(ref: np.ndarray,
                          deblur: np.ndarray,
                          blur_length: int,
                          direction: str = 'horizontal') -> np.ndarray:
    # crop ref to match deblurred shape
    if direction == 'horizontal':
        # Horizontal blur: crop left/right
        h_diff = ref.shape[0] - deblur.shape[0]
        w_diff = ref.shape[1] - deblur.shape[1]
        
        # Crop half from each side for width
        left_margin = w_diff // 2
        top_margin = h_diff // 2
        
        return ref[top_margin:top_margin+deblur.shape[0], 
                 left_margin:left_margin+deblur.shape[1]]
    else:
        # For vertical motion, remove pad_size pixels from top edge only
        ref_aligned = ref[pad_size : pad_size + deblur.shape[0], :]
    
    return ref_aligned

def pad_deblurred_for_psnr(deblur: np.ndarray,
                          target_shape: tuple,
                          blur_length: int,
                          bg_level: int = 0,
                          direction: str = 'horizontal') -> np.ndarray:

    delta = blur_length - 1
    padding = delta // 2  # For symmetrical padding
    
    if direction == 'horizontal':
        padded = cv2.copyMakeBorder(deblur,     # src
                                   top=0, bottom=0,
                                   left=padding, right=padding,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=bg_level)
    else:
        padded = cv2.copyMakeBorder(deblur,
                                   top=padding, bottom=padding,
                                   left=0, right=0,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=bg_level)
    
    # Final safety crop in case padding produced a different size
    height, width = target_shape[:2]
    return padded[:height, :width] 