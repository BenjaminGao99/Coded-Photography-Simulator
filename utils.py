import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Tuple, List, Optional, Dict, Any
import datetime

#constants
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_INPUT_DIR = "input"
DEFAULT_BLUR_LENGTH = 100
DEFAULT_CODE_TYPE = "optimal"
DEFAULT_ANGLE = 0
CODE_LENGTH = 52  # from paper

# file dialog filters
IMAGE_FILETYPES = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]

def get_timestamp() -> str:
    # timestamp for filenames
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir_exists(directory: str) -> None:
    # make dir if needed
    os.makedirs(directory, exist_ok=True)

def get_output_filename(code_method: str, blur_length: int, angle: float) -> str:
    # standard filename
    return f"blurred_{code_method}_blur{blur_length}_angle{angle}.png"

def get_full_output_path(output_dir: str, code_method: str, blur_length: int, angle: float) -> str:
    # full path
    ensure_dir_exists(output_dir)
    filename = get_output_filename(code_method, blur_length, angle)
    return os.path.join(output_dir, filename)

def validate_numeric_input(value: str, min_value: Optional[float] = None, 
                         max_value: Optional[float] = None, 
                         is_int: bool = False) -> Tuple[bool, Any]:
    # check if number is valid
    try:
        value = value.strip()
        
        if not value:
            return False, None
        
        if is_int:
            num_value = int(value)
        else:
            num_value = float(value)
        
        if min_value is not None and num_value < min_value:
            return False, None
        
        if max_value is not None and num_value > max_value:
            return False, None
        
        return True, num_value
            
    except ValueError:
        return False, None

def show_error(title: str, message: str) -> None:
    # show error msgbox
    messagebox.showerror(title, message)

def show_info(title: str, message: str) -> None:
    # show info msgbox
    messagebox.showinfo(title, message)

def browse_for_file(title: str, filetypes=None) -> Optional[str]:
    # file dialog
    if filetypes is None:
        filetypes = IMAGE_FILETYPES
    
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return path if path else None

def browse_for_directory(title: str) -> Optional[str]:
    # dir dialog
    directory = filedialog.askdirectory(title=title)
    return directory if directory else None 