import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import random 
import blur_core
import image_processing
import ui_components
import utils
import parameter_logger 

class CodedExposureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coded Exposure Photography Tool")
        self.root.geometry("1200x800")
        

        self.logger = parameter_logger.ParameterLogger()
        self.background_path = None
        self.object_path = None
        self.background_image = None
        self.object_image = None
        self.blurred_object = None
        self.result_image = None
        

        self.bg_path_display = tk.StringVar(value="")
        self.obj_path_display = tk.StringVar(value="")

        self.object_position = (0, 0)
        
        # resize tracking
        self.is_resizing = False
        


        # params
        self.blur_length = tk.StringVar(value=str(utils.DEFAULT_BLUR_LENGTH))
        self.code_type = tk.StringVar(value=utils.DEFAULT_CODE_TYPE)
        self.angle = tk.StringVar(value=str(utils.DEFAULT_ANGLE))
        self.output_dir = tk.StringVar(value=utils.DEFAULT_OUTPUT_DIR)
        
        # deblur params
        self.regularization_factor = tk.DoubleVar(value=0.005)
        
        #live preview toggle
        self.live_preview = tk.BooleanVar(value=False)
        
        # phase management
        self.current_phase = tk.StringVar(value="Image Creation")
        self.auto_advance = tk.BooleanVar(value=True)
        self.phases = ["Image Creation", "Motion Crop", "Deblurring"]
        
        self.create_ui()
        self.drag_handler = None
        self.crop_handler = None
        utils.ensure_dir_exists(self.output_dir.get())
        
        # setup var tracing for live preview
        self.blur_length.trace_add("write", self.parameter_changed)
        self.code_type.trace_add("write", self.parameter_changed)
        self.angle.trace_add("write", self.parameter_changed)
        
        # setup phase change tracing
        self.current_phase.trace_add("write", self.phase_changed)
        self.root.bind("<Configure>", self.on_window_resize)
    
        self.root.bind("<FocusIn>", self.on_focus_in)
        
        # Handle window close event to close logger properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)
        
        # Automatically load default images
        self.load_default_images()
    
    def on_window_close(self):
        """Handle window close event to properly close the logger."""
        if hasattr(self, 'logger'):
            self.logger.close()
        self.root.destroy()
    
    def load_default_images(self):
        """Try to load default background and foreground images from the input folder.
        Tries multiple common file formats."""
        input_dir = "input"
        utils.ensure_dir_exists(input_dir)
        
        # Common image formats to try
        formats = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
        
        # Try to load background image
        background_loaded = False
        for fmt in formats:
            bg_filename = "background" + fmt
            bg_path = os.path.join(input_dir, bg_filename)
            if os.path.exists(bg_path):
                self.background_path = bg_path
                self.load_background()
                background_loaded = True
                break
        
        # Try to load foreground image
        foreground_loaded = False
        for fmt in formats:
            fg_filename = "foreground" + fmt
            fg_path = os.path.join(input_dir, fg_filename)
            if os.path.exists(fg_path):
                self.object_path = fg_path
                self.load_object()
                foreground_loaded = True
                break
        
        # Update the canvas and path displays
        self.update_path_displays()
        self.update_canvas()
        
        # Report what was loaded
        if not background_loaded and not foreground_loaded:
            print("No default images found in the input folder.")
        else:
            if background_loaded:
                print(f"Loaded background image: {self.background_path}")
            if foreground_loaded:
                print(f"Loaded foreground image: {self.object_path}")
    
    def create_ui(self):
        # Create main frame with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top navigation frame for phase selection
        self.nav_frame = ttk.Frame(main_frame, padding="5")
        self.nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Phase selector
        ttk.Label(self.nav_frame, text="Current Phase:").pack(side=tk.LEFT, padx=(0, 5))
        phase_dropdown = ttk.Combobox(self.nav_frame, textvariable=self.current_phase, 
                                      values=self.phases, state="readonly", width=15)
        phase_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        # Auto-advance checkbox
        auto_advance_check = ttk.Checkbutton(self.nav_frame, text="Auto-advance to next phase", 
                                            variable=self.auto_advance)
        auto_advance_check.pack(side=tk.LEFT, padx=(10, 0))
        
        # Add save working image button for debugging
        ttk.Separator(self.nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.save_working_image_btn = ttk.Button(self.nav_frame, text="Save Working Image", 
                                              command=self.save_working_image)
        self.save_working_image_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Create a PanedWindow for the main content area
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Configure the sash (the draggable separator)
        style = ttk.Style()
        style.configure('TPanedwindow', background='#cccccc')
        style.configure('Sash', background='#999999', handlesize=8, sashthickness=6)
        
        # Left column for controls - use LabelFrame inside the PanedWindow
        self.controls_frame = ttk.LabelFrame(self.paned_window, text=f"Controls - {self.current_phase.get()}", padding="10", width=320)
        
        # Create a canvas container frame for the preview on the right
        self.preview_container = ttk.Frame(self.paned_window)
        
        # Add both frames to the PanedWindow
        self.paned_window.add(self.controls_frame, weight=0)  # Controls don't expand
        self.paned_window.add(self.preview_container, weight=1)  # Preview expands
        
        # Prevent controls frame from shrinking
        self.controls_frame.pack_propagate(False)
        
        # Create frames for each phase
        self.phase1_frame = ttk.Frame(self.controls_frame)
        self.phase2_frame = ttk.Frame(self.controls_frame)
        self.phase3_frame = ttk.Frame(self.controls_frame)
        
        # Create content for each phase
        self.create_phase1_controls(self.phase1_frame)
        self.create_phase2_controls(self.phase2_frame)
        self.create_phase3_controls(self.phase3_frame)
        
        # Initialize with Phase 1 visible
        self.phase1_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the canvas manager for the preview
        self.canvas_frame = ttk.LabelFrame(self.preview_container, text=f"Preview - {self.current_phase.get()}")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_manager = ui_components.CanvasManager(self.canvas_frame)
        self.canvas = self.canvas_manager.canvas
    
    def create_phase1_controls(self, parent):
        # Image selection
        ttk.Label(parent, text="Background Image:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        background_frame = ttk.Frame(parent)
        background_frame.grid(row=0, column=1, sticky=tk.W, pady=(0, 5))
        ttk.Button(background_frame, text="Browse...", command=self.browse_background).pack(side=tk.LEFT)
        ttk.Label(background_frame, textvariable=self.bg_path_display).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(parent, text="Foreground Image:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        foreground_frame = ttk.Frame(parent)
        foreground_frame.grid(row=1, column=1, sticky=tk.W, pady=(0, 5))
        ttk.Button(foreground_frame, text="Browse...", command=self.browse_object).pack(side=tk.LEFT)
        ttk.Label(foreground_frame, textvariable=self.obj_path_display).pack(side=tk.LEFT, padx=(5, 0))
        
        # Blur parameters
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(parent, text="Blur Parameters", font=("TkDefaultFont", 10, "bold")).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(parent, text="Blur Length (px):").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Entry(parent, textvariable=self.blur_length, width=10).grid(row=4, column=1, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(parent, text="Code Type:").grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        code_options = ["optimal", "box", "random", "mura"]
        ttk.Combobox(parent, textvariable=self.code_type, values=code_options, state="readonly", width=10).grid(row=5, column=1, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(parent, text="Angle (degrees):").grid(row=6, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Entry(parent, textvariable=self.angle, width=10).grid(row=6, column=1, sticky=tk.W, pady=(0, 5))
        
        # Directory settings
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(parent, text="Output Directory", font=("TkDefaultFont", 10, "bold")).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        output_dir_frame = ttk.Frame(parent)
        output_dir_frame.grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=(0, 5))
        ttk.Entry(output_dir_frame, textvariable=self.output_dir, width=15).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_dir_frame, text="...", command=self.browse_output_dir, width=2).pack(side=tk.RIGHT)
        
        # Position info
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(parent, text="Object Position", font=("TkDefaultFont", 10, "bold")).grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.position_label = ttk.Label(parent, text="X: 0, Y: 0")
        self.position_label.grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(parent, text="Drag the object in the preview\nto position it").grid(row=13, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Action buttons
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=14, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        # Live Preview toggle
        self.preview_button = ttk.Checkbutton(parent, text="Live Preview", 
                                            variable=self.live_preview, 
                                            command=self.toggle_live_preview)
        self.preview_button.grid(row=15, column=0, columnspan=2, sticky=tk.EW, pady=(0, 5))
        
        # Apply & Save button
        ttk.Button(parent, text="Apply & Save", command=self.apply_and_save).grid(row=16, column=0, columnspan=2, sticky=tk.EW, pady=(0, 5))
        
        # Reset button
        ttk.Button(parent, text="Reset", command=self.reset).grid(row=17, column=0, columnspan=2, sticky=tk.EW, pady=(0, 5))
        
        # Help text
        help_text = "Instructions:\n\n1. Select background and foreground images\n2. Set blur parameters\n3. Toggle 'Live Preview' to see changes in real-time\n4. Drag the object to position it\n5. Click 'Apply & Save' when satisfied"
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=200).grid(row=18, column=0, columnspan=2, sticky=tk.W, pady=(20, 0))
    
    def create_phase2_controls(self, parent):
        # Motion Crop controls
        ttk.Label(parent, text="Motion Crop", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        self.crop_button = ttk.Button(parent, text="Crop", command=self.start_crop)
        self.crop_button.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 5))
        
        self.confirm_crop_button = ttk.Button(parent, text="Confirm Crop", command=self.confirm_crop, state="disabled")
        self.confirm_crop_button.grid(row=2, column=0, columnspan=1, sticky=tk.EW, pady=(0, 5))
        
        self.cancel_crop_button = ttk.Button(parent, text="Cancel Crop", command=self.cancel_crop, state="disabled")
        self.cancel_crop_button.grid(row=2, column=1, columnspan=1, sticky=tk.EW, pady=(0, 5))
        
        # Add crop instructions
        self.crop_instructions = ttk.Label(parent, text="", wraplength=250, justify="left")
        self.crop_instructions.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Help text for Motion Crop
        help_text = "Instructions:\n\n1. Click 'Crop' to start crop mode\n2. Click to place 4 points defining the motion region:\n   - Points 1 & 2: Define motion direction\n   - Points 3 & 4: Define width boundaries\n3. You can drag points to adjust position\n4. Click 'Confirm Crop' when satisfied"
        ttk.Label(parent, text=help_text, justify=tk.LEFT, wraplength=200).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(20, 0))
    
    def create_phase3_controls(self, parent):
        # Deblurring controls
        ttk.Label(parent, text="Deblurring", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Deblurring parameters
        ttk.Label(parent, text="Blur Length:").grid(row=1, column=0, sticky=tk.W, pady=(5, 2))
        self.deblur_blur_length = tk.StringVar(value=self.blur_length.get())
        blur_length_entry = ttk.Entry(parent, textvariable=self.deblur_blur_length, width=10)
        blur_length_entry.grid(row=1, column=1, sticky=tk.W, pady=(5, 2))
        
        ttk.Label(parent, text="Code Type:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.deblur_code_type = tk.StringVar(value=self.code_type.get())
        code_options = ["optimal", "box", "random", "mura"]
        code_dropdown = ttk.Combobox(parent, textvariable=self.deblur_code_type, values=code_options, state="readonly", width=10)
        code_dropdown.grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Background type
        ttk.Label(parent, text="Background Type:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.background_type = tk.StringVar(value="none")
        background_options = ["none", "constant", "textured"]
        background_dropdown = ttk.Combobox(parent, textvariable=self.background_type, values=background_options, state="readonly", width=10)
        background_dropdown.grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Regularization factor
        ttk.Label(parent, text="Regularization Factor:").grid(row=4, column=0, sticky=tk.W, pady=(5, 2))
        reg_frame = ttk.Frame(parent)
        reg_frame.grid(row=4, column=1, sticky=tk.W, pady=(5, 2))

        # Create a StringVar to display the current value
        self.reg_value_display = tk.StringVar(value=f"{self.regularization_factor.get():.4f}")

        # Create the slider
        reg_slider = ttk.Scale(
            reg_frame, 
            from_=0.001, 
            to=0.05, 
            orient=tk.HORIZONTAL, 
            variable=self.regularization_factor,
            length=120,
            command=lambda val: self.reg_value_display.set(f"{float(val):.4f}")
        )
        reg_slider.pack(side=tk.LEFT, padx=(0, 5))

        # Display the current value
        reg_value_label = ttk.Label(reg_frame, textvariable=self.reg_value_display, width=6)
        reg_value_label.pack(side=tk.LEFT)
        
        # Background help text
        bg_help = ttk.Label(parent, text="'none': Must have black background or tight crop\n'constant': Best for most images\n'textured': Not yet implemented", 
                          justify="left", font=("TkDefaultFont", 8))
        bg_help.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Preview button
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        preview_frame = ttk.Frame(parent)
        preview_frame.grid(row=7, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        self.preview_deblur_button = ttk.Button(preview_frame, text="Preview Deblurring", command=self.preview_deblurring)
        self.preview_deblur_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Debug log area
        ttk.Label(parent, text="Debug Log:").grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(10, 2))
        
        # Create a frame to contain the Text widget and scrollbar
        log_frame = ttk.Frame(parent)
        log_frame.grid(row=9, column=0, columnspan=2, sticky=tk.NSEW, pady=5)
        parent.grid_rowconfigure(9, weight=1)  # Allow the log frame to expand
        
        # Create a Text widget with scrollbar
        self.debug_log = tk.Text(log_frame, wrap=tk.WORD, width=30, height=10, font=("Consolas", 9))
        self.debug_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.debug_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.debug_log.config(yscrollcommand=scrollbar.set)
        
        # Make the text widget read-only
        self.debug_log.config(state=tk.DISABLED)
        
        # Create a frame for the deblurring control buttons
        buttons_frame = ttk.Frame(parent)
        buttons_frame.grid(row=10, column=0, columnspan=2, sticky=tk.EW, pady=5)
        
        # Save button
        self.save_deblurred_button = ttk.Button(buttons_frame, text="Apply & Save", command=self.apply_deblurring_and_save)
        self.save_deblurred_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Cancel button (initially disabled)
        self.cancel_deblur_button = ttk.Button(buttons_frame, text="Cancel", command=self.cancel_deblurring, state=tk.DISABLED)
        self.cancel_deblur_button.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.reset_deblur_button = ttk.Button(buttons_frame, text="Reset", command=self.reset)
        self.reset_deblur_button.pack(side=tk.LEFT, padx=5)
        
        # Status & information
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=2, sticky=tk.EW, pady=10)
        
        self.deblur_status = ttk.Label(parent, text="Ready for deblurring", wraplength=250, justify="left")
        self.deblur_status.grid(row=12, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Initialize threading variables
        self.deblurring_thread = None
        self.deblurring_cancelled = False
        
        # Initialize variables to track original image for reset
        self.original_image_for_reset = None
        self.has_been_deblurred = False
    
    def phase_changed(self, *args):
        """Handle phase changes from dropdown."""
        # Get the target phase
        phase = self.current_phase.get()
        
        # Hide all phase frames
        self.phase1_frame.pack_forget()
        self.phase2_frame.pack_forget()
        self.phase3_frame.pack_forget()
        
        # Update LabelFrame titles to show current phase
        self.controls_frame.configure(text=f"Controls - {phase}")
        self.canvas_frame.configure(text=f"Preview - {phase}")
        
        # Force cleanup of any drag handlers before switching phases
        if self.drag_handler:
            self.drag_handler.unbind_events()
            self.drag_handler = None
            
        # Prepare for the specific phase
        self.prepare_phase(phase)
        
        # Update path displays after phase preparation
        self.update_path_displays()
        
        # Show the selected phase frame
        if phase == "Image Creation":
            self.phase1_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        elif phase == "Motion Crop":
            self.phase2_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        elif phase == "Deblurring":
            self.phase3_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def prepare_phase(self, phase):
        """Prepare the application for a specific phase by setting up appropriate state."""
        # Ensure baking occurs for non-Phase 1 preparation
        if phase != "Image Creation":
            # Always bake the composite when entering Phase 2 or 3
            self.bake_composite()
            # Ensure foreground object is completely cleared
            self.object_image = None
            self.blurred_object = None
        
        # Phase-specific preparation
        if phase == "Image Creation":
            # Cancel any ongoing crop operation
            if self.crop_handler and self.crop_handler.active:
                self.cancel_crop()
            
            # Refresh canvas
            self.update_canvas()
            print("Object dragging enabled for image creation phase")
            
        elif phase == "Motion Crop":
            # Disable live preview when entering crop phase
            if self.live_preview.get():
                self.live_preview.set(False)
            
            # Reset crop buttons to initial state
            self.crop_button.configure(state="normal")
            self.confirm_crop_button.configure(state="disabled")
            self.cancel_crop_button.configure(state="disabled")
            self.crop_instructions.configure(text="")
            
        elif phase == "Deblurring":
            # Cancel any ongoing crop operation
            if self.crop_handler and self.crop_handler.active:
                self.cancel_crop()
                
            # Prepare for deblurring phase
            self.prepare_phase3()
    
    def bake_composite(self):
        """Bake the current composite image into the background.
        This creates a composite if one doesn't exist and sets it as the background."""
        if self.background_image is None:
            return
            
        # Make sure we have a composite image to use as the new background
        if self.object_image is not None:
            # If we already have a result image (from blur), use that
            if self.result_image is not None:
                # Already have a composite, use it
                composite = self.result_image
            else:
                # Create a new composite with the unblurred foreground object
                composite = image_processing.composite_images(
                    self.background_image.copy(), 
                    self.object_image, 
                    self.object_position
                )
                
            # Set the composite as the new background
            self.background_image = composite.copy()
            # Clear the background path to indicate this is a working canvas
            self.background_path = None
                
        # Clear other image references to prevent them from being displayed or manipulated
        self.object_image = None
        self.blurred_object = None
        self.result_image = None
        self.object_position = (0, 0)
        self.object_path = None
            
        # Update path displays
        self.update_path_displays()
            
        # Redraw the canvas
        self.update_canvas()
    
    def advance_to_next_phase(self):
        """Advance to the next phase if auto-advance is enabled."""
        if not self.auto_advance.get():
            return
            
        current_idx = self.phases.index(self.current_phase.get())
        if current_idx < len(self.phases) - 1:
            self.current_phase.set(self.phases[current_idx + 1])
    
    def update_path_displays(self):
        """Update the path display labels based on current state."""
        # Background image path display
        if self.background_image is None:
            # No background image
            self.bg_path_display.set("")
        elif self.background_path is None:
            # Background exists but no path - this is a working canvas
            self.bg_path_display.set("Working Canvas")
        else:
            # Show the filename part of the path
            filename = os.path.basename(self.background_path)
            self.bg_path_display.set(filename)
        
        # Foreground image path display
        if self.object_image is None:
            # No foreground image
            self.obj_path_display.set("")
        else:
            # Show the filename part of the path
            if self.object_path:
                filename = os.path.basename(self.object_path)
                self.obj_path_display.set(filename)
            else:
                self.obj_path_display.set("")
    
    def browse_background(self):
        path = utils.browse_for_file("Select Background Image")
        if path:
            self.background_path = path
            self.load_background()
            self.update_path_displays()  # Update the path display
            self.update_canvas()
            # Apply live preview if enabled
            if self.live_preview.get() and self.object_image is not None:
                self.apply_blur()
    
    def browse_object(self):
        path = utils.browse_for_file("Select Foreground Image")
        if path:
            self.object_path = path
            self.load_object()
            self.update_path_displays()  # Update the path display
            self.update_canvas()
            # Apply live preview if enabled
            if self.live_preview.get() and self.background_image is not None:
                self.apply_blur()
    
    def browse_output_dir(self):
        directory = utils.browse_for_directory("Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            utils.ensure_dir_exists(directory)
    
    def load_background(self):
        try:
            # Load image with CV2
            cv_img = image_processing.load_image(self.background_path)
            
            # Convert from BGR to RGB for display
            self.background_image = image_processing.convert_bgr_to_rgb(cv_img)
            
            # Center the object if it exists
            if self.object_image is not None:
                self.center_object()
        except Exception as e:
            utils.show_error("Error", f"Failed to load background image: {str(e)}")
    
    def load_object(self):
        try:
            # Load image with transparency support
            cv_img = image_processing.load_image(self.object_path, with_alpha=True)
            
            # Convert from BGR(A) to RGB(A) for display
            self.object_image = image_processing.convert_bgr_to_rgb(cv_img)
            
            # Center the object if background exists
            if self.background_image is not None:
                self.center_object()
        except Exception as e:
            utils.show_error("Error", f"Failed to load foreground image: {str(e)}")
    
    def center_object(self):
        """Center the object on the background."""
        bg_h, bg_w = self.background_image.shape[:2]
        obj_h, obj_w = self.object_image.shape[:2]
        
        x = (bg_w - obj_w) // 2
        y = (bg_h - obj_h) // 2
        
        self.object_position = (x, y)
        self.update_position_label()
    
    def update_position_label(self):
        """Update the position label with current coordinates."""
        x, y = self.object_position
        self.position_label.config(text=f"X: {x}, Y: {y}")
    
    def update_canvas(self):
        """Update the canvas with current images."""
        # Check if we have valid images
        if self.background_image is None and self.result_image is None:
            # Display placeholder
            self.canvas_manager.display_placeholder("Select background image")
            return
        
        # Check if we should display a deblurred image in Phase 3
        if (self.current_phase.get() == "Deblurring" and 
            self.has_been_deblurred and 
            hasattr(self, 'deblurred_image') and 
            self.deblurred_image is not None):
            # Display the deblurred image
            self.canvas_manager.display_image(self.deblurred_image)
            return
        
        # If result image exists, display only that (after cropping, this should be the only visible image)
        if self.result_image is not None:
            self.canvas_manager.display_image(self.result_image, "result")
            return
        
        # If no result image but background exists, display background first
        if self.background_image is not None:
            # Display background
            self.canvas_manager.display_image(self.background_image, "background")
            
            # Only create composite and enable dragging in Phase 1
            if self.current_phase.get() == "Image Creation" and self.object_image is not None:
                # Create a composite image
                composite = image_processing.composite_images(
                    self.background_image, 
                    self.object_image, 
                    self.object_position
                )
                
                # Display the composite
                self.canvas_manager.display_image(composite, "composite")
                
                # Make sure the canvas has the current scale attribute set
                self.canvas.current_scale = self.canvas_manager.current_scale
                
                # Set up drag handling for the object
                if self.drag_handler is None:
                    self.drag_handler = ui_components.ObjectDragHandler(
                        self.canvas, 
                        on_drag_callback=self.on_object_drag
                    )
                
                self.drag_handler.set_object_info(
                    self.object_position,
                    (self.object_image.shape[0], self.object_image.shape[1]),
                    (self.background_image.shape[0], self.background_image.shape[1])
                )
    
    def on_object_drag(self, new_position):
        """Handle object dragging."""
        self.object_position = new_position
        self.update_position_label()
        
        # Reset any blur result when repositioning
        self.result_image = None
        
        # Update canvas with new composite
        if self.background_image is not None and self.object_image is not None:
            composite = image_processing.composite_images(
                self.background_image, 
                self.object_image, 
                self.object_position
            )
            self.canvas_manager.display_image(composite, "composite")
            
            # Apply live preview if enabled
            if self.live_preview.get():
                self.apply_blur()
    
    def on_window_resize(self, event):
        """Handle window resize event."""
        # Only update if it's a true window resize (not widget changes)
        if event.widget == self.root:
            # Check if we're already resizing
            if not self.is_resizing:
                # Set resizing flag
                self.is_resizing = True
                
                # Show placeholder gray box instead of image
                if hasattr(self, 'canvas_manager') and self.canvas_manager:
                    self.canvas_manager.show_resize_placeholder()
                
            # Cancel any existing after callback to avoid multiple updates
            try:
                self.root.after_cancel(self._resize_after_id)
            except AttributeError:
                pass
            
            # Schedule update after a delay
            self._resize_after_id = self.root.after(200, self.update_after_resize)
    
    def update_after_resize(self):
        """Update canvas after window resize is complete."""
        # Reset the resizing flag
        self.is_resizing = False
        
        # Handle deblurred image case for Phase 3
        if (self.current_phase.get() == "Deblurring" and 
            self.has_been_deblurred and 
            hasattr(self, 'deblurred_image') and 
            self.deblurred_image is not None):
            # Explicitly display the deblurred image after resize
            self.canvas_manager.display_image(self.deblurred_image)
            print("Restored deblurred image after resize")
            return
        
        # Trigger a refresh of the canvas with current images
        self.update_canvas()
        
        # If we have a drag handler, update it with the new scale
        if self.drag_handler and self.background_image is not None and self.object_image is not None:
            self.canvas.current_scale = self.canvas_manager.current_scale
            self.drag_handler.set_object_info(
                self.object_position,
                (self.object_image.shape[0], self.object_image.shape[1]),
                (self.background_image.shape[0], self.background_image.shape[1])
            )
    
    def parameter_changed(self, *args):
        """Called when any parameter is changed."""
        if self.live_preview.get() and self.background_image is not None and self.object_image is not None:
            # Add a small delay to avoid processing during typing
            self.root.after(500, self.apply_blur_if_valid)

    def apply_blur_if_valid(self):
        """Apply blur only if parameters are valid."""
        try:
            # Validate parameters
            is_valid, blur_length = utils.validate_numeric_input(
                self.blur_length.get(), min_value=1, is_int=True
            )
            if not is_valid:
                return
            
            is_valid, angle = utils.validate_numeric_input(
                self.angle.get()
            )
            if not is_valid:
                return
            
            # If we get here, parameters are valid, so apply blur
            self.apply_blur()
        except ValueError:
            # Invalid number format, just skip
            return

    def apply_blur(self):
        """Apply the blur effect."""
        if self.background_image is None or self.object_image is None:
            return
        
        try:
            # Validate parameters
            is_valid, blur_length = utils.validate_numeric_input(
                self.blur_length.get(), min_value=1, is_int=True
            )
            if not is_valid:
                return
            
            is_valid, angle = utils.validate_numeric_input(
                self.angle.get()
            )
            if not is_valid:
                return
            
            code_method = self.code_type.get()
            
            # Generate code
            code_length = utils.CODE_LENGTH
            code = blur_core.generate_code_array(code_length, code_method)
            
            # Convert RGB(A) to BGR(A) for processing
            object_cv = image_processing.convert_rgb_to_bgr(self.object_image)
            
            # Apply blur
            blurred_object_cv = blur_core.apply_motion_blur(object_cv, code, blur_length, angle)
            
            # Convert back to RGB(A) for display
            self.blurred_object = image_processing.convert_bgr_to_rgb(blurred_object_cv)
            
            # Calculate the center of mass offset of the blur
            offset_x, offset_y = blur_core.calculate_blur_offset(code, blur_length, angle)
            print(f"Blur offset calculated: x={offset_x}, y={offset_y}")
            
            # Adjust the position to compensate for the blur offset
            adjusted_position = (
                self.object_position[0] - offset_x,
                self.object_position[1] - offset_y
            )
            
            # Create composite image with the adjusted position
            self.result_image = image_processing.composite_images(
                self.background_image.copy(),
                self.blurred_object,
                adjusted_position
            )
            
            # Reset drag handler to ensure it's reinitialized in Phase 1
            if self.current_phase.get() == "Image Creation":
                self.drag_handler = None
            
            # Update canvas
            self.update_canvas()
            
        except Exception as e:
            print(f"Error applying blur: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def toggle_live_preview(self):
        """Toggle live preview on/off."""
        if self.live_preview.get():
            # If enabling live preview, apply blur immediately
            self.apply_blur_if_valid()
        else:
            # If disabling, show a message
            print("Live preview disabled")
    
    def apply_and_save(self):
        """Apply the blur and save the result."""
        # If we're in Phase 1, ensure blur is applied and composite is created
        if self.current_phase.get() == "Image Creation":
            # Apply blur if not already applied
            if self.result_image is None:
                self.apply_blur()
                
            if self.result_image is None:
                utils.show_error("Error", "Failed to create a preview. Please check your settings.")
                return
            
            # Also create an unblurred composite for saving
            unblurred_composite = None
            if self.background_image is not None and self.object_image is not None:
                unblurred_composite = image_processing.composite_images(
                    self.background_image.copy(),
                    self.object_image,
                    self.object_position
                )
                
                # Store the unblurred composite as class variable for PSNR calculation in Phase 3
                self.unblurred_reference_image = unblurred_composite.copy()
                print("Stored unblurred reference image for PSNR calculation")
        
        # For other phases, just make sure we have something to save
        elif self.result_image is None and self.background_image is None:
            utils.show_error("Error", "No image to save.")
            return
        
        try:
            # Make sure output directory exists
            output_dir = self.output_dir.get()
            utils.ensure_dir_exists(output_dir)
            
            # Generate output path
            output_path = utils.get_full_output_path(
                output_dir,
                self.code_type.get(),
                int(self.blur_length.get()),
                float(self.angle.get())
            )
            
            # Determine which image to save
            save_image = self.result_image if self.result_image is not None else self.background_image
            
            # Save the image
            image_processing.save_image(save_image, output_path)
            
            # If in Phase 1 and we have an unblurred composite, save it too
            if self.current_phase.get() == "Image Creation" and unblurred_composite is not None:
                # Create path for unblurred version
                unblurred_path = os.path.splitext(output_path)[0] + "_original_unblurred.png"
                
                # Save the unblurred composite (same dimensions as blurred image)
                image_processing.save_image(unblurred_composite, unblurred_path)
                print(f"Saved unblurred composite to: {unblurred_path}")
            
            utils.show_info("Success", f"Image saved to:\n{output_path}")
            
            # Log parameters for Phase 1 if applicable
            if self.current_phase.get() == "Image Creation":
                # Log the parameters at the end of Phase 1
                self.logger.log_parameters({
                    "phase": "Image Creation",
                    "background_image_path": self.background_path,
                    "foreground_image_path": self.object_path,
                    "object_position_x": self.object_position[0],
                    "object_position_y": self.object_position[1],
                    "blur_length": int(self.blur_length.get()),
                    "code_type": self.code_type.get(),
                    "blur_angle": float(self.angle.get()),
                    "output_directory": output_dir,
                    "output_file_path": output_path
                })
                print(f"Logged parameters for Image Creation phase")
            
            # In Phase 1, bake the composite after saving
            if self.current_phase.get() == "Image Creation":
                self.bake_composite()
            
            # Auto advance to next phase if enabled
            self.advance_to_next_phase()
            
        except Exception as e:
            utils.show_error("Error", f"Failed to save image: {str(e)}")
    
    def reset(self):
        """Reset the current phase."""
        current_phase = self.current_phase.get()
        
        if current_phase == "Image Creation":
            # Clear all images and reset parameters
            self.background_image = None
            self.background_path = None
            self.object_image = None
            self.object_path = None
            self.blurred_object = None
            self.result_image = None
            self.object_position = (0, 0)
            
            # Update displays
            self.update_path_displays()
            self.canvas_manager.clear_canvas()
            self.canvas_manager.display_placeholder("Select background and foreground images")
            
            utils.show_info("Reset", "Image creation phase has been reset.")
            
        elif current_phase == "Motion Crop":
            # Cancel any ongoing crop operation
            if self.crop_handler and self.crop_handler.active:
                self.cancel_crop()
            
            # Reset to the state at the beginning of Phase 2
            self.crop_button.configure(state="normal")
            self.confirm_crop_button.configure(state="disabled")
            self.cancel_crop_button.configure(state="disabled")
            self.crop_instructions.configure(text="")
            
            # Display the original composite
            self.update_canvas()
            
            utils.show_info("Reset", "Motion crop has been reset.")
            
        elif current_phase == "Deblurring":
            # Reset deblurring state
            if hasattr(self, 'deblurred_image'):
                delattr(self, 'deblurred_image')
            
            # Restore the original image from the saved copy
            if hasattr(self, 'original_image_for_reset') and self.original_image_for_reset is not None:
                print("Resetting to original blurred image")
                self.log_message("Resetting to original blurred image")
                self.background_image = self.original_image_for_reset.copy()
            else:
                print("WARNING: No original image to reset to")
                self.log_message("WARNING: No original image to reset to")
            
            # Reset deblurring tracking flag
            self.has_been_deblurred = False
            
            # Do not reset parameters to defaults from Phase 1
            #self.deblur_blur_length.set(self.blur_length.get())
            #self.deblur_code_type.set(self.code_type.get())
            #self.background_type.set("none")
            #self.regularization_factor.set(0.005)  # Reset to default value
            
            # Redisplay the original image
            self.canvas_manager.display_image(self.background_image)
            
            # Update status
            self.deblur_status.configure(text="Ready for deblurring")
            
            utils.show_info("Reset", "Deblurring has been reset.")
    
    def start_crop(self):
        """Start the cropping process."""
        # Make sure we have an image to crop
        if self.background_image is None:
            utils.show_error("Error", "Please load an image first.")
            return
        
        # Create crop handler if needed
        if self.crop_handler is None:
            self.crop_handler = ui_components.CropHandler(self.canvas)
            
        # Store the current scale factor in the canvas for the crop handler to access
        self.canvas.current_scale = self.canvas_manager.current_scale
        print(f"Setting canvas scale to: {self.canvas_manager.current_scale}")
        
        # Update button states
        self.crop_button.configure(state="disabled")
        self.confirm_crop_button.configure(state="normal")
        self.cancel_crop_button.configure(state="normal")
        
        # Display instructions
        self.crop_instructions.configure(text="Select 4 points:\n"
                                        "1 & 2: Define motion direction line\n"
                                        "3 & 4: Define width boundaries\n"
                                        "You can drag points to adjust.")
        
        # Start crop mode
        self.crop_handler.start_crop(on_complete_callback=self.on_crop_points_selected)
    
    def on_crop_points_selected(self, points):
        """Called when 4 crop points are selected."""
        # Enable confirm button
        self.confirm_crop_button.configure(state="normal")
    
    def confirm_crop(self):
        """Confirm the crop and apply perspective transformation."""
        print("DEBUG - confirm_crop method called")
        
        if not self.crop_handler or not self.crop_handler.active:
            print("DEBUG - crop_handler is None or not active, returning")
            return
            
        try:
            # Get the current displayed image
            current_image = self.result_image if self.result_image is not None else self.background_image.copy()
            
            if current_image is None:
                utils.show_error("Error", "No image to crop.")
                return
            
            print("DEBUG - About to calculate crop dimensions")
            
            # Calculate dimensions and corner points
            dst_width, dst_height, src_corners = self.crop_handler.calculate_crop_dimensions()
            
            # IMPORTANT: Save crop points BEFORE calling cancel_crop
            # Get the crop points (need to adjust for canvas scale later for logging)
            saved_crop_points = self.crop_handler.get_points().copy()
            saved_motion_angle = self.crop_handler.get_motion_angle() if hasattr(self.crop_handler, 'get_motion_angle') else None
            print(f"DEBUG - Saved crop points before cleanup: {saved_crop_points}")
            
            # Apply the perspective transformation to the blurred image
            warped = image_processing.apply_perspective_transform(
                current_image, src_corners, dst_width, dst_height
            )
            
            # Create an unblurred version with the same transformation if we have the original image
            # At this stage, we need to find or create an unblurred version to transform
            # Try to find original image or background from previous phase
            unblurred_source = None
            
            # Make sure output directory exists for saving the unblurred crop
            output_dir = self.output_dir.get()
            utils.ensure_dir_exists(output_dir)
            
            # Find a suitable source for the unblurred image:
            # If phase 1 result had both foreground and background, there should be
            # an associated _original_unblurred.png image in the output directory
            # Look through files in the output directory for unblurred images
            unblurred_paths = []
            for filename in os.listdir(output_dir):
                if "_original_unblurred.png" in filename:
                    unblurred_paths.append(os.path.join(output_dir, filename))
            
            # Use the most recent unblurred image if available
            if unblurred_paths:
                unblurred_paths.sort(key=os.path.getmtime, reverse=True)
                latest_unblurred = unblurred_paths[0]
                try:
                    unblurred_source = image_processing.load_image(latest_unblurred)
                    # Convert BGR to RGB for processing
                    unblurred_source = image_processing.convert_bgr_to_rgb(unblurred_source)
                    print(f"Found unblurred source image: {latest_unblurred}")
                except Exception as e:
                    print(f"Error loading unblurred source: {str(e)}")
            
            # If no unblurred source was found, try to reconstruct one
            if unblurred_source is None and hasattr(self, 'object_image') and self.object_image is not None:
                # Recreate the original composite
                unblurred_source = image_processing.composite_images(
                    self.background_image.copy(),
                    self.object_image,
                    self.object_position
                )
                print("Created unblurred source image from original components")
            
            # Apply the same perspective transformation to the unblurred source if available
            unblurred_warped = None
            if unblurred_source is not None:
                unblurred_warped = image_processing.apply_perspective_transform(
                    unblurred_source, src_corners, dst_width, dst_height
                )
            
            # Store the unblurred warped image as a class variable for PSNR calculation in Phase 3
            self.unblurred_reference_image = unblurred_warped.copy()
            print("Stored unblurred reference image for PSNR calculation")
            
            # Clean up crop UI elements
            self.cancel_crop()
            
            # Clear the canvas to ensure proper cleanup
            self.canvas.delete("all")
            self.canvas_manager.photo_image = None
            
            # Update the background image with the warped result
            self.background_image = warped
            
            # Clear other image references
            self.object_image = None
            self.blurred_object = None
            self.result_image = None
            self.object_position = (0, 0)
            
            # Clear paths since this is now a working canvas
            self.background_path = None
            self.object_path = None
            
            # Update path displays
            self.update_path_displays()
            
            # Update angle to 0 since the motion is now horizontal
            self.angle.set("0")
            
            # Add a short delay before updating the canvas to ensure clean redraw
            self.root.after(50, self.update_canvas)
            
            # Generate timestamp for filenames
            timestamp = utils.get_timestamp()
            
            # Save the unblurred warped image if available
            if unblurred_warped is not None:
                # Generate output filename
                unblurred_crop_path = os.path.join(
                    output_dir, 
                    f"cropped_{timestamp}_original_unblurred.png"
                )
                
                # Save the unblurred cropped image (same dimensions as cropped blurred image)
                image_processing.save_image(unblurred_warped, unblurred_crop_path)
                print(f"Saved unblurred cropped image to: {unblurred_crop_path}")
            
            # Save the blurred warped image (current background_image)
            blurred_crop_path = os.path.join(
                output_dir, 
                f"cropped_{timestamp}_blurred.png"
            )
            image_processing.save_image(warped, blurred_crop_path)
            print(f"Saved blurred cropped image to: {blurred_crop_path}")
            
            # Log parameters for Phase 2 (Motion Crop)
            # Check if we have saved crop points
            if len(saved_crop_points) == 0:
                print("WARNING: No saved crop points available")
                # Log without crop points
                self.logger.log_parameters({
                    "phase": "Motion Crop",
                    "motion_angle": saved_motion_angle,
                    "output_file_path": blurred_crop_path,
                    "crop_error": "No crop points available"
                })
                print(f"Logged Motion Crop phase parameters without crop points")
            else:
                # Use the saved points instead of trying to get them after they're cleared
                # Get the scale for adjustment
                scale = getattr(self.canvas, 'current_scale', 1.0)
                
                # Debug output to verify saved crop points
                print(f"DEBUG - Using saved crop points for logging: {saved_crop_points}")
                
                # Store each point as a separate parameter for better JSON serialization
                # Convert from canvas coordinates to image coordinates by dividing by scale
                crop_params = {}
                for i, (x, y) in enumerate(saved_crop_points):
                    # Store coordinates as floating point values for accuracy
                    crop_params[f"crop_point{i+1}_x"] = float(x/scale)
                    crop_params[f"crop_point{i+1}_y"] = float(y/scale)
                    print(f"DEBUG - Adding crop point {i+1}: x={float(x/scale)}, y={float(y/scale)}")
                
                # Add the rest of the parameters
                crop_params.update({
                    "phase": "Motion Crop",
                    "motion_angle": saved_motion_angle,
                    "output_file_path": blurred_crop_path
                })
                
                # Debug output to verify parameters before logging
                print(f"DEBUG - Parameters being logged: {crop_params}")
                
                self.logger.log_parameters(crop_params)
                print(f"Logged parameters for Motion Crop phase")
            
            utils.show_info("Success", "Image cropped and perspective corrected successfully.")
            
            # Auto advance to next phase if enabled
            self.advance_to_next_phase()
            
        except Exception as e:
            utils.show_error("Error", f"Failed to apply crop: {str(e)}")
            import traceback
            traceback.print_exc()
            self.cancel_crop()
    
    def cancel_crop(self):
        """Cancel the cropping process."""
        if self.crop_handler:
            self.crop_handler.cancel_crop()
        
        # Reset button states
        self.crop_button.configure(state="normal")
        self.confirm_crop_button.configure(state="disabled")
        self.cancel_crop_button.configure(state="disabled")
        
        # Clear instructions
        self.crop_instructions.configure(text="")

    def preview_deblurring(self):
        """Preview the deblurring result on the canvas using a separate thread."""
        # Check if deblurring has already been done and no parameters have changed
        if self.has_been_deblurred and hasattr(self, 'deblurred_image') and self.deblurred_image is not None:
            # Just display the existing deblurred image
            self.canvas_manager.display_image(self.deblurred_image)
            self.deblur_status.configure(text="Showing existing deblurred image")
            self.log_message("Using existing deblurred image")
            return
        
        # Check if we have a valid image to deblur
        if self.background_image is None:
            utils.show_error("Error", "No image to deblur. Please complete Phase 2 first.")
            return
        
        # Ensure we have the original image for deblurring
        if not hasattr(self, 'original_image_for_reset') or self.original_image_for_reset is None:
            # If original image is missing, store a copy of the current image as the original
            self.log_message("Original image for deblurring not found, using current image as original")
            self.original_image_for_reset = self.background_image.copy()
        
        # Get deblurring parameters
        try:
            blur_length = int(self.deblur_blur_length.get())
            code_type = self.deblur_code_type.get()
            background_type = self.background_type.get()
            
            # Validate parameters
            if blur_length <= 0:
                utils.show_error("Error", "Blur length must be a positive integer.")
                return
                
            if code_type not in ["optimal", "box", "random", "mura"]:
                utils.show_error("Error", "Invalid code type.")
                return
                
            if background_type not in ["none", "constant", "textured"]:
                utils.show_error("Error", "Invalid background type.")
                return
            
            # Display appropriate warnings based on background type
            if not self.validate_background_settings(background_type):
                return
                
            # Validate that blur length is not too large for the image
            if blur_length >= self.background_image.shape[1]:
                utils.show_error("Error", f"Blur length ({blur_length}) is too large for image width ({self.background_image.shape[1]}). Reduce blur length.")
                return
            
            # Clear the debug log
            self.debug_log.config(state=tk.NORMAL)
            self.debug_log.delete(1.0, tk.END)
            self.debug_log.config(state=tk.DISABLED)
            
            # Reset cancellation flag
            self.deblurring_cancelled = False
            
            # Disable controls and enable cancel button
            self.disable_controls_during_deblurring(True)
            
            # Update status
            self.deblur_status.configure(text="Deblurring in progress...")
            self.log_message(f"Starting deblurring with parameters:")
            self.log_message(f"- Blur length: {blur_length}")
            self.log_message(f"- Code type: {code_type}")
            self.log_message(f"- Background type: {background_type}")
            self.log_message(f"- Regularization factor: {self.regularization_factor.get()}")
            self.log_message(f"- Image shape: {self.background_image.shape}")
            
            # Start deblurring in a separate thread
            import threading
            self.deblurring_thread = threading.Thread(
                target=self._deblur_thread,
                args=(blur_length, code_type, background_type, "preview")
            )
            self.deblurring_thread.daemon = True
            self.deblurring_thread.start()
            
        except ValueError as e:
            utils.show_error("Parameter Error", f"Invalid parameter: {str(e)}")
        except Exception as e:
            utils.show_error("Deblurring Error", f"Unexpected error: {str(e)}")
            self.deblur_status.configure(text=f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def validate_background_settings(self, background_type):
        """Validate background settings and display appropriate warnings."""
        if background_type == "none":
            # Check if image appears to have non-black background
            # Look at edges of the image to determine if background is likely not black
            img = self.background_image
            h, w = img.shape[:2]
            
            # Sample points from edges of the image
            edge_samples = []
            edge_samples.extend(img[0, ::w//10])  # Top edge
            edge_samples.extend(img[h-1, ::w//10])  # Bottom edge
            edge_samples.extend(img[::h//10, 0])  # Left edge
            edge_samples.extend(img[::h//10, w-1])  # Right edge
            
            # Calculate average brightness of edges
            avg_brightness = np.mean([np.mean(sample) for sample in edge_samples])
            
            if avg_brightness > 30:  # Threshold for considering background as non-black
                # Ask user to confirm
                resp = messagebox.askyesno(
                    "Background Warning", 
                    "Background type 'none' requires a black background or tight crop around the "
                    "blurred object.\n\nYour image appears to have a non-black background, which may "
                    "cause poor deblurring results. Consider using 'constant' instead.\n\n"
                    "Do you want to continue anyway?"
                )
                return resp
        elif background_type == "textured":
            resp = messagebox.askyesno(
                "Background Warning",
                "Textured background handling is experimental and not fully implemented.\n\n"
                "For best results with textured backgrounds, ensure any strong edges lie "
                "outside the blur band.\n\n"
                "Consider using 'constant' background type instead.\n\n"
                "Do you want to continue anyway?"
            )
            return resp
            
        # For "constant" type or if checks pass, return True
        return True
        
    def apply_deblurring_and_save(self):
        """Apply deblurring and save the result using a separate thread."""
        # Check if deblurring is already done
        if hasattr(self, 'deblurred_image') and self.deblurred_image is not None:
            # Just save the existing result
            self._save_deblurred_image()
            
            # Do not make the deblurred image the new working image after saving
            # self.make_deblurred_current()  # Removed to retain original blurred image
            return
        
        # Otherwise, run deblurring and then save
        # Check if we have a valid image to deblur
        if self.background_image is None:
            utils.show_error("Error", "No image to deblur. Please complete Phase 2 first.")
            return
        
        # Get deblurring parameters
        try:
            blur_length = int(self.deblur_blur_length.get())
            code_type = self.deblur_code_type.get()
            background_type = self.background_type.get()
            
            # Validate parameters
            if blur_length <= 0:
                utils.show_error("Error", "Blur length must be a positive integer.")
                return
                
            if code_type not in ["optimal", "box", "random", "mura"]:
                utils.show_error("Error", "Invalid code type.")
                return
                
            if background_type not in ["none", "constant", "textured"]:
                utils.show_error("Error", "Invalid background type.")
                return
                
            # Display appropriate warnings based on background type
            if not self.validate_background_settings(background_type):
                return
                
            # Validate that blur length is not too large for the image
            if blur_length >= self.background_image.shape[1]:
                utils.show_error("Error", f"Blur length ({blur_length}) is too large for image width ({self.background_image.shape[1]}). Reduce blur length.")
                return
            
            # Clear the debug log
            self.debug_log.config(state=tk.NORMAL)
            self.debug_log.delete(1.0, tk.END)
            self.debug_log.config(state=tk.DISABLED)
            
            # Reset cancellation flag
            self.deblurring_cancelled = False
            
            # Disable controls and enable cancel button
            self.disable_controls_during_deblurring(True)
            
            # Update status
            self.deblur_status.configure(text="Deblurring and saving in progress...")
            self.log_message(f"Starting deblurring with parameters:")
            self.log_message(f"- Blur length: {blur_length}")
            self.log_message(f"- Code type: {code_type}")
            self.log_message(f"- Background type: {background_type}")
            self.log_message(f"- Regularization factor: {self.regularization_factor.get()}")
            self.log_message(f"- Image shape: {self.background_image.shape}")
            
            # Start deblurring in a separate thread
            import threading
            self.deblurring_thread = threading.Thread(
                target=self._deblur_thread,
                args=(blur_length, code_type, background_type, "save")
            )
            self.deblurring_thread.daemon = True
            self.deblurring_thread.start()
            
        except ValueError as e:
            utils.show_error("Parameter Error", f"Invalid parameter: {str(e)}")
        except Exception as e:
            utils.show_error("Deblurring Error", f"Unexpected error: {str(e)}")
            self.deblur_status.configure(text=f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _deblur_thread(self, blur_length, code_type, background_type, action="preview"):
        """Background thread for deblurring process."""
        try:
            import time
            import numpy as np
            
            # Start timing
            start_time = time.time()
            self.log_message("Generating code array...")
            
            # Generate the code array
            code = blur_core.generate_code_array(utils.CODE_LENGTH, code_type)
            self.log_message(f"Code generated ({len(code)} elements)")
            
            # Check if cancelled
            if self.deblurring_cancelled:
                self.log_message("Operation cancelled by user.")
                return
            
            # Always use the original image for deblurring, not potentially already deblurred ones
            if hasattr(self, 'original_image_for_reset') and self.original_image_for_reset is not None:
                # Use original blurred image for deblurring
                image_to_deblur = self.original_image_for_reset.copy()
                self.log_message("Using original blurred image for deblurring")
            else:
                # Fallback to background image if original isn't available
                image_to_deblur = self.background_image.copy()
                self.log_message("WARNING: Using current background image for deblurring (original not found)")
            
            # Log background type being used
            self.log_message(f"Using background type: {background_type}")
            
            # Handle different background types
            if background_type == "none":
                self.log_message("Warning: 'none' background type assumes blurred object is on a black background")
                self.log_message("If your image has a complex background, use 'constant' instead")
                # TODO: Future enhancement - allow user to draw tight crop around blurred object
            
            # Process with background estimation
            height, width, channels = image_to_deblur.shape
            self.log_message(f"Processing {channels} channels, image dimensions: {width}x{height}")
            
            # Calculate size of deblurred output
            n = width - blur_length + 1
            if n <= 0:
                error_msg = f"ERROR: Blur length too large, resulting width would be {n}"
                self.log_message(error_msg)
                self.root.after(0, lambda: utils.show_error("Error", f"Blur length ({blur_length}) too large for image width ({width})"))
                self.root.after(0, lambda: self.disable_controls_during_deblurring(False))
                return
            
            self.log_message(f"Output image width will be {n} pixels")
            
            # Apply deblurring with background estimation
            self.log_message("Starting deblurring process...")
            deblur_time = time.time()
            
            try:
                # Use the background-aware deblurring function
                deblurred = blur_core.deblur_with_background_estimation(
                    image_to_deblur, code, blur_length, background_type, 
                    regularization_factor=self.regularization_factor.get()
                )
                deblur_elapsed = time.time() - deblur_time
                self.log_message(f"Deblurring completed in {deblur_elapsed:.2f} seconds")
            except NotImplementedError as e:
                # Handle the textured background case
                error_msg = f"ERROR: {str(e)}"
                self.log_message(error_msg)
                self.root.after(0, lambda: utils.show_error("Not Implemented", str(e)))
                self.root.after(0, lambda: self.disable_controls_during_deblurring(False))
                return
            except ValueError as e:
                # Handle value errors (e.g., deblurred width too small)
                error_msg = f"ERROR: {str(e)}"
                self.log_message(error_msg)
                self.root.after(0, lambda: utils.show_error("Value Error", str(e)))
                self.root.after(0, lambda: self.disable_controls_during_deblurring(False))
                return
            
            # Check if cancelled
            if self.deblurring_cancelled:
                self.log_message("Operation cancelled by user.")
                return
            
            # Clip values to valid range and convert to uint8 (delay until the very end)
            self.log_message("Finalizing image...")
            deblurred_image = np.clip(deblurred, 0, 255).astype(np.uint8)
            
            # Store the result
            self.deblurred_image = deblurred_image
            self.has_been_deblurred = True
            
            # Calculate total time
            total_time = time.time() - start_time
            self.log_message(f"Total deblurring time: {total_time:.2f} seconds")
            
            # Calculate PSNR if we have a reference unblurred image
            if hasattr(self, 'unblurred_reference_image') and self.unblurred_reference_image is not None:
                try:
                    # First, crop the reference image to match the deblurred image dimensions
                    aligned_reference = image_processing.crop_reference_for_psnr(
                        self.unblurred_reference_image,
                        deblurred_image,
                        blur_length,
                        direction='horizontal'  # Default to horizontal blur direction
                    )
                    
                    # Log dimensions for debugging
                    self.log_message(f"Reference image dimensions: {self.unblurred_reference_image.shape}")
                    self.log_message(f"Aligned reference dimensions: {aligned_reference.shape}")
                    self.log_message(f"Deblurred image dimensions: {deblurred_image.shape}")
                    
                    # Calculate PSNR between aligned reference and deblurred result
                    psnr_value = image_processing.calculate_psnr(
                        aligned_reference, 
                        deblurred_image
                    )
                    
                    # Log the PSNR result
                    self.log_message(f"PSNR Calculation Results:")
                    self.log_message(f"Peak Signal-to-Noise Ratio: {psnr_value:.2f} dB")
                    
                except Exception as e:
                    self.log_message(f"Error calculating PSNR: {str(e)}")
                    import traceback
                    self.log_message(traceback.format_exc())
            else:
                self.log_message("No unblurred reference image available for PSNR calculation")
            
            # Update UI from main thread
            if not self.deblurring_cancelled:
                if action == "preview":
                    # Display the image
                    self.root.after(0, lambda: self.canvas_manager.display_image(self.deblurred_image))
                    self.root.after(0, lambda: self.deblur_status.configure(text="Deblurring preview completed."))
                elif action == "save":
                    # Save the image (but do not make it the current working image)
                    self.root.after(0, self._save_deblurred_image)
            
            # Re-enable controls from main thread
            self.root.after(0, lambda: self.disable_controls_during_deblurring(False))
            
        except Exception as e:
            error_msg = f"Error in deblurring thread: {str(e)}"
            self.log_message(error_msg)
            import traceback
            trace_str = traceback.format_exc()
            self.log_message(trace_str)
            
            # Ensure we update UI from main thread using root.after
            self.root.after(0, lambda: self.deblur_status.configure(text=f"Error: {str(e)}"))
            self.root.after(0, lambda: utils.show_error("Deblurring Error", f"{str(e)}\n\nSee debug log for details."))
            self.root.after(0, lambda: self.disable_controls_during_deblurring(False))
    
    def _save_deblurred_image(self):
        """Save the deblurred image to disk."""
        try:
            # Create the output directory if it doesn't exist
            output_dir = self.output_dir.get()
            utils.ensure_dir_exists(output_dir)
            
            # Generate output filename
            blur_length = int(self.deblur_blur_length.get())
            code_type = self.deblur_code_type.get()
            background_type = self.background_type.get()
            
            filename = f"deblurred_{code_type}_blur{blur_length}_{background_type}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save the deblurred image
            self.log_message(f"Saving result to {filename}...")
            image_processing.save_image(self.deblurred_image, output_path)
            
            # Also save properly cropped unblurred reference image if available
            if hasattr(self, 'unblurred_reference_image') and self.unblurred_reference_image is not None:
                try:
                    # Create path for aligned unblurred version
                    aligned_ref_filename = f"deblurred_{code_type}_blur{blur_length}_{background_type}_reference.png"
                    aligned_ref_path = os.path.join(output_dir, aligned_ref_filename)
                    
                    # Crop the reference to match the deblurred dimensions 
                    aligned_ref = image_processing.crop_reference_for_psnr(
                        self.unblurred_reference_image,
                        self.deblurred_image,
                        blur_length,
                        direction='horizontal'  # Default to horizontal blur direction
                    )
                    
                    # Save the aligned reference (same dimensions as deblurred image)
                    image_processing.save_image(aligned_ref, aligned_ref_path)
                    self.log_message(f"Saved aligned reference image to {aligned_ref_filename}")
                    
                except Exception as e:
                    self.log_message(f"Error saving aligned reference: {str(e)}")
            
            # Log parameters for Phase 3 (Deblurring)
            # Get PSNR value if it was calculated
            psnr_value = None
            for log_line in self.debug_log.get(1.0, tk.END).split('\n'):
                if "PSNR:" in log_line:
                    try:
                        psnr_text = log_line.split("PSNR:")[1].strip()
                        psnr_value = float(psnr_text.split()[0])  # Extract the numeric value
                    except (IndexError, ValueError):
                        pass
            
            # Log the parameters
            self.logger.log_parameters({
                "phase": "Deblurring",
                "blur_length": blur_length,
                "code_type": code_type,
                "background_type": background_type,
                "regularization_factor": self.regularization_factor.get(),
                "psnr_value": psnr_value,
                "output_file_path": output_path
            })
            print(f"Logged parameters for Deblurring phase")
            
            # Update status
            self.deblur_status.configure(text=f"Deblurred image saved to {filename} (Original image preserved)")
            self.log_message("Save completed successfully. Original image preserved for further deblurring.")
            
            # Show success message
            utils.show_info("Success", f"Deblurred image saved to {output_path}\n\nOriginal blurred image preserved for further deblurring.")
            
        except Exception as e:
            error_msg = f"Failed to save deblurred image: {str(e)}"
            self.log_message(error_msg)
            utils.show_error("Save Error", error_msg)
            self.deblur_status.configure(text=f"Error saving: {str(e)}")
            import traceback
            traceback.print_exc()

    def log_message(self, message):
        """Add a message to the debug log."""
        # Make sure this runs on the main thread
        def _update_log():
            # Enable text widget for editing
            self.debug_log.config(state=tk.NORMAL)
            
            # Add timestamp and message
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.debug_log.insert(tk.END, f"[{timestamp}] {message}\n")
            
            # Auto-scroll to the end
            self.debug_log.see(tk.END)
            
            # Make text widget read-only again
            self.debug_log.config(state=tk.DISABLED)
        
        # If called from a different thread, schedule for main thread
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, _update_log)
        else:
            _update_log()
    
    def disable_controls_during_deblurring(self, disabled=True):
        """Disable or enable UI controls during deblurring."""
        state = tk.DISABLED if disabled else tk.NORMAL
        cancel_state = tk.NORMAL if disabled else tk.DISABLED
        
        # Disable/enable phase selection dropdown
        for child in self.nav_frame.winfo_children():
            if isinstance(child, ttk.Combobox):
                child.config(state="readonly" if not disabled else tk.DISABLED)
            elif not isinstance(child, ttk.Separator):
                child.config(state=state)
        
        # Disable/enable deblurring controls
        self.preview_deblur_button.config(state=state)
        self.save_deblurred_button.config(state=state)
        self.reset_deblur_button.config(state=state)
        
        # Set cancel button state
        self.cancel_deblur_button.config(state=cancel_state)
        
        # Update UI
        self.root.update()
    
    def cancel_deblurring(self):
        """Cancel the deblurring process."""
        if self.deblurring_thread and self.deblurring_thread.is_alive():
            self.log_message("Cancelling deblurring process...")
            self.deblurring_cancelled = True
            
            # Wait for thread to finish (with a timeout)
            self.deblurring_thread.join(timeout=0.5)
            
            # Re-enable controls
            self.disable_controls_during_deblurring(False)
            
            # Update status
            self.deblur_status.configure(text="Deblurring cancelled")
            self.log_message("Deblurring process cancelled.")
    
    def prepare_phase3(self):
        """Prepare the application for Phase 3 (Deblurring)."""
        # Make sure we have a valid image to deblur from Phase 2
        if self.background_image is None:
            utils.show_error("Error", "No image to deblur. Please complete Phase 2 first.")
            return False
            
        # Set initial deblurring parameters based on blur settings from Phase 1
        self.deblur_blur_length.set(self.blur_length.get())
        self.deblur_code_type.set(self.code_type.get())
        
        # Save a copy of the original image for reset purposes
        self.original_image_for_reset = self.background_image.copy()
        
        # Reset deblurring state tracking
        if hasattr(self, 'deblurred_image'):
            delattr(self, 'deblurred_image')
        self.has_been_deblurred = False
        
        # Set up variable tracing for deblurring parameters
        self.deblur_blur_length.trace_add("write", self.deblurring_parameter_changed)
        self.deblur_code_type.trace_add("write", self.deblurring_parameter_changed)
        self.background_type.trace_add("write", self.deblurring_parameter_changed)
        self.regularization_factor.trace_add("write", self.deblurring_parameter_changed)
        
        # Update status
        self.deblur_status.configure(text="Ready for deblurring")
        
        return True

    def deblurring_parameter_changed(self, *args):
        """Called when any deblurring parameter is changed."""
        # Reset the flag to indicate deblurring needs to be redone
        self.has_been_deblurred = False
        
        # Log the parameter that changed
        param_name = args[0] if args else "unknown"
        
        # Get the parameter's new value for logging
        if param_name == self.deblur_blur_length._name:
            value = self.deblur_blur_length.get()
        elif param_name == self.deblur_code_type._name:
            value = self.deblur_code_type.get()
        elif param_name == self.background_type._name:
            value = self.background_type.get()
        elif param_name == self.regularization_factor._name:
            value = self.regularization_factor.get()
        else:
            value = "unknown"
        
        print(f"Deblurring parameter changed: {param_name} = {value}")
        
        # Update the status text
        self.deblur_status.configure(text="Parameters changed. Click Preview to update.")
    
    def save_working_image(self):
        """Save the currently displayed image for debugging purposes."""
        try:
            # Determine which image to save
            current_phase = self.current_phase.get()
            
            # By default, use background image
            image_to_save = self.background_image
            
            # Phase-specific image selection
            if current_phase == "Image Creation":
                if self.result_image is not None:  # If blurred result exists
                    image_to_save = self.result_image
                elif self.object_image is not None and self.background_image is not None:
                    # Create a composite for saving
                    image_to_save = image_processing.composite_images(
                        self.background_image.copy(),
                        self.object_image,
                        self.object_position
                    )
            elif current_phase == "Motion Crop":
                # Use background image (already set as default)
                pass
            elif current_phase == "Deblurring":
                # Use deblurred image if available
                if hasattr(self, 'deblurred_image') and self.deblurred_image is not None:
                    image_to_save = self.deblurred_image
            
            # Check if we have an image to save
            if image_to_save is None:
                utils.show_error("Error", "No image currently available to save.")
                return
                
            # Create output directory if needed
            output_dir = self.output_dir.get()
            utils.ensure_dir_exists(output_dir)
            
            # Generate a timestamp for unique filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename with phase and timestamp
            filename = f"debug_{current_phase.replace(' ', '_')}_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save the image
            image_processing.save_image(image_to_save, output_path)
            
            # Show success message
            utils.show_info("Debug Image Saved", f"Current working image saved to:\n{output_path}")
            
        except Exception as e:
            utils.show_error("Error Saving Debug Image", f"Failed to save debug image: {str(e)}")
            print(f"Error saving debug image: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_focus_in(self, event):
        """Handle window focus-in events to maintain deblurred image display"""
        # Only apply to the main window
        if event.widget == self.root:
            print(f"Focus in event received. Phase: {self.current_phase.get()}")
            print(f"  has_been_deblurred: {self.has_been_deblurred}")
            print(f"  hasattr(self, 'deblurred_image'): {hasattr(self, 'deblurred_image')}")
            if hasattr(self, 'deblurred_image'):
                print(f"  deblurred_image is None: {self.deblurred_image is None}")
            
            # If we're in the deblurring phase and have a deblurred image, restore it
            if (self.current_phase.get() == "Deblurring" and 
                self.has_been_deblurred and 
                hasattr(self, 'deblurred_image') and 
                self.deblurred_image is not None):
                # Redisplay the deblurred image
                print("  Restoring deblurred image")
                self.canvas_manager.display_image(self.deblurred_image)
                self.log_message("Restored deblurred image display after focus returned")

    def make_deblurred_current(self):
        """Make the current deblurred image the new working image for further processing.
        This is only called when explicitly requested by saving a deblurred image."""
        if hasattr(self, 'deblurred_image') and self.deblurred_image is not None:
            self.log_message("Making deblurred image the new working image")
            # Replace background image with deblurred image
            self.background_image = self.deblurred_image.copy()
            # Update original image reference to match
            self.original_image_for_reset = self.background_image.copy()
            # Reset deblurring state for new processing
            self.has_been_deblurred = False
            if hasattr(self, 'deblurred_image'):
                delattr(self, 'deblurred_image')
            # Update display
            self.canvas_manager.display_image(self.background_image)
            # Update status
            self.deblur_status.configure(text="Deblurred image set as current working image")
            return True
        return False

def main():
    root = tk.Tk()
    app = CodedExposureApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 