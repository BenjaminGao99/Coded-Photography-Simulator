import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
from typing import Tuple, List, Callable, Optional, Any
import math

class CanvasManager:
    """Manages canvas display and interactions for images."""
    
    def __init__(self, master: ttk.Frame, width: int = 800, height: int = 600):
        """Initialize canvas with scrollbars."""
        self.master = master
        
        # Create canvas with scrollbars
        self.container = ttk.Frame(master)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas with aspect ratio maintenance
        self.canvas = tk.Canvas(self.container, bg="gray", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(master, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = ttk.Scrollbar(self.container, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        # Display variables
        self.current_scale = 1.0
        self.scale_notification = None
        self.image_id = None
        self.photo_image = None  # Store PhotoImage to prevent garbage collection
        
        # Display placeholder message
        self.display_placeholder("Select background and foreground images to preview")
    
    def display_placeholder(self, message: str = ""):
        """Display a placeholder message on the canvas."""
        self.clear_canvas()
        if message:
            self.canvas.create_text(300, 200, text=message, fill="white", font=("TkDefaultFont", 12))
    
    def clear_canvas(self):
        """Clear all items from the canvas."""
        self.canvas.delete("all")
        self.scale_notification = None
        self.image_id = None
        self.photo_image = None  # Release memory
    
    def display_image(self, image_array: np.ndarray, tag: str = "image") -> None:
        """Display an image on the canvas, properly scaled to fit the available space."""
        if image_array is None:
            self.display_placeholder("No image to display")
            return
        
        # Calculate scale factor for display
        self.current_scale = 1.0  # Default to no scaling
        
        # Get available space in the container
        available_width = max(100, self.master.winfo_width() - 30)  # 30 pixels for scrollbar and padding
        available_height = max(100, self.master.winfo_height() - 50)  # 50 pixels for title, scrollbar and padding
        
        # Get image dimensions
        img_h, img_w = image_array.shape[:2]
        
        # Calculate scale factor
        scale_w = available_width / img_w if img_w > available_width else 1.0
        scale_h = available_height / img_h if img_h > available_height else 1.0
        scale = min(scale_w, scale_h)  # Use the smaller scale to ensure entire image fits
        
        # Only scale down, never scale up
        if scale >= 1.0:
            scale = 1.0
        
        self.current_scale = scale
        
        # Clear existing content
        self.clear_canvas()
        
        # If we need to scale, resize the image
        if scale < 1.0:
            # Calculate new dimensions
            new_width = int(img_w * scale)
            new_height = int(img_h * scale)
            
            # Resize the image using PIL
            pil_img = Image.fromarray(image_array)
            resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(resized_img)
            
            # Show scaling notification
            self.scale_notification = self.canvas.create_text(
                new_width - 10, 10, 
                text=f"Image scaled to {int(scale*100)}%", 
                fill="white", font=("TkDefaultFont", 10),
                anchor=tk.NE)
        else:
            # No scaling needed
            self.photo_image = ImageTk.PhotoImage(Image.fromarray(image_array))
        
        # Display the image
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image, tags=tag)
        
        # Set canvas size and scroll region
        canvas_width = int(img_w * scale)
        canvas_height = int(img_h * scale)
        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))
        
        # Center the canvas in the available space
        self.center_in_container(canvas_width, canvas_height, available_width, available_height)
    
    def center_in_container(self, canvas_width: int, canvas_height: int, 
                         available_width: int, available_height: int) -> None:
        """Center the canvas in the available container space."""
        # Calculate padding to center the canvas
        padx = max(0, (available_width - canvas_width) // 2)
        pady = max(0, (available_height - canvas_height) // 2)
        
        # Apply padding to center the canvas
        self.container.pack_configure(padx=padx, pady=pady)
    
    def update_on_resize(self, image_array: np.ndarray = None) -> None:
        """Update canvas after container resize."""
        if image_array is not None and self.photo_image is not None:
            self.display_image(image_array)
    
    def show_resize_placeholder(self):
        """Clear the canvas and set it to gray color for resize operations."""
        self.clear_canvas()
        self.canvas.config(bg="gray")
        # Display a resizing message
        self.canvas.create_text(
            self.canvas.winfo_width() // 2 or 300, 
            self.canvas.winfo_height() // 2 or 200,
            text="Resizing...", 
            fill="white", 
            font=("TkDefaultFont", 14, "bold")
        )


class CropHandler:
    """Handles crop functionality with point selection and manipulation."""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.active = False
        self.points = []
        self.point_tags = []
        self.visual_elements = []
        self.rect = None
        self.motion_line = None
        self.motion_arrow = None
        self.dragging_point_index = None
        self.motion_angle = 0  # Store the motion angle
        
        # Bindings will be set when activating crop mode
    
    def start_crop(self, on_complete_callback: Callable[[List[Tuple[float, float]]], None] = None):
        """Start the cropping process."""
        self.active = True
        self.points = []
        self.point_tags = []
        self.visual_elements = []
        self.rect = None
        self.motion_line = None
        self.motion_arrow = None
        self.motion_angle = 0
        self.on_complete_callback = on_complete_callback
        
        # Check for and print canvas scale factor
        try:
            self.canvas_scale = getattr(self.canvas, 'current_scale', 1.0)
            print(f"CropHandler detected canvas scale: {self.canvas_scale}")
        except AttributeError:
            self.canvas_scale = 1.0
            print("No canvas scale found, defaulting to 1.0")
        
        # Bind canvas click for point selection
        self.canvas.bind("<Button-1>", self.add_point)
    
    def cancel_crop(self):
        """Cancel the cropping process and clean up."""
        self.active = False
        
        # Clear all visual elements
        for element in self.visual_elements:
            self.canvas.delete(element)
        
        # Clear data
        self.points = []
        self.point_tags = []
        self.visual_elements = []
        
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
            
        if self.motion_line:
            self.canvas.delete(self.motion_line)
            self.motion_line = None
            
        if self.motion_arrow:
            self.canvas.delete(self.motion_arrow)
            self.motion_arrow = None
        
        # Unbind events
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
    
    def add_point(self, event):
        """Add a point to the crop selection."""
        if not self.active:
            return
        
        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Add point to list
        self.points.append((x, y))
        
        # Draw point
        point_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red", outline="white", width=2, 
                                        tags=f"crop_point_{len(self.points)}")
        self.visual_elements.append(point_id)
        self.point_tags.append(f"crop_point_{len(self.points)}")
        
        # Add point number label
        label_id = self.canvas.create_text(x, y-15, text=str(len(self.points)), fill="white", 
                                        font=("Arial", 12, "bold"))
        self.visual_elements.append(label_id)
        
        # If we have the first two points, draw the motion direction line
        if len(self.points) == 2:
            self.draw_motion_line()
        
        # If we have at least 3 points, update the rectangle
        if len(self.points) >= 3:
            self.update_crop_rectangle()
        
        # If we have 4 points, enable dragging and call callback if complete
        if len(self.points) == 4:
            # Unbind click event to prevent adding more points
            self.canvas.unbind("<Button-1>")
            
            # Bind events for dragging points
            self.canvas.bind("<ButtonPress-1>", self.start_drag_point)
            self.canvas.bind("<B1-Motion>", self.drag_point)
            self.canvas.bind("<ButtonRelease-1>", self.stop_drag_point)
            
            # If we have a callback for completion, call it
            if self.on_complete_callback:
                self.on_complete_callback(self.points)
    
    def draw_motion_line(self):
        """Draw a line between points 1 and 2 to show motion direction."""
        if len(self.points) < 2:
            return  # Not enough points yet
            
        # Create or update motion line
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        
        if self.motion_line:
            # Update existing line
            self.canvas.coords(self.motion_line, x1, y1, x2, y2)
        else:
            # Create new line
            self.motion_line = self.canvas.create_line(
                x1, y1, x2, y2, 
                fill="yellow", 
                width=2,
                arrow=tk.LAST,
                arrowshape=(16, 20, 6),
                tags="crop"
            )
            self.visual_elements.append(self.motion_line)
        
        # Calculate the angle
        if x2 - x1 != 0:  # Avoid division by zero
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            self.motion_angle = angle_deg  # Store the angle
            
            # Add or update an angle label
            angle_text = f"Motion angle: {angle_deg:.1f}°"
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Offset the text position
            offset = 20
            offset_x = mid_x + offset * math.sin(angle_rad)
            offset_y = mid_y - offset * math.cos(angle_rad)
            
            # Create or update the angle text
            if hasattr(self, 'angle_text') and self.angle_text in self.visual_elements:
                self.canvas.coords(self.angle_text, offset_x, offset_y)
                self.canvas.itemconfig(self.angle_text, text=angle_text)
            else:
                self.angle_text = self.canvas.create_text(
                    offset_x, offset_y,
                    text=angle_text,
                    fill="white",
                    font=("Arial", 10, "bold"),
                    tags="crop"
                )
                self.visual_elements.append(self.angle_text)
    
    def update_crop_rectangle(self):
        """Update the crop rectangle based on current points."""
        if len(self.points) < 3:
            return
        
        # Get the first two points that define motion direction
        p1 = np.array(self.points[0])
        p2 = np.array(self.points[1])
        
        # Calculate motion direction vector
        motion_vector = p2 - p1
        motion_length = np.linalg.norm(motion_vector)
        
        if motion_length == 0:
            return  # Avoid division by zero
        
        # Normalize motion vector
        motion_unit = motion_vector / motion_length
        
        # Calculate perpendicular unit vector (90 degrees counterclockwise)
        perp_unit = np.array([-motion_unit[1], motion_unit[0]])
        
        # Process all available points for width calculation
        if len(self.points) >= 3:
            # Calculate perpendicular projection of point 3
            p3 = np.array(self.points[2])
            p3_proj_dist = np.dot(p3 - p1, perp_unit)
            
            # If we have point 4, use it too
            if len(self.points) >= 4:
                p4 = np.array(self.points[3])
                p4_proj_dist = np.dot(p4 - p1, perp_unit)
                
                # Use min/max to determine the width boundaries
                min_dist = min(p3_proj_dist, p4_proj_dist)
                max_dist = max(p3_proj_dist, p4_proj_dist)
            else:
                # If only point 3 is available, use symmetric boundaries
                min_dist = -abs(p3_proj_dist)
                max_dist = abs(p3_proj_dist)
        
        # Delete existing rectangle if it exists
        if self.rect:
            self.canvas.delete(self.rect)
            if self.rect in self.visual_elements:
                self.visual_elements.remove(self.rect)
        
        # Calculate the four corners of the rectangle
        corner1 = p1 + min_dist * perp_unit
        corner2 = p2 + min_dist * perp_unit
        corner3 = p2 + max_dist * perp_unit
        corner4 = p1 + max_dist * perp_unit
        
        # Convert to integer coordinates
        corners = [tuple(corner1.astype(int)), tuple(corner2.astype(int)), 
                  tuple(corner3.astype(int)), tuple(corner4.astype(int))]
        
        # Draw the rectangle
        self.rect = self.canvas.create_polygon(
            corners[0][0], corners[0][1],
            corners[1][0], corners[1][1], 
            corners[2][0], corners[2][1], 
            corners[3][0], corners[3][1],
            outline="cyan", fill="", width=2, tags="crop_rect")
        self.visual_elements.append(self.rect)
    
    def start_drag_point(self, event):
        """Start dragging a crop point."""
        if not self.active or len(self.points) != 4:
            return
        
        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Check if click is near any crop point
        for i, (px, py) in enumerate(self.points):
            # Check if click is within the point's area
            if abs(x - px) <= 10 and abs(y - py) <= 10:
                self.dragging_point_index = i
                return
        
        self.dragging_point_index = None
    
    def drag_point(self, event):
        """Drag a crop point."""
        if self.dragging_point_index is None:
            return
        
        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Update the point coordinates
        self.points[self.dragging_point_index] = (x, y)
        
        # Update the visual representation of the point
        tag = self.point_tags[self.dragging_point_index]
        point_items = self.canvas.find_withtag(tag)
        
        if point_items:
            # Update the point oval
            self.canvas.coords(point_items[0], x-5, y-5, x+5, y+5)
            
            # Find the text label (should be the point number)
            point_idx = self.dragging_point_index + 1  # Point numbers are 1-based
            for item in self.visual_elements:
                if self.canvas.type(item) == "text" and self.canvas.itemcget(item, "text") == str(point_idx):
                    self.canvas.coords(item, x, y-15)
                    break
        
        # Redraw the motion line if points 1 or 2 were moved
        if self.dragging_point_index < 2:
            self.draw_motion_line()
        
        # Update the crop rectangle
        self.update_crop_rectangle()
    
    def stop_drag_point(self, event):
        """Stop dragging a crop point."""
        self.dragging_point_index = None
        
        # If we have a callback for completion, call it with the updated points
        if self.on_complete_callback and len(self.points) == 4:
            self.on_complete_callback(self.points)
    
    def get_points(self) -> List[Tuple[float, float]]:
        """Get the current crop points."""
        print(f"DEBUG - CropHandler.get_points: returning {self.points}")
        return self.points.copy()
    
    def calculate_crop_dimensions(self) -> Tuple[int, int, np.ndarray]:
        """Calculate dimensions and corner points for a crop based on the current points.
        
        Returns:
            Tuple of (width, height, src_corners) where src_corners is a numpy array
            of the four corner points in the correct order for perspective transform.
        """
        if len(self.points) != 4:
            raise ValueError("Need exactly 4 points to calculate crop dimensions")
        
        # Use the class's canvas_scale property that was set during start_crop
        canvas_scale = getattr(self, 'canvas_scale', 1.0)
        print(f"Using canvas scale: {canvas_scale}")
        
        # Correct points for canvas scaling if needed
        scaled_points = []
        for point in self.points:
            if canvas_scale != 1.0:
                # Convert canvas coordinates back to original image coordinates
                scaled_point = (point[0] / canvas_scale, point[1] / canvas_scale)
                scaled_points.append(scaled_point)
            else:
                scaled_points.append(point)
                
        # Get points
        p1 = np.array(scaled_points[0])
        p2 = np.array(scaled_points[1])
        p3 = np.array(scaled_points[2])
        p4 = np.array(scaled_points[3])
        
        # Calculate motion direction vector
        motion_vector = p2 - p1
        motion_length = np.linalg.norm(motion_vector)
        
        if motion_length == 0:
            raise ValueError("Motion direction points must be different")
        
        # Normalize motion vector
        motion_unit = motion_vector / motion_length
        
        # Calculate perpendicular unit vector
        perp_unit = np.array([-motion_unit[1], motion_unit[0]])
        
        # Calculate perpendicular distances of points 3 and 4
        p3_dist = np.dot(p3 - p1, perp_unit)
        p4_dist = np.dot(p4 - p1, perp_unit)
        
        # Use min/max to determine width boundaries
        min_dist = min(p3_dist, p4_dist)
        max_dist = max(p3_dist, p4_dist)
        width = max_dist - min_dist
        
        # Calculate the four corners of the source rectangle
        corner1 = (p1 + min_dist * perp_unit).astype(int)
        corner2 = (p2 + min_dist * perp_unit).astype(int)
        corner3 = (p2 + max_dist * perp_unit).astype(int)
        corner4 = (p1 + max_dist * perp_unit).astype(int)
        
        # Print for debugging
        print(f"Original Points: {self.points}")
        print(f"Scaled Points: {scaled_points}")
        print(f"Crop corners: {corner1}, {corner2}, {corner3}, {corner4}")
        
        # Create source points array with proper ordering
        # Order MUST be: top-left, top-right, bottom-right, bottom-left
        src_corners = np.array([
            tuple(corner1),  # Top-left
            tuple(corner2),  # Top-right
            tuple(corner3),  # Bottom-right
            tuple(corner4)   # Bottom-left
        ], dtype=np.float32)
        
        # Use actual motion length and width (not length of vector)
        dst_width = int(motion_length)
        dst_height = int(width)
        
        # Return the dimensions and source corner points
        return dst_width, dst_height, src_corners
    
    def get_motion_angle(self) -> float:
        """
        Get the motion angle in degrees.
        
        Returns:
            The motion direction angle in degrees
        """
        # Calculate angle from points 1 and 2
        if len(self.points) >= 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            
            if x2 - x1 != 0:  # Avoid division by zero
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = math.degrees(angle_rad)
                return angle_deg
            
        return 0.0  # Default angle if not calculable


class ObjectDragHandler:
    """Handles dragging objects on a canvas."""
    
    def __init__(self, canvas: tk.Canvas, on_drag_callback: Callable[[Tuple[int, int]], None] = None):
        self.canvas = canvas
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.position = (0, 0)
        self.object_size = (0, 0)
        self.container_size = (0, 0)
        self.on_drag_callback = on_drag_callback
        self.canvas_scale = 1.0
        
        # Bind mouse events
        self.bind_events()
    
    def bind_events(self):
        """Bind mouse events for dragging."""
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
    
    def unbind_events(self):
        """Unbind mouse events."""
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
    
    def update_scale(self):
        """Update the canvas scale from the canvas attribute."""
        self.canvas_scale = getattr(self.canvas, 'current_scale', 1.0)
        return self.canvas_scale
    
    def set_object_info(self, position: Tuple[int, int], object_size: Tuple[int, int], 
                      container_size: Tuple[int, int]):
        """Set the current object information for drag constraints."""
        self.position = position
        self.object_size = object_size
        self.container_size = container_size
        
        # Get current canvas scale from the canvas attribute if available
        self.canvas_scale = getattr(self.canvas, 'current_scale', 1.0)
        print(f"Object drag handler using scale: {self.canvas_scale}")
    
    def on_mouse_down(self, event):
        """Handle mouse button press."""
        # Update scale before processing mouse event
        self.update_scale()
        
        # Get canvas coordinates
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Check if click is on object
        obj_x, obj_y = self.position
        obj_h, obj_w = self.object_size
        
        # Scale object position and size by canvas scale
        scaled_obj_x = int(obj_x * self.canvas_scale)
        scaled_obj_y = int(obj_y * self.canvas_scale)
        scaled_obj_w = int(obj_w * self.canvas_scale)
        scaled_obj_h = int(obj_h * self.canvas_scale)
        
        if (scaled_obj_x <= x <= scaled_obj_x + scaled_obj_w and 
            scaled_obj_y <= y <= scaled_obj_y + scaled_obj_h):
            self.dragging = True
            self.drag_start_x = x - scaled_obj_x
            self.drag_start_y = y - scaled_obj_y
    
    def on_mouse_drag(self, event):
        """Handle mouse dragging."""
        if not self.dragging:
            return
        
        # Update scale before processing mouse event
        self.update_scale()
        
        # Get canvas coordinates
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Calculate new object position, accounting for scaling
        new_x = max(0, (x - self.drag_start_x) / self.canvas_scale)
        new_y = max(0, (y - self.drag_start_y) / self.canvas_scale)
        
        # Limit to container boundaries
        container_h, container_w = self.container_size
        obj_h, obj_w = self.object_size
        
        new_x = min(new_x, container_w - obj_w)
        new_y = min(new_y, container_h - obj_h)
        
        # Calculate the change in position
        old_x, old_y = self.position
        position_changed = (int(new_x) != old_x or int(new_y) != old_y)
        
        if position_changed:
            # Update position
            self.position = (int(new_x), int(new_y))
            
            # Call the callback if provided
            if self.on_drag_callback:
                self.on_drag_callback(self.position)
    
    def on_mouse_up(self, event):
        """Handle mouse button release."""
        self.dragging = False
    
    def get_position(self) -> Tuple[int, int]:
        """Get the current object position."""
        return self.position 