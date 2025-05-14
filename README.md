Coded Exposure Photography Tool

![alt text](https://github.com/BenjaminGao99/Coded-Photography-Simulator/blob/main/output/aligned_reference_angle0.0.png?raw=true)
![alt text](https://github.com/BenjaminGao99/Coded-Photography-Simulator/blob/main/output/deblurred_angle0.0.png?raw=true)

This app simulates coded exposure blur as described in the paper "Coded Exposure Photography: Motion Deblurring using Fluttered Shutter" by Ramesh Raskar et al. The tool allows users to apply motion blur with various exposure codes to foreground objects and composite them onto background images.

Files:
- blur_core.py - core algorithms for blur
- image_processing.py - image manipulation functions 
- ui_components.py - ui components for canvas and crop
- utils.py - helper funcs
- coded_exposure_app.py - main ui app
- run_app.py - launcher script

Features
- load background and foreground imgs
- apply different types of blur (optimal, box, random, MURA)
- customize blur length and angle and code
- preview results
- interactive motion crop tool
- perspective correction
- deblurring
- save results to local file

Usage

To start the ui app:
```
python run_app.py
```

Workflow
Phase 1: Image Creation
- select background and foreground images
- adjust blur length, code type, angle
- position the object
- toggle live preview

Phase 2: Motion Crop
- use crop tool to define motion region
- automatically transform the cropped region

Phase 3: Deblurring
- set deblurring parameters
- preview result
- adjust regularization factor and deblur as many times as needed
- save the deblurred image

Requirements
- Python 3.6+
- NumPy
- OpenCV (cv2)
- PIL/Pillow
- tkinter 
