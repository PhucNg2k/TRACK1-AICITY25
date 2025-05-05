import h5py
import cv2 
import numpy as np
file_path = r'..\DEPTHMAP\\MTMC_Tracking_2025\\train\Warehouse_000\depth_maps\\Camera_0000.h5'

with h5py.File(file_path, 'r') as f:
    print(f.keys())  # See top-level datasets/groups

    # Example: read a depth map
    depth_img = f['distance_to_image_plane_00000.png'][()] # [()] is HDF5-specific syntax: Read the entire dataset as a NumPy array
    
    cv2.imshow("Depth Map", depth_img)
    cv2.waitKey(0)
