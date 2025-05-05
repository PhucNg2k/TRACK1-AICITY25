import cv2
import numpy as np

# Load the image
map_path = r"D:\Contest\AICITY\DATASET\MTMC_Tracking_2025\train\Warehouse_000\map.png"
image = cv2.imread(map_path, cv2.IMREAD_COLOR)  # Load in color mode

# Create a mask where black pixels (0,0,0 in BGR) are detected
mask = (image == [0, 0, 0]).all(axis=2)

# Create a copy of the original image and modify only black pixels
image_transformed = image.copy()
image_transformed[mask] = [255, 255, 255]  # Change black pixels to white

# Show the images
cv2.imshow("Original Map", image)
cv2.imshow("Transformed Map", image_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()