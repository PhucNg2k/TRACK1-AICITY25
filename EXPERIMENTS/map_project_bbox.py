import json
import os
import cv2
import random
import numpy as np
import math

WAREHOUSE_ID = "Warehouse_000"
BASE_URL = f"..\DATASET\MTMC_Tracking_2025\\train\{WAREHOUSE_ID}"
bbox_path = f"{BASE_URL}\BBoxFolderSorted\Warehouse_000_Camera_bbox_frame_0_sorted.json"
cali_path = f"{BASE_URL}\calibration.json"
map_path = f"{BASE_URL}\map.png"

def get_color():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

def get_camera_calibration(camera_id):
    camera_num = camera_id.split('_')
    if len(camera_num) > 1:
        num = camera_num[1]
        num = int(num)
        if num == 0:
            camera_id = "Camera"
        else:
            camera_id = f"Camera_{num:02d}"

    print(camera_id)
    # Load JSON while preserving key order
    with open(cali_path, 'r') as f:
        cali_data = json.load(f)

    sensors = cali_data["sensors"]
    print("Number of cameras:", len(sensors))
    
    for sensor in sensors:
        if sensor.get("type") == "camera" and sensor.get("id") == camera_id:
            return sensor

    print(f"Camera with ID {camera_id} not found.")
    return None


def main():
    camera_id = "Camera_0000"
    video_file = f"{BASE_URL}\\videos\{camera_id}.mp4"
    bbox_path = f"{BASE_URL}\BBoxFolderSorted\Warehouse_000_Camera_bbox_frame_0_sorted.json"
    
    bbox_data = []
    with open(bbox_path,'r') as f:
        data = json.load(f)
        bbox_data = data[camera_id]
    


    camera_data = get_camera_calibration(camera_id)
    if not camera_data:
        print(f"[ERROR] Camera calibration not found for ID: {camera_id}")
        return

    homographyMat = np.array(camera_data['homography']) 
    inv_homoMat = np.linalg.inv(homographyMat)

    cam_id = camera_data['id']
    cam_local = camera_data['coordinates']
    cam_scaleFactor = camera_data['scaleFactor']
    cam_translateGlobal  = camera_data['translationToGlobalCoordinates']

    points_ls = []
    for bbox in bbox_data:
        centerX = (bbox[0] + bbox[2])/2
        centerY = (bbox[1] + bbox[3])/2
        print("Center: ", (centerX,centerY))

        pixel_point = np.array([centerX, centerY, 1])
        reprojected_pixel = inv_homoMat @ pixel_point
        reprojected_pixel /= reprojected_pixel[2]

        reprojected_pixel[0] = (reprojected_pixel[0] + cam_translateGlobal['x'])*cam_scaleFactor
        reprojected_pixel[1] = (reprojected_pixel[1] + cam_translateGlobal['y'])*cam_scaleFactor

        points_ls.append(tuple(reprojected_pixel[:2]))

    map_image = cv2.imread(map_path, cv2.IMREAD_COLOR)
    map_height, map_width, map_channels = map_image.shape
    print(f"Map:\n Width {map_height}, Height {map_width}, Channels {map_channels}\n")



    # draw camera
    cam_direction_deg = float( camera_data["attributes"][1]["value"])
    adjusted_deg = (90 - cam_direction_deg) % 360
    cam_direction_rad = math.radians(adjusted_deg)
    
    cam_x = (cam_local['x'] + cam_translateGlobal['x']) * cam_scaleFactor
    cam_y = (cam_local['y'] + cam_translateGlobal['y']) * cam_scaleFactor
    point = (int(cam_x), map_height - int(cam_y))

    arrow_length = 20
    dx = int(arrow_length * math.cos(cam_direction_rad))
    dy = int(-arrow_length * math.sin(cam_direction_rad))  # minus because image Y axis is downward
    # Compute end point of the arrow
    arrow_end = (point[0] + dx, point[1] + dy)

    f_x = float(camera_data["intrinsicMatrix"][0][0])
    fov_rad = 2 * math.atan(map_width / (2 * f_x))
    fov_deg = math.degrees(fov_rad)

    arrow_length = 800  # How far the FOV lines extend (tweak as needed)
    left_angle = cam_direction_rad - fov_rad / 2
    right_angle = cam_direction_rad + fov_rad / 2

    left_point = (
        int(point[0] + arrow_length * math.cos(left_angle)),
        int(point[1] - arrow_length * math.sin(left_angle))
    )
    right_point = (
        int(point[0] + arrow_length * math.cos(right_angle)),
        int(point[1] - arrow_length * math.sin(right_angle))
    )

    cam_id = cam_id.split("_")
    n_id = 0
    if len(cam_id)>1:
        n_id = int(cam_id[1])

    cv2.circle(map_image, point, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.arrowedLine(map_image, point, arrow_end, color=(0, 255, 0), thickness=2, tipLength=0.3)
    cv2.putText(map_image, str(n_id), (point[0] + 10, point[1] - 15), # top left
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    f_str = f"({cam_local['x']:.2f},{cam_local['y']:.2f})"
    f_str = f"({point[0]:.2f},{point[1]:.2f})"
    cv2.putText(map_image, f_str, (point[0], point[1] + 15), # top left
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    
    cv2.line(map_image, (point[0], point[1]), left_point, (255, 0, 0), 2)
    cv2.line(map_image, (point[0], point[1]), right_point, (255, 0, 0), 2)
    #cv2.line(map_image, left_point, right_point, (100, 100, 255), 1, cv2.LINE_AA)


    # draw bbox mapping
    for px, py in points_ls:
        print((px,py))
        point = (int(px), map_height - int(py))
        cv2.circle(map_image, point, radius=5, color=get_color(), thickness=-1)
    

    cv2.imshow("Plot map 2d", map_image)
    cv2.imwrite(f"Map_{camera_id}_view.jpg", map_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()