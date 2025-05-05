

import numpy as np
import cv2
import json
import os
import random
WAREHOUSE_ID = "Warehouse_000"
BASE_URL = f"..\DATASET\MTMC_Tracking_2025\\train\{WAREHOUSE_ID}"
gt_path = f"{BASE_URL}\ground_truth.json"
cali_path = f"{BASE_URL}\calibration.json"
map_path = f"{BASE_URL}\map.png"

def get_color():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

def get_camera_calibration(camera_id):
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


def get_gt():
    # Load JSON while preserving key order
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    gt_keys = list(gt_data.keys())  # Keep key order
    num_frames = len(gt_keys)
    
    print("Numbers of frames: ", num_frames)

    bbox_folder = f"{BASE_URL}\BBox3DFolder"
    os.makedirs(bbox_folder, exist_ok=True)

    bbox_data = []

    for idx in range(num_frames):
        frame_data = gt_data[str(idx)]
        print("Number of detected objects: ", len(frame_data))
        for objdata in frame_data:
            object_id = objdata['object id']
            wx, wy, wz = objdata['3d location']
            w, h, d = objdata['3d bounding box scale'] 
            pitch, roll, yaw = objdata['3d bounding box rotation']
            sample = {
                "obj_id": object_id,
                "obj_pos": [wx,wy,wz],
                "obj_scale": [w,h,d],
                "yaw": yaw
            }
            bbox_data.append(sample)
        break
    
    return bbox_data




def get_yaw_rotation_matrix(yaw):
    return np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

def process(bbox_data, camera_data, image, connected):
    intrinsicMat = np.array(camera_data['intrinsicMatrix'])
    extrinsicMat = np.array(camera_data['extrinsicMatrix'])
    cameraMat = np.array(camera_data['cameraMatrix'])
    homographyMat = np.array(camera_data['homography']) 

    inv_homoMat = np.linalg.inv(homographyMat)

    cam_scaleFactor = camera_data['scaleFactor']
    cam_translateGlobal  = camera_data['translationToGlobalCoordinates']

    map_data = []

    for bbox in bbox_data:
        bbox_wpos = bbox['obj_pos']
        bbox_wscale = bbox['obj_scale']
        bbox_yaw = bbox['yaw']

        '''
        # === Project center point (bbox_wpos) ===
        center_world = np.array(bbox_wpos + [1])  # make it homogeneous
        center_camera = np.dot(extrinsicMat, center_world)
        center_2d = np.dot(intrinsicMat, center_camera)
        center_2d /= center_2d[2]  # normalize        

        world_coords = process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw)
        camera_coords = convertWorldToCamera(world_coords, extrinsicMat)
        projected_coords = projectCamera(camera_coords, intrinsicMat)

        
        image = draw_3d_bbox(image, projected_coords, connected)
         # === Draw label at center ===
        text = f"({bbox_wpos[0]:.2f}, {bbox_wpos[1]:.2f}, {bbox_wpos[2]:.2f})"
        text_position = tuple(center_2d[:2].astype(int))
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        '''
        
        # === Project world coord to 2D map using homography ===
        # Convert [x, y] world pos → image 2D → map 2D
        
        map_data.append()
    print("Number of bbox: ", len(map_data))        
    return image, map_data


def convert_map_homography(cord2D, homographyMat):
    # cord2D: shape (N, 2)
    cord2D = np.array(cord2D, dtype=np.float32)
    if cord2D.ndim == 1:
        cord2D = cord2D.reshape(1, 2)
    num_pts = cord2D.shape[0]
    cord2D_homog = np.hstack([cord2D, np.ones((num_pts, 1))])  # (N, 3)
    transformed = (homographyMat @ cord2D_homog.T).T  # (N, 3)
    transformed /= transformed[:, 2:3]  # normalize
    return transformed[:, :2]  # return (x, y)


def process_world_coordinates(bbox_wpos, bbox_wscale, bbox_yaw):
    # CACULATE BBOX 8 corners
    x, y, z = bbox_wpos
    w, h, d = bbox_wscale
    yaw = bbox_yaw # rotation around vertical axis

    # 8 corners before rotation (in local object frame)
    corners = np.array([
        [ w/2,  h/2,  d/2],
        [-w/2,  h/2,  d/2],
        [-w/2, -h/2,  d/2],
        [ w/2, -h/2,  d/2],
        [ w/2,  h/2, -d/2],
        [-w/2,  h/2, -d/2],
        [-w/2, -h/2, -d/2],
        [ w/2, -h/2, -d/2],
    ])

    # apply rotation
    rot_mat = get_yaw_rotation_matrix(yaw)
    rotated_corners = (rot_mat @ corners.T).T + np.array([x, y, z])
    return rotated_corners

def convertWorldToCamera(points, extrinsic_mat):
    """
    Convert world coordinates to camera coordinates using an extrinsic matrix.
    Supports both single 3D point (shape: (3,)) and multiple points (shape: (N, 3)).
    """
    points = np.asarray(points)

    # Ensure shape is (N, 3)
    if points.ndim == 1 and points.shape[0] == 3:
        points = points.reshape(1, 3)
    elif points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be shape (3,) or (N, 3)")

    # Convert to homogeneous coordinates
    homogeneous_corners = np.hstack([points, np.ones((points.shape[0], 1))])
    camera_coords = (np.dot(extrinsic_mat, homogeneous_corners.T)).T
    return camera_coords

def projectCamera(camera_coords, intrinsic_mat): # using Intrinsic matrix
    # Normalize and project
    projected = (intrinsic_mat @ camera_coords.T).T
    projected_2d = projected[:, :2] / projected[:, 2:]
    return projected_2d


def draw_3d_bbox(image, projected_2d, connected):
    

    for i, pt in enumerate(projected_2d.astype(int)):
        cv2.circle(image, tuple(pt), 3, (0, 255, 0), -1)
    # Optionally connect corners with lines to form cube 

    print("2D:\n", projected_2d)
    print("2D shape:\n", projected_2d.shape)

    if connected:
        color = get_color()
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom square
            (4, 5), (5, 6), (6, 7), (7, 4),  # top square
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical lines
        ]
        
        for start, end in edges:
            pt1 = tuple(map(int, projected_2d[start]))
            pt2 = tuple(map(int, projected_2d[end]))

            cv2.line(image, pt1, pt2, color, 1)
    
    return image

def get_camera_id(number):
    num_str = str(int(number)) if number != "0000" else ""
    return num_str

def scale_map_coords_to_image(map_coords, image):
    img_h, img_w = image.shape[:2]

    coords = np.array(map_coords)
    x_vals, y_vals = coords[:, 0], coords[:, 1]

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    coord_w = x_max - x_min
    coord_h = y_max - y_min

    # Uniform scale to fit image while preserving aspect ratio
    scale = min(img_w / coord_w, img_h / coord_h)

    # Center in image
    x_scaled = (x_vals - x_min) * scale + (img_w - coord_w * scale) / 2
    y_scaled = (y_vals - y_min) * scale + (img_h - coord_h * scale) / 2

    return list(zip(x_scaled.astype(int), y_scaled.astype(int)))

def get_map_bounds(image):
    # Get mask of non-white pixels (i.e., where map exists)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray < 250  # nearly white areas ignored

    coords = np.column_stack(np.where(mask))
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    return xmin, xmax, ymin, ymax

def crop_and_save_map(image, bounds, save_path):
    xmin, xmax, ymin, ymax = bounds
    cropped = image[ymin:ymax, xmin:xmax]
    cv2.imwrite(save_path, cropped)
    return cropped

def draw_map_distribution(map_data, padding=5, point_radius=3):
    coords = np.array(map_data)
    x_vals, y_vals = coords[:, 0], coords[:, 1]

    # Get min/max
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # Compute width and height for the image
    width = int(np.ceil(x_max - x_min)) + 2 * padding
    height = int(np.ceil(y_max - y_min)) + 2 * padding

    # Create black background
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for coord in coords:
        x, y = coord
        # Translate and flip y (because image origin is top-left)
        x_img = int((x - x_min) + padding)
        y_img = int((y_max - y) + padding)  # invert Y-axis for display
        color = tuple(random.randint(0, 255) for _ in range(3))
        cv2.circle(image, (x_img, y_img), radius=point_radius, color=color, thickness=-1)

    return image


def draw_map(map_data, map_path):
    '''
    print("MAP DATA:\n", map_data)

    map_distri = draw_map_distribution(map_data)
    cv2.imwrite("map_distribution.jpg", map_distri)    

    image = cv2.imread(map_path, cv2.IMREAD_COLOR)
    image = flip_image_black(image)

    bounds = get_map_bounds(image)
    cropped_map = crop_and_save_map(image, bounds, "cropped_map.jpg")       
    # save the image inside bound (cut)

    scaled = scale_map_coords_to_image(map_data, cropped_map)
    '''
    image = cv2.imread(map_path, cv2.IMREAD_COLOR)
    for map_coord in map_data: #  map_coord is [X,Y]
        map_coord_int = tuple(int(x) for x in map_coord)
        print("Map coord:", map_coord_int)
        cv2.circle(image, map_coord_int, radius=3, color=get_color(), thickness=-1)
    
    return image


def flip_image_black(image):
    # Create a mask where black pixels (0,0,0 in BGR) are detected
    mask = (image == [0, 0, 0]).all(axis=2)
    # modify only black pixels
    image[mask] = [255, 255, 255]
    return image


def main():
    bbox_data = get_gt()

    camera_id = "Camera_0000"

    video_file = f"{BASE_URL}\\videos\{camera_id}.mp4"

    # Open video and extract frames
    vidcap = cv2.VideoCapture(video_file)
    success, frame = vidcap.read()
    
    if not success:
        print(f"[ERROR] Failed to read from video: {video_file}")
        return

    camera_num = camera_id.split('_')
    if len(camera_num) > 1:
        num = camera_num[1]
        num = int(num)
        if num == 0:
            camera_id = "Camera"
        else:
            camera_id = f"Camera_{str(num)}"
    
    print(camera_id)
    camera_data = get_camera_calibration(camera_id)
    if not camera_data:
        print(f"[ERROR] Camera calibration not found for ID: {camera_id}")
        return

    frame_copy = frame.copy()
    connected = True
    img, map_data = process(bbox_data, camera_data, frame_copy, connected)

    if map_data is not None:
        map_image = draw_map(map_data, map_path)
        cv2.imshow("Project map: ", map_image)
        file_name = "Camera_0_homo.jpg"
        cv2.imwrite( file_name, map_image)
        

    if img is not None:
        #cv2.imshow("3D BBOX", img)
        pass

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

