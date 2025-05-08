import json
import pprint
import pandas as pd
import numpy as np


WAREHOUSE_ID = "Warehouse_000"
BASE_URL = f"../DATASET/MTMC_Tracking_2025/train/{WAREHOUSE_ID}"
gt_path = f"{BASE_URL}/ground_truth.json"
cali_path = f"{BASE_URL}/calibration.json"

with open(gt_path, 'r') as f:
    gt_data = json.load(f)
    
with open(cali_path, 'r') as f:
    cali = json.load(f)

def get_camera_calibration(camera_id, sensors):
    
    cam_num = int(camera_id.split('_')[1])
    if cam_num == 0:
        camera_id = "Camera"
    else:
        camera_id = f"Camera_{cam_num:02d}"
    
    for sensor in sensors:
        if sensor.get("type") == "camera" and sensor.get("id") == camera_id:
            #print(f"Found {camera_id}")
            return sensor

    print(f"Camera with ID {camera_id} not found.")
    return None

def extract_camera_cali(camera_info):
    camera_id = camera_info['id']
    coord_3d = camera_info['coordinates']
    
    camMat = camera_info['cameraMatrix']
    homoMat = camera_info['homography']
    
    attrib = camera_info['attributes']

    for attr in attrib:
        if attr['name'] == 'direction':
            cam_2d_direct = attr['value']
        if attr['name'] == 'direction3d':
            cam_3d_direct = attr['value']
        if attr['name'] == 'frameWidth':
            frameWidth = attr['value']
        if attr['name'] == 'frameHeight':
            frameHeight = attr['value']
    
    sample = {
        "cam_id": camera_id,
        "cam_pos_3d":  [coord_3d['x'], coord_3d['y']],
        "cam_2d_direct": float(cam_2d_direct),
        "cam_3d_direct": [float(x) for x in cam_3d_direct.split(',')],
        "cam_frameWidth": int(frameWidth),
        "cam_frameHeight": int(frameHeight)
    }
    
    camera_detail = {
        "intrinsic": camera_info['intrinsicMatrix'],
        "extrinsic": camera_info['extrinsicMatrix'],
        "homography": camera_info['homography'],
        "camMat": camera_info['cameraMatrix'],
        "cam_pos_3d":  [coord_3d['x'], coord_3d['y']]
    }

    return sample, camera_detail
            

def get_data(frame_idx, camera_id, camera_detail):
    
    cam_num = int(camera_id.split('_')[1])
    camera_id = f"Camera_{cam_num:04d}"
    
    intrinsicMat = camera_detail['intrinsic']
    extrinsicMat = camera_detail['extrinsic']
    camMat = camera_detail['camMat']
    homoMat = camera_detail['homography']
    camPos3d = camera_detail['cam_pos_3d']
    inv_homoMat = np.linalg.inv(homoMat)
    
    
    frame_data = gt_data[str(frame_idx)]
    #print("Number of detected objects: ", len(frame_data))
    
    
    bbox_data = []
    
    for objdata in frame_data:
        object_id = objdata['object id']
        loc3d = objdata['3d location']
        bbox_visible = objdata['2d bounding box visible']
        
        if camera_id not in bbox_visible:
            continue
        
        coord = bbox_visible[camera_id]
        
        centerX = (coord[0] + coord[2])/2
        centerY = (coord[1] + coord[3])/2
        bbox_on_cam = [int(centerX), int(centerY)]
        
        # capture homography
        pixel_point = np.array([centerX, centerY, 1])
        reprojected_pixel = inv_homoMat @ pixel_point
        reprojected_pixel /= reprojected_pixel[2] 
        reprojected_pixel = reprojected_pixel[:2]
        
        # capture camera matrix
        cam_2d_pos = camMat @ np.array(camPos3d + [1,1])
        cam_2d_pos /= cam_2d_pos[2]
        cam_2d_pos = cam_2d_pos[:2]
        
        sample = {
            'obj_id': object_id,
            '2d_visible': bbox_on_cam,
            '3d_loc': loc3d,
            'map_2d': list(map(int, cam_2d_pos)),
            'reproject_pix': list(map(int, reprojected_pixel)),
            'cam_2d_pos': list(map(int, cam_2d_pos))
        }
        bbox_data.append(sample)
        
    return bbox_data


sensors = cali['sensors']
    
target_frame = 0
df = []
for target_frame in range(100):
    for c_num in range(51):
        target_camera = f"Camera_{c_num}"
        camera_config = get_camera_calibration(target_camera, sensors)
        camera_info, camera_detail = extract_camera_cali(camera_config)
        camera_bbox = get_data(frame_idx=target_frame, camera_id=target_camera, camera_detail=camera_detail)

        for sample in camera_bbox:
            row = {**camera_info, **sample}
            df.append(row)
        
        #print(f"Bbox count for {target_camera}: {len(camera_bbox)}")
    
    print(f"Processed frame {target_frame}.")
    
df = pd.DataFrame(df)
print("\nDF heads:\n", df.head())

print('\nDF rows: ', len(df))

#df.to_json('out.json')
df.to_csv('out.csv')

print("Saving done!")
# cam_id |  cam_pos_3d | cam_3d_direct | cam_2d_direct | cam_frameWidth | cam_frameHeight | object id | 2d_bbox_visible -> 3d_locc
# map_4_corners