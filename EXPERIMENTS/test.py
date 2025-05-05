import json
import pprint
import random
from collections import OrderedDict
import cv2
import os

WAREHOUSE_ID = "Warehouse_000"
BASE_URL = f"..\DATASET\MTMC_Tracking_2025\\train\{WAREHOUSE_ID}"
gt_path = f"{BASE_URL}\ground_truth.json"
map_path = f"{BASE_URL}\map.png"


def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def convertBBoxCoord(xmin, ymin, xmax, ymax):
    x = xmin
    y = ymin
    width = xmax - xmin
    height = ymax - ymin
    return [x, y, width, height]


def apply_bbox_map(map_path, bbox_path):
    bbox_data = []
    with open(bbox_path, 'r') as f:
        data = json.load(f)

    for bbox in data:
        sample = bbox['bbox_value']
        bbox_data.append(sample)
    
    map_image = cv2.imread(map_path, cv2.IMREAD_COLOR)
    map_height, map_width, map_channels = map_image.shape

    for bbox_value in bbox_data:
        # draw bbox onto map_image
        xmin,ymin,xmax,ymax = bbox_value
        ymin = map_height - ymin
        ymax = map_height - ymax
        color = get_random_color()
        cv2.rectangle(map_image, (xmin,ymin), (xmax, ymax), color, 1)  # Draw green bbox
    
    return map_image


def collect_2d_bbox(frame_data, save_file):
    # Reset save_file (overwrite if exists)
    with open(save_file, 'w') as f:
        json.dump([], f) 

    data_list = []  
    
    for obj in frame_data:

        object_type = obj["object type"]
        object_id  = obj["object id"]
        camera_ls = obj["2d bounding box visible"]

        for cam_id, bbox_value in camera_ls.items():
            rect_value = convertBBoxCoord(*bbox_value)  # Unpack bbox correctly

            sample = {
                "object_type": object_type,
                "object_id": object_id,
                "cam_id": cam_id,
                "bbox_value": bbox_value,  # xmin, ymin, xmax, ymax
                "rect_value": rect_value  # x, y, width, height
            }

            data_list.append(sample) 

    # Save collected data to JSON file
    with open(save_file, 'w') as f:
        json.dump(data_list, f, indent=4)

    print(f"Saved {len(data_list)} bounding boxes to {save_file}")


def main():
    # Load JSON while preserving key order
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    gt_keys = list(gt_data.keys())  # Keep key order
    num_frames = len(gt_keys)

    print("Numbers of frames: ", num_frames)
    
    #camera_id = "Camera_0000"
    bbox_folder = f"{BASE_URL}\BBoxFolder"
    os.makedirs(bbox_folder, exist_ok=True)

    for idx in range(num_frames):
        frame_data = gt_data[str(idx)]

        print("Number of detected objects: ", len(frame_data))

        #save_frame_file = f"test_frame_{idx}.json"
        #with open(save_frame_file,'w') as f:
         #   json.dump(frame_data, f, indent=2)


        file_name = f"{WAREHOUSE_ID}_BBox_frame_{idx}.json"
        save_bbox_file = os.path.join(bbox_folder, file_name)
        collect_2d_bbox(frame_data, save_bbox_file)
        
    # Pretty-print while maintaining key order
    #pprint.pprint(frame_data, sort_dicts=False)


def show_image_bbox(map_path, save_bbox_file):
    applied_map = apply_bbox_map(map_path, save_bbox_file)
    cv2.imshow("Bbox on map: ", applied_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def sort_bbox_camera(bbox_file, idx):
    target_bbox_folder =  f"{BASE_URL}\BBoxFolderSorted"
    file_name = f"{WAREHOUSE_ID}_Camera_bbox_frame_{idx}_sorted.json"
    
    save_file = os.path.join(target_bbox_folder, file_name)

    with open(save_file, 'w') as f:
        json.dump([], f) 
    
    with open(bbox_file, 'r') as f:
        data = json.load(f)
    
    data_dict = {}

    for obj in data:
        cam_id = obj['cam_id']
        data_dict[cam_id] = []

    for obj in data:
        cam_id = obj['cam_id']
        
        bbox_sample = obj['bbox_value']
        
        data_dict[cam_id].append(bbox_sample)

    with open(save_file, 'w') as f:
        json.dump(data_dict, f, indent=2)

    print("Saved: ", save_file)
    return

    '''
     {
        "object_type": "Forklift",
        "object_id": 38,
        "cam_id": "Camera_0050",
        "bbox_value": [
            1143,
            568,
            1449,
            866
        ],
        "rect_value": [
            1143,
            568,
            306,
            298
        ]
    }
    '''

if __name__ == "__main__":
    #main()
    
    src_bbox_folder = f"{BASE_URL}\BBoxFolder"
    os.makedirs(src_bbox_folder, exist_ok=True)
    target_bbox_folder =  f"{BASE_URL}\BBoxFolderSorted"
    os.makedirs(target_bbox_folder, exist_ok=True)

    num_frames = 9000
    for idx in range(num_frames):
        save_bbox_file = f"{WAREHOUSE_ID}_BBox_frame_{idx}.json"
        src_file = os.path.join(src_bbox_folder, save_bbox_file)

        sort_bbox_camera(src_file, idx)





    






