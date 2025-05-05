import cv2
import json
import os
import random
import re


WAREHOUSE_ID = "Warehouse_000"
BASE_URL = f"..\DATASET\MTMC_Tracking_2025\\train\{WAREHOUSE_ID}"
camera_calibrate_path = f"{BASE_URL}\calibration.json"
map_path = f"{BASE_URL}\map.png"


def get_color(idx):
    random.seed(idx)
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

camera_id = "Camera_0000"


video_file = f"{BASE_URL}\\videos\{camera_id}.mp4"
frames_folder = f"{BASE_URL}\\FrameFolder\{camera_id}_frames"

os.makedirs(frames_folder, exist_ok=True)

def process_frame(frame, bboxes):
    for i, bbox_value in enumerate(bboxes):
        xmin, ymin, xmax, ymax = map(int, bbox_value)
        color = get_color(i)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    return frame


def extract_bbox_info(frame_id, camera_id):
    target_bbox_folder =  f"{BASE_URL}\BBoxFolderSorted"
    bbox_file = f"{WAREHOUSE_ID}_Camera_bbox_frame_{frame_id}_sorted.json"
    file_name = os.path.join(target_bbox_folder, bbox_file)
    with open(file_name, 'r') as f:
        bbox_data = json.load(f)

    return bbox_data[camera_id]

def foo():

    # Open video and extract frames
    vidcap = cv2.VideoCapture(video_file)
    success, frame = vidcap.read()

    frame_id = 0

    #frame_count = 1
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total number of frames:", frame_count)

    while success and frame_id < frame_count:
        
        bboxes = extract_bbox_info(frame_id, camera_id)
        frame = process_frame(frame, bboxes)

        file_name = f"{WAREHOUSE_ID}_{camera_id}_frame_{frame_id}.jpg"
        save_path = os.path.join(frames_folder, file_name)
        cv2.imwrite(save_path, frame)
        print("Frame saved to:", save_path)

        success, frame = vidcap.read()
        frame_id += 1


def get_frame_index(filename):
    # Extract the number after '_frame_' using regex
    match = re.search(r'_frame_(\d+)\.jpg', filename)
    return int(match.group(1)) if match else -1

def main():
    # open frame_folder (contains images), make mp4 video from those images
    output_video = "output_video.mp4"
    fps = 30


 # List and numerically sort the images
    images = sorted([
        img for img in os.listdir(frames_folder)
        if img.endswith(".jpg") or img.endswith(".png")
    ], key=get_frame_index)

    if not images:
        print("No images found in folder.")
        return

    # Read first image to get dimensions
    first_image_path = os.path.join(frames_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print("Failed to read the first image.")
        return

    height, width, layers = frame.shape

     # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for idx, image_name in enumerate(images):
        img_path = os.path.join(frames_folder, image_name)
        frame = cv2.imread(img_path)

        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Failed to read {img_path}")
        
        print(f"Write frame {idx} done")

    video.release()
    print(f"Video saved to: {output_video}")


if __name__ == "__main__":
    foo()
    main()