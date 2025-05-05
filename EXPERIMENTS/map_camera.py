import json
import pprint
import cv2
import math

WAREHOUSE_ID = "Warehouse_000"
BASE_URL = f"..\DATASET\MTMC_Tracking_2025\\train\{WAREHOUSE_ID}"
camera_calibrate_path = f"{BASE_URL}\calibration.json"
map_path = f"{BASE_URL}\map.png"


def convertLocalToGlobal(local_coord, translate_global, scaleFactor):
    return (local_coord + translate_global) * scaleFactor

def scaleImage(image):
    # scale image to 1920x1080
    target_width = 1920
    target_height = 1080
    scaledImage = cv2.resize(image, (target_width,target_height), interpolation=cv2.INTER_LINEAR)
    return scaledImage
    
def flip_image_black(image):
    # Create a mask where black pixels (0,0,0 in BGR) are detected
    mask = (image == [0, 0, 0]).all(axis=2)
    # modify only black pixels
    image[mask] = [255, 255, 255]
    return image

def main():
    

    map_image = cv2.imread(map_path, cv2.IMREAD_COLOR)
    map_height, map_width, map_channels = map_image.shape
    print(f"Map:\n Width {map_height}, Height {map_width}, Channels {map_channels}\n")

    with open(camera_calibrate_path, 'r') as f:
        camera_calibration = json.load(f)

    # Extract sensors data
    sensors_data = camera_calibration.get('sensors', [])
    num_sensors = len(sensors_data)
    print("Total numbers of cameras: ", num_sensors)

    for idx, target_cam in enumerate(sensors_data):
        if idx < num_sensors:
            cam_id = target_cam['id']
            cam_local = target_cam['coordinates']
            cam_scaleFactor = target_cam['scaleFactor']
            cam_translateGlobal  = target_cam['translationToGlobalCoordinates']

            global_x = convertLocalToGlobal(cam_local['x'], cam_translateGlobal['x'], cam_scaleFactor)
            global_y = convertLocalToGlobal(cam_local['y'], cam_translateGlobal['y'], cam_scaleFactor)

            cam_global = {'x': global_x, 'y': global_y}
            print( "Cam global coordinate:\n",cam_global)

        

            # Flip Y to match image coordinate system
            # POSITION
            point = (int(global_x), map_height - int(global_y))

            # ROTATION
            arrow_length = 20
            cam_direction_deg = float( target_cam["attributes"][1]["value"])
            adjusted_deg = (90 - cam_direction_deg) % 360
            cam_direction_rad = math.radians(adjusted_deg)
            # Calculate direction vector (cos, sin) — adjust y to match image coordinate
            dx = int(arrow_length * math.cos(cam_direction_rad))
            dy = int(-arrow_length * math.sin(cam_direction_rad))  # minus because image Y axis is downward
            # Compute end point of the arrow
            arrow_end = (point[0] + dx, point[1] + dy)

            #FOV ESTIMATION
            f_x = float(target_cam["intrinsicMatrix"][0][0])
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

            # ID
            cam_id = cam_id.split("_")
            n_id = 0
            if len(cam_id)>1:
                n_id = int(cam_id[1])

            cv2.circle(map_image, point, radius=5, color=(0, 0, 255), thickness=-1)
            cv2.arrowedLine(map_image, point, arrow_end, color=(0, 255, 0), thickness=2, tipLength=0.3)
            cv2.putText(map_image, str(n_id), (point[0] + 10, point[1] - 15), # top left
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            '''
            f_str = f"({cam_local['x']:.2f},{cam_local['y']:.2f})"
            f_str = f"({point[0]:.2f},{point[1]:.2f})"
            cv2.putText(map_image, f_str, (point[0], point[1] + 15), # top left
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            
            cv2.line(map_image, (point[0], point[1]), left_point, (255, 0, 0), 2)
            cv2.line(map_image, (point[0], point[1]), right_point, (255, 0, 0), 2)
            cv2.line(map_image, left_point, right_point, (100, 100, 255), 1, cv2.LINE_AA)
            '''
    # Display the map
    # save map image as "camera_map.png"
    cv2.imwrite(f"{BASE_URL}\camera_map.png", map_image)
    cv2.imshow("Map with Camera", map_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()