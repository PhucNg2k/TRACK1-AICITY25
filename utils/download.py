from huggingface_hub import snapshot_download

# Constants
REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
data_folder = "MTMC_Tracking_2025"
DATASET_DIR = "DATASET"

token = ""

# Download snapshot excluding the undesired folder
'''
snapshot_path = snapshot_download(
    repo_id=REPO_ID,
     repo_type="dataset",
    allow_patterns=[
        f"{data_folder}/train/**/", 
        f"{data_folder}/val/**"
    ],
    ignore_patterns=[
        f"{data_folder}/**/depth_maps/**"
    ],
    local_dir=DATASET_DIR,
    local_dir_use_symlinks=False,
    token=token
)
'''
DATASET_DIR = "DEPTHMAP"
# Download snapshot excluding the undesired folder
snapshot_path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=[
        f"{data_folder}/train/Warehouse_000/depth_maps/Camera_0000.h5", 
    ],
    local_dir=DATASET_DIR,
    local_dir_use_symlinks=False,
    token=token
)

print(f"Dataset downloaded to: {snapshot_path}")
