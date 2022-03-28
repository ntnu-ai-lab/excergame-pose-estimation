import os

# IMPORTANT: this file must be located in and be ran from the openpose installation directory
# e.g. C:/path/to/installation/openpose

# Paths
source_folder = r'[INSERT_PATH]' + '\\'  # Directory containing images from COCO dataset, e.g. C:/path/to/directory/coco/images
dest_folder = r'[INSERT_PATH]' + '\\'  # Directory where openpose output ends up, e.g. C:/path/to/directory/coco/output

# Other strings
IMAGE_SET = 'val2014' # Which COCO dataset, e.g. 'val2014' or 'val2017'

source_folder = source_folder + IMAGE_SET + '\\'
dest_folder = dest_folder + IMAGE_SET + '\\'

os.system(r'build\x64\Release\OpenPoseDemo.exe --image_dir ' + source_folder + ' --write_json ' + dest_folder + ' --display 0 --render_pose 0')
