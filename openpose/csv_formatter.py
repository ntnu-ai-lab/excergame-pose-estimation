import os
import csv
import json

# IMPORTANT: this file must be located in and be ran from the openpose installation directory
# e.g. C:/path/to/installation/openpose

# Paths
openpose_path = r'D:\Master\openpose'
source_folder = r'[INSERT_PATH]' + '\\'  # Root directory of dataset, e.g. C:/path/to/directory/dataset/
temp_json = r'.\temp_json' + '\\'  # Temporary directory for json objects
csv_dest = r'[INSERT_PATH]' + '\\'  # Directory of resulting csv files, e.g. C:/path/to/directory/csv_output

# Other strings
DELETE_CONTENTS = r'del /Q /S '  # Command to delete contents of a folder, adjust as necessary

# POINT_ORDER is the order the columns will be in in the csv files, but both x, and y coords, e.g. NoseX, NoseY, NeckX, NeckY, ...
POINT_ORDER = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']


def main():
    # Get all participants
    participants = os.listdir(source_folder)

    for par in participants:
    
        par_path = source_folder + par + '\\'
        
        # Get only the camera angles we need
        cam5_vids = [v for v in os.listdir(par_path) if v.endswith('Cam5.avi')]
        
        # Create folder for csv files
        csv_path = csv_dest + par

        # Check if this par is already done
        try:
            if len(cam5_vids) == len(os.listdir(csv_path)):
                print(par + ' done')
                continue
        except FileNotFoundError:
            pass
        
        print(par + ' in progress...')
        
        try:
            os.mkdir(csv_path)
        except FileExistsError:
            os.system(DELETE_CONTENTS + csv_path)
            os.rmdir(csv_path)
            os.mkdir(csv_path)

        for vid in cam5_vids:
            # Delete contents of temporary folder
            os.system(DELETE_CONTENTS + temp_json)
            
            vid_source = par_path + vid

            # This command runs openpose pose estimation on a video, and is why the script must be run from the specific folder
            os.system(r'build\x64\Release\OpenPoseDemo.exe --video ' + vid_source + ' --write_json ' + temp_json + ' --display 0 --render_pose 0')
            print()
            
            # Create csv file with header
            f = open(csv_path + '\\' +  vid[:-4] + '.csv', 'w', newline='')
            writer = csv.writer(f)
            HEADER = []
            for po in POINT_ORDER:
                HEADER.extend([po+'X', po+'Y'])
            writer.writerow(HEADER)
            
            # Get list of JSON objects (each represents an individual frame from the video)
            frames = sorted(os.listdir(temp_json))
            
            # Iterate through JSON objects
            for frame in frames:
                g = open(temp_json + frame, 'r')
                data = json.load(g)
                person = data['people'][0]['pose_keypoints_2d']  # Get keypoints for the person
                
                # Remove every third element (the ones that are not coords) and write data to csv
                del person[2::3]
                writer.writerow(person)
                
                g.close()
                # break
            f.close()
            # break
        # break


main()
