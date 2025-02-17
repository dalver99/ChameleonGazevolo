import cv2
import os
import zipfile

def extract_frames(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    #count = 40
    while success:
        cv2.imwrite(os.path.join(output_dir, f"{video_path[:-4]}_frame_{count}.jpg"), image)
        #cv2.imwrite(os.path.join(output_dir, f"{video_path[:-5]}1_frame_{count}.jpg"), image)
        cv2.waitKey(1)  # wait for a key press
        success, image = vidcap.read()
        count += 1

def zip_frames(output_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for folder, _, filenames in os.walk(output_dir):
            for filename in filenames:
                zipf.write(os.path.join(folder, filename),
                           arcname=os.path.basename(filename))


video_path = 'ys_ipark.mp4'  # input video file
output_dir = 'frames'     # directory to store frames
zip_path = 'frames.zip'   # output zip file

# Ensure output directory exists.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

extract_frames(video_path, output_dir)
#zip_frames(output_dir, zip_path)