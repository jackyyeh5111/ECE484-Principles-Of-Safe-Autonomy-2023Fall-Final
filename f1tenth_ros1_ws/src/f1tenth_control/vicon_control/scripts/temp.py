import numpy as np
import cv2
import os
import pathlib
def imshow(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

dir1 = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/f1tenth_control/vicon_control/scripts/debug_results/debug_results_jay'
dir2 = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/f1tenth_control/vicon_control/scripts/debug_results/debug_results_new'

OUTPUT_DIR = 'debug_results/compare'
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

paths1 = []
paths2 = []
for path in os.listdir(dir1):
    if not path.endswith('png'):
        continue
    path = os.path.join(dir1, path)
    paths1.append(path)
for path in os.listdir(dir2):
    if not path.endswith('png'):
        continue
    path = os.path.join(dir2, path)
    paths2.append(path)

for path1 in paths1:
    img_name = path1.split('/')[-1]
    path2 = os.path.join(dir2, img_name)
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    concat = cv2.hconcat([img1, img2])
    
    output_path = os.path.join(OUTPUT_DIR, img_name)
    print ('output_path:', output_path)    
    cv2.imwrite(output_path, concat)
    # imshow('concat', concat)
    
    