import time
import math
import numpy as np
import cv2
import rospy
import os
import pathlib

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_name', '-o', type=str, required=True)
args = parser.parse_args()

bridge = CvBridge()
OUTPUT_DIR = "test_images"
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def img_callback(data):
    try:
        # Convert a ROS image message into an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    output_path = os.path.join(OUTPUT_DIR, '{}.jpg'.format(args.output_name))
    cv2.imwrite(output_path, cv_image)

if __name__ == '__main__':    
    sub_image = rospy.Subscriber('/D435I/color/image_raw', Image, img_callback, queue_size=1)

    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

