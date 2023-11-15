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
parser.add_argument('--num_save', '-n', type=int, default=1)
parser.add_argument('--output_name', '-o', type=str, default='test')
args = parser.parse_args()

bridge = CvBridge()
OUTPUT_DIR = "test_images"
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

cnt = 0
def img_callback(data):
    global cnt
    try:
        # Convert a ROS image message into an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    
    if cnt >= args.num_save:
        return
    
    cnt += 1
    output_path = os.path.join(OUTPUT_DIR, '{}_{}.png'.format(args.output_name, cnt))
    print ('Output image: {}'.format(output_path))
    cv2.imwrite(output_path, cv_image)
    
if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    rate = rospy.Rate(30)  # Hz

    # Subscribe to a topic
    print ('Waiting images...')
    sub_image = rospy.Subscriber('/D435I/color/image_raw', Image, img_callback, queue_size=1)

    while not rospy.core.is_shutdown():
        if cnt >= args.num_save:
            break
        
        # Main ROS loop to handle callbacks
        # rospy.spin()
        rate.sleep()