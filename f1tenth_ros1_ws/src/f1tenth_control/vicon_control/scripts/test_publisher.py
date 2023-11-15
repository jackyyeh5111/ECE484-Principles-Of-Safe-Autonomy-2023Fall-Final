import numpy as np
import cv2
import rospy
import os
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_publish', '-n', type=str, default=1)
parser.add_argument('--image_folder', '-d', type=str, default='./test_images')
args = parser.parse_args()

IMAGE_CHANNEL = '/D435I/color/image_raw'

if __name__ == '__main__':
    rospy.init_node('publish_node', anonymous=True)
    pub_image = rospy.Publisher(IMAGE_CHANNEL, Image, queue_size=1)

    print('Ready to publish images...')
    bridge = CvBridge()
    for image_path in os.listdir(args.image_folder):
        if not image_path.endswith('png') and not image_path.endswith('jpg'):
            continue
        image_path = os.path.join(args.image_folder, image_path)
        img = cv2.imread(image_path)
        out_img_msg = bridge.cv2_to_imgmsg(img, 'bgr8')
        
        # Publish image message in ROS
        pub_image.publish(out_img_msg)
        print ('Publish image: {}'.format(image_path))
        time.sleep(0.5)
