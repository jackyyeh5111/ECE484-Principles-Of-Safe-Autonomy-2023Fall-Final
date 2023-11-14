import time
import math
import numpy as np
import cv2
import rospy
import os
import pathlib

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
# from skimage import morphology

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gradient_thresh', '-g', type=str, default='75,150')
parser.add_argument('--sat_thresh', type=str, default='60,255')
parser.add_argument('--val_thres_percentile', type=int, default=65)
parser.add_argument('--hue_thresh', type=str, default='15,40')
parser.add_argument('--dilate_size', type=int, default=5)
parser.add_argument('--hist_y_begin', type=int, default=30)
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='218,467,348,0')
args = parser.parse_args()

OUTPUT_DIR = './output'

# TMP_DIR = './vis_{}'.format(args.append_str)
grad_thres_min, grad_thres_max = args.gradient_thresh.split(',')
grad_thres_min, grad_thres_max = int(grad_thres_min), int(grad_thres_max)
assert grad_thres_min < grad_thres_max

sat_thres_min, sat_thres_max = args.sat_thresh.split(',')
sat_thres_min, sat_thres_max = int(sat_thres_min), int(sat_thres_max)

hue_thres_min, hue_thres_max = args.hue_thresh.split(',')
hue_thres_min, hue_thres_max = int(hue_thres_min), int(hue_thres_max)
assert hue_thres_min < hue_thres_max

src_leftx, src_rightx, laney, offsety = args.perspective_pts.split(',')
src_leftx, src_rightx, laney, offsety = int(
    src_leftx), int(src_rightx), int(laney), int(offsety)

INCH2METER = 0.0254
PIX2METER_X = 0.0009525 # meter
PIX2METER_Y = 0.0018518 # meter
DIST_CAM2FOV_INCH = 21 # inch

def imshow(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class LaneDetector():
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        if not self.test_mode:
            self.bridge = CvBridge()
            # NOTE
            self.sub_image = rospy.Subscriber('/D435I/color/image_raw', Image, self.img_callback, queue_size=1)
            # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)
            self.pub_image = rospy.Publisher(
                "lane_detection/annotate", Image, queue_size=1)
            self.pub_bird = rospy.Publisher(
                "lane_detection/birdseye", Image, queue_size=1)
            self.left_line = Line(n=5)
            self.right_line = Line(n=5)
            self.detected = False
            self.hist = True
            self.counter = 0
            self.way_pts = []

    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()
        self.detection(raw_img)
        # mask_image, bird_image = self.detection(raw_img)

        # if mask_image is not None and bird_image is not None:
        #     # Convert an OpenCV image into a ROS image message
        #     out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
        #     out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

        #     # Publish image message in ROS
        #     self.pub_image.publish(out_img_msg)
        #     self.pub_bird.publish(out_bird_msg)

    def gradient_thresh(self, img, thresh_min=grad_thres_min, thresh_max=grad_thres_max):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        # 1. Convert the image to gray scale
        # 2. Gaussian blur the image
        # 3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        # 4. Use cv2.addWeighted() to combine the results
        # 5. Convert each pixel to uint8, then apply threshold to get binary image

        # Step 1: Load the image and convert it to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 2: Apply Gaussian blur to the grayscale image
        blurred_image = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # Step 3: Use cv2.Sobel() to find derivatives for both X and Y axes
        sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

        # Step 4: Combine the results using cv2.addWeighted()
        sobel_combined = cv2.addWeighted(np.absolute(
            sobel_x), 0.5, np.absolute(sobel_y), 0.5, 0)

        # Step 5: Convert each pixel to uint8 and apply a threshold to get a binary image
        sobel_combined = np.uint8(sobel_combined)
        binary_output = np.zeros_like(sobel_combined)
        binary_output[(thresh_min < sobel_combined) &
                      (sobel_combined < thresh_max)] = 1

        # vis
        # vis = cv2.cvtColor(binary_output*255, cv2.COLOR_GRAY2BGR)
        # imshow("binary_output", cv2.hconcat([img, vis]))

        return binary_output

    def color_thresh(self, img, val_thres):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # 1. Convert the image from RGB to HSL
        # 2. Apply threshold on S channel to get binary image
        # Hint: threshold on H to remove green grass
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # For HSL
        # ref: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
        #   image format: (8-bit images) V ← 255⋅V, S ← 255⋅S, H ← H/2(to fit to 0 to 255)

        # Step 2: Apply threshold on the S (Saturation) channel to get a binary image
        h, l, s = cv2.split(hls_img)
        binary_output = np.zeros_like(l)
        sat_cond = ((sat_thres_min <= s) & (s <= sat_thres_max))
        
        # Use gray image instead of L channel of HLS (their images are different!)
        val_cond = (val_thres <= gray_img)
        hue_cond = (hue_thres_min <= h) & (h <= hue_thres_max)
        
        binary_output[val_cond & sat_cond & hue_cond] = 1
        
        # closing
        kernel = np.ones((5, 5), np.uint8)
        binary_output = cv2.morphologyEx(
            binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return binary_output

    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        # 1. Apply sobel filter and color filter on input image
        # 2. Combine the outputs
        # Here you can use as many methods as you want.

        SobelOutput = self.gradient_thresh(img)
        ColorOutput = self.color_thresh(
            img, val_thres_min, val_thres_max, sat_thres_min, sat_thres_max)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.dilate_size, args.dilate_size))
        ColorOutput = cv2.dilate(ColorOutput, kernel, iterations=1)
        # imshow("dilate ColorOutput", ColorOutput*255)

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput == 1)] = 1
        # binaryImage[(ColorOutput == 1) & (SobelOutput == 1)] = 1

        # Remove noise from binary image
        # binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

        return binaryImage

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        # Define four points as (x, y) coordinates
        src_height, src_width = img.shape[:2]

        src_pts = np.array([[src_leftx, laney],
                            [0, src_height - offsety],
                            [src_width, src_height - offsety],
                            [src_rightx, laney]], dtype=np.int32)

        # dst_width, dst_height = 720, 1250
        dst_width, dst_height = src_width, src_height
        dst_pts = np.array([[0, 0],
                            [0, dst_height],
                            [dst_width, dst_height],
                            [dst_width, 0]], dtype=np.int32)

        def calc_warp_points():
            src = np.float32(src_pts)
            dst = np.float32(dst_pts)

            return src, dst

        src, dst = calc_warp_points()
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        # keep same size as input image
        warped_img = cv2.warpPerspective(
            img, M, (dst_width, dst_height), flags=cv2.INTER_NEAREST)

        return warped_img, M, Minv

    def findContourForColor(self, color_warped):
        # Find the contours for color (ideeally two contours along the trajectory)
        contours, _ = cv2.findContours(
            color_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_canvas = np.zeros_like(color_warped)
        cv2.drawContours(contour_canvas, contours, -1, 1, 1)

        # remove contours along the image border
        contour_canvas[-5:, :] = 0
        contour_canvas[:5, :] = 0

        return contour_canvas

    def get_latest_waypoints(self):
        return self.way_pts
    
    def update_waypoints(self, ret, width, height, look_ahead_dist = 1.0):
            lanex = ret['lanex']
            laney = ret['laney']
            
            # transform from image coord (x, y) to camera coord in meters
            lanex = [(x - width // 2) * PIX2METER_X for x in lanex]
            laney = [(height - y) * PIX2METER_Y + DIST_CAM2FOV_INCH * INCH2METER for y in laney]
            
            print ('\n--------- waypoints ---------')
            for i, (x, y) in enumerate(zip(lanex, laney)):
                print ('{} => Jacky coord: ({:.2f}, {:.2f}), Ricky coord: ({:.2f}, {:.2f})'
                       .format(i+1, x, y, y, -x))
            
            # change to Ricky's coordinate    
            way_pts = [(y, -x) for x, y in zip(lanex, laney)]
            
            # only update way pts when succefully fit lines
            if len(way_pts) != 0:
                self.way_pts = way_pts
        
    def detection(self, img):

        # get dynamic value thres
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_warped, M, Minv = self.perspective_transform(gray_img)
        val_thres = np.percentile(gray_img_warped, args.val_thres_percentile)
        
        # use color_thresh result only
        color_output = self.color_thresh(img, val_thres)
        color_warped, M, Minv = self.perspective_transform(color_output)
        contour_warped = self.findContourForColor(color_warped)
        
        # line fit
        hist = np.sum(color_warped[-args.hist_y_begin:, :], axis=0)
        ret = line_fit(contour_warped, hist, gray_img_warped)
        if ret is None:
            return
            
        # get get_waypoints
        height, width = img.shape[:2]
        self.update_waypoints(ret, width, height, look_ahead_dist = 1.0)
        
        # if not self.hist:
        #     # Fit lane without previous result
        #     ret = line_fit(img_birdeye)
        #     left_fit = ret['left_fit']
        #     right_fit = ret['right_fit']
        #     nonzerox = ret['nonzerox']
        #     nonzeroy = ret['nonzeroy']
        #     left_lane_inds = ret['left_lane_inds']
        #     right_lane_inds = ret['right_lane_inds']

        # else:
        #     # Fit lane with previous result
        #     if not self.detected:
        #         ret = line_fit(img_birdeye)

        #         if ret is not None:
        #             left_fit = ret['left_fit']
        #             right_fit = ret['right_fit']
        #             nonzerox = ret['nonzerox']
        #             nonzeroy = ret['nonzeroy']
        #             left_lane_inds = ret['left_lane_inds']
        #             right_lane_inds = ret['right_lane_inds']

        #             left_fit = self.left_line.add_fit(left_fit)
        #             right_fit = self.right_line.add_fit(right_fit)

        #             self.detected = True
        #     else:
        #         left_fit = self.left_line.get_fit()
        #         right_fit = self.right_line.get_fit()
        #         ret = tune_fit(img_birdeye, left_fit, right_fit)

        #         if ret is not None:
        #             left_fit = ret['left_fit']
        #             right_fit = ret['right_fit']
        #             nonzerox = ret['nonzerox']
        #             nonzeroy = ret['nonzeroy']
        #             left_lane_inds = ret['left_lane_inds']
        #             right_lane_inds = ret['right_lane_inds']

        #             left_fit = self.left_line.add_fit(left_fit)
        #             right_fit = self.right_line.add_fit(right_fit)

        #         else:
        #             self.detected = False

        #     # Annotate original image
        #     bird_fit_img = None
        #     combine_fit_img = None
        #     if ret is not None:
        #         bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
        #         combine_fit_img = final_viz(img, left_fit, right_fit, Minv)
        #         print("Lanes detected!")
        #     else:
        #         print("Unable to detect lanes")

        #     return combine_fit_img, bird_fit_img


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    print ('Start to detect...')
    LaneDetector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
