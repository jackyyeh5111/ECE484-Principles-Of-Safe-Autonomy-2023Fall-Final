import time
import math
import numpy as np
import cv2
import rospy
import os
import pathlib
from vicon_tracker_pp import F1tenth_controller

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32

import argparse

INCH2METER = 0.0254
PIX2METER_X = 0.0009525 # meter
PIX2METER_Y = 0.0018518 # meter
DIST_CAM2FOV_INCH = 21 # inch

class LaneDetector():
    def __init__(self, args, debug_mode=False):
        self.parse_params(args)
        self.debug_mode = debug_mode
        self.output_dir = args.output_dir
        self.output_freq = args.output_freq
        
        self.way_pts = []
        self.cnt = 0
        self.controller = F1tenth_controller(args)
        if not self.debug_mode:
            self.bridge = CvBridge()
            self.sub_image = rospy.Subscriber('/D435I/color/image_raw', Image, self.img_callback)
            
    def parse_params(self, args):
        # parse params
        # self.sat_thres_min, self.sat_thres_max = args.sat_thresh.split(',')
        # self.sat_thres_min, self.sat_thres_max = int(self.sat_thres_min), int(self.sat_thres_max)
        # assert self.sat_thres_min < self.sat_thres_max

        self.hue_thres_min, self.hue_thres_max = args.hue_thresh.split(',')
        self.hue_thres_min, self.hue_thres_max = int(self.hue_thres_min), int(self.hue_thres_max)
        assert self.hue_thres_min < self.hue_thres_max

        src_leftx, src_rightx, laney, offsety = args.perspective_pts.split(',')
        self.src_leftx, self.src_rightx, self.laney, self.offsety = int(
            src_leftx), int(src_rightx), int(laney), int(offsety)
        
        self.val_thres_percentile = args.val_thres_percentile
        self.dilate_size = args.dilate_size
        self.sat_cdf_lower_thres = args.sat_cdf_lower_thres
        self.window_height = args.window_height
        
    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        start_time = time.time()
        
        raw_img = cv_image.copy()
        way_pts = self.detection(raw_img)
        print("Detection takes time: {:.3f} seconds".format(time.time() - start_time))

        # output images for debug
        self.cnt += 1
        OUTPUT_DIR = os.path.join('test_images', self.output_dir)
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        if self.output_dir != '' and self.cnt % self.output_freq == 0:
            output_path = '{}/{}.png'.format(OUTPUT_DIR, self.cnt)
            print ('output to {}'.format(output_path))
            cv2.imwrite(output_path, raw_img)
         
        # Do not update control signal. 
        # Because it cannot fit polyline if way points < 3
        if way_pts is None or len(way_pts) < 3:
            return
        else:
            self.controller.run(way_pts)
        
    def line_fit(self, binary_warped):
        """
        Find and fit lane lines
        """
        ### 1. sliding window to find the base point
        height, width = binary_warped.shape
        nwindows = 15
        sliding_offset = 5
        margin = 70
        best_base_x = -1
        best_num_pixels = 0
        
        for basex in range(margin, width-margin, sliding_offset):
            left = basex - margin
            right = basex + margin
            total_num_pixels = cv2.countNonZero(
                binary_warped[-self.window_height:, left:right])
            
            if total_num_pixels > best_num_pixels:
                best_num_pixels = total_num_pixels
                best_base_x = basex
        
        if best_base_x == -1:
            return None
        
        # visualize
        # vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
        # vis = cv2.rectangle(
        #     vis, (best_base_x - margin, height - self.window_height), 
        #     (best_base_x + margin, height), 
        #     (0, 0, 255))
        # imshow("vis", vis)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        # Set minimum number of pixels found to recenter window
        minpix = 200
        
        # Step through the windows one by one
        color_warped = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
        color_warped[color_warped > 0] = 255
            
        basex = best_base_x
        lane_pts = []
        prev_basex_list = []
        for i in range(nwindows):
            win_top = height - (i + 1) * self.window_height
            win_bottom = win_top + self.window_height
            
            # adjust basex based on the slope of previous two basex
            if i >= 2:
                slope = prev_basex_list[-1] - prev_basex_list[-2]
                basex = prev_basex_list[-1] + slope
                
            window_inds = np.where((nonzerox > basex - margin) &
                                (nonzerox < basex + margin) &
                                (nonzeroy > win_top) &
                                (nonzeroy < win_bottom))
            window_nonzerox = nonzerox[window_inds]
            window_nonzeroy = nonzeroy[window_inds]
            # print (len(window_nonzerox))
            
            # finish fitting condition
            reach_boundary = basex - margin < 0 or \
                            basex + margin >= width or \
                            win_top < 0
            if reach_boundary or len(window_nonzerox) < minpix:
                break
            
            # correct basex by average and use average (x, y) as way points
            basex = int(np.mean(window_nonzerox))
            basey = int(np.mean(window_nonzeroy))
            lane_pts.append([basex, basey])
            prev_basex_list.append(basex)
                
            # visualization
            color_warped = cv2.rectangle(
                color_warped, (basex - margin, win_top), (basex +margin, win_bottom), (0, 0, 255))
            # imshow("color_warped", color_warped)
        
        lanex = [pt[0] for pt in lane_pts]
        laney = [pt[1] for pt in lane_pts]
        try:
            lane_fit = np.polyfit(laney, lanex, deg=2)
            
            ### vis lane points ###
            for x, y in zip(lanex, laney):
                color_warped = cv2.circle(color_warped, (x, y), 2, (0,255, 0), -1)
                
            ### vis points nonzero ###
            # for x, y in zip(rightx, righty):
            #     color_warped = cv2.circle(color_warped, (x, y), 1, (0,255, 0), -1)
            # imshow("points", color_warped )

        except TypeError:
            print("Unable to detect lanes")
            return None

        ret = {}
        ret['vis_warped'] = color_warped
        ret['lane_fit'] = lane_fit
        ret['lanex'] = lanex
        ret['laney'] = laney
        return ret
    
    def color_thresh(self, img):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        # 1. Convert the image from RGB to HSL
        # 2. Apply threshold on S channel to get binary image

        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        # For HSL
        # ref: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
        #   image format: (8-bit images) V ← 255⋅V, S ← 255⋅S, H ← H/2(to fit to 0 to 255)

        # Step 2: Apply threshold on the S (Saturation) channel to get a binary image
        h, l, s = cv2.split(hls_img)
        binary_output = np.zeros_like(l)
        
        # dynamic search sat_thres_min
        s_warped, M, Minv = self.perspective_transform(s)
        sat_hist, bins = np.histogram(s_warped.flatten(), bins=256, range=[0, 256])

        # Calculate the cumulative distribution function (CDF) of the histogram
        cdf = sat_hist.cumsum()
        cdf_normalized = cdf / cdf.max() # Normalize the CDF to the range [0, 1]
        bin_idxs = \
            np.where((cdf_normalized > self.sat_cdf_lower_thres) & (cdf_normalized < 0.95))[0]
        sat_thres_min = np.argmin( [sat_hist[idx] for idx in bin_idxs] ) + bin_idxs[0]
        sat_cond = ((sat_thres_min <= s) & (s <= 255))
        
        
        # Steps 2: Apply value threshold on image
        # Use red channel of raw_image instead of l channel to do the value filtering
        # Because red channel for yellow lane is much different from background
        red_channel = img[:, :, 2] # red channel
        red_channel_warped, M, Minv = self.perspective_transform(red_channel)
        val_thres_min = np.percentile(red_channel_warped, self.val_thres_percentile)
        # val_mean = np.mean(red_channel_warped)
        val_cond = (val_thres_min <= red_channel) & (red_channel <= 255)

        # Step 3: Apply predefined hue threshold on image
        hue_cond = (self.hue_thres_min <= h) & (h <= self.hue_thres_max)
        
        # combine conditions and get final output
        binary_output[val_cond & sat_cond & hue_cond] = 1

        # closing
        kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)
        binary_output = cv2.morphologyEx(
            binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return binary_output

    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        # 1. Visually determine 4 source points and 4 destination points
        # 2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        # 3. Generate warped image in bird view using cv2.warpPerspective()

        # Define four points as (x, y) coordinates
        src_height, src_width = img.shape[:2]

        src_pts = np.array([[self.src_leftx, self.laney],
                            [0, src_height - self.offsety],
                            [src_width, src_height - self.offsety],
                            [self.src_rightx, self.laney]], dtype=np.int32)

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

    def get_latest_waypoints(self):
        return self.way_pts
    
    def update_waypoints(self, ret, width, height, look_ahead_dist = 1.0):
            lanex = ret['lanex']
            laney = ret['laney']
            
            # transform from image coord (x, y) to camera coord in meters
            lanex = [(x - width // 2) * PIX2METER_X for x in lanex]
            laney = [(height - y) * PIX2METER_Y + DIST_CAM2FOV_INCH * INCH2METER for y in laney]
            
            # print ('\n--------- waypoints ---------')
            # for i, (x, y) in enumerate(zip(lanex, laney)):
            #     print ('{} => Jacky coord: ({:.2f}, {:.2f}), Ricky coord: ({:.2f}, {:.2f})'
            #            .format(i+1, x, y, y, -x))
            
            # change to Ricky's coordinate    
            way_pts = [(y, -x) for x, y in zip(lanex, laney)]
            
            # only update way pts when succefully fit lines
            if len(way_pts) != 0:
                self.way_pts = way_pts
        
    def detection(self, img):
        # use color_thresh result only
        color_output = self.color_thresh(img)
        color_warped, M, Minv = self.perspective_transform(color_output)
        
        # line fit
        ret = self.line_fit(color_warped)
        if ret is None: # fail to polyfit waypoints
            return None
            
        # get get_waypoints
        height, width = img.shape[:2]
        self.update_waypoints(ret, width, height, look_ahead_dist = 1.0)
        
        return self.way_pts

if __name__ == '__main__':
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

    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    rate = rospy.Rate(30)  # Hz
    print ('Start to detect...')
    lane_detector = LaneDetector(args)
    while not rospy.core.is_shutdown():
        rate.sleep()
