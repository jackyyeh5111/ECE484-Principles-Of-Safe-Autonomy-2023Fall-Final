#!/usr/bin/env python3

from __future__ import print_function

# ROS Headers
import rospy
from vicon_tracker_pp import F1tenth_controller
from lane_detector import LaneDetector
import argparse
import pathlib
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser()

# lane detector arguments
parser.add_argument('--gradient_thresh', '-g', type=str, default='75,150')
parser.add_argument('--sat_thresh', type=str, default='60,255')
parser.add_argument('--val_thres_percentile', type=int, default=65)
parser.add_argument('--hue_thresh', type=str, default='15,40')
parser.add_argument('--dilate_size', type=int, default=5)
parser.add_argument('--hist_y_begin', type=int, default=30)
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='218,467,348,0')

# controller arguments
parser.add_argument('--steering_k', type=float, default=0.6)
parser.add_argument('--steering_i', type=float, default=4.0)
parser.add_argument('--angle_limit', type=float, default=80)
parser.add_argument('--curv_min', type=float, default=0.0)
parser.add_argument('--curv_max', type=float, default=0.4)
parser.add_argument('--vel_min', type=float, default=0.6)
parser.add_argument('--vel_max', type=float, default=1.0)
parser.add_argument('--look_ahead', type=float, default=1.0)

# debug params
parser.add_argument('--specified_name', '-s', type=str)
parser.add_argument('--num_samples', '-n', type=int, default=-1,
                    help="-1 means check all files in the input folder")
parser.add_argument('--input_dir', '-i', type=str, default='test_images')

args = parser.parse_args()

OUTPUT_DIR = 'debug_results'

ctrl_params = [
    'steering_k: {}'.format(args.steering_k),
    'steering_i: {}'.format(args.steering_i),
    'angle_limit: {}'.format(args.angle_limit),
    'curv_min: {}'.format(args.curv_min),
    'curv_max: {}'.format(args.curv_max),
    'vel_min: {}'.format(args.vel_min),
    'vel_max: {}'.format(args.vel_max),
    'look_ahead: {}'.format(args.look_ahead),
]


def get_output_img(raw_img, vis_warped, ctrl_msgs, way_pts):
    height, width = raw_img.shape[:2]
    canvas = np.zeros((height//3, width, 3), dtype=raw_img.dtype) + 50
    concat = cv2.vconcat([raw_img, canvas, vis_warped])

    # puttext params
    font_scale = 0.6
    font_color = (0, 0, 255)
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 10

    # top left
    pos = [padding, height + 30]
    for msg in ctrl_params:
        cv2.putText(concat, msg, pos, font,
                    font_scale, font_color, font_thickness)
        (box_x, box_y), _ = cv2.getTextSize(
            msg, font, font_scale, font_thickness)
        pos[1] = pos[1] + box_y + padding

    # top center
    pos = [padding + width//3, height + 30]
    for msg in ctrl_msgs:
        cv2.putText(concat, msg, pos, font,
                    font_scale, font_color, font_thickness)
        (box_x, box_y), _ = cv2.getTextSize(
            msg, font, font_scale, font_thickness)
        pos[1] = pos[1] + box_y + padding

    # top right
    pos = [width - 150, height + 30]
    for way_pt in way_pts:
        text = '({:.2f}, {:.2f})'.format(way_pt[0], way_pt[1])
        cv2.putText(concat, text, pos, font,
                    font_scale, (0, 255, 0), font_thickness)
        
        (box_x, box_y), _ = cv2.getTextSize(
            msg, font, font_scale, font_thickness)
        pos[1] = pos[1] + box_y + padding

    return concat

def run(img_path, lane_detector, controller):
        img_name = img_path.split('/')[-1]
        raw_img = cv2.imread(img_path)
        vis_warped, color_warped, way_pts = lane_detector.detection(raw_img)
        ctrl_msgs = controller.run(way_pts)

        out_img = get_output_img(raw_img, vis_warped, ctrl_msgs, way_pts)
        output_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(output_path, out_img)
    
def main():
    args = parser.parse_args()
    print('======= Initial arguments =======')
    for key, val in vars(args).items():
        print(f"{key} => {val}")

    assert args.curv_min < args.curv_max
    assert args.vel_min < args.vel_max

    lane_detector = LaneDetector(args, debug_mode=True)
    controller = F1tenth_controller(args, debug_mode=True)

    pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        
    if args.specified_name:
        img_path = os.path.join(
            args.input_dir, '{}.png'.format(args.specified_name))
        print ('img_path:', img_path)
        run(img_path, lane_detector, controller)
    else:
        paths = sorted(os.listdir(args.input_dir))
        for i, img_path in enumerate(paths):
            if i == args.num_samples:
                break
            if not img_path.endswith('png') and not img_path.endswith('jpg'):
                continue
            
            img_path = os.path.join(args.input_dir, img_path)
            print ('img_path:', img_path)
            run(img_path, lane_detector, controller)

if __name__ == '__main__':
    main()
