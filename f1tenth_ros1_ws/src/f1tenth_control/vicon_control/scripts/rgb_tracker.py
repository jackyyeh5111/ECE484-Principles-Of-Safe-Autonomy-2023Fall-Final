#!/usr/bin/env python3

from __future__ import print_function

# ROS Headers
import rospy
from vicon_tracker_pp import F1tenth_controller
from lane_detector import LaneDetector
import argparse
import pathlib
import os

parser = argparse.ArgumentParser()

# lane detector arguments
parser.add_argument('--output_dir', '-o', type=str, default='')
parser.add_argument('--output_freq', type=int, default=1)
parser.add_argument('--gradient_thresh', '-g', type=str, default='75,150')
# parser.add_argument('--sat_thresh', type=str, default='60,255')
parser.add_argument('--sat_cdf_lower_thres', type=float, default=0.5)
parser.add_argument('--val_thres_percentile', type=int, default=65)
parser.add_argument('--hue_thresh', type=str, default='15,40')
parser.add_argument('--dilate_size', type=int, default=5)
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='218,467,348,0')
parser.add_argument('--window_height', type=int, default=20)

# controller arguments
parser.add_argument('--steering_k', type=float, default=0.6)
parser.add_argument('--steering_i', type=float, default=1.0)
parser.add_argument('--angle_limit', type=float, default=80)
parser.add_argument('--curv_min', type=float, default=0.0)
parser.add_argument('--curv_max', type=float, default=0.4)
parser.add_argument('--vel_min', type=float, default=0.6)
parser.add_argument('--vel_max', type=float, default=1.0)
parser.add_argument('--look_ahead', type=float, default=1.0)
parser.add_argument('--angle_diff_thres', type=float, default=2.0)
parser.add_argument('--kp', type=float, default=1.5)
parser.add_argument('--kd', type=float, default=0.05)
parser.add_argument('--ki', type=float, default=0.0)
parser.add_argument('--enable_pid', action='store_true')

def main():
    args = parser.parse_args()
    print ('======= Initial arguments =======')
    params = []
    for key, val in vars(args).items():
        param = f"--{key} {val}"
        print(f"{key} => {val}")
        params.append(param)

    # save params for debug
    if args.output_dir != '':
        OUTPUT_DIR = os.path.join('test_images', args.output_dir)
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  
        with open(os.path.join(OUTPUT_DIR, 'params.txt'), 'w') as f:
            commands = ['python debug_helper.py -i {}'.format(OUTPUT_DIR)] + params
            f.write(' \\\n  '.join(commands))

    assert args.curv_min <= args.curv_max
    assert args.vel_min <= args.vel_max
    
    rospy.init_node('rgb_track_node', anonymous=True)
    rate = rospy.Rate(30)  # Hz

    lane_detector = LaneDetector(args)
    controller = F1tenth_controller(args)
    try:
        print ('\nStart navigation...')
        while not rospy.is_shutdown():
            way_pts = lane_detector.get_latest_waypoints()
            
            # Do not update control signal. 
            # Because it cannot fit polyline if way points < 3
            if len(way_pts) >= 3:
                controller.run(way_pts)
            rate.sleep()  # Wait a while before trying to get a new waypoints
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
