#!/usr/bin/env python3

from __future__ import print_function

# ROS Headers
import rospy
from vicon_tracker_pp import F1tenth_controller
from lane_detector import LaneDetector
import argparse

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

def main():
    args = parser.parse_args()
    print ('======= Initial arguments =======')
    for key, val in vars(args).items():
        print (f"{key} => {val}")
    
    assert args.curv_min < args.curv_max
    assert args.vel_min < args.vel_max
    
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
