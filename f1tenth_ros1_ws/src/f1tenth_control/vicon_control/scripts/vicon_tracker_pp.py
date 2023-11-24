#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
import argparse

###################################################################################################

from way_pts import way_pts
class F1tenth_controller(object):
    def __init__(self, args, debug_mode=False):
        self.steering_k = args.steering_k
        self.steering_i = args.steering_i
        self.angle_limit = args.angle_limit
        self.curv_min = args.curv_min
        self.curv_max = args.curv_max
        self.vel_min = args.vel_min
        self.vel_max = args.vel_max
        self.look_ahead = args.look_ahead
        self.kp = args.kp
        self.kd = args.kd
        self.ki = args.ki
        self.wheelbase = 0.325
        self.debug_mode = debug_mode
        self.prev_error = 0.0 
        self.integral = 0.0
        self.dt = 0.03
        
        if not debug_mode:
            self.rate = rospy.Rate(30)  # Hz
            self.ctrl_pub = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
            self.drive_msg = AckermannDriveStamped()
            self.drive_msg.header.frame_id = "f1tenth_control"

            self.read_waypoints()

            self.car_state_sub = rospy.Subscriber('/car_state', Float64MultiArray, self.carstate_callback)
            self.car_x   = 0.0
            self.car_y   = 0.0
            self.car_yaw = 0.0

    def carstate_callback(self, data):
        self.car_x = data.data[0]
        self.car_y = data.data[1]
        self.car_yaw = np.radians(data.data[3])

    def read_waypoints(self):
        ## sample a waypoint every "wp_dist" meters
        wp_dist = 0.2
        waypoints_x, waypoints_y, waypoints_yaw, dist_list = [], [], [], []
        for i in range(len(way_pts)-1):
            dist_list.append(np.hypot(way_pts[i+1][0] - way_pts[i][0], way_pts[i+1][1] - way_pts[i][1]))
        cumsum_dist = np.cumsum(dist_list)
        count = 0
        for i in range(len(cumsum_dist)):
            if cumsum_dist[i] >= count * wp_dist:
                waypoints_x.append(way_pts[i][0])
                waypoints_y.append(way_pts[i][1])
                waypoints_yaw.append(way_pts[i][2])
                count = count + 1
        self.wp_x = np.array(waypoints_x)
        self.wp_y = np.array(waypoints_y)
        self.wp_yaw = np.array(waypoints_yaw)  # degree
    
    def get_targ_points(self):
        ## coordinate transformation
        curr_x, curr_y, curr_yaw = self.car_x, self.car_y, self.car_yaw
        rot_mtx = np.array([[np.cos(-curr_yaw), -np.sin(-curr_yaw)], [np.sin(-curr_yaw), np.cos(-curr_yaw)]])
        pts_arr = np.dot(rot_mtx, np.array([self.wp_x, self.wp_y]) - np.array([[curr_x], [curr_y]]))
        ## find the distance of each waypoint from current position
        dist_list, angle_list = [], []
        for i in range(pts_arr.shape[1]):
            dist_list.append(np.hypot(pts_arr[0,i], pts_arr[1,i]))
            angle_list.append(np.arctan2(pts_arr[1,i], pts_arr[0,i]))
        dist_arr = np.array(dist_list)
        angle_arr = np.degrees(np.array(angle_list))
        ## find those points which are less than lookahead distance (behind and ahead the vehicle)
        targ_idx = np.where((abs(angle_arr) < 90) & (dist_arr < self.look_ahead))[0]
        self.targ_pts = list(pts_arr[:,targ_idx].transpose())

    def get_steering_based_point(self, targ_pts, max_step = 100):
        """ 
            Extend the curve to find the most suitable way point
        """
        self.look_ahead
        lanex = [pt[0] for pt in targ_pts]
        laney = [pt[1] for pt in targ_pts]
        lane_fit = np.polyfit(lanex, laney, deg=2)
        max_x = np.max(lanex)
        min_x = np.min(lanex)
        step = (max_x - min_x) / 50
        
        steering_based_pt = [-1, -1]
        for i in range(max_step):
            x = min_x + i*step
            y = np.polyval(lane_fit, x)
            dist = np.hypot(x, y)
            
            steering_based_pt = [x, y]
            if dist > self.look_ahead:
                break
            
        return steering_based_pt
        
        
    def run(self, way_pts=None):
        ## find the goal point which is the last in the set of points less than lookahead distance
        if way_pts is None: # way_pts is provided by the perload file
            self.get_targ_points()
        else:
            self.targ_pts = way_pts
            
        # for targ_pt in self.targ_pts[::-1]:
        #     angle = np.arctan2(targ_pt[1], targ_pt[0])
        #     ## find correct look-ahead point by using heading information
        #     if abs(angle) < np.pi/2:
        #         self.goal_x, self.goal_y = targ_pt[0], targ_pt[1]
        #         break

        ## lateral control using pure pursuit
        self.goal_x, self.goal_y = self.get_steering_based_point(self.targ_pts)
        # self.goal_x = self.targ_pts[-1][0]
        # self.goal_y = self.targ_pts[-1][1]
        
        ## true look-ahead distance between a waypoint and current position
        ld = np.hypot(self.goal_x, self.goal_y)
        
        # find target steering angle (tuning this part as needed)
        # alpha = np.arctan2(self.goal_y, self.goal_x)
        # angle = np.arctan2((self.steering_k * 2 * self.wheelbase * np.sin(alpha)) / ld, 1) * self.steering_i
        # target_steering = round(np.clip(angle, -np.radians(self.angle_limit), np.radians(self.angle_limit)), 3)
        # target_steering_deg = round(np.degrees(target_steering))
        
        alpha = np.arctan2(self.goal_y, self.goal_x)
        
        # PID control
        ct_error = self.targ_pts[0][1]  
        pid_ang = self.kp * ct_error + self.kd * ((ct_error - self.prev_error) / self.dt)
        self.prev_error = ct_error
        pp_angle = np.arctan2((2 * self.wheelbase * np.sin(alpha)) / ld, 1)
        target_steering = round(np.clip(pp_angle + pid_ang, -np.radians(self.angle_limit), np.radians(self.angle_limit)), 3)
        target_steering_deg = round(np.degrees(target_steering))
        
        ## compute track curvature for longititudal control
        num_waypts = len(self.targ_pts)
        idxs = [0, num_waypts // 2, num_waypts - 1]
            
        if len(self.targ_pts) >= 3:
            dx0 = self.targ_pts[idxs[1]][0] - self.targ_pts[idxs[0]][0]
            dy0 = self.targ_pts[idxs[1]][1] - self.targ_pts[idxs[0]][1]
            dx1 = self.targ_pts[idxs[2]][0] - self.targ_pts[idxs[1]][0]
            dy1 = self.targ_pts[idxs[2]][1] - self.targ_pts[idxs[1]][1]
    
            # dx0 = self.targ_pts[-2][0] - self.targ_pts[-3][0]
            # dy0 = self.targ_pts[-2][1] - self.targ_pts[-3][1]
            # dx1 = self.targ_pts[-1][0] - self.targ_pts[-2][0]
            # dy1 = self.targ_pts[-1][1] - self.targ_pts[-2][1]
            ddx, ddy = dx1 - dx0, dy1 - dy0
            curvature = np.inf if dx1 == 0 and dy1 == 0 else abs((dx1*ddy - dy1*ddx) / (dx1**2 + dy1**2) ** (3/2))
        else:
            curvature = np.inf

        ## adjust speed according to curvature and steering angle
        curvature = min(self.curv_max, curvature)
        curvature = max(self.curv_min, curvature)
        target_velocity = self.vel_max - (self.vel_max - self.vel_min) * curvature / (self.curv_max - self.curv_min)
        steering_limit = 60
        if target_steering >= np.radians(steering_limit):
            target_velocity = self.vel_min
        
        if not self.debug_mode:
            self.drive_msg.header.stamp = rospy.get_rostime()
            self.drive_msg.drive.steering_angle = target_steering
            self.drive_msg.drive.speed = target_velocity
            self.ctrl_pub.publish(self.drive_msg)
        
        # ctrl msgs displayed on results
        msgs = [
            "first waypt: ({:.2f}, {:.2f})".format(self.targ_pts[0][0], self.targ_pts[0][1]),
            "lookahead_pt: ({:.2f}, {:.2f})".format(self.goal_x, self.goal_y),
            "ct_error: {:.3f}".format(ct_error),
            "steering(deg): {}".format(target_steering_deg),
            "pp_angle: {:.2f}".format(pp_angle),
            "pid_ang: {:.2f}".format(pid_ang),
            "curvature: {:.3f}".format(curvature),
            "target_vel: {:.2f}".format(target_velocity),
        ]
        
        # print msgs
        print ('\n----- control msgs -----')
        for msg in msgs:
            print (msg)

        return msgs # return msgs for debug
        
    def controller(self):
        while not rospy.is_shutdown():
            self.run()
            self.rate.sleep()

###################################################################################################

def control_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steering_k', type=float, default=0.6)
    parser.add_argument('--steering_i', type=float, default=4.0)
    parser.add_argument('--angle_limit', type=float, default=80)
    parser.add_argument('--curv_min', type=float, default=0.0)
    parser.add_argument('--curv_max', type=float, default=0.4)
    parser.add_argument('--vel_min', type=float, default=0.6)
    parser.add_argument('--vel_max', type=float, default=1.0)
    parser.add_argument('--look_ahead', type=float, default=1.0)
    args = parser.parse_args()

    rospy.init_node('vicon_pp_node', anonymous=True)
    ctrl = F1tenth_controller(args)
    try:
        ctrl.controller()
    except rospy.ROSInterruptException:
        pass

if __name__== '__main__':
    control_main()
