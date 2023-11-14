#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

###################################################################################################

from way_pts import way_pts
class F1tenth_controller(object):
    def __init__(self, 
                 steering_k = 0.6,
                 steering_i = 4.0,
                 angle_limit = 80,
                 curv_min = 0.0,
                 curv_max = 0.4,
                 vel_min = 0.6,
                 vel_max = 1.0,):
        self.steering_k = steering_k
        self.steering_i = steering_i
        self.angle_limit = angle_limit
        self.curv_min = curv_min
        self.curv_max = curv_max
        self.vel_min = vel_min
        self.vel_max = vel_max
        
        self.rate = rospy.Rate(30)  # Hz
        self.ctrl_pub = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/navigation", AckermannDriveStamped, queue_size=1)
        self.drive_msg = AckermannDriveStamped()
        self.drive_msg.header.frame_id = "f1tenth_control"

        self.look_ahead = 1.0
        self.wheelbase = 0.325
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
        self.goal_x = self.targ_pts[-1][0]
        self.goal_y = self.targ_pts[-1][1]
        
        ## true look-ahead distance between a waypoint and current position
        ld = np.hypot(self.goal_x, self.goal_y)
        
        # find target steering angle (tuning this part as needed)
        alpha = np.arctan2(self.goal_y, self.goal_x)
        angle = np.arctan2((self.steering_k * 2 * self.wheelbase * np.sin(alpha)) / ld, 1) * self.steering_i
        target_steering = round(np.clip(angle, -np.radians(self.angle_limit), np.radians(self.angle_limit)), 3)
        target_steering_deg = round(np.degrees(target_steering))
        
        ## compute track curvature for longititudal control
        if len(self.targ_pts) >= 3:
            dx0 = self.targ_pts[-2][0] - self.targ_pts[-3][0]
            dy0 = self.targ_pts[-2][1] - self.targ_pts[-3][1]
            dx1 = self.targ_pts[-1][0] - self.targ_pts[-2][0]
            dy1 = self.targ_pts[-1][1] - self.targ_pts[-2][1]
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
        
        ct_error = round(np.sin(alpha) * ld, 3)
        print("Lookahead distance: ", str(ld))
        print("Crosstrack Error: " + str(ct_error))
        print("Steering angle: " + str(target_steering_deg) + " degrees\n")
        print("Velocity: " + str(target_velocity))
        
        self.drive_msg.header.stamp = rospy.get_rostime()
        self.drive_msg.drive.steering_angle = target_steering
        self.drive_msg.drive.speed = target_velocity
        self.ctrl_pub.publish(self.drive_msg)
        
    def controller(self):
        while not rospy.is_shutdown():
            self.run()
            self.rate.sleep()

###################################################################################################

def control_main():
    rospy.init_node('vicon_pp_node', anonymous=True)
    ctrl = F1tenth_controller()
    try:
        ctrl.lateral_controller()
    except rospy.ROSInterruptException:
        pass

if __name__== '__main__':
    control_main()
