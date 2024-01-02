# ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final

Welcome! This is the final project for course ECE484-Principles-Of-Safe-Autonomy in 2023 Fall. The course page can be found [here](https://publish.illinois.edu/robotics-autonomy-resources/f1tenth/).

The project implements a vision-based lane following system. Our aim to make vehicle follow the lane accurately and quickly without collision using Pure Pursuit Controller given RGB images. Our vehicle platform is build on [F1TENTH](https://f1tenth.org/).

Please check out my portfolio post or our [final presentation video](https://www.youtube.com/watch?v=mselI6W_V-o) for a greater detailed description.

## Overview
The vehicle is able to follow the lane accurately without collision:
![demo](pics/demo.gif?raw=true  "demo")

Lane detection result:  

<img src='pics/lane_detect.gif' width='200'>

## Method
<img src='pics/diagram.png' width='750'>

The project built vision-based lane following system from scratch. Lane detector identifies the lane from the captured frame and provides imaginary waypoints candidates for the controller. Next, the controller selects the best waypoint based on the vehicle state, and sends out next control signal.

The whole system is integrated with ROS. It consists of four primary components:
1. Camera calibration
2. Lane detection
3. State estimation
4. Controller

## Quick Starter Guide
Testing environment: Ubuntu 20.04 LTS

### Installation
1. Install ROS Noetic ([link](https://wiki.ros.org/noetic/Installation/Ubuntu))
2. Clone repo
    ```
    $ git clone https://github.com/jackyyeh5111/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final.git lane-follow-project
    $ cd lane-follow-project
    ```
3. Activate virtualenv and install dependencies
    ```
    $ pip install -r requirements.txt
    ```

### Usage

#### online usage
```python
$ python3 lane_detector.py [params...]
```

important params:
- `--perspective_pts`: param for perpective projection. Four number represents respectively for src_leftx, src_rightx, laney, offsety.
- `--angle_limit`: limit vechicle turing angle.
- `--vel_min`: min velocity
- `--vel_max`: max velocity
- `--look_ahead`: fixed look ahead distance for pure pursuit controller. -1 denotes dynamic lookahead distance, which directly use the most distant waypoint as target point.
- `--obstacle_tolerate_dist`: car change velocity if obstacle is within this distance.

#### offline usage (for testing)
For offline, you have to prepare data folder beforehand. Two types of input data format is allowed. One is rosbag, the other option is to put sequential images under the same folder.

- Use rosbag (testing rosbag [download link](https://uofi.box.com/s/ivq5gv9ffxyqpugf4f5c0p13gmado4pe))
    ```
    $ python debug_helper.py --use_rosbag [params...] # Run program
    $ rosbag play <rosbag_path> # Run rosbag
    ```
    
- Sequential images (testing images [download link](https://uofi.box.com/s/82lk65dg8a9vkvc4hn17ffag5car7dva))
    ```
    $ python debug_helper.py -i <source dir> [-s <specified_image_id> -n <num_samples>]
    ```

## Simulation

## Acknowledgement
My great team members:
- Chu-Lin Huang
- Huey-Chii Liang
- Jay Patel

And the support from Professor & TA of ECE484 course.