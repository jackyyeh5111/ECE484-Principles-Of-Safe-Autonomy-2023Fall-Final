import time
import math
import numpy as np
import cv2
from skimage import morphology
import argparse
import matplotlib.pyplot as plt
import pathlib
import os
import sympy as sp

# import sys
# sys.path.insert(0, "../")
# import lane_detector


parser = argparse.ArgumentParser()
parser.add_argument('--vis_mode', '-v', action='store_true')
parser.add_argument('--vis_output', '--vo', action='store_true')
parser.add_argument('--num_samples', '-n', type=int, default=-1)
parser.add_argument('--vis_hue', action='store_true')
parser.add_argument('--append_str', '-a', type=str, default='tmp')
parser.add_argument('--specified_name', '-s', type=str)
parser.add_argument('--gradient_thresh', '-g', type=str, default='75,150')
parser.add_argument('--sat_thresh', type=str, default='60,255')
# parser.add_argument('--val_thresh', type=str, default='80,255')
# parser.add_argument('--val_thres_offset', type=int, default=20)
parser.add_argument('--val_thres_percentile', type=int, default=65)
parser.add_argument('--hue_thresh', type=str, default='15,40')
parser.add_argument('--dilate_size', type=int, default=5)
parser.add_argument('--hist_y_begin', type=int, default=30)
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='218,467,348,0')
parser.add_argument('--base_algo', type=str, required=True)

args = parser.parse_args()

TEST_DIR = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/frames'
TMP_DIR = './vis_{}'.format(args.append_str)
grad_thres_min, grad_thres_max = args.gradient_thresh.split(',')
grad_thres_min, grad_thres_max = int(grad_thres_min), int(grad_thres_max)
assert grad_thres_min < grad_thres_max

# val_thres_min, val_thres_max = args.val_thresh.split(',')
# val_thres_min, val_thres_max = int(val_thres_min), int(val_thres_max)
# assert val_thres_min < val_thres_max

sat_thres_min, sat_thres_max = args.sat_thresh.split(',')
sat_thres_min, sat_thres_max = int(sat_thres_min), int(sat_thres_max)

hue_thres_min, hue_thres_max = args.hue_thresh.split(',')
hue_thres_min, hue_thres_max = int(hue_thres_min), int(hue_thres_max)
assert hue_thres_min < hue_thres_max

src_leftx, src_rightx, laney, offsety = args.perspective_pts.split(',')
src_leftx, src_rightx, laney, offsety = int(
    src_leftx), int(src_rightx), int(laney), int(offsety)

img_name = ""

INCH2METER = 0.0254
PIX2METER_X = 0.0009525 # meter
PIX2METER_Y = 0.0018518 # meter
DIST_CAM2FOV_INCH = 21 # inch

def putText(img, text,
            font_scale=1,
            font_color=(0, 0, 255),
            font_thickness=2,
            font=cv2.FONT_HERSHEY_SIMPLEX):

    # Calculate the position for bottom right corner
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = img.shape[1] - text_size[0] - 10  # Adjust for padding
    text_y = img.shape[0] - 10  # Adjust for padding

    cv2.putText(img, text, (text_x, text_y), font,
                font_scale, font_color, font_thickness)


def imshow(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bird_fit(binary_warped, ret, save_file=None):
    """
    Visualize the predicted lane lines with margin, on binary warped image
    save_file is a string representing where to save the image (if None, then just display)
    """
    # Grab variables from ret dictionary
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # Create an image to draw on and an image to show the selection window
    out_img = (np.dstack((binary_warped, binary_warped,
               binary_warped))*255).astype('uint8')
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # imshow("out_img", out_img)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    margin = 100  # NOTE: Keep this in sync with *_fit()
    left_line_window1 = np.array(
        [np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array(
        [np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # imshow("window_img", window_img)

    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    # cv2.imshow('bird',result)
    # cv2.imwrite('bird_from_cv2.png', result)

    # if save_file is None:
    #     plt.show()
    # else:
    #     plt.savefig(save_file)
    # plt.gcf().clear()

    return result


def final_viz(undist, left_fit, right_fit, m_inv):
    """
    Final lane line prediction visualized and overlayed on top of original image
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    # warp_zero = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # NOTE: Hard-coded image dimensions
    color_warp = np.zeros((720, 1280, 3), dtype='uint8')

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    # Convert arrays to 8 bit for later cv to ros image transfer
    undist = np.array(undist, dtype=np.uint8)
    newwarp = np.array(newwarp, dtype=np.uint8)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


def gradient_thresh(img, thresh_min=grad_thres_min, thresh_max=grad_thres_max):
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

    # closing
    # kernel = np.ones((5, 5), np.uint8)
    # binary_output = cv2.morphologyEx(
    #     binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # vis
    # if args.vis_mode:
    #     vis = cv2.cvtColor(binary_output*255, cv2.COLOR_GRAY2BGR)
    #     imshow("binary_output", cv2.hconcat([img, vis]))

    return binary_output


def color_thresh(img, val_thres):
    """
    Convert RGB to HSL and threshold to binary image using S channel
    """
    # 1. Convert the image from RGB to HSL
    # 2. Apply threshold on S channel to get binary image
    # Hint: threshold on H to remove green grass
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = img[:, :, 2] # red channel

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

    ### visualization three channels result ###
    # vis = np.zeros_like(l)
    # vis[sat_cond] = 255
    # imshow("sat", vis)
    # vis = np.zeros_like(l)
    # vis[val_cond] = 255
    # imshow("val", vis)
    # vis = np.zeros_like(l)
    # vis[hue_cond] = 255
    # imshow("hue", vis)
    
    ### visualization for hue testing ###
    if args.vis_hue:
        OUTPUT_DIR = "./hue-test"
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        step = 10
        MAX_HUE = 180
        for i in range(0, MAX_HUE - step, step):
            mask = cv2.inRange(hls_img[:, :, 0], i, i + step)
            mask[mask > 0] = 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis = cv2.hconcat([img, mask])
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'L{}.jpg'.format(i)), vis)

    # closing
    kernel = np.ones((5, 5), np.uint8)
    binary_output = cv2.morphologyEx(
        binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return binary_output


def combinedBinaryImage(SobelOutput, ColorOutput):
    """
    Get combined binary image from color filter and sobel filter
    """
    # 1. Apply sobel filter and color filter on input image
    # 2. Combine the outputs
    # Here you can use as many methods as you want.

    # SobelOutput = gradient_thresh(img)
    # ColorOutput = color_thresh(
    #     img, val_thres_min, val_thres_max, sat_thres_min, sat_thres_max)

    # img[ColorOutput == 1] = (0, 0, 255)
    # img[SobelOutput == 1] = (0, 255, 0)
    # imshow("img", img)

    # imshow("ColorOutput", ColorOutput*255)
    # if args.dilate_size > 0:
    #     kernel = cv2.getStructuringElement(
    #         cv2.MORPH_ELLIPSE, (args.dilate_size, args.dilate_size))
    #     ColorOutput = cv2.dilate(ColorOutput, kernel, iterations=1)
    # imshow("dilate ColorOutput", ColorOutput*255)

    # use color and gradient
    binaryImage = np.zeros_like(ColorOutput)
    binaryImage[(ColorOutput == 1) & (SobelOutput == 1)] = 1
    # vis_SobelOutput = cv2.cvtColor(SobelOutput*255, cv2.COLOR_GRAY2BGR)
    # putText(vis_SobelOutput, "gradient thres")
    # vis_ColorOutput = cv2.cvtColor(ColorOutput*255, cv2.COLOR_GRAY2BGR)
    # putText(vis_ColorOutput, "color thres")
    # concat = cv2.vconcat([img, vis_SobelOutput, vis_ColorOutput])

    # Remove noise from binary image
    # kernel = np.ones((5, 5), np.uint8)
    # binaryImage = cv2.morphologyEx(
    #     binaryImage.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # imshow("closing", binaryImage*255)

    # imshow("combined result", concat)
    # cv2.imwrite(os.path.join(TMP_DIR, 'combined.png'), concat)
    return binaryImage


def perspective_transform(img, raw_img):
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

    ### vis poly lines ###
    vis_src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis_dst = cv2.warpPerspective(
        raw_img, M, (dst_width, dst_height), flags=cv2.INTER_NEAREST)
    # concat = cv2.vconcat([raw_img, vis_dst])
    # if args.vis_mode:
    #     imshow("img", concat)

    return warped_img, M, Minv


def plotHist(hist, suffix):
    # Create a line chart for the histogram
    plt.plot(range(len(hist)), hist, color='blue', linestyle='-')
    plt.title('Histogram')
    plt.xlabel('x-axis')
    plt.ylabel('# of nonzero')

    # Display the line chart
    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(TMP_DIR, 'hist_{}.png'.format(suffix)))
    plt.clf()


def fixedAspectRatioResize(img, desired_height=None, desired_width=None):
    if desired_width:
        # Calculate the new height to maintain the aspect ratio
        height, width, _ = img.shape
        aspect_ratio = width / height
        desired_height = int(desired_width / aspect_ratio)

        # Resize the image
        return cv2.resize(img, (desired_width, desired_height))
    else:
        exit(1)
        return -1


def getCurvaturePts(x_arr, coeffs):
    x_fit = np.linspace(np.min(x_arr), np.max(x_arr), 100)
    y_fit = np.polyval(coeffs, x_fit)
    return np.array(list(zip(x_fit, y_fit)))


def drawCurvature(color_warped, x_arr, coeffs, thickness=10):
    # Generate points on the fitted curve
    x_fit = np.linspace(np.min(x_arr), np.max(x_arr), 100)
    y_fit = np.polyval(coeffs, x_fit)
    curve = color_warped
    for i in range(len(x_fit)-1):
        curve = cv2.line(curve, (int(x_fit[i]), int(y_fit[i])),
                         (int(x_fit[i+1]), int(y_fit[i+1])),
                         (0, 0, 255), thickness)
    return curve


def line_fit(binary_warped, histogram, raw_img_warped):
    """
    Find and fit lane lines
    """
    ### 1. sliding window to find the base point
    height, width = binary_warped.shape
    sliding_offset = 5
    margin = 30
    best_left_base_x = -1
    best_right_base_x = -1
    best_num_pixels = -1
    best_even_ratio = -1
    side_xdist = 120
    assert margin*2 < side_xdist
    for leftbase in range(margin, width-margin-side_xdist, sliding_offset):
        rightbase = leftbase + side_xdist
        left_num_pixels = np.sum(histogram[leftbase-margin:leftbase+margin])
        right_num_pixels = np.sum(histogram[rightbase-margin:rightbase+margin])
        total_num_pixels = left_num_pixels + right_num_pixels
        if total_num_pixels == 0:
            continue
        even_ratio = float(left_num_pixels) / total_num_pixels * float(right_num_pixels) / total_num_pixels

        # We matter even_ratio more than num_pixels
        if even_ratio > best_even_ratio and total_num_pixels > best_num_pixels * 0.8:
            best_even_ratio = even_ratio
            best_num_pixels = total_num_pixels
            best_left_base_x = leftbase
            best_right_base_x = rightbase

    if best_left_base_x == -1:
        return None
    
    blurred_img = cv2.GaussianBlur(raw_img_warped, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)

    window_height = 20
    nwindows = 15
    
    ### 2. correct base point
    leftx_base = np.argmax(histogram[best_left_base_x-margin:best_left_base_x+margin]) + best_left_base_x-margin
    rightx_base = np.argmax(histogram[best_right_base_x-margin:best_right_base_x+margin]) + best_right_base_x-margin
    # rightx_base = np.argmax(histogram[best_base_x:best_base_x+margin]) + best_base_x
    print("leftx_base:", leftx_base)
    print("rightx_base:", rightx_base)
    # vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    # vis = cv2.rectangle(
    #     vis, (leftx_base, height - window_height), (rightx_base, height), (0, 0, 255))
    # imshow("vis", vis)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 15
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    color_warped = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    color_warped[color_warped > 0] = 255
    
    def correct(currentx, start_y, end_y, func):
        y = (start_y + end_y) // 2
        
        grads = sobel_x.copy()
        grads[(binary_warped == 0)] = 0
        grads = grads[y, currentx-margin:currentx+margin]
        
        idx = func(grads)
        return idx + currentx - margin
        
    lane_pts = []
    for i in range(nwindows):
        win_top = binary_warped.shape[0] - (i + 1) * window_height
        win_bottom = win_top + window_height
        
        try:
            leftx_current = correct(leftx_current, win_top, win_bottom, np.argmax)
            rightx_current = correct(rightx_current, win_top, win_bottom, np.argmin)
        except:
            print ("Lane reaches the boundary.")
            break
        lane_pt = ((leftx_current + rightx_current) // 2, (win_top + win_bottom) // 2)
        lane_pts.append(lane_pt)
        
        # Identify window boundaries in x and y (and right and left)
        # left
        left_lt = [leftx_current - margin, win_top]
        left_rb = [leftx_current + margin, win_bottom]
        # right
        right_lt = [rightx_current - margin, win_top]
        right_rb = [rightx_current + margin, win_bottom]

        ### Draw the windows on the visualization image using cv2.rectangle() ###
        color_warped = cv2.rectangle(
            color_warped, left_lt, left_rb, (0, 255, 0))
        color_warped = cv2.rectangle(
            color_warped, right_lt, right_rb, (0, 255, 0))
        # imshow("color_warped", color_warped)

    ### vis color_warped ###
    putText(color_warped, "warp & lanefit")
    
    lanex = [pt[0] for pt in lane_pts]
    laney = [pt[1] for pt in lane_pts]
    try:
        lane_fit = np.polyfit(laney, lanex, deg=2)
        
        ### vis lane points ###
        for x, y in zip(lanex, laney):
            color_warped = cv2.circle(color_warped, (x, y), 1, (0,255, 0), -1)
            
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


def findContourForColor(color_warped):
    # Find the contours for color (ideeally two contours along the trajectory)
    contours, _ = cv2.findContours(
        color_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_canvas = np.zeros_like(color_warped)
    cv2.drawContours(contour_canvas, contours, -1, 1, 1)

    # remove contours along the image border
    contour_canvas[-5:, :] = 0
    contour_canvas[:5, :] = 0

    # imshow("binary_warped", binary_warped)
    return contour_canvas


def getHistogram(binary_warped):
    # Draw the contours on the copy of the original image
    histogram = np.sum(binary_warped[-args.hist_y_begin:, :], axis=0)
    return histogram

def run(img_path):
    global img_name
    img_name = img_path.strip('.png').split('/')[-1]
    print("img_path:", img_path)

    img = cv2.imread(img_path)
    # gray_img = cv2.imread(img_path, 0)
    gray_img = img[:, :, 2] # red channel
    gray_img_warped, M, Minv = perspective_transform(gray_img, img)
    
    # val_mean = np.mean(gray_img_warped)
    val_thres = np.percentile(gray_img_warped, args.val_thres_percentile)
    print ("mean:", val_thres)
    if args.vis_mode:
        imshow("gray_img_warped", gray_img_warped)
    
    
    SobelOutput = gradient_thresh(img)
    ColorOutput = color_thresh(img, val_thres)
    combinedOutput = combinedBinaryImage(SobelOutput, ColorOutput)
    sobel_warped, M, Minv = perspective_transform(SobelOutput, img)
    color_warped, M, Minv = perspective_transform(ColorOutput, img)
    color_warped = findContourForColor(color_warped)
    combined_warped, M, Minv = perspective_transform(combinedOutput, img)
    

    # imshow("color_warped", color_warped*255)
    
    if args.base_algo == "color":
        color_hist = getHistogram(color_warped)
        hist = color_hist.copy()
        contour_warped = color_warped.copy()
    elif args.base_algo == "grad":
        sobel_hist = getHistogram(sobel_warped)
        hist = sobel_hist.copy()
        contour_warped = sobel_warped.copy()
    elif args.base_algo == "combine":
        combined_hist = getHistogram(combined_warped)
        hist = combined_hist.copy()
        contour_warped = combined_warped.copy()

    plotHist(hist, "target")
    vis_hist = fixedAspectRatioResize(
        cv2.imread(os.path.join(TMP_DIR, 'hist_target.png')), desired_width=color_warped.shape[1])
    if args.vis_mode:
        imshow("vis_hist", vis_hist)
    
    
    ### visualization hist ###
    # plotHist(sobel_hist, "grad")
    # plotHist(color_hist, "color")
    # plotHist(combined_hist, "combined")

    # vis_hist_grad = fixedAspectRatioResize(
    #     cv2.imread(os.path.join(TMP_DIR, 'hist_grad.png')), desired_width=color_warped.shape[1])
    # vis_hist_color = fixedAspectRatioResize(
    #     cv2.imread(os.path.join(TMP_DIR, 'hist_color.png')), desired_width=color_warped.shape[1])
    # vis_hist_combined = fixedAspectRatioResize(
    #     cv2.imread(os.path.join(TMP_DIR, 'hist_combined.png')), desired_width=color_warped.shape[1])

    # vis_grad = np.hstack(
    #     (cv2.cvtColor(sobel_warped*255, cv2.COLOR_GRAY2BGR), vis_hist_grad))
    # vis_color = np.hstack(
    #     (cv2.cvtColor(color_warped*255, cv2.COLOR_GRAY2BGR), vis_hist_color))
    # vis_combined = np.hstack(
    #     (cv2.cvtColor(combined_warped*255, cv2.COLOR_GRAY2BGR), vis_hist_combined))
    # if args.vis_mode:
    #     imshow("vis_grad", vis_grad)
    #     imshow("vis_color", vis_color)
    #     imshow("vis_combined", vis_combined)

    ret = line_fit(contour_warped, hist, gray_img_warped)
    if ret is None:
        print ("Fail to fit line")
        exit(1)
    
    def get_waypoints(ret, width, height, look_ahead_dist = 1.0):
        lanex = ret['lanex']
        laney = ret['laney']
        
        # transform from image coord (x, y) to camera coord in meters
        lanex = [(x - width // 2) * PIX2METER_X for x in lanex]
        laney = [(height - y) * PIX2METER_Y + DIST_CAM2FOV_INCH * INCH2METER for y in laney]
        # lane_fit = np.polyfit(laney, lanex, deg=2)
        for x, y in zip(lanex, laney):
            print (x, y)
        way_pts = [(y, -x) for x, y in zip(lanex, laney)]
        return way_pts
        
    height, width = img.shape[:2]
    way_pts = get_waypoints(ret, width, height, look_ahead_dist = 1.0)
    
    ### vis all ###
    SobelOutput = cv2.cvtColor(SobelOutput*255, cv2.COLOR_GRAY2BGR)
    ColorOutput = cv2.cvtColor(ColorOutput*255, cv2.COLOR_GRAY2BGR)
    concat = cv2.vconcat([img, SobelOutput, ColorOutput, vis_hist, ret['vis_warped']])
    if args.vis_mode:
        imshow("warped", ret['vis_warped'])
        # imshow("concat", concat)
    if args.vis_output:
        cv2.imwrite(os.path.join(
            TMP_DIR, 'result_{}.png').format(img_name), concat)


if __name__ == '__main__':

    print('======= Initial parameters =======')
    params = []
    for key, val in vars(args).items():
        param = f"{key} => {val}"
        print(param)
        params.append(param)

    pathlib.Path(TMP_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(TMP_DIR, 'a_params.txt'), 'w') as f:
        f.write('\n'.join(params))

    if args.specified_name:
        img_path = os.path.join(TEST_DIR, '{}.png'.format(args.specified_name))
        run(img_path)
    else:
        paths = sorted(os.listdir(TEST_DIR))
        for i, path in enumerate(paths):
            if i == args.num_samples:
                break
            if not path.endswith('png'):
                continue

            img_path = os.path.join(TEST_DIR, path)
            run(img_path)
