import time
import math
import numpy as np
import cv2
from skimage import morphology
import argparse
import matplotlib.pyplot as plt
import pathlib
import os

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
parser.add_argument('--sat_thresh', type=str, default='50,255')
parser.add_argument('--val_thresh', type=str, default='80,255')
parser.add_argument('--hue_thresh', type=str, default='10,40')
parser.add_argument('--dilate_size', type=int, default=5)
parser.add_argument('--hist_y_begin', type=int, default=30)
parser.add_argument('--perspective_pts', '-p',
                    type=str, default='218,467,348,0')
# parser.add_argument('--algo', type=str, required=True)

args = parser.parse_args()

TEST_DIR = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/frames'
TMP_DIR = './vis_{}'.format(args.append_str)
grad_thres_min, grad_thres_max = args.gradient_thresh.split(',')
grad_thres_min, grad_thres_max = int(grad_thres_min), int(grad_thres_max)
assert grad_thres_min < grad_thres_max

val_thres_min, val_thres_max = args.val_thresh.split(',')
val_thres_min, val_thres_max = int(val_thres_min), int(val_thres_max)
assert val_thres_min < val_thres_max

sat_thres_min, sat_thres_max = args.sat_thresh.split(',')
sat_thres_min, sat_thres_max = int(sat_thres_min), int(sat_thres_max)

hue_thres_min, hue_thres_max = args.hue_thresh.split(',')
hue_thres_min, hue_thres_max = int(hue_thres_min), int(hue_thres_max)
assert hue_thres_min < hue_thres_max

src_leftx, src_rightx, laney, offsety = args.perspective_pts.split(',')
src_leftx, src_rightx, laney, offsety = int(
    src_leftx), int(src_rightx), int(laney), int(offsety)

img_name = ""


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


def color_thresh(img):
    """
    Convert RGB to HSL and threshold to binary image using S channel
    """
    # 1. Convert the image from RGB to HSL
    # 2. Apply threshold on S channel to get binary image
    # Hint: threshold on H to remove green grass
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # For HSL
    # ref: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_hls
    #   image format: (8-bit images) V ← 255⋅V, S ← 255⋅S, H ← H/2(to fit to 0 to 255)

    # Step 2: Apply threshold on the S (Saturation) channel to get a binary image
    h, l, s = cv2.split(hls_img)
    binary_output = np.zeros_like(l)
    sat_cond = ((sat_thres_min <= s) & (s <= sat_thres_max)) | (s == 0)
    val_cond = (val_thres_min <= l) & (l <= val_thres_max)
    hue_cond = (hue_thres_min <= h) & (h <= hue_thres_max)
    binary_output[val_cond & sat_cond & hue_cond] = 1

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


def line_fit(binary_warped, histogram, color_warped, img):
    """
    Find and fit lane lines
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image

    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped,
               binary_warped))*255).astype('uint8')

    # # sliding window
    # height, width = binary_warped.shape
    # sliding_offset = 5
    # margin = 30
    # best_left_base_x = -1
    # best_right_base_x = -1
    # best_num_pixels = -1
    # best_even_ratio = -1
    # side_xdist = 120
    # assert margin*2 < side_xdist
    # for leftbase in range(margin, width-margin-side_xdist, sliding_offset):
    #     rightbase = leftbase + side_xdist
    #     left_num_pixels = np.sum(histogram[leftbase-margin:leftbase+margin])
    #     right_num_pixels = np.sum(histogram[rightbase-margin:rightbase+margin])
    #     total_num_pixels = left_num_pixels + right_num_pixels
    #     if total_num_pixels == 0:
    #         continue
    #     even_ratio = float(left_num_pixels) / total_num_pixels * float(right_num_pixels) / total_num_pixels

    #     # We matter even_ratio more than num_pixels
    #     if even_ratio > best_even_ratio and total_num_pixels > best_num_pixels * 0.8:
    #         best_even_ratio = even_ratio
    #         best_num_pixels = total_num_pixels
    #         best_left_base_x = leftbase
    #         best_right_base_x = rightbase

    # sliding window
    height, width = binary_warped.shape
    sliding_offset = 5
    window_width = 180
    diff_grad_thres = 20
    window_height = 20
    margin = 30
    side_xdist = 120

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_warped, M, Minv = perspective_transform(gray_img, img)
    blurred_img = cv2.GaussianBlur(gray_warped, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)

    def getSidePoints(y):
        max_num_pixel = -1
        base_leftx = -1
        base_rightx = -1
        is_success = False

        # image border contains gradient noise, so +-10 pixels here
        for leftbase in range(margin, width-margin-side_xdist, sliding_offset):
            rightbase = leftbase + side_xdist
            grads = sobel_x[y]

            # min_grad, max_grad, min_loc, max_loc = cv2.minMaxLoc(grads)
            
            leftx = np.argmax(grads[leftbase-margin:leftbase+margin]) +leftbase-margin# left part
            rightx = np.argmin(grads[rightbase-margin:rightbase+margin]) +rightbase-margin# right part
            min_grad = grads[rightx]
            max_grad = grads[leftx]
            
            diff_grad = max_grad - min_grad

            num_pixel = np.count_nonzero(color_warped[y, leftx:rightx])
            # num_pixels = np.sum(histogram[leftbase-margin:leftbase+margin])

            ### visualize box ###
            print ('----------')
            print ("min loc(right):{} min_grad:".format(rightx), min_grad)
            print ("max loc(left):{} max_grad:".format(leftx), max_grad)
            print ("num_pixel:", num_pixel)
            print ("diff_grad:", diff_grad)
            
            # vis = img.copy()
            # vis = cv2.rectangle(
            #     vis, (leftx, y), (rightx, y), (0, 0, 255))
            # imshow("vis", vis)

            # vis = cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
            if num_pixel > max_num_pixel and diff_grad > diff_grad_thres:
                base_leftx = leftx
                base_rightx = rightx
                max_num_pixel = num_pixel
                is_success = True
                
                # print ('----------')
                # print ("left loc:{} max_grad:".format(leftx), max_grad)
                # print ("right loc:{} min_grad:".format(rightx), min_grad)
                # print ("num_pixel:", num_pixel)

                # vis = img.copy()
                # vis = cv2.rectangle(vis, (leftx, y), (rightx, y), (0, 0, 255))
                # imshow("vis", vis)

        return is_success, base_leftx, base_rightx

    start_y = height - 5
    is_success, best_left_base_x, best_right_base_x = getSidePoints(start_y)
    print ("best_left_base_x:", best_left_base_x)
    print ("best_right_base_x:", best_right_base_x)
    # exit(1)
    
    if not is_success:
        assert ("Cannot get base points")

    # Choose the number of sliding windows
    nwindows = 15
    # Set height of windows
    # window_height = int(binary_warped.shape[0]/nwindows)
    window_height = 20

    # visualization base

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # midpoint = int(histogram.shape[0]/2)
    # leftx_base = np.argmax(histogram[100:midpoint]) + 100
    # rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint
    
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
    for i in range(nwindows):
        win_top = binary_warped.shape[0] - (i + 1) * window_height
        win_bottom = win_top + window_height

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

        ####
        # Identify the nonzero pixels in x and y within the window
        left_window_inds = np.where((nonzerox > leftx_current - margin) &
                                    (nonzerox < leftx_current + margin) &
                                    (nonzeroy > win_top) &
                                    (nonzeroy < win_bottom))
        right_window_inds = np.where((nonzerox > rightx_current - margin) &
                                     (nonzerox < rightx_current + margin) &
                                     (nonzeroy > win_top) &
                                     (nonzeroy < win_bottom))
        ####
        # Append these indices to the lists
        left_lane_inds.append(left_window_inds[0])
        right_lane_inds.append(right_window_inds[0])

        ####
        # If you found > minpix pixels, recenter next window on their mean position
        left_nonzerox = nonzerox[left_window_inds]
        if len(left_nonzerox) > minpix:
            leftx_current = int(np.mean(left_nonzerox))

        right_nonzerox = nonzerox[right_window_inds]
        if len(right_nonzerox) > minpix:
            rightx_current = int(np.mean(right_nonzerox))

    ### vis color_warped ###
    putText(color_warped, "warp & lanefit")
    # cv2.imwrite(os.path.join(TMP_DIR, 'warped.png'), color_warped)
    # imshow("color_warped", color_warped)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each using np.polyfit()
    # If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
    # the second order polynomial is unable to be sovled.
    # Thus, it is unable to detect edges.
    try:
        # left_fit = np.polyfit(leftx, lefty, deg=2)
        # right_fit = np.polyfit(rightx, righty, deg=2)
        left_fit = np.polyfit(lefty, leftx, deg=2)
        right_fit = np.polyfit(righty, rightx, deg=2)

        ### vis points nonzero ###
        # for x, y in zip(rightx, righty):
        #     color_warped = cv2.circle(color_warped, (x, y), 1, (0,255, 0), -1)
        # imshow("points", color_warped )

        ### vis draw curvature ###
        # drawCurvature(color_warped, leftx, left_fit)
        # drawCurvature(color_warped, rightx, right_fit)

        ### vis fill poly ###
        # coordinates = np.array(list(zip(leftx, lefty)))
        # cv2.fillPoly(color_warped, [coordinates], (0, 255, 0))
        # newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0]))
        # imshow("fill_poly", color_warped)

    except TypeError:
        print("Unable to detect lanes")
        return None

    return color_warped


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
    SobelOutput = gradient_thresh(img)
    ColorOutput = color_thresh(img)
    combinedOutput = combinedBinaryImage(SobelOutput, ColorOutput)
    sobel_warped, M, Minv = perspective_transform(SobelOutput, img)
    color_warped, M, Minv = perspective_transform(ColorOutput, img)
    color_contour_warped = findContourForColor(color_warped)
    combined_warped, M, Minv = perspective_transform(combinedOutput, img)

    # imshow("color_warped", color_warped*255)
    sobel_hist = getHistogram(sobel_warped)
    color_hist = getHistogram(color_contour_warped)
    combined_hist = getHistogram(combined_warped)
    
    hist = combined_hist.copy()
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

    warped_fit = line_fit(combined_warped, hist, color_warped, img)

    # color_warped = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)
    # leftx = ret['nonzerox'][ret['left_lane_inds']]
    # rightx = ret['nonzerox'][ret['right_lane_inds']]

    ### vis curvature ###
    # vis_bird_fit = bird_fit(binary_warped, ret)
    # imshow("bird_fit", result)
    # img_with_curve = final_viz(img, ret['left_fit'], ret['right_fit'], Minv)
    # imshow("final_viz", img_with_curve)

    ### vis curvature ###
    # left_curve = drawCurvature(color_warped, leftx, ret['left_fit'])
    # right_curve = drawCurvature(color_warped, rightx, ret['right_fit'])
    # left_curve = cv2.warpPerspective(
    #     left_curve, Minv, (img.shape[1], img.shape[0]))
    # right_curve = cv2.warpPerspective(
    #     right_curve, Minv, (img.shape[1], img.shape[0]))
    # img_with_curve = cv2.addWeighted(img, 1, left_curve + right_curve, 0.5, 0)

    ### vis filled lane ###
    # left_pts = getCurvaturePts(leftx, ret['left_fit'])
    # right_pts = getCurvaturePts(rightx, ret['right_fit'])
    # pts = np.vstack([left_pts, right_pts])
    # canvas = np.zeros_like(img)
    # cv2.fillPoly(canvas, np.int_([pts]), (0, 255, 0))
    # canvas = cv2.warpPerspective(canvas, Minv, (img.shape[1], img.shape[0]))
    # img_with_curve = cv2.addWeighted(img_with_curve, 1, canvas, 0.3, 0)

    ### vis all ###
    SobelOutput = cv2.cvtColor(SobelOutput*255, cv2.COLOR_GRAY2BGR)
    ColorOutput = cv2.cvtColor(ColorOutput*255, cv2.COLOR_GRAY2BGR)
    concat = cv2.vconcat([img, SobelOutput, ColorOutput, warped_fit])
    if args.vis_mode:
        imshow("concat", concat)
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
