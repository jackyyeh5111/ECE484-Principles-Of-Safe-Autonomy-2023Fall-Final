import cv2
import sys

# Create a callback function to capture mouse events

img_name = sys.argv[1]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        # Get the pixel color at the current (x, y) coordinates
        pixel_color = img[y, x]
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(
            f"Mouse Hover - X: {x}, Y: {y}, RGB Color: {pixel_color} | HLS Color: {hls_img[y][x]} | HSV Color: {hsv_img[y][x]}")


# Load an image
# PATH = "/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/assignments/MP1/test_imgs/{}.png".format(img_name)
PATH = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/frames/{}.png'.format(img_name)
img = cv2.imread(PATH)

# Create a window to display the image
cv2.namedWindow('Image')

# Set the mouse callback function for the window
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
