import cv2
import numpy as np

# Function to compute average motion vector using Farneback optical flow
def compute_average_motion_vector(prev_frame, next_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute average motion vector
    avg_motion_vector = np.mean(flow, axis=(0, 1))

    return avg_motion_vector

# Loop through the video frames

prev_path = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/frames/frame000365.png'
next_path = '/Users/jackyyeh/Desktop/Courses/UIUC/ECE484-Principles-Of-Safe-Autonomy/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/f1tenth_ros1_ws/src/frames/frame000370.png'
prev_frame = cv2.imread(prev_path)
next_frame = cv2.imread(next_path)

# prev_frame = cv2.imread(next_path)
# next_frame = cv2.imread(prev_path)

# Initialize the total motion vector
total_motion_vector = np.zeros(2)

while True:
    
    # Compute the average motion vector
    avg_motion_vector = compute_average_motion_vector(prev_frame, next_frame)
    print ('avg_motion_vector:', avg_motion_vector)
    
    # Update the total motion vector
    total_motion_vector += avg_motion_vector

    # Display the frame with motion vectors (optional)
    draw_frame = next_frame.copy()
    draw_frame = cv2.putText(draw_frame, f'Motion Vector: {avg_motion_vector}', (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Video with Motion Vectors', draw_frame)

    # Update the previous frame
    prev_frame = next_frame

    exit(1)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cv2.destroyAllWindows()

# Calculate and print the average motion vector over all frames
average_motion_vector = total_motion_vector / cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f'Average Motion Vector: {average_motion_vector}')
