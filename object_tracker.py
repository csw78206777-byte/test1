import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# --- 卡尔曼滤波器设置 ---
dt = 1.0
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

# --- 文件夹和路径 ---
input_folder = 'test_images'
output_folder = 'output_images'
output_video_path = 'tracking_video.mp4'
output_plot_path = 'trajectory_plot.png'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- 数据收集 ---
image_files = sorted(glob.glob(os.path.join(input_folder, '*.png')))
trajectory = []
detected_points = []
kalman_points = []

# --- 视频写入器设置 ---
first_image = cv2.imread(image_files[0])
height, width, _ = first_image.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 10.0, (width, height))

first_frame = True

for img_path in image_files:
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    # --- 物体检测 ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detection_center = None
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        detection_center = np.array([x + w / 2, y + h / 2], dtype=np.float32)
        detected_points.append(detection_center)
    else:
        detected_points.append(None) # 如果没有检测到，也记录下来

    # --- 卡尔曼滤波 ---
    predicted_state = kalman.predict()
    px, py = int(predicted_state[0]), int(predicted_state[1])
    cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    if detection_center is not None:
        if first_frame:
            kalman.statePost = np.array([detection_center[0], detection_center[1], 0, 0], dtype=np.float32)
            first_frame = False
        else:
            kalman.correct(detection_center)
    
    corrected_state = kalman.statePost
    cx, cy = int(corrected_state[0]), int(corrected_state[1])
    kalman_points.append((cx, cy))
    
    trajectory.append((cx, cy))
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

    # 写入视频和图片
    video_writer.write(frame)
    output_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(output_path, frame)

video_writer.release()
print(f"Processing complete. Output video saved to '{output_video_path}'")
print(f"Output images saved in '{output_folder}' folder.")

# --- 绘制图表 ---
frames = range(len(image_files))
detected_x = [p[0] if p is not None else np.nan for p in detected_points]
detected_y = [p[1] if p is not None else np.nan for p in detected_points]
kalman_x = [p[0] for p in kalman_points]
kalman_y = [p[1] for p in kalman_points]

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(frames, detected_x, 'g.', label='Detected X')
plt.plot(frames, kalman_x, 'b-', label='Kalman Filter X')
plt.title('X Coordinate vs. Frame')
plt.xlabel('Frame')
plt.ylabel('X Coordinate')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frames, detected_y, 'g.', label='Detected Y')
plt.plot(frames, kalman_y, 'b-', label='Kalman Filter Y')
plt.title('Y Coordinate vs. Frame')
plt.xlabel('Frame')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(output_plot_path)
print(f"Trajectory plot saved to '{output_plot_path}'")