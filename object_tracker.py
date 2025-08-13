import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import re
from scipy.optimize import linear_sum_assignment

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

class KalmanFilter:
    def __init__(self):
        self.A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.R = np.eye(2, dtype=np.float32) * 1
        self.state = np.zeros((4, 1), np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict(self):
        if self.initialized:
            return np.dot(self.A, self.state)
        return self.state

    def step(self):
        if self.initialized:
            self.state = np.dot(self.A, self.state)
            self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def correct(self, measurement):
        measurement = np.array(measurement, dtype=np.float32).reshape(2, 1)
        if not self.initialized:
            self.state[0:2] = measurement
            self.initialized = True
        else:
            y = measurement - np.dot(self.H, self.state)
            S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
            self.state = self.state + np.dot(K, y)
            self.P = self.P - np.dot(np.dot(K, self.H), self.P)

class Tracker:
    def __init__(self, tracker_id, initial_box):
        self.id = tracker_id
        self.kf = KalmanFilter()
        self.box = initial_box
        self.misses = 0
        center_x = initial_box[0] + initial_box[2] / 2
        center_y = initial_box[1] + initial_box[3] / 2
        self.kf.correct((center_x, center_y))

    def predict(self):
        return self.kf.predict()
    
    def step(self):
        self.kf.step()

    def update(self, box):
        self.box = box
        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2
        self.kf.correct((center_x, center_y))
        self.misses = 0

    def mark_missed(self):
        self.misses += 1

class MultiObjectTracker:
    def __init__(self):
        self.prototxt = "mobilenet_ssd.prototxt"
        self.model = "mobilenet_iter_73000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        
        self.trackers = []
        self.next_tracker_id = 0
        self.max_misses = 7
        # 直接记录所有点
        self.all_real_points = []
        self.all_kalman_points = []

    def process_frame(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        detected_boxes = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            if self.CLASSES[idx] == "person" and confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                detected_boxes.append([startX, startY, endX - startX, endY - startY])

        for t in self.trackers:
            t.step()

        matched_trackers = set()
        if detected_boxes:
            cost_matrix = np.zeros((len(self.trackers), len(detected_boxes)))
            for i, tracker in enumerate(self.trackers):
                predicted_state = tracker.predict()
                pred_x, pred_y = predicted_state[0,0], predicted_state[1,0]
                for j, box in enumerate(detected_boxes):
                    det_x, det_y = box[0] + box[2]/2, box[1] + box[3]/2
                    cost_matrix[i, j] = np.sqrt((pred_x - det_x)**2 + (pred_y - det_y)**2)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_detections = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 75: # 放宽距离阈值
                    self.trackers[r].update(detected_boxes[c])
                    matched_trackers.add(r)
                    matched_detections.add(c)

            unmatched_detections = set(range(len(detected_boxes))) - matched_detections
            for i in unmatched_detections:
                self.trackers.append(Tracker(self.next_tracker_id, detected_boxes[i]))
                self.next_tracker_id += 1
        
        for i in range(len(self.trackers)):
            if i not in matched_trackers:
                self.trackers[i].mark_missed()

        self.trackers = [t for t in self.trackers if t.misses <= self.max_misses]

        for tracker in self.trackers:
            x, y, w, h = tracker.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tracker.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            pred_x, pred_y = int(tracker.kf.state[0,0]), int(tracker.kf.state[1,0])
            cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)
            
            # 记录所有点
            self.all_real_points.append((x + w/2, y + h/2))
            self.all_kalman_points.append((pred_x, pred_y))
        
        return frame

def plot_trajectories(real_points, kalman_points):
    plt.figure(figsize=(12, 9))
    
    if real_points:
        real_x, real_y = zip(*real_points)
        plt.scatter(real_x, real_y, s=10, color='blue', label='Detected Points')
        
    if kalman_points:
        kalman_x, kalman_y = zip(*kalman_points)
        plt.scatter(kalman_x, kalman_y, s=10, color='red', label='Kalman Filtered Points')
            
    plt.title('Detected vs. Kalman Filtered Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.savefig('trajectory_plot.png')
    print("Trajectory plot saved to trajectory_plot.png")

if __name__ == '__main__':
    multi_tracker = MultiObjectTracker()
    
    input_dir = 'images'
    output_dir = 'output_images'
    video_output_path = 'tracking_video.mp4'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))], key=natural_sort_key)

    if not image_files:
        print(f"Error: No images found in the '{input_dir}' directory.")
    else:
        first_frame = cv2.imread(os.path.join(input_dir, image_files[0]))
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output_path, fourcc, 10.0, (width, height))

        for image_file in image_files:
            frame = cv2.imread(os.path.join(input_dir, image_file))
            if frame is None:
                continue
            tracked_frame = multi_tracker.process_frame(frame)
            video_writer.write(tracked_frame)
            cv2.imwrite(os.path.join(output_dir, image_file), tracked_frame)
            print(f"Processed {image_file}")

        video_writer.release()
        print(f"Tracking video saved to {video_output_path}")
        
        plot_trajectories(multi_tracker.all_real_points, multi_tracker.all_kalman_points)
        print("Processing complete.")
