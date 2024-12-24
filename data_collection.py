import os
import cv2
import dlib
import math
import json
import statistics
from PIL import Image
import imageio.v2 as imageio
import numpy as np
from collections import deque

# Constants as specified
TOTAL_FRAMES = 22
VALID_WORD_THRESHOLD = 1
NOT_TALKING_THRESHOLD = 10
PAST_BUFFER_SIZE = 4
LIP_WIDTH = 112
LIP_HEIGHT = 80

class LipReadingDataCollector:
    def __init__(self, predictor_path="../model/face_weights.dat"):
        # Initialize face and landmark detection
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(predictor_path)
        
        # Configuration parameters using the provided constants
        self.TOTAL_FRAME_COUNT = TOTAL_FRAMES
        self.VALID_WORD_MIN_FRAMES = VALID_WORD_THRESHOLD
        self.SILENCE_FRAME_THRESHOLD = NOT_TALKING_THRESHOLD
        self.PAST_BUFFER_SIZE = PAST_BUFFER_SIZE
        self.LIP_OUTPUT_WIDTH = LIP_WIDTH
        self.LIP_OUTPUT_HEIGHT = LIP_HEIGHT

    def _preprocess_lip_region(self, frame, landmarks):
        """
        Extract and preprocess the lip region from the frame
        """
        # Identify lip boundaries
        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        # Calculate padding
        width_diff = self.LIP_OUTPUT_WIDTH - (lip_right - lip_left)
        height_diff = self.LIP_OUTPUT_HEIGHT - (lip_bottom - lip_top)
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top

        # Ensure that the padding doesn't extend beyond the original frame
        pad_left = min(pad_left, lip_left)
        pad_right = min(pad_right, frame.shape[1] - lip_right)
        pad_top = min(pad_top, lip_top)
        pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

        # Extract lip region with padding
        padded_lip_region = frame[
            lip_top - pad_top:lip_bottom + pad_bottom, 
            lip_left - pad_left:lip_right + pad_right
        ]

        # Resize to standard dimensions
        processed_lip = cv2.resize(padded_lip_region, (self.LIP_OUTPUT_WIDTH, self.LIP_OUTPUT_HEIGHT))

        # Enhanced image processing
        return self._enhance_image(processed_lip)

    def _enhance_image(self, lip_frame):
        """
        Apply advanced image enhancement techniques
        """
        # Convert to LAB color space for contrast enhancement
        lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
        
        # Split the LAB channels
        l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
        l_channel_eq = clahe.apply(l_channel)

        # Merge the equalized L channel with original A and B channels
        lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
        lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
        
        # Apply various filters
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
        lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
        
        # Sharpening kernel
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)

        return lip_frame_eq

    def collect_training_data(self):
        """
        Main method to collect lip reading training data
        """
        # Video capture setup
        video_capture = cv2.VideoCapture(0)
        
        # Data collection variables
        collected_word_sequences = []
        collected_labels = []
        current_word_frames = []
        past_frame_buffer = deque(maxlen=self.PAST_BUFFER_SIZE)
        
        # User input for word selection
        available_words = ["here", "is", "a", "demo", "can", "you", "read", "my", "lips", "cat", "dog", "hello", "bye"]
        print("Available words:", ", ".join(available_words))
        selected_word = input("Enter the word to collect data for: ")
        
        # Lip distance calibration
        lip_distance_threshold = self._calibrate_lip_distance(video_capture)
        
        not_talking_counter = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_detector(gray_frame)

            for face in detected_faces:
                # Detect facial landmarks
                landmarks = self.landmark_predictor(image=gray_frame, box=face)
                
                # Calculate lip distance
                mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
                mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
                lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])

                # Check if speaking
                if lip_distance > lip_distance_threshold:
                    processed_lip = self._preprocess_lip_region(frame, landmarks)
                    current_word_frames.append(processed_lip.tolist())
                    not_talking_counter = 0
                else:
                    not_talking_counter += 1

                    # Word recording logic
                    if (not_talking_counter >= self.SILENCE_FRAME_THRESHOLD and 
                        len(current_word_frames) + self.PAST_BUFFER_SIZE == self.TOTAL_FRAME_COUNT):
                        
                        # Combine past frames with current frames
                        complete_word_sequence = list(past_frame_buffer) + current_word_frames
                        
                        collected_word_sequences.append(complete_word_sequence)
                        collected_labels.append(selected_word)
                        
                        # Reset for next word
                        current_word_frames = []
                        not_talking_counter = 0

                # Update past frame buffer
                past_frame_buffer.append(processed_lip.tolist())

            # Visualization and user interface
            cv2.imshow("Lip Reading Data Collection", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                break

        video_capture.release()
        cv2.destroyAllWindows()

        return collected_word_sequences, collected_labels

    def _calibrate_lip_distance(self, video_capture, calibration_frames=50):
        """
        Automatically calibrate lip distance threshold
        """
        lip_distances = []

        while calibration_frames > 0:
            ret, frame = video_capture.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_detector(gray_frame)

            for face in detected_faces:
                landmarks = self.landmark_predictor(image=gray_frame, box=face)
                distance = landmarks.part(58).y - landmarks.part(50).y
                lip_distances.append(distance)
                calibration_frames -= 1

        return sum(lip_distances) / len(lip_distances) + 2

    def save_collected_data(self, word_sequences, labels, base_output_dir="../collected_data"):
        """
        Save collected word sequences as text and image/video files
        """
        for idx, word_frames in enumerate(word_sequences):
            # Create unique directory for each word sequence
            word_dir = os.path.join(base_output_dir, f"{labels[idx]}_{idx+1}")
            os.makedirs(word_dir, exist_ok=True)

            # Save data as text
            with open(os.path.join(word_dir, "data.txt"), "w") as f:
                json.dump(word_frames, f)

            # Save individual frames and create video
            image_paths = []
            for frame_idx, frame_data in enumerate(word_frames):
                img = Image.fromarray(np.uint8(frame_data))
                img_path = os.path.join(word_dir, f"{frame_idx}.png")
                img.save(img_path)
                image_paths.append(img_path)

            # Create video from frames
            video_path = os.path.join(word_dir, "video.mp4")
            imageio.mimsave(video_path, [imageio.imread(path) for path in image_paths], fps=30)

def main():
    lip_data_collector = LipReadingDataCollector()
    word_sequences, labels = lip_data_collector.collect_training_data()
    lip_data_collector.save_collected_data(word_sequences, labels)

if __name__ == "__main__":
    main()