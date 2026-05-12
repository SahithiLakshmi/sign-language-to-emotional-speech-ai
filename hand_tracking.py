import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from gesture_inference import HandGestureInference
from utils import GestureUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HandTracker:
    def __init__(self, static_mode=False, max_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        """
        Initialize the hand tracker with MediaPipe.
        
        Args:
            static_mode (bool): Whether to treat input as static images
            max_hands (int): Maximum number of hands to detect
            min_detection_confidence (float): Minimum detection confidence threshold
            min_tracking_confidence (float): Minimum tracking confidence threshold
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hand detection model with optimized settings for better performance
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize gesture inference system
        try:
            self.gesture_inference = HandGestureInference()
            logger.info("Gesture inference system loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load gesture inference system: {str(e)}")
            self.gesture_inference = None
        
        # Tracking variables
        self.detected_gestures = []
        
    def process_frame(self, frame):
        """
        Process a video frame to detect hands and gestures.
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            tuple: (processed_frame, hand_landmarks, detected_gesture)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        detected_gesture = None
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmarks for gesture detection
                landmarks = self.extract_landmarks(hand_landmarks)
                hand_landmarks_list.append(landmarks)
                
                # Extract features for gesture prediction
                if self.gesture_inference:
                    # Pass the landmarks directly to avoid re-processing the frame
                    features = GestureUtils.extract_geometric_features_from_landmarks(hand_landmarks)
                    if features is not None:
                        # Get prediction from our trained model
                        gesture = self.gesture_inference.predict_gesture(features)
                        if gesture:
                            # 1. Update the display gesture (always show current detection)
                            detected_gesture = gesture
                            # 2. Add to list only if it's a new gesture (duplicate filter)
                            self.filter_stable_gesture(gesture)
                        else:
                            # Debug: Show when features are extracted but no gesture detected
                            logger.debug(f"Features extracted but no stable gesture detected")
        
        return frame, hand_landmarks_list, detected_gesture
    
    def extract_landmarks(self, hand_landmarks):
        """
        Extract landmark coordinates from MediaPipe hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            list: Flattened list of landmark coordinates (63 features)
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    
    def filter_stable_gesture(self, gesture):
        """
        Filter gestures to ensure stability before accepting them.
        Since gesture_inference already applies stability filtering,
        we just need to prevent duplicates.
        
        Args:
            gesture (str): Detected gesture name
            
        Returns:
            str: Gesture name or None if duplicate
        """
        # Check for repetition (no duplicates)
        if not self.detected_gestures or gesture != self.detected_gestures[-1]:
            self.detected_gestures.append(gesture)
            logger.info(f"Stable gesture detected: {gesture}")
            return gesture
        
        return None
    
    def get_detected_gestures(self):
        """Get list of detected gestures."""
        return self.detected_gestures.copy()
    
    def clear_gestures(self):
        """Clear the detected gestures list."""
        self.detected_gestures.clear()
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
        if self.gesture_inference:
            # Note: The gesture inference system doesn't have a release method
            pass

def main():
    """Main function for testing hand tracking"""
    # Initialize hand tracker
    tracker = HandTracker()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    # Set frame properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    logger.info("Hand tracking started. Press 'q' to quit, 'c' to clear gestures.")
    
    try:
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame")
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, landmarks, gesture = tracker.process_frame(frame)
            
            # Display detected gesture
            if gesture:
                cv2.putText(processed_frame, f"Gesture: {gesture}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instruction
            cv2.putText(processed_frame, "Press 'q' to quit, 'c' to clear", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Hand Tracking", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                tracker.clear_gestures()
                logger.info("Gestures cleared")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in hand tracking: {str(e)}")
    finally:
        # Cleanup
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()
        
        # Display final results
        final_gestures = tracker.get_detected_gestures()
        logger.info(f"Final detected gestures: {final_gestures}")

if __name__ == "__main__":
    main()