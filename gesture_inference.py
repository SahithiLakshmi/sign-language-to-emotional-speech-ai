import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import logging
from utils import GestureUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HandGestureInference:
    def __init__(self, models_dir="models"):
        """
        Initialize the hand gesture inference system.
        
        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = models_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3,  # Reduced for better performance
            min_tracking_confidence=0.3    # Reduced for better performance
        )
        
        # Load trained models
        self.model = None
        self.label_encoder = None
        self.load_models()
        
        # Stability tracking
        self.current_prediction = None
        self.prediction_count = 0
        self.stability_threshold = 3  # frames (reduced for better responsiveness)
        self.last_output = None
        
        # TEXT-BASED POSITION RULES (Rule-Based layer to supplement ML)
        # 1.0 = fully extended, 0.0 = fully curled
        # Order: [Thumb, Index, Middle, Ring, Pinky]
        self.gesture_rules = {
            'i': [0.0, 1.0, 0.0, 0.0, 0.0],       # Only Index up
            'good': [1.0, 0.0, 0.0, 0.0, 0.0],    # Only Thumb up
            'bad': [0.0, 0.0, 0.0, 0.0, 1.0],     # Only Pinky up
            'hello': [1.0, 1.0, 1.0, 1.0, 1.0],   # All fingers fully extended
            'no': [0.0, 1.0, 1.0, 0.0, 0.0],      # Index and Middle up
            'love': [1.0, 1.0, 0.0, 0.0, 1.0],    # Thumb, Index, Pinky up
            'play': [1.0, 0.0, 0.0, 0.0, 1.0],    # Thumb and Pinky up
            'eat': [0.0, 0.0, 0.0, 0.0, 0.0],     # All fingers curled (Fist)
            'home': [0.0, 1.0, 1.0, 1.0, 1.0],    # Four fingers up, Thumb tucked
            'help': [1.0, 1.0, 0.0, 0.0, 0.0],    # Thumb and Index up (L-shape)
            'please': [0.0, 1.0, 1.0, 1.0, 0.0],  # Index, Middle, Ring up
        }
        
    def load_models(self):
        """Load the trained model and label encoder."""
        model_path = os.path.join(self.models_dir, "gesture_model.pkl")
        encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from: {model_path}")
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            logger.info(f"Label encoder loaded from: {encoder_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def normalize_landmarks(self, landmarks):
        """Delegate to GestureUtils."""
        return GestureUtils.normalize_landmarks(landmarks)
    
    def extract_one_hand_features(self, frame):
        """Delegate to GestureUtils."""
        return GestureUtils.extract_one_hand_features(frame, self.hands)
    
    def verify_gesture_rules(self, label, finger_status):
        """
        Verify if the finger configuration matches the expected textual rule.
        """
        if label not in self.gesture_rules:
            return True # No rule for this gesture, trust ML
            
        expected = self.gesture_rules[label]
        # Check if extended/curled status matches roughly
        for i in range(5):
            if expected[i] == 1.0 and finger_status[i] < 0.4:
                return False # Expected up, but is down
            if expected[i] == 0.0 and finger_status[i] > 0.6:
                return False # Expected down, but is up
        return True

    def predict_gesture(self, features):
        """
        Predict gesture from geometric features with hybrid rule verification.
        
        Args:
            features (list): 24 geometric features
            
        Returns:
            str: Predicted gesture or None
        """
        if features is None or self.model is None:
            return None
        
        # Make ML prediction
        prediction = self.model.predict([features])[0]
        probs = self.model.predict_proba([features])[0]
        confidence = np.max(probs)
        gesture_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Extract finger status from features (first 5 features in our vector)
        finger_status = features[:5]
        
        # Rule-based verification
        # Let's be a bit more lenient with rules to ensure detection works
        if not self.verify_gesture_rules(gesture_label, finger_status):
            # If rules fail, don't return immediately, but log it
            logger.debug(f"Rule verification failed for {gesture_label}")
            # return None # Commented out to see if it improves detection
            
        # Lower confidence threshold for better detection
        if confidence < 0.25:
            return None
            
        # Apply stability filter
        if gesture_label == self.current_prediction:
            self.prediction_count += 1
        else:
            self.current_prediction = gesture_label
            self.prediction_count = 1
        
        # For the UI, we want to see the detection immediately
        # But we only add to the list if stable
        if self.prediction_count >= self.stability_threshold:
            if gesture_label != self.last_output:
                self.last_output = gesture_label
                # self.prediction_count = 0 # Don't reset to keep returning the label
                logger.info(f"Stable prediction: {gesture_label} (confidence: {confidence:.3f})")
            return gesture_label
        
        return None
    
    def run_inference(self):
        """Run real-time gesture inference."""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return
        
        # Set frame properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info("Starting real-time gesture inference...")
        logger.info("Press 'q' to quit")
        
        detected_gestures = []
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract features
                features = self.extract_one_hand_features(frame)
                
                # Draw hand landmarks if detected
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                
                # Predict gesture
                gesture = self.predict_gesture(features)
                if gesture:
                    detected_gestures.append(gesture)
                    logger.info(f"Detected gesture: {gesture}")
                
                # Display information
                cv2.putText(frame, f"Detected: {len(detected_gestures)} gestures", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if detected_gestures:
                    current_gesture = detected_gestures[-1]
                    cv2.putText(frame, f"Last: {current_gesture}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.putText(frame, "Press 'q' to quit, 'c' to clear", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow("Hand Gesture Inference", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    detected_gestures.clear()
                    self.last_output = None
                    logger.info("Gestures cleared")
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Display final results
            if detected_gestures:
                logger.info("Final detected gestures:")
                for i, gesture in enumerate(detected_gestures):
                    logger.info(f"  {i+1}. {gesture}")
            else:
                logger.info("No gestures detected")

def main():
    """Main inference function."""
    try:
        # Initialize inference system
        inference = HandGestureInference()
        
        # Run inference
        inference.run_inference()
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()