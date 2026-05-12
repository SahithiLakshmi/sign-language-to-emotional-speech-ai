import cv2
import numpy as np
import onnxruntime as ort
import requests
import os
import logging
from pathlib import Path
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, model_path="models/emotion_model.onnx"):
        """
        Initialize the emotion detector with ONNX model.
        
        Args:
            model_path (str): Path to the ONNX emotion model
        """
        self.model_path = Path(model_path)
        self.session = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # New model parameters from Shohruh72/Emotion_onnx (HSEmotion)
        self.img_size = 260
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # New labels in the correct order (0-6)
        self.emotion_labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        
        # Bias correction weights (default is 1.0)
        # Higher values make the emotion more likely to be detected
        self.emotion_bias = {
            'neutral': 2.0,    # Reduced from 3.5 - less dominant
            'happiness': 1.8,  # Increased from 1.2 - more responsive to smiles
            'sadness': 1.2,    # Increased from 0.4 - better sadness detection
            'anger': 1.0,      # Increased from 0.6 - more balanced
            'disgust': 1.0,    # Increased from 0.8
            'fear': 1.0,       # Increased from 0.8
            'surprise': 1.3    # Increased from 1.0
        }
        
        # Stabilization buffer
        self.emotion_history = deque(maxlen=15)  # Store last 15 detections (~0.5s at 30fps)
        self.last_stable_emotion = "neutral"
        
        # Load model
        self.load_model()
    
    def download_model(self):
        """Download the emotion detection model from Shohruh72 repository."""
        if self.model_path.exists():
            logger.info("Model already exists")
            return True
        
        logger.info("Downloading Shohruh72 emotion detection model...")
        
        # Create models directory
        self.model_path.parent.mkdir(exist_ok=True)
        
        # Correct URL for Shohruh72 emotion model
        model_url = "https://github.com/Shohruh72/Emotion_onnx/releases/download/v.1.0.0/emotion.onnx"
        
        try:
            response = requests.get(model_url, timeout=60, allow_redirects=True)
            response.raise_for_status()
            
            with open(self.model_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Model downloaded successfully to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            return False
    
    def load_model(self):
        """Load the ONNX model."""
        try:
            # Check if model exists
            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}. Attempting download...")
                if not self.download_model():
                    logger.error("Could not obtain emotion model.")
                    return False
            
            # Initialize ONNX runtime session
            # Try CUDA if available, otherwise CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            logger.info("Shohruh72 emotion model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            return False
    
    def detect_face(self, frame):
        """
        Detect the largest face in the frame using Haar Cascade.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List containing the largest face bounding box [(x, y, w, h)] or empty list
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)  # Larger min size for better reliability
        )
        
        if len(faces) == 0:
            return []
            
        # Find the largest face by area (w * h)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return [largest_face]
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image for Shohruh72/HSEmotion model.
        
        Args:
            face_image (numpy.ndarray): Cropped face image (BGR)
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input
        """
        # Resize to model input size (260x260)
        resized = cv2.resize(face_image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values (0-1) and apply mean/std
        # HSEmotion normalization: (x / 255 - mean) / std
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        
        # Change layout to NCHW (batch, channel, height, width)
        # Transpose from (H, W, C) to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        input_tensor = transposed.reshape(1, 3, self.img_size, self.img_size)
        
        return input_tensor
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict_emotion(self, face_image):
        """
        Predict emotion from face image with temporal stabilization and custom bias.
        
        Args:
            face_image (numpy.ndarray): Preprocessed face image
            
        Returns:
            tuple: (emotion_label, confidence_score)
        """
        if self.session is None:
            return self.last_stable_emotion, 0.0
        
        try:
            # Run inference
            input_name = self.session.get_inputs()[0].name
            result = self.session.run(None, {input_name: face_image})
            
            # Get output (logits)
            output = result[0][0]  # Remove batch dimension
            
            # Apply softmax to get probabilities
            probabilities = self.softmax(output)
            
            # Apply custom emotion bias
            weighted_probs = np.copy(probabilities)
            for i, label in enumerate(self.emotion_labels):
                weighted_probs[i] *= self.emotion_bias.get(label, 1.0)
            
            # Normalize again after weighting
            weighted_probs = weighted_probs / np.sum(weighted_probs)
                
            emotion_idx = np.argmax(weighted_probs)
            confidence = probabilities[emotion_idx] # Keep original confidence for display
            current_emotion = self.emotion_labels[emotion_idx]
            
            # Hard override for stability: if neutral has any significant presence, default to it
            # Reduced threshold from 0.15 to 0.25 for less aggressive neutral override
            if probabilities[self.emotion_labels.index('neutral')] > 0.25:
                current_emotion = 'neutral'
            
            # Add to history
            self.emotion_history.append(current_emotion)
            
            # Stabilization logic
            if len(self.emotion_history) >= 5:
                from collections import Counter
                counts = Counter(self.emotion_history)
                
                # Priority: if neutral is even slightly common, stay neutral
                # Increased threshold from 0.3 to 0.4 for better emotion diversity
                neutral_ratio = counts.get('neutral', 0) / len(self.emotion_history)
                if neutral_ratio > 0.4:
                    stable_emotion = 'neutral'
                else:
                    stable_emotion = counts.most_common(1)[0][0]
                
                self.last_stable_emotion = stable_emotion
            
            return self.last_stable_emotion, confidence
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {str(e)}")
            return self.last_stable_emotion, 0.0
    
    def detect_emotion(self, frame):
        """
        Detect emotion from a single frame.
        This is a convenience method for main.py.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            str: Detected emotion label or None
        """
        faces = self.detect_face(frame)
        if len(faces) == 0:
            return None
            
        # Process the first detected face
        (x, y, w, h) = faces[0]
        face_region = frame[y:y+h, x:x+w]
        processed_face = self.preprocess_face(face_region)
        emotion, confidence = self.predict_emotion(processed_face)
        
        return emotion

    def process_frame(self, frame, frame_count=0):
        """
        Process frame to detect emotions.
        
        Args:
            frame (numpy.ndarray): Input frame
            frame_count (int): Frame counter for optimization
            
        Returns:
            tuple: (processed_frame, detected_emotion)
        """
        detected_emotion = None
        
        # Process every 3rd frame for performance optimization
        if frame_count % 3 != 0:
            return frame, None
        
        # Detect faces
        faces = self.detect_face(frame)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = self.preprocess_face(face_region)
            
            # Predict emotion
            emotion, confidence = self.predict_emotion(processed_face)
            detected_emotion = emotion
            
            # Display emotion on frame
            emotion_text = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, emotion_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display emotion emoji
            emoji = self.get_emotion_emoji(emotion)
            cv2.putText(frame, emoji, (x + w - 30, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame, detected_emotion
    
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion."""
        emoji_map = {
            'anger': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happiness': '😊',
            'sadness': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }
        return emoji_map.get(emotion.lower(), '😐')

def main():
    """Main function for testing emotion detection"""
    # Initialize emotion detector
    detector = EmotionDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    # Set frame properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    logger.info("Emotion detection started. Press 'q' to quit.")
    
    try:
        while True:
            # Read frame
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame")
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for emotion detection
            processed_frame, emotion = detector.process_frame(frame, frame_count)
            frame_count += 1
            
            # Display emotion
            if emotion:
                cv2.putText(processed_frame, f"Detected Emotion: {emotion}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instruction
            cv2.putText(processed_frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow("Emotion Detection", processed_frame)
            
            # Handle key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in emotion detection: {str(e)}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()