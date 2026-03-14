import sys
import os
import logging
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('sklearn', 'scikit-learn'),
        ('ort', 'onnxruntime'),
        ('pyttsx3', 'pyttsx3'),
        ('np', 'numpy'),
        ('pd', 'pandas'),
        ('PIL', 'Pillow'),
        ('ctk', 'customtkinter')
    ]
    
    # Custom mapping for some modules where import name != package name
    import_to_pkg = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'sklearn': 'scikit-learn',
        'onnxruntime': 'onnxruntime',
        'pyttsx3': 'pyttsx3',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'PIL': 'Pillow',
        'customtkinter': 'customtkinter'
    }
    
    missing_modules = []
    for module_name in import_to_pkg.keys():
        try:
            __import__(module_name)
        except ImportError:
            missing_modules.append(import_to_pkg[module_name])
    
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        logger.info("Please install missing modules using:")
        logger.info("pip install " + " ".join(missing_modules))
        return False
    
    logger.info("All dependencies are available")
    return True

def check_models():
    """Check if required models are available."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    required_models = [
        "models/gesture_model.pkl",
        "models/label_encoder.pkl",
        "models/emotion_model.onnx"
    ]
    
    missing_models = []
    for model_path in required_models:
        if not Path(model_path).exists():
            if "emotion_model.onnx" in model_path:
                logger.warning(f"Optional model missing: {model_path}. Emotion detection will use a dummy model.")
                continue
            missing_models.append(model_path)
    
    if missing_models:
        logger.error(f"Missing required models: {missing_models}")
        logger.info("Please train the model first using: python train_gesture.py")
        return False
    
    logger.info("All required models are available")
    return True

def initialize_system():
    """Initialize the complete system."""
    logger.info("=" * 50)
    logger.info("Initializing Sign to Emotional Speech System")
    logger.info("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("System initialization failed: Missing dependencies")
        return False
    
    # Check models
    if not check_models():
        logger.error("System initialization failed: Missing models")
        return False
    
    logger.info("System initialization successful")
    return True

def run_training():
    """Run the training process for gesture classification."""
    logger.info("Starting training process...")
    
    try:
        import train_gesture
        logger.info("Training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return False

def run_application():
    """Run the main application with gesture detection and speech conversion."""
    logger.info("Starting main application...")
    
    try:
        # Import required modules
        from gesture_inference import HandGestureInference
        from emotion_detection import EmotionDetector
        from sentence_builder import SentenceBuilder
        from speech_engine import SpeechEngine
        
        # Initialize components
        gesture_inference = HandGestureInference()
        emotion_detector = EmotionDetector()
        sentence_builder = SentenceBuilder()
        speech_engine = SpeechEngine()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return False
        
        # Set frame properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        logger.info("Starting Sign to Emotional Speech Converter...")
        logger.info("Press 'q' to quit, 'c' to clear, 's' to speak")
        
        detected_gestures = []
        session_emotions = []  # To track majority emotion in the session
        current_sentence = ""
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract features and predict gesture
                features = gesture_inference.extract_one_hand_features(frame)
                gesture = gesture_inference.predict_gesture(features)
                
                # Draw hand landmarks if detected
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = gesture_inference.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, gesture_inference.mp_hands.HAND_CONNECTIONS
                        )
                
                # Add detected gesture to list
                if gesture:
                    detected_gestures.append(gesture)
                    logger.info(f"Detected gesture: {gesture}")
                    
                    # Build sentence
                    current_sentence = sentence_builder.build_sentence(detected_gestures)
                
                # Detect and record emotion
                emotion = emotion_detector.detect_emotion(frame)
                if emotion:
                    session_emotions.append(emotion)
                
                # Display information
                cv2.putText(frame, f"Gestures: {len(detected_gestures)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if detected_gestures:
                    current_gesture = detected_gestures[-1]
                    cv2.putText(frame, f"Last: {current_gesture}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if session_emotions:
                    current_emotion = session_emotions[-1]
                    cv2.putText(frame, f"Emotion: {current_emotion}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if current_sentence:
                    cv2.putText(frame, f"Sentence: {current_sentence[:30]}...", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.putText(frame, "Press: q=quit, c=clear, s=speak", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow("Sign to Emotional Speech Converter", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    detected_gestures.clear()
                    session_emotions.clear()
                    current_sentence = ""
                    gesture_inference.last_output = None
                    logger.info("Gestures and emotions cleared")
                elif key == ord('s') and current_sentence:
                    # Determine majority emotion for the final output
                    final_emotion = "neutral"
                    if session_emotions:
                        from collections import Counter
                        final_emotion = Counter(session_emotions).most_common(1)[0][0]
                    
                    logger.info(f"Speaking sentence with majority emotion: {final_emotion}")
                    success = speech_engine.speak_text(current_sentence, final_emotion)
                    if success:
                        logger.info("Sentence spoken successfully")
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error during application: {str(e)}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Display final results
            if detected_gestures:
                logger.info("Final detected gestures:")
                for i, gesture in enumerate(detected_gestures):
                    logger.info(f"  {i+1}. {gesture}")
                logger.info(f"Final sentence: {current_sentence}")
            else:
                logger.info("No gestures detected")
        
        return True
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        return False

def main():
    """Main entry point for the application - launches UI directly."""
    # Initialize system
    if not initialize_system():
        logger.error("Failed to initialize system")
        return
    
    # Launch UI directly - this is the only way to run the project
    logger.info("Launching Sign to Emotional Speech Converter with UI...")
    try:
        from app import SignToSpeechApp
        app = SignToSpeechApp()
        app.run()
    except Exception as e:
        logger.error(f"UI application failed: {str(e)}")
        sys.exit(1)

def print_system_info():
    """Print system information and usage instructions."""
    print("\n" + "="*60)
    print("SIGN TO EMOTIONAL SPEECH CONVERSION SYSTEM")
    print("="*60)
    print("System Status: Ready")
    print("\nFeatures:")
    print("  • Real-time hand gesture recognition")
    print("  • Emotion detection from facial expressions")
    print("  • Automatic sentence building with grammar correction")
    print("  • Emotion-based text-to-speech output")
    print("\nUsage:")
    print("  python main.py    - Launch Sign to Emotional Speech Converter (UI mode only)")
    print("  python train_gesture.py - Train the gesture model separately")
    print("\nThis is the ONLY way to run the project - UI with all integrated functionality")
    print("\nRequirements:")
    print("  • Webcam for hand gesture detection")
    print("  • Good lighting conditions")
    print("  • Clear hand movements in camera view")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Print system information
    print_system_info()
    
    # Run main application
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)