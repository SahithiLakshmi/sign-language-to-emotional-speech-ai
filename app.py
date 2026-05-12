import customtkinter as ctk
import cv2
import threading
import time
import logging
from PIL import Image, ImageTk
import numpy as np

# Import our modules
from hand_tracking import HandTracker
from emotion_detection import EmotionDetector
from sentence_builder import SentenceBuilder
from speech_engine import SpeechEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignToSpeechApp:
    def __init__(self):
        """Initialize the main application window."""
        # Initialize CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Sign to Emotional Speech Converter")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
        
        # Force window to top and show it
        self.root.lift()
        self.root.focus_force()
        
        # Application state
        self.is_running = False
        self.is_detecting = False
        self.current_emotion = "neutral"
        self.session_emotions = []  # Store emotions for current session
        self.frame_count = 0
        self.cap = None
        
        # 1. Setup UI first so the user sees something
        self.setup_ui()
        self.update_status("● Loading models...", "orange")
        self.root.update() # Update UI to show the loading status
        
        # 2. Initialize components (heavy loading)
        try:
            self.hand_tracker = HandTracker()
            self.emotion_detector = EmotionDetector()
            self.sentence_builder = SentenceBuilder()
            self.speech_engine = SpeechEngine()
            
            # 3. Start camera
            if not self.start_camera():
                self.update_status("● Camera Error", "red")
            else:
                self.update_status("● Ready", "green")
                
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            self.update_status(f"● Error: {str(e)[:20]}", "red")
        
        # 4. Schedule the update loop (don't call it directly in __init__)
        self.root.after(100, self.update_frame)
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="Sign to Emotional Speech Converter", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(10, 20))
        
        # Content frame
        content_frame = ctk.CTkFrame(main_frame)
        content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Left panel - Camera and controls
        left_frame = ctk.CTkFrame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Camera frame
        self.camera_label = ctk.CTkLabel(
            left_frame, 
            text="Camera Feed", 
            font=ctk.CTkFont(size=16),
            fg_color="gray20",
            corner_radius=10,
            height=400
        )
        self.camera_label.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Control buttons
        button_frame = ctk.CTkFrame(left_frame)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.start_button = ctk.CTkButton(
            button_frame,
            text="Start Detection",
            command=self.start_detection,
            fg_color="green",
            hover_color="dark green",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.stop_button = ctk.CTkButton(
            button_frame,
            text="Stop Detection",
            command=self.stop_detection,
            fg_color="red",
            hover_color="dark red",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            state="disabled"
        )
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(5, 0))
        
        # Clear button
        self.clear_button = ctk.CTkButton(
            button_frame,
            text="Clear Gestures",
            command=self.clear_gestures,
            fg_color="orange",
            hover_color="dark orange",
            font=ctk.CTkFont(size=12),
            height=30
        )
        self.clear_button.pack(fill="x", padx=10, pady=(10, 0))
        
        # Middle panel - Information display
        middle_frame = ctk.CTkScrollableFrame(content_frame, width=300)
        middle_frame.pack(side="left", fill="y", padx=10)
        
        # Status panel
        status_frame = ctk.CTkFrame(middle_frame)
        status_frame.pack(fill="x", padx=5, pady=10)
        
        status_label = ctk.CTkLabel(
            status_frame,
            text="System Status",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        status_label.pack(pady=5)
        
        self.status_indicator = ctk.CTkLabel(
            status_frame,
            text="● Ready",
            text_color="green",
            font=ctk.CTkFont(size=14)
        )
        self.status_indicator.pack(pady=(0, 10))
        
        # Detection info
        info_frame = ctk.CTkFrame(middle_frame)
        info_frame.pack(fill="x", padx=5, pady=(0, 10))
        
        ctk.CTkLabel(
            info_frame,
            text="Current Detection",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Detected gesture
        self.gesture_label = ctk.CTkLabel(
            info_frame,
            text="Gesture: None",
            font=ctk.CTkFont(size=12)
        )
        self.gesture_label.pack(pady=2)
        
        # Detected emotion
        self.emotion_label = ctk.CTkLabel(
            info_frame,
            text="Emotion: Neutral 😐",
            font=ctk.CTkFont(size=12)
        )
        self.emotion_label.pack(pady=2)
        
        # Emotion emoji
        self.emotion_emoji = ctk.CTkLabel(
            info_frame,
            text="😐",
            font=ctk.CTkFont(size=24)
        )
        self.emotion_emoji.pack(pady=5)
        
        # Gesture counter
        self.gesture_count_label = ctk.CTkLabel(
            info_frame,
            text="Gestures: 0",
            font=ctk.CTkFont(size=12)
        )
        self.gesture_count_label.pack(pady=2)
        
        # Detected gestures list
        gestures_frame = ctk.CTkFrame(middle_frame)
        gestures_frame.pack(fill="both", expand=True, padx=5, pady=(0, 10))
        
        ctk.CTkLabel(
            gestures_frame,
            text="Detected Gestures",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Textbox for gestures
        self.gestures_textbox = ctk.CTkTextbox(
            gestures_frame,
            height=150,
            font=ctk.CTkFont(size=11)
        )
        self.gestures_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.gestures_textbox.insert("0.0", "No gestures detected yet...\n")
        self.gestures_textbox.configure(state="disabled")

        # Output sentence (Moved to middle panel)
        output_frame = ctk.CTkFrame(middle_frame)
        output_frame.pack(fill="x", padx=5, pady=(0, 10))
        
        ctk.CTkLabel(
            output_frame,
            text="Generated Sentence",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        self.sentence_label = ctk.CTkLabel(
            output_frame,
            text="",
            font=ctk.CTkFont(size=12),
            wraplength=280,
            justify="left"
        )
        self.sentence_label.pack(pady=5, padx=10)
        
        # Speak button
        self.speak_button = ctk.CTkButton(
            output_frame,
            text="Speak Sentence",
            command=self.speak_current_sentence,
            fg_color="purple",
            hover_color="dark purple",
            font=ctk.CTkFont(size=12),
            height=30,
            state="disabled"
        )
        self.speak_button.pack(fill="x", padx=10, pady=(0, 10))

    def start_camera(self):
        """Initialize and start the camera."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logger.error("Cannot open webcam")
                self.update_status("● Camera Error", "red")
                return False
            
            # Set frame properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Camera started successfully")
            self.update_status("● Camera Ready", "green")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            self.update_status("● Camera Error", "red")
            return False
    
    def start_detection(self):
        """Start the detection process."""
        if not self.is_running:
            self.is_running = True
            self.is_detecting = True
            self.session_emotions = []  # Clear session emotions
            
            # Start fresh with no previous gestures
            self.hand_tracker.clear_gestures()
            self.update_gestures_display()
            self.gesture_count_label.configure(text="Gestures: 0")
            self.sentence_label.configure(text="")
            
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.speak_button.configure(state="disabled")
            self.update_status("● Detecting...", "yellow")
            logger.info("Detection started - New session")
    
    def stop_detection(self):
        """Stop the detection process and determine majority emotion."""
        if self.is_running:
            self.is_running = False
            self.is_detecting = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            
            # Determine majority emotion from the session
            if self.session_emotions:
                from collections import Counter
                counts = Counter(self.session_emotions)
                majority_emotion = counts.most_common(1)[0][0]
                self.current_emotion = majority_emotion
                logger.info(f"Session finished. Majority emotion: {majority_emotion}")
                
                # Update UI with the final session emotion
                emoji = self.emotion_detector.get_emotion_emoji(majority_emotion)
                self.emotion_label.configure(text=f"Final Emotion: {majority_emotion.title()} {emoji}")
                self.emotion_emoji.configure(text=emoji)
            
            # Generate and display final sentence
            final_sentence = self.generate_final_sentence()
            if final_sentence:
                self.speak_button.configure(state="normal")
            
            self.update_status("● Ready", "green")
            logger.info("Detection stopped")
    
    def clear_gestures(self):
        """Clear all detected gestures."""
        self.hand_tracker.clear_gestures()
        self.update_gestures_display()
        self.gesture_count_label.configure(text="Gestures: 0")
        self.sentence_label.configure(text="")
        self.speak_button.configure(state="disabled")
        logger.info("Gestures cleared")
    
    def generate_final_sentence(self):
        """Generate final sentence from detected gestures."""
        try:
            gestures = self.hand_tracker.get_detected_gestures()
            if not gestures:
                self.sentence_label.configure(text="No gestures detected")
                return ""
            
            sentence = self.sentence_builder.build_sentence(gestures)
            if not sentence:
                sentence = " ".join(gestures) + "." # Fallback to raw gestures
            
            self.sentence_label.configure(text=sentence)
            logger.info(f"Generated sentence: {sentence}")
            return sentence
            
        except Exception as e:
            logger.error(f"Error in generate_final_sentence: {str(e)}")
            # Fallback if building fails
            gestures = self.hand_tracker.get_detected_gestures()
            if gestures:
                sentence = " ".join(gestures).capitalize() + "."
                self.sentence_label.configure(text=sentence)
                return sentence
            return ""
    
    def speak_current_sentence(self):
        """Speak the current generated sentence in a separate thread."""
        sentence = self.sentence_label.cget("text")
        if sentence and sentence != "No gestures detected":
            # Use a thread to prevent UI freezing during speech
            speech_thread = threading.Thread(
                target=self.speech_engine.speak_sentence,
                args=(sentence, self.current_emotion)
            )
            speech_thread.daemon = True
            speech_thread.start()
            logger.info(f"Speaking sentence with {self.current_emotion} emotion (async)")
    
    def update_frame(self):
        """Update camera frame and perform detection."""
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    processed_frame = self.process_frame(frame)
                    
                    # Convert to PhotoImage for display
                    photo = self.convert_frame_to_photo(processed_frame)
                    if photo:
                        self.camera_label.configure(image=photo)
                        self.camera_label.image = photo  # Keep reference
                    
            except Exception as e:
                logger.error(f"Error updating frame: {str(e)}")
        
        # Schedule next update
        self.root.after(30, self.update_frame)  # ~33 FPS
    
    def process_frame(self, frame):
        """Process frame for hand and emotion detection."""
        if not self.is_detecting:
            return frame
        
        self.frame_count += 1
        
        # Process hand tracking
        frame, landmarks, gesture = self.hand_tracker.process_frame(frame)
        
        # Update gesture display
        if gesture:
            self.gesture_label.configure(text=f"Gesture: {gesture}")
            self.update_gestures_display()
            gesture_count = len(self.hand_tracker.get_detected_gestures())
            self.gesture_count_label.configure(text=f"Gestures: {gesture_count}")
        else:
            self.gesture_label.configure(text="Gesture: None")
        
        # Process emotion detection
        # No frame_count filtering here to ensure it's responsive in the UI
        frame, emotion = self.emotion_detector.process_frame(frame, 0)
        
        # Update emotion display and collect for session
        if emotion:
            self.session_emotions.append(emotion)
            # Live display of current frame's emotion
            emoji = self.emotion_detector.get_emotion_emoji(emotion)
            self.emotion_label.configure(text=f"Emotion: {emotion.title()} {emoji}")
            self.emotion_emoji.configure(text=emoji)
        
        return frame
    
    def update_gestures_display(self):
        """Update the gestures display textbox."""
        gestures = self.hand_tracker.get_detected_gestures()
        self.gestures_textbox.configure(state="normal")
        self.gestures_textbox.delete("0.0", "end")
        
        if gestures:
            gesture_text = "\n".join([f"{i+1}. {gesture}" for i, gesture in enumerate(gestures)])
            self.gestures_textbox.insert("0.0", gesture_text)
        else:
            self.gestures_textbox.insert("0.0", "No gestures detected yet...")
        
        self.gestures_textbox.configure(state="disabled")
    
    def update_status(self, text, color):
        """Update the status indicator."""
        self.status_indicator.configure(text=text, text_color=color)
    
    def convert_frame_to_photo(self, frame):
        """Convert OpenCV frame to PhotoImage for display."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            return photo
            
        except Exception as e:
            logger.error(f"Error converting frame: {str(e)}")
            return None
    
    def run(self):
        """Start the application."""
        try:
            logger.info("Starting Sign to Speech application...")
            # Make sure the window is visible before starting mainloop
            self.root.deiconify()
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Error in application: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        # Stop detection
        self.is_running = False
        self.is_detecting = False
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Release hand tracker
        self.hand_tracker.release()
        
        # Destroy window
        if self.root:
            self.root.destroy()

def main():
    """Main function to run the application."""
    app = SignToSpeechApp()
    app.run()

if __name__ == "__main__":
    main()