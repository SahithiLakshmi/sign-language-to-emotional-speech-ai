import pyttsx3
import logging
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechEngine:
    def __init__(self):
        """Initialize the speech engine with pyttsx3."""
        try:
            self.engine = pyttsx3.init()
            self.setup_voice_properties()
            logger.info("Speech engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing speech engine: {str(e)}")
            self.engine = None
    
    def setup_voice_properties(self):
        """Setup default voice properties."""
        if self.engine is None:
            return
        
        try:
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to set a female voice (commonly index 1)
                if len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
                else:
                    self.engine.setProperty('voice', voices[0].id)
            
            # Set default properties
            self.engine.setProperty('rate', 200)    # Words per minute
            self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            logger.info("Voice properties configured")
            
        except Exception as e:
            logger.warning(f"Could not configure voice properties: {str(e)}")
    
    def get_emotion_voice_settings(self, emotion):
        """
        Get voice settings based on detected emotion.
        
        Args:
            emotion (str): Detected emotion
            
        Returns:
            dict: Voice settings (rate, volume, pitch)
        """
        # Default settings
        settings = {
            'rate': 200,
            'volume': 0.9,
            'pitch': 50  # pyttsx3 doesn't directly support pitch, but we can simulate it
        }
        
        # Emotion-based adjustments
        emotion_settings = {
            'happiness': {
                'rate': 220,    # Faster speech
                'volume': 1.0,  # Louder
                'pitch': 70     # Higher pitch
            },
            'sadness': {
                'rate': 160,    # Slower speech
                'volume': 0.7,  # Quieter
                'pitch': 30     # Lower pitch
            },
            'anger': {
                'rate': 240,    # Much faster
                'volume': 1.0,  # Louder
                'pitch': 80     # Higher pitch
            },
            'fear': {
                'rate': 230,
                'volume': 0.8,
                'pitch': 75
            },
            'disgust': {
                'rate': 180,
                'volume': 0.8,
                'pitch': 40
            },
            'surprise': {
                'rate': 250,
                'volume': 1.0,
                'pitch': 90
            },
            'neutral': {
                'rate': 200,    # Normal rate
                'volume': 0.9,  # Normal volume
                'pitch': 50     # Normal pitch
            }
        }
        
        # Apply emotion-specific settings
        if emotion.lower() in emotion_settings:
            settings.update(emotion_settings[emotion.lower()])
        
        logger.info(f"Voice settings for {emotion}: rate={settings['rate']}, volume={settings['volume']}")
        return settings
    
    def set_voice_settings(self, settings):
        """
        Apply voice settings to the engine.
        
        Args:
            settings (dict): Voice settings to apply
        """
        if self.engine is None:
            return
        
        try:
            self.engine.setProperty('rate', settings['rate'])
            self.engine.setProperty('volume', settings['volume'])
            # Note: pyttsx3 doesn't have direct pitch control
            # We can only approximate it through rate and volume adjustments
        except Exception as e:
            logger.warning(f"Error setting voice properties: {str(e)}")
    
    def speak_text(self, text, emotion='neutral', blocking=True):
        """
        Speak text with emotion-based voice modulation.
        
        Args:
            text (str): Text to speak
            emotion (str): Emotion to convey
            blocking (bool): Whether to block until speech completes
        """
        if self.engine is None:
            logger.error("Speech engine not available")
            return False
        
        if not text or not text.strip():
            logger.warning("Empty text provided for speech")
            return False
        
        try:
            # Get emotion-based voice settings
            voice_settings = self.get_emotion_voice_settings(emotion)
            
            # Apply settings
            self.set_voice_settings(voice_settings)
            
            logger.info(f"Speaking: '{text}' with {emotion} emotion")
            
            # Speak the text
            if blocking:
                self.engine.say(text)
                self.engine.runAndWait()
            else:
                # Non-blocking speech in separate thread
                def speak_async():
                    self.engine.say(text)
                    self.engine.runAndWait()
                
                thread = threading.Thread(target=speak_async)
                thread.daemon = True
                thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return False
    
    def speak_sentence(self, sentence, emotion='neutral', add_pause=True):
        """
        Speak a complete sentence with optional pause.
        
        Args:
            sentence (str): Sentence to speak
            emotion (str): Emotion to convey
            add_pause (bool): Whether to add pause after speaking
        """
        if not sentence or not sentence.strip():
            return False
        
        success = self.speak_text(sentence, emotion, blocking=True)
        
        if success and add_pause:
            # Add a brief pause after speaking
            time.sleep(0.5)
        
        return success
    
    def stop_speaking(self):
        """Stop current speech output."""
        if self.engine is None:
            return
        
        try:
            self.engine.stop()
            logger.info("Speech stopped")
        except Exception as e:
            logger.error(f"Error stopping speech: {str(e)}")
    
    def get_available_voices(self):
        """Get list of available voices."""
        if self.engine is None:
            return []
        
        try:
            voices = self.engine.getProperty('voices')
            voice_info = []
            for voice in voices:
                voice_info.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages
                })
            return voice_info
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}")
            return []
    
    def set_voice_by_id(self, voice_id):
        """Set voice by ID."""
        if self.engine is None:
            return False
        
        try:
            self.engine.setProperty('voice', voice_id)
            logger.info(f"Voice set to: {voice_id}")
            return True
        except Exception as e:
            logger.error(f"Error setting voice: {str(e)}")
            return False
    
    def test_emotions(self):
        """Test different emotion-based speech outputs."""
        test_sentences = [
            "Hello, how are you today?",
            "I am very happy to see you!",
            "This makes me feel quite sad.",
            "I am really angry about this situation!"
        ]
        
        emotions = ['neutral', 'happy', 'sad', 'anger']
        
        print("Testing emotion-based speech...")
        for emotion in emotions:
            print(f"\n--- Testing {emotion.upper()} emotion ---")
            for sentence in test_sentences:
                print(f"Speaking: {sentence}")
                self.speak_text(sentence, emotion, blocking=True)
                time.sleep(1)  # Pause between sentences

def main():
    """Main function for testing speech engine"""
    # Initialize speech engine
    speech_engine = SpeechEngine()
    
    # Test available voices
    voices = speech_engine.get_available_voices()
    print(f"Available voices: {len(voices)}")
    for voice in voices[:3]:  # Show first 3 voices
        print(f"  - {voice['name']} ({voice['id']})")
    
    # Test basic speech
    print("\nTesting basic speech...")
    speech_engine.speak_text("Hello! This is a test of the speech engine.", blocking=True)
    
    # Test emotion-based speech
    print("\nTesting emotion-based speech...")
    speech_engine.test_emotions()
    
    print("\nSpeech engine test completed!")

if __name__ == "__main__":
    main()