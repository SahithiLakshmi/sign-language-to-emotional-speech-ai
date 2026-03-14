import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import logging
from utils import GestureUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HandGestureTrainer:
    def __init__(self, dataset_path="Hand_gestures"):
        """
        Initialize the hand gesture trainer.
        
        Args:
            dataset_path (str): Path to the hand gestures dataset
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_absolute():
            self.dataset_path = Path(os.getcwd()) / self.dataset_path
            
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Storage for training data
        self.X = []  # Features (63 per sample)
        self.y = []  # Labels
        
        # Model components
        self.model = None
        self.label_encoder = None
        
    def normalize_landmarks(self, landmarks):
        """Delegate to GestureUtils."""
        return GestureUtils.normalize_landmarks(landmarks)
    
    def extract_one_hand_features(self, image_path):
        """
        Extract 63 features from image using GestureUtils.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            list: 63 features or None if no hand detected
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not read image: {image_path}")
            return None
            
        return GestureUtils.extract_one_hand_features(image, self.hands)
    
    def load_dataset(self):
        """
        Load and process the expanded subset of the dataset.
        """
        logger.info("Loading dataset from: %s", self.dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Expanded subset for a comprehensive demo (11 physically distinct gestures)
        demo_gestures = [
            'bad', 'eat', 'good', 'hello', 'help', 'home', 'i', 'love', 'no', 'play', 'please'
        ]
        
        # Delete old model files if they exist to ensure a fresh start
        old_models = ["models/gesture_model.pkl", "models/label_encoder.pkl"]
        for model_file in old_models:
            if os.path.exists(model_file):
                try:
                    os.remove(model_file)
                    logger.info(f"Deleted old model file: {model_file}")
                except Exception as e:
                    logger.warning(f"Could not delete {model_file}: {e}")
        
        # Process each gesture class in the subset
        for gesture_name in demo_gestures:
            gesture_dir = self.dataset_path / gesture_name
            if not gesture_dir.exists():
                # Try with underscores if spaces are used
                gesture_dir = self.dataset_path / gesture_name.replace(' ', '_')
                if not gesture_dir.exists():
                    logger.warning(f"Gesture directory not found: {gesture_name}")
                    continue
                
            logger.info(f"Processing gesture: {gesture_name}")
            
            # Get all image files
            image_files = list(gesture_dir.glob("*.png")) + list(gesture_dir.glob("*.jpg"))
            logger.info(f"Found {len(image_files)} images for {gesture_name}")
            
            # Process each image
            processed_count = 0
            for image_file in image_files:
                # Read image
                image = cv2.imread(str(image_file))
                if image is None:
                    continue
                
                # 1. Original image
                features = GestureUtils.extract_one_hand_features(image, self.hands)
                if features is not None:
                    self.X.append(features)
                    self.y.append(gesture_name)
                    processed_count += 1
                
                # 2. Flipped image (Data Augmentation)
                flipped_image = cv2.flip(image, 1)
                features_flipped = GestureUtils.extract_one_hand_features(flipped_image, self.hands)
                if features_flipped is not None:
                    self.X.append(features_flipped)
                    self.y.append(gesture_name)
                    processed_count += 1
            
            logger.info(f"Successfully processed {processed_count} samples for {gesture_name}")
        
        logger.info(f"Total dataset size: {len(self.X)} samples")
        logger.info(f"Number of classes: {len(set(self.y))}")
        logger.info(f"Classes: {sorted(list(set(self.y)))}")
        
        if len(self.X) == 0:
            raise ValueError("No valid samples found in dataset")
    
    def train_model(self):
        """
        Train the RandomForest classifier.
        """
        logger.info("Starting model training...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train RandomForest
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        logger.info("Training RandomForest classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, 
                                        target_names=self.label_encoder.classes_))
        
        return accuracy
    
    def save_model(self, models_dir="models"):
        """
        Save the trained model and label encoder.
        
        Args:
            models_dir (str): Directory to save models
        """
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, "gesture_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to: {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(models_dir, "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info(f"Label encoder saved to: {encoder_path}")

def main():
    """Main training function."""
    try:
        # Initialize trainer
        trainer = HandGestureTrainer()
        
        # Load dataset
        trainer.load_dataset()
        
        # Train model
        accuracy = trainer.train_model()
        
        # Save model
        trainer.save_model()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final accuracy: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()