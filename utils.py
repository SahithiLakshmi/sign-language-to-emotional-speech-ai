import numpy as np
import cv2
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class GestureUtils:
    @staticmethod
    def get_finger_status(points):
        """
        Calculate if each finger is extended or curled.
        Returns 5 values (0.0 to 1.0) representing extension.
        """
        # points shape: (21, 3)
        # Finger tips: 4, 8, 12, 16, 20
        # Finger MCPs: 2, 5, 9, 13, 17
        tips = [4, 8, 12, 16, 20]
        mcps = [2, 5, 9, 13, 17]
        wrist = points[0]
        
        finger_statuses = []
        for tip, mcp in zip(tips, mcps):
            # Distance from wrist to tip vs distance from wrist to MCP
            dist_tip = np.linalg.norm(points[tip] - wrist)
            dist_mcp = np.linalg.norm(points[mcp] - wrist)
            
            # Ratio describes extension
            if dist_mcp > 0:
                ratio = dist_tip / dist_mcp
            else:
                ratio = 0.0
            
            # Clip and normalize (typical range is 1.0 to 2.5)
            status = np.clip((ratio - 1.0) / 1.5, 0.0, 1.0)
            finger_statuses.append(float(status))
            
        return finger_statuses

    @staticmethod
    def get_geometric_features(landmarks):
        """
        Extract descriptive geometric features from landmarks.
        Returns a vector of 20 features.
        """
        points = np.array(landmarks).reshape(21, 3)
        
        # 1. Finger extension status (5 features)
        extension = GestureUtils.get_finger_status(points)
        
        # 2. Inter-finger distances (tips) (4 features)
        tips = [4, 8, 12, 16, 20]
        inter_finger = []
        for i in range(len(tips) - 1):
            dist = np.linalg.norm(points[tips[i]] - points[tips[i+1]])
            inter_finger.append(float(dist))
            
        # 3. Relative positions of tips to wrist (15 features)
        wrist = points[0]
        relative_tips = []
        max_dist = 0.001
        for tip in tips:
            vec = points[tip] - wrist
            max_dist = max(max_dist, np.linalg.norm(vec))
            relative_tips.extend(vec.tolist())
            
        # Normalize relative tips by scale
        relative_tips = [v / max_dist for v in relative_tips]
        
        # Combined feature vector (5 + 4 + 15 = 24 features)
        return extension + inter_finger + relative_tips

    @staticmethod
    def extract_geometric_features_from_landmarks(hand_landmarks):
        """
        Extract geometric features from MediaPipe hand landmarks.
        """
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return GestureUtils.get_geometric_features(landmarks)

    @staticmethod
    def extract_one_hand_features(frame, mp_hands_model):
        """
        Extract geometric features from frame (one hand).
        Used for training where we don't have results yet.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands_model.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
            
        return GestureUtils.extract_geometric_features_from_landmarks(results.multi_hand_landmarks[0])
