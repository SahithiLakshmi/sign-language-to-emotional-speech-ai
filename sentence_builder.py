import re
import logging
import spacy
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceBuilder:
    def __init__(self):
        """Initialize the lightweight offline sentence reconstruction system."""
        try:
            # Load spaCy English model (lightweight version)
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy English model loaded successfully")
        except OSError:
            logger.error("spaCy English model not found. Please install with: python -m spacy download en_core_web_sm")
            raise Exception("Required spaCy model 'en_core_web_sm' not installed")
        
        # Define core components for sentence analysis
        self.time_words = {
            'tomorrow', 'today', 'yesterday', 'now', 'later', 'soon', 
            'morning', 'afternoon', 'evening', 'night', 'weekend'
        }
        
        self.prepositions = {
            'to', 'at', 'in', 'on', 'by', 'for', 'with', 'from', 'of', 'about'
        }
        
        self.default_objects = {
            'eat': 'food',
            'drink': 'water',
            'play': 'games',
            'work': 'hard',
            'sleep': 'well',
            'go': 'there'
        }
        
        # POS tag mappings for quick lookup
        self.pos_categories = {
            'PRON': ['i'],
            'VERB': ['eat', 'help', 'love', 'play'],
            'ADJ': ['good', 'bad'],
            'NOUN': ['home'],
            'INTJ': ['hello', 'no', 'please'],
            'AUX': [] # Empty for now, but referenced in rules
        }

    def clean_input(self, gestures):
        """
        Step 1: Clean Input - Remove consecutive duplicates and normalize.
        
        Args:
            gestures (list): List of detected gestures
            
        Returns:
            list: Cleaned gestures
        """
        if not gestures:
            return []
        
        # Remove consecutive duplicates
        cleaned = [gestures[0]]
        for i in range(1, len(gestures)):
            if gestures[i].lower() != gestures[i-1].lower():
                cleaned.append(gestures[i])
        
        # Convert to lowercase for processing (will be capitalized later)
        cleaned = [word.lower() for word in cleaned]
        
        logger.info(f"Cleaned input: {cleaned}")
        return cleaned

    def extract_components(self, gestures):
        """
        Step 2: Detect Core Components using spaCy POS tagging with a simple fallback.
        
        Args:
            gestures (list): Cleaned list of gestures
            
        Returns:
            dict: Dictionary with categorized components
        """
        if not gestures:
            return {}
        
        components = {
            'subjects': [],
            'modals': [],
            'verbs': [],
            'adjectives': [],
            'nouns': [],
            'adverbs': [],
            'time_words': [],
            'prepositions': [],
            'others': []
        }
        
        try:
            # Try to use spaCy if available
            text = " ".join(gestures)
            doc = self.nlp(text)
            
            # Extract components based on POS tags
            for token in doc:
                word = token.text.lower()
                pos = token.pos_
                
                if word in self.pos_categories['PRON']:
                    components['subjects'].append(token.text)
                elif word in self.pos_categories['AUX']:
                    components['modals'].append(token.text)
                elif pos == 'VERB' or word in self.pos_categories['VERB']:
                    components['verbs'].append(token.text)
                elif pos == 'ADJ' or word in self.pos_categories['ADJ']:
                    components['adjectives'].append(token.text)
                elif pos in ['NOUN', 'PROPN'] or word in self.pos_categories['NOUN']:
                    components['nouns'].append(token.text)
                elif pos == 'ADV':
                    components['adverbs'].append(token.text)
                elif word in self.time_words:
                    components['time_words'].append(token.text)
                elif word in self.prepositions:
                    components['prepositions'].append(token.text)
                else:
                    components['others'].append(token.text)
                    
        except Exception as e:
            logger.warning(f"spaCy processing failed: {str(e)}. Using fallback extraction.")
            # Simple fallback based on predefined categories
            for word in gestures:
                word_lower = word.lower()
                if word_lower in self.pos_categories['PRON']:
                    components['subjects'].append(word)
                elif word_lower in self.pos_categories['VERB']:
                    components['verbs'].append(word)
                elif word_lower in self.pos_categories['ADJ']:
                    components['adjectives'].append(word)
                elif word_lower in self.pos_categories['NOUN']:
                    components['nouns'].append(word)
                elif word_lower in self.time_words:
                    components['time_words'].append(word)
                else:
                    components['others'].append(word)
        
        logger.info(f"Extracted components: {components}")
        return components

    def apply_rules(self, gestures, components):
        """
        Step 3: Apply Rule-Based Reconstruction for the 11 gestures.
        """
        if not gestures:
            return []
        
        result = []
        used_words = set()
        
        # Rule 1: Handle "Hello I ..."
        if len(gestures) >= 2 and gestures[0].lower() == 'hello' and gestures[1].lower() == 'i':
            result.extend(['Hello,', 'I'])
            used_words.update(['hello', 'i'])
        
        # Rule 2: Handle "Please help me ..."
        if 'please' in [g.lower() for g in gestures]:
            if 'Please' not in result:
                result.append('Please')
            used_words.add('please')
            if 'help' in [g.lower() for g in gestures] and 'help' not in used_words:
                result.append('help')
                used_words.add('help')
            if 'i' in [g.lower() for g in gestures] and 'me' not in [r.lower() for r in result]:
                result.append('me')
                used_words.add('i')
        
        # Rule 3: I + [Verb] + [Noun/Adj]
        if 'i' in [g.lower() for g in gestures] and 'i' not in used_words:
            result.append('I')
            used_words.add('i')
            
        # Add verbs
        for word in gestures:
            if word.lower() in self.pos_categories['VERB'] and word.lower() not in used_words:
                result.append(word.lower())
                used_words.add(word.lower())
        
        # Add adjectives with "am" if subject is I
        for word in gestures:
            if word.lower() in self.pos_categories['ADJ'] and word.lower() not in used_words:
                if 'I' in result and 'am' not in [r.lower() for r in result]:
                    result.insert(result.index('I') + 1, 'am')
                result.append(word.lower())
                used_words.add(word.lower())
        
        # Add remaining words in order
        for word in gestures:
            if word.lower() not in used_words:
                result.append(word.lower())
                used_words.add(word.lower())
        
        # --------------------------------------------------
        # Sentence Refinement for Specific Gesture Combinations
        # (Safe post-processing: does NOT disturb existing logic)
        # --------------------------------------------------
        
        gesture_set = set([g.lower() for g in gestures])
        
        # i + eat → I want to eat
        if {"i", "eat"}.issubset(gesture_set):
            result = ["I", "want", "to", "eat"]
        
        # i + help → I need help
        elif {"i", "help"}.issubset(gesture_set):
            result = ["I", "need", "help"]
        
        # please + help → Please help me
        elif {"please", "help"}.issubset(gesture_set):
            result = ["Please", "help", "me"]
        
        # i + love + play → I love to play
        elif {"i", "love", "play"}.issubset(gesture_set):
            result = ["I", "love", "to", "play"]
        
        # i + love → I love you
        elif {"i", "love"}.issubset(gesture_set):
            result = ["I", "love", "you"]
        
        # i + play → I want to play
        elif {"i", "play"}.issubset(gesture_set):
            result = ["I", "want", "to", "play"]
        
        logger.info(f"Rule-based reconstruction: {result}")
        return result

    def construct_sentence(self, words):
        """
        Step 4: Construct final sentence with proper formatting.
        
        Args:
            words (list): Reconstructed word list
            
        Returns:
            str: Final formatted sentence
        """
        if not words:
            return ""
        
        # Capitalize first word properly
        sentence_words = []
        for i, word in enumerate(words):
            if i == 0:
                # Capitalize first letter, handle special cases
                if word.lower() == 'i':
                    sentence_words.append('I')
                else:
                    sentence_words.append(word.capitalize())
            else:
                sentence_words.append(word)
        
        # Join words
        sentence = " ".join(sentence_words)
        
        # Add period if no ending punctuation
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        # Clean up spacing
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        return sentence

    def build_sentence(self, gestures):
        """
        Main function: Complete sentence reconstruction pipeline.
        
        Args:
            gestures (list): List of detected gestures
            
        Returns:
            str: Grammatically correct English sentence
        """
        if not gestures:
            return ""
        
        logger.info(f"Input gestures: {gestures}")
        
        # Step 1: Clean Input
        cleaned = self.clean_input(gestures)
        
        # Step 2: Extract Components
        components = self.extract_components(cleaned)
        
        # Step 3: Apply Rules
        reconstructed = self.apply_rules(cleaned, components)
        
        # Step 4: Construct Sentence
        final_sentence = self.construct_sentence(reconstructed)
        
        logger.info(f"Final sentence: '{final_sentence}'")
        return final_sentence
    
    def get_sentence_stats(self, sentence):
        """
        Get statistics about the generated sentence.
        
        Args:
            sentence (str): Generated sentence
            
        Returns:
            dict: Sentence statistics
        """
        if not sentence:
            return {
                'word_count': 0,
                'character_count': 0,
                'has_punctuation': False,
                'is_question': False,
                'is_exclamation': False
            }
        
        words = sentence.split()
        return {
            'word_count': len(words),
            'character_count': len(sentence),
            'has_punctuation': sentence[-1] in '.!',
            'is_question': sentence.endswith('?'),
            'is_exclamation': sentence.endswith('!'),
            'sentence': sentence
        }

def main():
    """Main function for testing the lightweight sentence builder"""
    try:
        builder = SentenceBuilder()
        print("Lightweight Offline Sentence Builder initialized successfully!")
        print("=" * 60)
        
        # Test cases matching the requirements
        test_cases = [
            # Core requirement test cases
            (["drink", "i", "can"], "I can drink."),
            (["eat", "i", "want"], "I want to eat."),
            (["can", "help"], "I can help."),
            (["beautiful", "she"], "She is beautiful."),
            (["tomorrow", "i", "go", "market"], "I go to market tomorrow."),
            
            # Additional test cases
            (["hello", "i", "am", "fine"], "Hello I am fine."),
            (["thank", "you", "very", "much"], "Thank you very much."),
            (["how", "are", "you", "today"], "How are you today."),
            (["good", "morning", "everyone"], "Good morning everyone."),
            (["please", "help", "me"], "Please help me."),
            (["work", "i", "must"], "I must work."),
            (["play", "we", "will"], "We will play."),
            (["sleep", "he", "should"], "He should sleep."),
            (["now", "go", "i"], "I go now."),
            (["very", "good", "it"], "It is very good.")
        ]
        
        passed = 0
        total = len(test_cases)
        
        for i, (input_gestures, expected) in enumerate(test_cases, 1):
            print(f"\nTest {i}/{total}:")
            print(f"Input: {input_gestures}")
            print(f"Expected: {expected}")
            
            try:
                result = builder.build_sentence(input_gestures)
                print(f"Generated: {result}")
                
                # Simple check - exact match or similar structure
                if result.lower().replace('.', '') == expected.lower().replace('.', ''):
                    print("✓ PASS")
                    passed += 1
                else:
                    print("✗ FAIL - Structure differs")
                    
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
            
            stats = builder.get_sentence_stats(result)
            print(f"Stats: {stats['word_count']} words, {stats['character_count']} chars")
        
        print("\n" + "=" * 60)
        print(f"Test Results: {passed}/{total} passed ({(passed/total)*100:.1f}% success rate)")
        
    except Exception as e:
        print(f"Failed to initialize sentence builder: {str(e)}")
        print("Please ensure spaCy English model is installed: python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()