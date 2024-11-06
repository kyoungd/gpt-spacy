import re
import string
from typing import List, Dict
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from spellchecker import SpellChecker
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
except Exception as e:
    logging.error(f"Error downloading NLTK data: {str(e)}")
    raise

class NormalizationLevel(Enum):
    BASIC = "basic"  # just lowercase and punctuation
    INTERMEDIATE = "intermediate"  # includes spell check
    FULL = "full"  # includes lemmatization/stemming

@dataclass
class NormalizationResult:
    """Container for normalization results"""
    original_text: str
    normalized_text: str
    corrections: Dict[str, str]  # original -> corrected words
    tokens: List[str]
    normalization_level: NormalizationLevel

class TextNormalizer:
    """
    Handles text normalization for cigar-related queries
    Includes: lowercase conversion, punctuation removal, spell checking,
    and optional stemming/lemmatization
    """
    
    def __init__(self, custom_dictionary: List[str] = None):
        self.spell_checker = SpellChecker()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Add custom cigar-related terms to spell checker
        if custom_dictionary:
            self.spell_checker.word_frequency.load_words(custom_dictionary)
        
        # Common cigar-related terms to preserve
        self.preserve_terms = {
            'mild', 'medium', 'full', 
            'maduro', 'claro', 'colorado', 'oscuro',
            'corona', 'robusto', 'torpedo', 'churchill',
            'nicaraguan', 'dominican', 'cuban', 'honduran'
        }

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation while preserving hyphenated words and numbers"""
        # Preserve hyphens between words
        text = re.sub(r'(?<![\w-])-|-(?![\w-])', ' ', text)
        # Remove other punctuation
        translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        return text.translate(translator)

    def _spell_check(self, tokens: List[str]) -> Dict[str, str]:
        """
        Perform spell checking while preserving domain-specific terms
        Returns dictionary of corrections made
        """
        corrections = {}
        corrected_tokens = []
        
        for token in tokens:
            # Skip preserved terms, numbers, and hyphenated words
            if (token.lower() in self.preserve_terms or 
                any(c.isdigit() for c in token) or
                '-' in token):
                corrected_tokens.append(token)
                continue
            
            # Check spelling
            if token not in self.spell_checker.known([token]):
                correction = self.spell_checker.correction(token)
                if correction and correction != token:
                    corrections[token] = correction
                    corrected_tokens.append(correction)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
                
        return ' '.join(corrected_tokens), corrections

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens while preserving domain-specific terms"""
        return [
            self.lemmatizer.lemmatize(token) if token.lower() not in self.preserve_terms
            else token
            for token in tokens
        ]

    def _stem(self, tokens: List[str]) -> List[str]:
        """Stem tokens while preserving domain-specific terms"""
        return [
            self.stemmer.stem(token) if token.lower() not in self.preserve_terms
            else token
            for token in tokens
        ]

    def normalize(self, 
                 text: str, 
                 level: NormalizationLevel = NormalizationLevel.INTERMEDIATE,
                 use_lemmatization: bool = True) -> NormalizationResult:
        try:
            if not text:
                raise ValueError("Input text cannot be empty.")
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation
            text = self._remove_punctuation(text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            corrections = {}
            
            # Apply intermediate level (spell checking)
            if level in [NormalizationLevel.INTERMEDIATE, NormalizationLevel.FULL]:
                text, corrections = self._spell_check(tokens)
                tokens = word_tokenize(text)
            
            # Apply full level (lemmatization/stemming)
            if level == NormalizationLevel.FULL:
                if use_lemmatization:
                    tokens = self._lemmatize(tokens)
                else:
                    tokens = self._stem(tokens)
                text = ' '.join(tokens)
            
            return NormalizationResult(
                original_text=text,
                normalized_text=text,
                corrections=corrections,
                tokens=tokens,
                normalization_level=level
            )
        except Exception as e:
            logging.error(f"Error during text normalization: {str(e)}")
            raise

def main():
    # Example usage
    custom_dictionary = [
        'padron', 'cohiba', 'montecristo', 'davidoff',
        'nicaraguan', 'dominican', 'honduran', 'maduro',
        'robusto', 'torpedo', 'churchill'
    ]
    
    normalizer = TextNormalizer(custom_dictionary)
    
    # Test cases
    test_queries = [
        "Show me STRONG Padron cigars under $15",
        "Find mild-to-medium Nicaraguan robustos",
        "What maduro wrappd cigars are availible?",  # intentional typos
        "Compare corona gordas with churchills"
    ]
    
    print("Text Normalization Examples")
    print("=========================")
    
    try:
        for query in test_queries:
            print(f"\nOriginal Query: {query}")
            
            # Try different normalization levels
            for level in NormalizationLevel:
                result = normalizer.normalize(query, level=level)
                print(f"\n{level.value.title()} Normalization:")
                print(f"Normalized: {result.normalized_text}")
                if result.corrections:
                    print("Corrections made:", result.corrections)
                print(f"Tokens: {result.tokens}")
            
            print("-" * 50)
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
    main()