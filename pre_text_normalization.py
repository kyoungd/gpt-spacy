import spacy
from transformers import AutoModel, AutoTokenizer
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load models and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces=True)
    model = AutoModel.from_pretrained("bert-base-uncased")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    # Fallback to a simple tokenizer if spaCy fails to load
    nlp = lambda text: type('obj', (object,), {'sents': [text], 'doc': [type('token', (object,), {'text': word, 'is_punct': False, 'is_stop': False, 'lemma_': word.lower()}) for word in text.split()]})()

def safe_text_processing(func):
    def wrapper(text: str, *args, **kwargs) -> str:
        if not isinstance(text, str):
            logger.warning(f"Input is not a string. Attempting to convert. Type: {type(text)}")
            try:
                text = str(text)
            except Exception as e:
                logger.error(f"Failed to convert input to string: {str(e)}")
                return text  # Return original input if conversion fails
        
        if not text.strip():
            logger.warning("Input text is empty or whitespace")
            return text  # Return original text if it's empty or whitespace
        
        try:
            return func(text, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return text  # Return original text if processing fails
    return wrapper

@safe_text_processing
def text_normalization_with_boundaries(text: str) -> str:
    doc = nlp(text[:1000000])  # Limit input size to prevent memory issues
    normalized_sents: List[str] = []
    
    for sent in doc.sents:
        normalized_tokens: List[str] = []
        for token in sent:
            if token.is_punct:
                if token.text in ['.', '?', '!']:
                    normalized_tokens.append(token.text)
            else:
                normalized_token = ''.join(char.lower() for char in token.text if char.isalnum())
                if normalized_token:
                    normalized_tokens.append(normalized_token)
        normalized_sents.append(' '.join(normalized_tokens))
    
    return '\n'.join(normalized_sents)

@safe_text_processing
def text_remove_stop_words_lemmatized(text: str) -> str:
    doc = nlp(text[:1000000])  # Limit input size to prevent memory issues
    cleaned_sents: List[str] = []
    
    for sent in doc.sents:
        cleaned_tokens = [token.lemma_ if not token.is_punct else token.text
                          for token in sent
                          if not token.is_stop or token.text in ['.', '?', '!']]
        cleaned_sents.append(' '.join(cleaned_tokens))
    
    return '\n'.join(cleaned_sents)

def process_woocommerce_text(text: str) -> Optional[dict]:
    try:
        normalized_text = text_normalization_with_boundaries(text)
        cleaned_text = text_remove_stop_words_lemmatized(text)
        
        return {
            "original": text,
            "normalized": normalized_text,
            "cleaned": cleaned_text
        }
    except Exception as e:
        logger.error(f"Error processing WooCommerce text: {str(e)}")
        return {
            "original": text,
            "normalized": text,
            "cleaned": text
        }

if __name__ == "__main__":
    input_text = "Text normalization involves converting text into a more uniform format, which can help in reducing the variability of the input data. I found this to be an excellent tutorial - very clear, great examples and thorough. thank you for sharing this and i look forward to seeing you continue with another covering machine learning in spacy."
    
    result = process_woocommerce_text(input_text)
    if result:
        print("Original text:", result["original"])
        print("Normalized text:", result["normalized"])
        print("Cleaned and lemmatized text:", result["cleaned"])
    else:
        print("Failed to process text")