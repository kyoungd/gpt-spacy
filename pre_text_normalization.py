import asyncio
import spacy
from transformers import AutoModel, AutoTokenizer
import logging
from typing import List, Optional, Dict, Any

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

@safe_text_processing
async def ner_and_pos_tagging_async(text: str) -> Dict[str, Any]:
    # Use asyncio.to_thread to run the NLP processing in a separate thread
    return await asyncio.to_thread(ner_and_pos_tagging, text)

# Synchronous version of the function, which is run in a separate thread
def ner_and_pos_tagging(text: str) -> Dict[str, Any]:
    doc = nlp(text[:1000000])  # Limit input size to prevent memory issues
    
    ner_results = []
    pos_results = []
    
    for ent in doc.ents:
        ner_results.append({
            "text": ent.text,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "label": ent.label_
        })
    
    for token in doc:
        pos_results.append({
            "text": token.text,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_
        })
    
    return {
        "ner": ner_results,
        "pos": pos_results
    }
def process_woocommerce_text(text: str) -> Optional[dict]:
    try:
        normalized_text = text_normalization_with_boundaries(text)
        cleaned_text = text_remove_stop_words_lemmatized(text)
        ner_pos_results = ner_and_pos_tagging(text)
        
        return {
            "original": text,
            "normalized": normalized_text,
            "cleaned": cleaned_text,
            "ner_pos": ner_pos_results
        }
    except Exception as e:
        logger.error(f"Error processing WooCommerce text: {str(e)}")
        return {
            "original": text,
            "normalized": text,
            "cleaned": text,
            "ner_pos": {"ner": [], "pos": []}
        }

if __name__ == "__main__":
    input_text = """
Title:
Men's Winter Jacket with Detachable Hoodie – Windproof, Insulated, and Waterproof

Short Description:
Stay warm and stylish this winter with our versatile men's winter jacket featuring a detachable hoodie. Designed for ultimate protection against the cold, this jacket is windproof, waterproof, and insulated for maximum warmth. Available in multiple sizes (S-XXL) and colors (Black, Navy, Grey, Olive Green), this jacket is perfect for everyday wear or outdoor adventures.

Long Description:
Stay Warm, Stay Dry, Stay Stylish
Our men's winter jacket with a detachable hoodie is your go-to choice for enduring the coldest months of the year while looking sharp. Meticulously designed for both comfort and functionality, this jacket combines top-tier materials and craftsmanship to provide excellent insulation and protection against harsh winter conditions.

Key Features:

Insulated for Warmth: Crafted with high-density synthetic insulation, this jacket traps heat to keep you warm in freezing temperatures, making it perfect for both everyday use and outdoor activities.
Windproof and Waterproof: Made from durable, high-quality fabric, the jacket is designed to shield you from strong winds and heavy rain or snow. The waterproof outer layer ensures that moisture stays out while maintaining breathability.
Detachable Hoodie: The jacket features a detachable hoodie with adjustable drawstrings, giving you flexibility based on your needs. Whether you want extra protection during storms or a more streamlined look on milder days, the hoodie can easily be removed or attached.
Multiple Pockets for Storage: The jacket comes with four spacious, zippered pockets — two on the chest and two on the sides — providing ample space for your essentials like your phone, wallet, gloves, and more. Additionally, there’s a secure interior pocket to keep your valuables safe.
Adjustable Fit: Equipped with adjustable cuffs and a drawstring hem, the jacket allows you to customize the fit and lock in heat, ensuring you stay comfortable no matter how cold it gets.
Soft Lining: The jacket’s interior is lined with a soft, fleece-like fabric that adds an extra layer of warmth and comfort, making it ideal for extended wear.
Stylish Design: Available in a variety of colors — Black, Navy, Grey, and Olive Green — the jacket complements any wardrobe. Its sleek design ensures that you can transition seamlessly from a casual outing to more rugged outdoor activities without compromising style.
Available Sizes:

Small (S): Chest 36-38 inches, Waist 29-31 inches
Medium (M): Chest 39-41 inches, Waist 32-34 inches
Large (L): Chest 42-44 inches, Waist 35-37 inches
Extra-Large (XL): Chest 45-47 inches, Waist 38-40 inches
Double Extra-Large (XXL): Chest 48-50 inches, Waist 41-43 inches
Color Options:

Black: A classic choice for a sleek and timeless look.
Navy: A versatile, deep blue tone perfect for any occasion.
Grey: A subtle and neutral shade that pairs well with a variety of outfits.
Olive Green: For those who want a rugged, outdoorsy feel with a splash of color.
Perfect for All Winter Activities: Whether you’re heading to work, running errands, or embarking on a winter hike, this jacket provides the right balance of warmth, protection, and style. The durable construction ensures long-lasting use, while the modern design keeps you looking sharp no matter where the day takes you.
"""
    
    result = process_woocommerce_text(input_text)
    if result:
        print("Original text:", result["original"])
        print("Normalized text:", result["normalized"])
        print("Cleaned and lemmatized text:", result["cleaned"])
        print("NER results:", result["ner_pos"]["ner"])
        print("POS tagging results:", result["ner_pos"]["pos"])
    else:
        print("Failed to process text")