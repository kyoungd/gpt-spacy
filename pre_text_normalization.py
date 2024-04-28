import spacy

# Load the SpaCy model outside of your function to improve efficiency
nlp = spacy.load("en_core_web_md")

def text_normalization_with_boundaries(text):
    # Process the input text
    doc = nlp(text)
    
    # Initialize a list to store normalized tokens
    normalized_tokens = []
    
    # Iterate through each token in the processed document
    for token in doc:
        if token.is_punct:
            if token.text in ['.', '?', '!']:  # Check if the punctuation should be preserved
                normalized_tokens.append(token.text)
        else:
            # Lowercase and remove special characters from non-punctuation tokens
            normalized_token = ''.join(char for char in token.text.lower() if char.isalnum())
            if normalized_token:  # Ensure the token is not empty
                normalized_tokens.append(normalized_token)
    
    # Join the normalized tokens back into a single string
    normalized_text = ' '.join(normalized_tokens)
    
    return normalized_text

def text_remove_stop_words_lemmanized(text) -> str:
    doc = nlp(text)
    cleaned_text: str = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    return cleaned_text


# Example text
input_text = "Text normalization involves converting text into a more uniform format, which can help in reducing the variability of the input data. I found this to be an excellent tutorial - very clear, great examples and thorough. thank you for sharing this and i look forward to seeing you continue with another covering machine learning in spacy."

# Perform text normalization
normalized_text = text_normalization_with_boundaries(input_text)

print("Original text:", input_text)
print("Normalized text:", normalized_text)


