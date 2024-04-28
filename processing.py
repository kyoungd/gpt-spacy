import spacy
import re

greetings_and_phrases = [
    "hi", "hello", "good morning", "good afternoon", "good evening", "good day",
    "hey", "hey there", "hiya", "howdy", "greetings", "salutations",
    "how are you", "how's it going", "how are things", "how do you do", "what's up",
    "how have you been", "are you okay", "hope you're well", "hope this finds you well",
    "i was wondering", "could you please", "would you mind", "can you",
    "it's been a while", "long time no see", "nice to meet you", "pleased to meet you",
    "how can I help you", "what can I do for you", "is there anything",
    "just a quick question", "just wondering", "out of curiosity",
    "by the way", "incidentally", "for your information", "for your reference",
    "kind regards", "best regards", "sincerely", "yours truly", "yours sincerely",
    "looking forward to hearing from you", "awaiting your response", "let me know",
    "thank you", "thanks", "much appreciated", "thank you in advance",
    "sorry to bother you", "apologies for the inconvenience", "excuse me",
    "take care", "all the best", "best wishes", "cheers",
    "talk to you soon", "catch you later", "see you soon", "until next time",
    "goodbye", "bye", "farewell", "see ya",
    "have a good day", "have a nice day", "have a great day", "enjoy your day",
    "take it easy", "keep in touch", "stay safe", "stay well",
]

# Precompiled regex pattern for efficiency
pattern = re.compile(r'\b(?:' + '|'.join(re.escape(phrase) for phrase in greetings_and_phrases) + r')\b', re.IGNORECASE)

def remove_greetings_and_phrases(text, pattern=pattern):
    # Load the spaCy model
    nlp = spacy.load('en_core_web_md')
    
    # Disable components for efficiency except tokenizer and sentencizer
    # Add the sentencizer component to the pipeline
    if not nlp.has_pipe("sentencizer"):
        nlp.add_pipe("sentencizer")

    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize an empty list to hold the cleaned sentences
    cleaned_sentences = []
    
    # Iterate over sentences in the document
    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        # Check the entire sentence for any matching phrases
        if not pattern.search(sentence_text):
            cleaned_sentences.append(sentence_text)
        else:
            # Attempt to remove matching phrases from the sentence
            cleaned_sentence = pattern.sub("", sentence_text).strip()
            if cleaned_sentence:  # Only add non-empty sentences
                cleaned_sentences.append(cleaned_sentence)
    
    # Join the cleaned sentences back into a single string
    cleaned_text = " ".join(cleaned_sentences).replace("  ", " ")
    return cleaned_text

if __name__ == "__main__":
    # Example text
    text = "Hi, Amy. How are you? I was wondering, what are the services you offer?"

    # Remove greetings and common phrases
    cleaned_text = remove_greetings_and_phrases(text)

    print("Original Text:", text)
    print("Cleaned Text:", cleaned_text)
