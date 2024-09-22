import spacy
from spacy.matcher import Matcher
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load models and tools
nlp = spacy.load('en_core_web_md')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def SentenceSimilarityScore(sentence1, sentence2):
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
    return similarity_score

if __name__ == "__main__":
    # Initialize the classifier
    classifier = pipeline("text-classification", model="textattack/roberta-base-CoLA")

    # Test data: (sentence, is_complete)
    test_data = [
        ('A sentence with a error in the Hitchhiker’s Guide tot he Galaxy', True),
        ("My address is ", False),
        ("I am looking for ", False),
        ("This is a complete sentence", True),
        ("Although it was raining", False),
        ("Yes", True),
        ("The quick brown fox", False),
        ("Where are you going", True),
        ("I went to the store to buy", True),
        ("The cat sat on the", False),
        ("When the alarm went off", False)
    ]

    # Variables to track correct predictions
    correct_predictions = 0
    total_predictions = len(test_data)

    # Test the sentences
    for sentence, expected in test_data:
        result = classifier(sentence)
        label = result[0]['label']
        score = result[0]['score']
        is_complete = label == 'LABEL_0'
        correct = is_complete == expected
        correct_predictions += correct
        # if not correct:
        print(f"'{sentence}': {'Complete' if is_complete else 'Fragment'} (Expected: {'Complete' if expected else 'Fragment'}) - Incorrect, Score: {score}")

    # Calculate accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy:.2f}%")
