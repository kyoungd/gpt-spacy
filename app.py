from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence import SentenceSimilarityScore
from fuzzywuzzy import fuzz, process

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

def SentenceSimilarity(sentence1, sentence2):
    # Implement your sentence similarity function here.
    # The dummy function below returns a fixed score.
    return 0.8

@app.route('/score', methods=['POST'])
def compare_sentences():
    data = request.get_json()
    sentence1 = data.get('sentence1')
    sentence2 = data.get('sentence2')

    if sentence1 is None or sentence2 is None:
        return jsonify({'error': 'Invalid input data'}), 400

    score = SentenceSimilarityScore(sentence1, sentence2)
    return jsonify({'score': score})

@app.route('/test', methods=['POST'])
def test():
    sentence1 = "Hi there"
    sentence2 = "Good morning"

    score = SentenceSimilarityScore(sentence1, sentence2)
    return jsonify({'score': score})

@app.route('/address-match', methods=['POST'])
def address_match():
    data = request.get_json()
    sentence1 = data.get('address')
    available_addresses = data.get('available_addresses')

    if sentence1 is None or available_addresses is None:
        return jsonify({'error': 'Invalid input data'}), 400

    best_match = process.extractOne(sentence1, available_addresses)
    return jsonify({'best_match': best_match[0], 'score': best_match[1]})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})

if __name__ == '__main__':
    app.run(port=3011)
