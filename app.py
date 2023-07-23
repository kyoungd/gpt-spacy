from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence import SentenceSimilarityScore

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

if __name__ == '__main__':
    app.run(port=3011)
