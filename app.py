from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from sentence import SentenceSimilarityScore
from fuzzywuzzy import fuzz, process
from pre_text_normalization import text_normalization_with_boundaries, text_remove_stop_words_lemmanized

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

@app.route('/text-clean', methods=['POST'])
def text_clean():
    try:
        data = request.get_json()
        text = data.get('text_block')
        text_block = text_normalization_with_boundaries(text)
        return jsonify({'text': text_block})
    except Exception as e:
        print(f"Error deleting data: {e}")
        abort(str(e), 501)

@app.route('/text-normalize', methods=['POST'])
def text_normalize():
    try:
        data = request.get_json()
        text = data.get('text_block')
        text_block = text_remove_stop_words_lemmanized(text)
        return jsonify({'text': text_block})
    except Exception as e:
        print(f"Error deleting data: {e}")
        abort(str(e), 501)


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})

if __name__ == '__main__':
    app.run(port=3011)
