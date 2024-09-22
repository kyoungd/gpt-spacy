from unittest import mock, TestCase
from unittest.mock import patch
# ------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(sys.path[0]))
# ------------------------------------
from app import app

class Test_Sentence(TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.url = '/tools/is-sentence-complete';
    
    def test_is_sentence_complete_1(self):
        question = "What is your name?"
        text = "My name is"
        with app.test_client() as client:
            result = client.post(self.url, json={"text_block": {
                "assistant" : question,
                "user": text }})
            self.assertTrue(result.status_code == 200)
            self.assertTrue(result['is_sentence_complete'] == True)

        # "My address is ",
        # "I am looking for ",
        # "Although it was raining",
        # "Go home",
        # "The quick brown fox",
        # "Where are you going",
        # "I went to the store to buy",
        # "The cat sat on the",
        # "She said that she would",
        # "Despite the challenges, we persevered",
        # "When the alarm went off"
