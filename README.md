
### Virtual Environment
pip install virtualenv
virtualenv --version
virtualenv <env-name>
source <env-name>/bin/activate
deactivate
python3.10 -m venv venv


### run a version
ngrok
# launch ngroke for local testing.
# and copy the URL to twilio website webhook for this number.
ngrok http 3000


### heroku and flask
Maks sure Procfile is there and it is correct.
Add gunicorn with pip and freeze it to requirements.txt


### create a requirement file
pip3 freeze > requirements.txt
pip3 install -r requirements.txt


### Preprocess text file
Put all your text file into a folder and name them .txt
and them call this funciton
this is under /texts folder

## change directory
cd texts
## Extract text from pdf and text files and combine it to a single file
python extractor.py <folder-path>
<folder-path> = path to the textdata folder.

## Convert the text file into pinecone matched json file.
python rewrite_to_json.py <file-path> <url> <company-code> <is-rewrite>
<file-path> = path to the consolidated text file.
<url> = url of the customer
<comapny-code> = company code of the customer
<is-rewrite> = Use chatGpt to rewrite the text.

## Rewrite content for clarity and simplicity
python airewrite.py <file-path>
<file-path> = path to the consolidated json file.

## Create summary content for better searching
python rewrite_summarize.py <file-path> <company-code>
## Create addtional summary content for better searching.
python rewrite_summary_different.py <file-path>
## Move back to home directoy
cd ..

## Push the json file to pinecone
python push_to_pinecone.py <file-path> <company_code>
<file-path> = path to the consolidated json file.
<comapny-code> = company code of the customer

failed_value = "I do not have answer to that question"
input
[
    { "id": "qefa11234", "datetime": "2023-01-01 00:01:00", "name": "User", "text": "hello there, Amy.  What services do you have." },
    { "id": "qefa11234", "datetime": "2023-01-01 00:01:00", "name": "AI", "text": "We have AI Receptionist and AI Answers.  They are AI That is designed to help your business run more effectively." },
    { "id": "qefa11234", "datetime": "2023-01-01 00:01:00", "name": "User", "text": "What is AI Receptionist?" },
    { "id": "qefa11234", "datetime": "2023-01-01 00:01:00", "name": "AI", "text": "AI Receptionist is designed to work as an Receptionist that can work on web and phone." },
    { "id": "qefa11234", "datetime": "2023-01-01 00:01:00", "name": "User", "text": "How much is it??" }
]
write a function that takes the input and return the last text from 'AI' that does not contain the failed_value text. Do a case insentive search. If there is no such text, return None.

output
[
    {"role": "system", "content": "answer as truthfully as you can."}
    {"role": "assistant", "content": "Pi is circumference of circle divided by diameter."},
    {"role": "user", "content": "What is its approximte number"},
    {"role": "assistant", "content": "text": "Who discovered it?"},
    {"role": "user", "content":  "Who discovered it?"},
]
The output always contains the "system" message.  And given the input, it takes the last four items, and copies them into the output.
Write a function that will take an input and creates an output.


Use <context> as context for entity coreference of the sentence in the <query>.  Rewrite the query by substituting pronoun with noun.
<context>AI Receptionist is a product or service offered by us. It is an artificial intelligence-powered system that can handle receptionist duties such as answering calls, scheduling appointments, and providing information to callers.</context>  <query>How much does it cost?</query>

cd texts
python combiner.py ./texts/-qdata/overall.txt https://2human.ai timdplr-gmail-com true
cd ..
push_to_pinecone.py ./texts/-qdata/overall.json 



Summarize the text.  Make it clear.
""" 

"""
# gpt_spacy


To install all the necessary packages listed in your code snippet, you need to run the pip install command with the package names. Some of these packages might require specific versions for compatibility, and additional dependencies will be installed automatically by pip.

Here is the command to install all the mentioned packages:

Install All Required Packages
sh
Copy code
pip install flask flask-cors fuzzywuzzy spacy sentence-transformers transformers instructor openai python-dotenv pydantic
Explanation of Each Package
flask: A lightweight web framework for Python.
flask-cors: Extension for handling Cross-Origin Resource Sharing (CORS), making cross-domain AJAX possible.
fuzzywuzzy: A library for fuzzy string matching.
spacy: A popular library for natural language processing.
sentence-transformers: A library to compute sentence embeddings using BERT-like models.
transformers: Hugging Face's library for state-of-the-art machine learning models (e.g., BERT, GPT).
instructor: A library that may be required for text-related tasks.
openai: The library for accessing OpenAI’s APIs.
python-dotenv: A library to read key-value pairs from .env files for environment variables.
pydantic: A data validation and settings management library using Python type annotations.
Additional Steps
Install SpaCy Model: After installing spacy, you will need to download the language model if you plan to use it:

sh
Copy code
python -m spacy download en_core_web_sm
Install fuzzywuzzy dependencies: fuzzywuzzy might need python-Levenshtein for better performance:

sh
Copy code
pip install python-Levenshtein

pip install tiktoken



sudo apt update
sudo apt install redis-server

pip install rq
rq worker
