
### Virtual Environment
pip install virtualenv
virtualenv --version
virtualenv <env-name>
source <env-name>/bin/activate
deactivate


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
