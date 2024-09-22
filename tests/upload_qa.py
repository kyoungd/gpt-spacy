import json
import requests
import re

def make_valid_json_string(string_data):
    placeholder_prefix = "PLACEHOLDER_"
    counter = 0
    placeholders = {}

    def replace_with_placeholder(match):
        nonlocal counter
        placeholder = f"{placeholder_prefix}{counter}"
        placeholders[placeholder] = match.group(0)
        counter += 1
        return placeholder

    # Identify and replace URLs and emails with placeholders
    string_data = re.sub(r'(https?://[^\s]+)', replace_with_placeholder, string_data)
    string_data = re.sub(r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9._-]+)', replace_with_placeholder, string_data)

    # Sanitize the string, preserving specified characters
    sanitized_string = re.sub(r'[^a-zA-Z0-9 _\-.,\'?!]', '', string_data)

    # Restore URLs and emails from placeholders
    sanitized_string = re.sub(f'{placeholder_prefix}(\\d+)', lambda match: placeholders[match.group(0)], sanitized_string)

    # Escape backslashes, double quotes, and control characters for JSON
    escaped_string = (
        sanitized_string
        .replace('\\', '\\\\')
        .replace('"', '\\"')
        .replace('\n', '\\n')
        .replace('\r', '\\r')
        .replace('\t', '\\t')
    )

    return escaped_string

# Replace with your actual bearer token
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzYsImlhdCI6MTcxNzY0NTE3NiwiZXhwIjoxNzIwMjM3MTc2fQ.zyoGYv2uKXpIoqAuPZzHp7vXzShrgWqrZdCmFET3K6o"

# Replace with your actual Strapi server URL
api_url = 'https://talkee.ai/api/customer-answers/customized/0'
# api_url = 'http://localhost:1337/api/customer-answers/customized/0'

headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}

qa_file_path = './tests/qa.json'

def main():
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as file:
            qa_data = json.load(file)

        for idx, item in enumerate(qa_data, start=1):
            question = item['question']
            answer = item['answer']

            # Perform PUT request to update the data
            put_response = requests.put(
                f'{api_url}',
                headers=headers,
                json={
                    'question': make_valid_json_string(question), 
                    'answer': make_valid_json_string(answer),
                    'type': 'customized',
                    'is_answered': True,
                    'is_answer_approved': False,
                    'is_sync_with_pinecone': False,
                    'is_pushed_to_pinecone': False
                }
            )

            if put_response.status_code in (200, 201):
                print(f'Row {idx} : Posted question: {question}')
            else:
                print(f'Row {idx} : Failed to post question: {question}. Status code: {put_response.status_code}')
                print(f'Response: {put_response.text}')

    except Exception as e:
        print(f'Error processing the data: {e}')

if __name__ == '__main__':
    main()
