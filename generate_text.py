import openai
import os
from dotenv import load_dotenv

load_dotenv()

model = os.get("OPENAI_MODEL_GENERATE_TEXT")

class GenerativeAnswers:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_for_entire_document(self, text):
        prompt = f"""Given the following entire product description:

{text}

Please provide:
1. Additional relevant information about this product (2-3 paragraphs).
2. 5-7 potential customer questions and detailed answers about this product.
3. A list of 10-15 relevant tags for this product (comma-separated).
4. 5-7 key features of the product (comma-separated list).

Format your response as follows:
Additional Information:
[Your generated additional information here]

Q&A:
[List of potential questions and answers]

Tags:
[comma-separated list of tags]

Key Features:
[comma-separated list of key features]
"""

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes product descriptions to generate comprehensive information, Q&A, tags, and key features."},
                {"role": "user", "content": prompt}
            ]
        )

        return self.parse_response(response.choices[0].message['content'])

    def generate_for_chunk(self, text):
        prompt = f"""Given the following portion of a product description:

{text}

Please provide:
1. A brief summary of the information in this chunk (1-2 sentences).
2. 1-2 potential customer questions and concise answers related to this chunk.
3. 3-5 relevant tags for this chunk of information (comma-separated).
4. 1-2 key features mentioned in this chunk (comma-separated list).

Format your response as follows:
Summary:
[Your generated summary here]

Q&A:
[List of potential questions and answers]

Tags:
[comma-separated list of tags]

Key Features:
[comma-separated list of key features]
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes portions of product descriptions to generate concise summaries, Q&A, tags, and key features."},
                {"role": "user", "content": prompt}
            ]
        )

        return self.parse_response(response.choices[0].message['content'])

    def parse_response(self, response_text):
        sections = response_text.split('\n\n')
        result = {}
        
        for section in sections:
            if section.startswith('Additional Information:') or section.startswith('Summary:'):
                result['info'] = section.split(':', 1)[1].strip()
            elif section.startswith('Q&A:'):
                result['qa'] = section.replace('Q&A:', '').strip()
            elif section.startswith('Tags:'):
                result['tags'] = [tag.strip() for tag in section.replace('Tags:', '').strip().split(',')]
            elif section.startswith('Key Features:'):
                result['key_features'] = [feature.strip() for feature in section.replace('Key Features:', '').strip().split(',')]
        
        return result

    def process_product(self, product_info):
        return self.generate_for_entire_document(product_info)

    def process_chunk(self, chunk_text):
        return self.generate_for_chunk(chunk_text)
    
if __name__ == "__main__":
    api_key = "your-openai-api-key"
    gen_answers = GenerativeAnswers(api_key)

    product_info = "Your product title, short description, description, and metadata here"

    # Process the entire product
    product_result = gen_answers.process_product(product_info)
    print("Product Info:", product_result['info'])
    print("Product Q&A:", product_result['qa'])
    print("Product Tags:", product_result['tags'])
    print("Product Key Features:", product_result['key_features'])

    # Process individual chunks
    chunks = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
    chunk_results = []
    for chunk in chunks:
        chunk_result = gen_answers.process_chunk(chunk)
        chunk_results.append(chunk_result)
        print(f"Chunk Summary: {chunk_result['info']}")
        print(f"Chunk Q&A: {chunk_result['qa']}")
        print(f"Chunk Tags: {chunk_result['tags']}")
        print(f"Chunk Key Features: {chunk_result['key_features']}")

    # If you need to aggregate tags and key features
    all_tags = set(product_result['tags'])
    all_key_features = set(product_result['key_features'])
    for result in chunk_results:
        all_tags.update(result['tags'])
        all_key_features.update(result['key_features'])

    print("All Unique Tags:", list(all_tags))
    print("All Unique Key Features:", list(all_key_features))