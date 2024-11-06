import asyncio
import instructor
from openai import OpenAI
import os
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv
from pre_text_normalization import ner_and_pos_tagging_async

load_dotenv()

ai_model = os.getenv("OPENAI_MODEL_70B")
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = instructor.from_openai(OpenAI())

def Call(messages: List[Dict[str, str]], res_model: Any) -> Tuple[Optional[str], Optional[Any]]:
    try:
        data = client.chat.completions.create(
            model=ai_model,
            response_model=res_model,
            messages=messages,
            temperature=0,
         )
        return None, data
    except ValidationError as e:
        next_message = e.errors()[0]['msg']
        return next_message, None
    except Exception as ex:
        print(ex)
        return None, None

async def CallAsync(messages: List[Dict[str, str]], res_model: Any) -> Tuple[Optional[str], Optional[Any]]:
    try:
        data = await asyncio.to_thread(
            client.chat.completions.create,
            model=ai_model,
            response_model=res_model,
            messages=messages,
            temperature=0,
        )
        return None, data
    except ValidationError as e:
        next_message = e.errors()[0]['msg']
        return next_message, None
    except Exception as ex:
        print(ex)
        return None, None

class ProductInfo(BaseModel):
    summary: str = Field(..., description="A brief summary of the information in this product chunk (1-2 sentences)")
    additional_info: str = Field(..., description="Additional relevant information about the product (2-3 paragraphs)")
    generated_questions_answers: List[str] = Field(..., min_items=0, max_items=7, description="0-7 potential customer questions and answers about the product. Only include factual information based on the provided product description.")
    tags: List[str] = Field(..., min_items=10, max_items=15, description="10-15 relevant tags for the product")
    key_features: List[str] = Field(..., min_items=5, max_items=7, description="5-7 key features of the product")

    @model_validator(mode="before")
    def validate_all(cls, values):
        additional_info = values.get('additional_info', '')
        paragraphs = additional_info.split("\n\n")
        if not (2 <= len(paragraphs) <= 3):
            raise ValueError('additional_info must contain 2-3 paragraphs.')

        qa_list = values.get('generated_questions_answers', [])
        if not (0 <= len(qa_list) <= 7):
            raise ValueError('generated_questions_answers must contain 0-7 Q&A pairs.')
        for qa_item in qa_list:
            if not isinstance(qa_item, str) or not qa_item.strip():
                raise ValueError("Each item in 'generated_questions_answers' must be a non-empty string.")
        
        return values

    @classmethod
    def run(cls, product_info: str) -> Tuple[Optional[str], Optional['ProductInfo']]:
        conv = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that analyzes product descriptions to generate comprehensive information, Q&A, tags, and key features."
            },
            {
                "role": "user",
                "content": f"""Given the following entire product description:

{product_info}

Please provide:
1. Additional information (2-3 paragraphs)
2. 5-7 potential customer questions and answers about the product (each as a single string)
3. 10-15 relevant tags for the product
4. 5-7 key features of the product

Format your response as follows:
additional_info: [Your 2-3 paragraphs here]

generated_questions_answers:
- [Question 1 and Answer 1]
- [Question 2 and Answer 2]
[And so on for 5-7 Q&A pairs]

tags: [tag1, tag2, tag3, ..., tag15]

key_features:
- [Feature 1]
- [Feature 2]
[And so on for 5-7 key features]
"""
            }
        ]
        
        try:
            error, data = Call(conv, ProductInfo)
            if error:
                print(f"Error from Call: {error}")
                return error, None
            return None, data
        except ValidationError as e:
            print(f"Validation Error: {e}")
            return str(e), None
        except Exception as ex:
            print(f"Unexpected Error: {ex}")
            return str(ex), None

class Chunking(BaseModel):
    chunks: List[str] = Field(..., description="List of text chunks from the product description")

    @field_validator('chunks')
    @classmethod
    def validate_chunks(cls, chunks):
        if not chunks:
            raise ValueError("At least one chunk must be provided")
        if any(not chunk.strip() for chunk in chunks):
            raise ValueError("All chunks must contain non-empty strings")
        if len(chunks) > 10:
            raise ValueError("Too many chunks. Maximum allowed is 10")
        return chunks

    @classmethod
    def run(cls, text_block: str, ner_and_pos: Dict[str, Any]):
        """
        Perform chunking on the input text based on best fit.

        Args:
            text_block (str): The original product description text.
            ner_and_pos (Dict[str, Any]): Named Entity Recognition and Part-of-Speech tagging results.

        Returns:
            tuple: A tuple containing an error message if applicable, and a Chunking object with chunks.
        
        Example:
            error, data = Chunking.run("The new XYZ-1000 is a game-changer. It features advanced AI capabilities. This product is perfect for both home and office use. Its sleek design fits any decor.", ner_and_pos_data)
            # Returns: (None, Chunking object with chunks)
        """
        conv = [
            {
                "role": "system",
                "content": (
                    "You are an AI model specializing in text analysis and processing for WooCommerce product descriptions. "
                    "Your task is to perform the following operation on the given text:\n"
                    "Chunk the text into logical sections, considering semantic coherence, paragraph structure, and NER/POS information.\n\n"
                    "For chunking, consider the following guidelines:\n"
                    "- Use NER to identify product names, features, and specifications, and ensure these are kept together in chunks.\n"
                    "- Use POS tagging to identify noun phrases and verb phrases that describe product attributes or actions.\n"
                    "- Create chunks that capture coherent ideas or features about the product.\n"
                    "- Each chunk should be substantial enough to convey meaningful information.\n"
                    "- Balance between respecting natural paragraph breaks and maintaining semantic coherence.\n"
                    "- Aim for 3-7 chunks in total, depending on the length and complexity of the text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please process the following WooCommerce product description text:\n\n"
                    f"\"{text_block}\"\n\n"
                    f"NER and POS tagging results: {ner_and_pos}\n\n"
                    f"Provide the text chunks, utilizing the NER and POS information."
                )
            }
        ]
    
        error, data = Call(conv, Chunking)
        if error:
            return error, None
        return None, data

class ProductChunkInfo(BaseModel):
    generated_questions_answers: List[str] = Field(..., min_items=0, max_items=2, description="0-2 potential customer questions and answers related to this product chunk. Only include factual information based on the provided chunk.")
    tags: List[str] = Field(..., min_items=3, max_items=5, description="3-5 relevant tags for this chunk of product information")
    key_features: List[str] = Field(..., min_items=1, max_items=2, description="1-2 key features mentioned in this product chunk")
    ner: List[Any] = []
    text_chunk: str = ""

    @classmethod
    async def run(cls, chunk_text: str) -> Tuple[Optional[str], Optional['ProductChunkInfo']]:
        # Assume ner_and_pos_tagging is an async function or wrap it if necessary
        ner_and_pos = await ner_and_pos_tagging_async(chunk_text)

        conv = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that analyzes portions of product descriptions to generate concise summaries, Q&A, tags, and key features. You also interpret NER and POS tagging results."
            },
            {
                "role": "user",
                "content": f"""Given the following portion of a product description and its NER/POS tagging:

Chunk text:
{chunk_text}

NER and POS tagging:
{ner_and_pos}

Please provide:
1. 1-2 potential customer questions and answers (each as a single string)
2. 3-5 relevant tags
3. 1-2 key features
4. Interpret and organize the NER results
5. Interpret and organize the POS tagging results

generated_questions_answers:
- [Question 1 and Answer 1]
- [Question 2 and Answer 2] (if applicable)

tags: [tag1, tag2, tag3, tag4, tag5]

key_features:
- [Feature 1]
- [Feature 2] (if applicable)

ner:
- ["PERSON", "John Doe"]
- ["ORG", "Acme Inc"]
- ["PRODUCT", "Winter Jacket"]
... (other relevant NER entities)

"""
            }
        ]

        # Assume Call is an async function
        error, data = await CallAsync(conv, ProductChunkInfo)
        if data:
            data.ner = ner_and_pos["ner"]
            data.text_chunk = chunk_text
        return error, data
    

if __name__ == '__main__':
    pass
    # # Print chunk information
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"Chunk {i} Type: {chunk['type']}")
    #     print(f"Chunk {i} Text: {chunk['chunk_text']}")
    #     print(f"Chunk {i} Tags: {chunk['tags']}")
    #     print(f"Chunk {i} Key Features: {chunk['key_features']}")
    #     print(f"Chunk {i} NER: {chunk['ner']}")
    #     print("---")
