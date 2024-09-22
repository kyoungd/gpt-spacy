import instructor
from openai import OpenAI
import os
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv

load_dotenv()

ai_model = os.getenv("OPENAI_MODEL_8B")
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = instructor.from_openai(OpenAI())

def Call(messages: List[Dict[str, str]], res_model: Any, model_use: str = None) -> Tuple[Optional[str], Optional[Any]]:
    if model_use is None:
        model_use = ai_model
    try:
        data = client.chat.completions.create(
            model=model_use,
            response_model=res_model,
            messages=messages,
            temperature=0,
         )
        return None, data
    except ValidationError as e:
        next_message = e.errors()[0]['msg']
        return next_message, None
    except Exception as ex:
        next_message = str(ex)
        return next_message, None

class CoreferenceResolution(BaseModel):
    clean_text: str = Field(..., description="The text block where pronouns have been replaced with appropriate nouns. This transformation aims to clarify subjects and objects in the text.")

    @classmethod
    def run(cls, text_block: str):
        conv = [
            {
                "role": "system",
                "content": (
                    "You are an AI model specializing in coreference resolution. "
                    "Your task is to replace pronouns in the given text block with the appropriate nouns. "
                    "When resolving pronouns, consider the entire context and replace each pronoun with the most specific noun possible. "
                    "Ensure the replacements are accurate, maintain the original meaning, and enhance clarity. "
                    "For example:\n"
                    "- Input: 'She said that she would help her.'\n"
                    "- Output: 'The woman said that the woman would help the other woman.'\n\n"
                    "If the context does not provide enough information to determine the specific noun, make a best guess while keeping the text coherent."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please replace all pronouns with the correct nouns in the following text, preserving the original context and meaning:\n\n"
                    f"Text block: \"{text_block}\""
                )
            }
        ]
        
        error, data = Call(conv, CoreferenceResolution)
        if error:
            return error, ""
        return None, data.clean_text

class Chunking(BaseModel):
    chunks: List[str] = Field(..., description="List of text chunks from the product description")
    resolved_text: str = Field(..., description="The full text after coreference resolution")

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

    @field_validator('resolved_text')
    @classmethod
    def validate_resolved_text(cls, text):
        if not text.strip():
            raise ValueError("Resolved text cannot be empty")
        if len(text) < 50:
            raise ValueError("Resolved text is too short. Minimum length is 50 characters")
        return text

    @model_validator(mode='after')
    def check_consistency(self):
        total_chunk_length = sum(len(chunk) for chunk in self.chunks)
        resolved_text_length = len(self.resolved_text)
        
        if not (0.8 * total_chunk_length <= resolved_text_length <= 1.2 * total_chunk_length):
            raise ValueError("The total length of chunks should be approximately equal to the length of the resolved text")
        
        return self

    @classmethod
    def run(cls, text_block: str):
        conv = [
            {
                "role": "system",
                "content": (
                    "You are an AI model specializing in text analysis and processing for WooCommerce product descriptions. "
                    "Your task is to perform the following operations on the given text:\n"
                    "1. Chunk the text into logical sections, considering both semantic coherence and paragraph structure.\n"
                    "2. Perform coreference resolution on the entire text.\n\n"
                    "For chunking, consider the following guidelines:\n"
                    "- Create chunks that capture coherent ideas or features about the product.\n"
                    "- Each chunk should be substantial enough to convey meaningful information.\n"
                    "- Balance between respecting natural paragraph breaks and maintaining semantic coherence.\n"
                    "- Aim for 3-7 chunks in total, depending on the length and complexity of the text.\n\n"
                    "For coreference resolution:\n"
                    "- Replace pronouns with appropriate nouns, focusing on product-related terms.\n"
                    "- Ensure the replacements maintain the original meaning and enhance clarity.\n"
                    "- If the context is unclear, use general terms like 'the product' or 'this item'."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Please process the following WooCommerce product description text:\n\n"
                    f"\"{text_block}\"\n\n"
                    f"Provide the text chunks and perform coreference resolution on the entire text."
                )
            }
        ]
        
        error, data = Call(conv, Chunking)
        if error:
            return error, None
        return None, data

class ChunkComparisonWithOriginalText(BaseModel):
    similarity_score: int = Field(..., ge=0, le=100, description="Estimate the similarity between original text and the list of chunks. 100 means all content is preserved in the chunks, 0 means no content is preserved.")
    difference_text: str = Field(..., description="A description of the differences between the original text and the list of chunks.")
    difference_details: List[str] = Field(..., description="A detail list of differences between the original text and the list of chunks.")

    @classmethod
    def run(cls, original_text: str, chunks: List[str]):
        conv = [
            {
                "role": "system",
                "content": (
                    "You are an AI model analyzing the results of a text chunking process. "
                    "Your task is to compare the original text with the chunks of paragraphs and provide a similarity score. "
                    "The similarity score should be an integer between 0 and 100, where:"
                    "\n- 100 indicates that all meaningful content from the original text is preserved in the chunks."
                    "\n- 0 indicates that none of the meaningful content from the original text is present in the chunks."
                    "\n- Scores in between represent the percentage of meaningful content preserved."
                    "\nFocus on content preservation rather than exact word matching. Consider:"
                    "\n1. Key information and main ideas"
                    "\n2. Important details and examples"
                    "\n3. Overall meaning and context"
                    "\n4. Logical flow and structure of the content"
                )
            },
            {
                "role": "assistant",
                "content": (
                    f"Original text:\n\"{original_text}\"\n\n"
                )
            }
        ]
        for chunk in chunks:
            conv.append({
                "role": "user",
                "content": (
                    f"Chunk : \n\"{chunk}\"\n\n"
                )
            })
        
        model70b = os.getenv("OPENAI_MODEL_70B")
        error, data = Call(conv, ChunkComparisonWithOriginalText, model70b)
        return error, data

class ProductInfo(BaseModel):
    additional_info: str = Field(default="", description="Additional relevant information about the product (up to 3 paragraphs)")
    generated_questions_answers: List[str] = Field(default_factory=list, description="0-7 potential customer questions and answers about the product")
    tags: List[str] = Field(default_factory=list, description="Up to 15 relevant tags for the product")
    key_features: List[str] = Field(default_factory=list, description="Up to 7 key features of the product")

    @model_validator(mode="before")
    def validate_all(cls, values):
        additional_info = values.get('additional_info', '')
        paragraphs = additional_info.split("\n\n")
        values['additional_info'] = "\n\n".join(paragraphs[:3])

        qa_list = values.get('generated_questions_answers', [])
        values['generated_questions_answers'] = [qa for qa in qa_list[:7] if isinstance(qa, str) and qa.strip()]

        tags = values.get('tags', [])
        values['tags'] = [tag for tag in tags[:15] if isinstance(tag, str) and tag.strip()]

        features = values.get('key_features', [])
        values['key_features'] = [feature for feature in features[:7] if isinstance(feature, str) and feature.strip()]

        return values

    @classmethod
    def run(cls, product_info: str) -> Tuple[Optional[str], Optional['ProductInfo']]:
        conv = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that analyzes product descriptions to generate comprehensive information, Q&A, tags, and key features. If the input is insufficient or unclear, provide as much information as possible based on what's given."
            },
            {
                "role": "user",
                "content": f"""Given the following product description (which may be incomplete or unclear):

{product_info}

Please provide:
1. Additional information (up to 3 paragraphs)
2. Up to 7 potential customer questions and answers about the product (each as a single string)
3. Up to 15 relevant tags for the product
4. Up to 7 key features of the product

If the description is insufficient, provide as much information as you can infer or generate placeholder content.

Format your response as follows:
additional_info: [Your paragraphs here]

generated_questions_answers:
- [Question 1 and Answer 1]
- [Question 2 and Answer 2]
[And so on for up to 7 Q&A pairs]

tags: [tag1, tag2, tag3, ..., up to tag15]

key_features:
- [Feature 1]
- [Feature 2]
[And so on for up to 7 key features]
"""
            }
        ]
        
        try:
            error, data = Call(conv, ProductInfo)
            if error:
                print(f"Error from Call: {error}")
                return f"Error in API call: {error}", None
            return None, data
        except ValidationError as e:
            error_msg = f"Validation Error: {str(e)}"
            print(error_msg)
            return error_msg, None
        except Exception as ex:
            error_msg = f"Unexpected Error: {str(ex)}"
            print(error_msg)
            return error_msg, None

class ProductChunkInfo(BaseModel):
    chunk_summaries: List[str] = Field(..., description="Brief summaries of the information in each product chunk (up to 2 sentences each)")
    generated_questions_answers: List[List[str]] = Field(..., description="0-2 potential customer questions and answers related to each product chunk")
    tags: List[List[str]] = Field(..., description="Up to 5 relevant tags for each chunk of product information")
    key_features: List[List[str]] = Field(..., description="Up to 2 key features mentioned in each product chunk")

    @model_validator(mode="before")
    def validate_all(cls, values):
        values['chunk_summaries'] = [summary.strip() for summary in values.get('chunk_summaries', []) if isinstance(summary, str) and summary.strip()]
        values['generated_questions_answers'] = [qa_list if isinstance(qa_list, list) else [qa_list] for qa_list in values.get('generated_questions_answers', []) if qa_list]
        values['tags'] = [tag_list if isinstance(tag_list, list) else [tag_list] for tag_list in values.get('tags', []) if tag_list]
        values['key_features'] = [feature_list if isinstance(feature_list, list) else [feature_list] for feature_list in values.get('key_features', []) if feature_list]
        return values

    @classmethod
    def run(cls, chunks: List[str]) -> Tuple[Optional[str], Optional['ProductChunkInfo']]:
        conv = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that analyzes portions of product descriptions to generate concise summaries, Q&A, tags, and key features. Process multiple chunks of text at once, providing separate analysis for each chunk."
            },
            {
                "role": "user",
                "content": f"""Given the following chunks of a product description:

{chunks}

For each chunk, please provide:
1. A brief summary (up to 2 sentences)
2. Up to 2 potential customer questions and answers (each as a single string)
3. Up to 5 relevant tags
4. Up to 2 key features

Format your response as follows:
chunk_summaries:
- [Summary for chunk 1]
- [Summary for chunk 2]
...

generated_questions_answers:
- [Q&As for chunk 1]
- [Q&As for chunk 2]
...

tags:
- [Tags for chunk 1]
- [Tags for chunk 2]
...

key_features:
- [Features for chunk 1]
- [Features for chunk 2]
...

Ensure that each list has the same number of items, corresponding to the number of input chunks.
"""
            }
        ]
        
        try:
            error, data = Call(conv, ProductChunkInfo)
            if error:
                print(f"Error from Call: {error}")
                return f"Error in API call: {error}", None
            return None, data
        except ValidationError as e:
            error_msg = f"Validation Error: {str(e)}"
            print(error_msg)
            return error_msg, None
        except Exception as ex:
            error_msg = f"Unexpected Error: {str(ex)}"
            print(error_msg)
            return error_msg, None



if __name__ == '__main__':
    product_info = """
Title:
Men's Winter Jacket with Detachable Hoodie – Windproof, Insulated, and Waterproof

Short Description:
Stay warm and stylish this winter with our versatile men's winter jacket featuring a detachable hoodie. Designed for ultimate protection against the cold, this jacket is windproof, waterproof, and insulated for maximum warmth. Available in multiple sizes (S-XXL) and colors (Black, Navy, Grey, Olive Green), this jacket is perfect for everyday wear or outdoor adventures.

Long Description:
Stay Warm, Stay Dry, Stay Stylish
Our men's winter jacket with a detachable hoodie is your go-to choice for enduring the coldest months of the year while looking sharp. Meticulously designed for both comfort and functionality, this jacket combines top-tier materials and craftsmanship to provide excellent insulation and protection against harsh winter conditions.

Key Features:

Insulated for Warmth: Crafted with high-density synthetic insulation, this jacket traps heat to keep you warm in freezing temperatures, making it perfect for both everyday use and outdoor activities.
Windproof and Waterproof: Made from durable, high-quality fabric, the jacket is designed to shield you from strong winds and heavy rain or snow. The waterproof outer layer ensures that moisture stays out while maintaining breathability.
Detachable Hoodie: The jacket features a detachable hoodie with adjustable drawstrings, giving you flexibility based on your needs. Whether you want extra protection during storms or a more streamlined look on milder days, the hoodie can easily be removed or attached.
Multiple Pockets for Storage: The jacket comes with four spacious, zippered pockets — two on the chest and two on the sides — providing ample space for your essentials like your phone, wallet, gloves, and more. Additionally, there’s a secure interior pocket to keep your valuables safe.
Adjustable Fit: Equipped with adjustable cuffs and a drawstring hem, the jacket allows you to customize the fit and lock in heat, ensuring you stay comfortable no matter how cold it gets.
Soft Lining: The jacket’s interior is lined with a soft, fleece-like fabric that adds an extra layer of warmth and comfort, making it ideal for extended wear.
Stylish Design: Available in a variety of colors — Black, Navy, Grey, and Olive Green — the jacket complements any wardrobe. Its sleek design ensures that you can transition seamlessly from a casual outing to more rugged outdoor activities without compromising style.
Available Sizes:

Small (S): Chest 36-38 inches, Waist 29-31 inches
Medium (M): Chest 39-41 inches, Waist 32-34 inches
Large (L): Chest 42-44 inches, Waist 35-37 inches
Extra-Large (XL): Chest 45-47 inches, Waist 38-40 inches
Double Extra-Large (XXL): Chest 48-50 inches, Waist 41-43 inches
Color Options:

Black: A classic choice for a sleek and timeless look.
Navy: A versatile, deep blue tone perfect for any occasion.
Grey: A subtle and neutral shade that pairs well with a variety of outfits.
Olive Green: For those who want a rugged, outdoorsy feel with a splash of color.
Perfect for All Winter Activities: Whether you’re heading to work, running errands, or embarking on a winter hike, this jacket provides the right balance of warmth, protection, and style. The durable construction ensures long-lasting use, while the modern design keeps you looking sharp no matter where the day takes you.
"""
    
    # Process the entire product
    error, product_result = ProductInfo.run(product_info)
    if error:
        print(f"Error processing product: {error}")
    else:
        print("Product Info:", product_result.additional_info)
        print("Product Q&A:", product_result.generated_questions_answers)
        print("Product Tags:", product_result.tags)
        print("Product Key Features:", product_result.key_features)

    total_text = product_info + "\n\n" + product_result.additional_info
    # Use Chunking class to chunk the product description
    error, chunking_result = Chunking.run(total_text)
    if error:
        print(f"Error chunking product description: {error}")
    else:
        print("Chunked and resolved text:", chunking_result.resolved_text)
        
        # Process all chunks at once
        error, chunk_process_results = ProductChunkInfo.run(chunking_result.chunks)
        if error:
            print(f"Error processing chunks: {error}")
        else:
            for summary in chunk_process_results.chunk_summaries:
                print(f"Chunk Summary: {summary}")
            for qa in chunk_process_results.generated_questions_answers:
                print(f"Chunk Q&A: {qa}")
            for tags in chunk_process_results.tags:
                print(f"Chunk Tags: {tags}")
            for features in chunk_process_results.key_features:
                print(f"Chunk Key Features: {features}")

            # Aggregate tags, key features, and Q&As
            all_tags = set(product_result.tags)
            all_key_features = set(product_result.key_features)
            all_generated_questions_answers = product_result.generated_questions_answers.copy()

            for tas in chunk_process_results.tags:
                all_tags.update(tas)
            for kfs in chunk_process_results.key_features:
                all_key_features.update(kfs)
            for qas in chunk_process_results.generated_questions_answers:
                all_generated_questions_answers.extend(qas)

            # Remove duplicates from all_generated_questions_answers while preserving order
            all_generated_questions_answers = list(dict.fromkeys(all_generated_questions_answers))

        # Create a chunks list that combines all chunks and all_generated_questions_answers
        chunks = chunking_result.chunks + all_generated_questions_answers

        print("\nAll Unique Tags:", list(all_tags))
        print("All Unique Key Features:", list(all_key_features))
        print("All Generated Questions and Answers:", all_generated_questions_answers)
        print("Combined Chunks and Q&As:", chunks)

    # Compare original text with chunks
    error, comparison_result = ChunkComparisonWithOriginalText.run(product_info, chunks)
    if error:
        print(f"Error comparing original text with chunks: {error}")
    else:
        print(f"\nSimilarity Score: {comparison_result.similarity_score}")
        print(f"Difference Text: {comparison_result.difference_text}")
        print(f"Difference Details: {comparison_result.difference_details}")