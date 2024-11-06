import instructor
from openai import OpenAI
import os
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv

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

class CoreferenceResolution(BaseModel):
    clean_text: str = Field(..., description="""
        The text block where pronouns have been replaced with appropriate nouns. This transformation aims to clarify subjects and objects in the text.
    """)

    @classmethod
    def run(cls, text_block: str):
        """
        Replace pronouns in a given text block with nouns to enhance clarity.

        Args:
            text_block (str): The original text containing pronouns.

        Returns:
            tuple: A tuple containing an error message if applicable, and a JSON object with the cleaned text.
        
        Example:
            error, data = CoreferenceResolution.run("She said that she would help her.")
            # Returns: (None, {'clean_text': 'The woman said that the woman would help the other woman.'})
        """
        # Constructing the conversation with improved instructions for the AI model
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
        
        # Call the AI model with the improved prompt
        error, data = Call(conv, CoreferenceResolution)
        if error:
            return error, ""
        return None, data.clean_text

class ChunkComparisonWithOriginalText(BaseModel):
    similarity_score: int = Field(..., ge=0, le=100, description="Estimate the similarity between original text and the list of chunks. 100 means all content is preserved in the chunks, 0 means no content is preserved.")
    difference_text : str = Field(..., description="A description of the differences between the original text and the list of chunks.")
    difference_details : List[str] = Field(..., description="A detail list of differences between the original text and the list of chunks.")

    @classmethod
    def run(cls, original_text: str, chunks: str):
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
                    "\nProvide only the integer score as your response, without any explanation."
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
        
        # Call the AI model with the improved prompt
        error, data = Call(conv, ChunkComparisonWithOriginalText)
        return error, data


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
