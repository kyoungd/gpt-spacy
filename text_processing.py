import asyncio
import time
import json
import os
from datetime import datetime
from ai import ChunkComparisonWithOriginalText
from call_ai import ProductInfo, ProductChunkInfo, Chunking
from pre_text_normalization import text_normalization_with_boundaries, text_remove_stop_words_lemmatized, ner_and_pos_tagging
from util import http_put
from dotenv import load_dotenv

load_dotenv()

class PreprocessTextForRAG:

    def __init__(self):
        self.global_chunks = []

    def init_chunking(self, final_text, ner_and_pos):
        error, chunking_result = Chunking.run(final_text, ner_and_pos)
        if error:
            print(f"Error processing chunking: {error}")
            return self.global_chunks
        # Process individual global_chunks
        for chunk in chunking_result.chunks:
            self.global_chunks.append(chunk)

    def init_qa_chunks(self, qa_chunks):
        for chunk in qa_chunks:
            self.global_chunks.append(chunk)

    async def run_chunking(self):
        all_chunks = []
        tasks = []
        for i, chunk in enumerate(self.global_chunks, 1):
            tasks.append(ProductChunkInfo.run(chunk))

        # Run all tasks concurrently and wait until they are all complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"Error processing chunk {i}: {result}")
                continue
            error, chunk_result = result
            if error:
                print(f"Error processing chunk {i}: {error}")
                continue
            new_chunk = {
                "type": "content",
                "chunk_text": chunk_result.text_chunk + ' '.join(chunk_result.generated_questions_answers),
                "tags": chunk_result.tags,
                "key_features": chunk_result.key_features,
                "ner": chunk_result.ner,
            }
            all_chunks.append(new_chunk)
        return all_chunks

    def run(self, text_block, wp_action_id):
        try:
            text1 = text_normalization_with_boundaries(text_block)
            text2 = text_remove_stop_words_lemmatized(text1)
            ner_and_pos = ner_and_pos_tagging(text2)
            
            # Process the entire product
            error, product_result = ProductInfo.run(text2)
            if error:
                return {
                    "error": f"Error processing product: {error}",
                    "chunks": [],
                }

            # Create summary chunk
            summary_chunk = {
                "type": "summary",
                "chunk_text": product_result.summary,
                "tags": product_result.tags,
                "key_features": product_result.key_features,
                "ner": ner_and_pos['ner']
            }

            # Process original text
            self.init_chunking(text2, ner_and_pos)

            # Process additional info
            text_ai1 = text_normalization_with_boundaries(product_result.additional_info)
            text_ai2 = text_remove_stop_words_lemmatized(text_ai1)
            self.init_chunking(text_ai2, ner_and_pos)

            # Process Q&As as separate chunks
            self.init_qa_chunks(product_result.generated_questions_answers)

            final_chunks = asyncio.run(self.run_chunking())

            # Combine all chunks
            chunks = [summary_chunk] + final_chunks

            url = f"{os.getenv('BASE_URL_ADMIN')}/api/wp-actions/{wp_action_id}"
            http_put(url, chunks)
            return { "chunks": chunks }

        except Exception as e:
            print(f"Error in rag_pipeline_processing: {e}")
            return {
                "error": f"Error in rag_pipeline_processing: {e}",
                "chunks": []
            }

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
    start_time = time.time()
    # Process the entire product
    process_text = PreprocessTextForRAG()
    result = process_text.run(product_info)
    if "error" in result:
        print(result["error"])
    else:
        print("Combined Chunks and Q&As:", result['chunks'])

        print('chunk size: ', len(result['chunks']))
        # Compare original text with chunks
        error, comparison_result = ChunkComparisonWithOriginalText.run(product_info, result['chunks'])
        if error:
            print(f"Error comparing original text with chunks: {error}")
        else:
            print(f"\nSimilarity Score: {comparison_result.similarity_score}")

    # Stop the timer
    end_time = time.time()
    
    # Calculate the elapsed time in seconds
    elapsed_time = end_time - start_time
    # Print the elapsed time in seconds with 1/10 second accuracy
    print(f"\nTime taken: {elapsed_time:.1f} seconds")

    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"product_processing_result_{timestamp}.json"

    # Save result to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {filename}")
