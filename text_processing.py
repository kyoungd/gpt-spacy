from ai import ProductInfo, ProductChunkInfo, Chunking, ChunkComparisonWithOriginalText
from pre_text_normalization import text_normalization_with_boundaries, text_remove_stop_words_lemmatized

def rag_pipeline_processing(product_info):
    try:
        # Process the entire product
        error, product_result = ProductInfo.run(product_info)
        if error:
            return {"error": f"Error processing product: {error}"}
        
        total_text = product_info + "\n\n" + product_result.additional_info
        # Use Chunking class to chunk the product description
        error, chunking_result = Chunking.run(total_text)
        if error:
            return {"error": f"Error chunking product description: {error}"}
        
        # Process all chunks at once
        error, chunk_process_results = ProductChunkInfo.run(chunking_result.chunks)
        if error:
            return {"error": f"Error processing chunks: {error}"}
        
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

        # Return the required data
        return {
            "all_tags": list(all_tags),
            "all_key_features": list(all_key_features),
            "chunks": chunks
        }
    
    except Exception as e:
        return {
            "all_tags": [],
            "all_key_features": [],
            "chunks": [product_info],
            "error": f"An error occurred: {str(e)}"
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
    
    result = rag_pipeline_processing(product_info)
    if "error" in result:
        print(result["error"])
    else:
        print("\nAll Unique Tags:", result['all_tags'])
        print("All Unique Key Features:", result['all_key_features'])
        print("Combined Chunks and Q&As:", result['chunks'])

        # Compare original text with chunks
        error, comparison_result = ChunkComparisonWithOriginalText.run(product_info, result['chunks'])
        if error:
            print(f"Error comparing original text with chunks: {error}")
        else:
            print(f"\nSimilarity Score: {comparison_result.similarity_score}")
            print(f"Difference Text: {comparison_result.difference_text}")
            print(f"Difference Details: {comparison_result.difference_details}")
