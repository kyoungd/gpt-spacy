import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import re

# ------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(sys.path[0]))
# ------------------------------------
from chunking.test_text import TestText
from chunking.ichunker import IChunker
from pre_text_normalization import text_normalization_with_boundaries, text_remove_stop_words_lemmanized
from ai import CoreferenceResolution
from coreference import coreference_resolution

class SemanticChunker(IChunker):

    def __init__(self, model_name=None, breakpoint_percentile=None, max_chunk_length=None, buffer_size=None, min_chunk_length=None):
        if model_name is None:
            model_name = "all-MiniLM-L6-v2"
        if breakpoint_percentile is None:
            breakpoint_percentile = 0.9  # Increased from 0.8 to 0.9
        if max_chunk_length is None:
            max_chunk_length = 3000  # Increased from 1500 to 3000
        if buffer_size is None:
            buffer_size = 3  # Increased from 2 to 3
        if min_chunk_length is None:
            min_chunk_length = 600  # Increased from 300 to 600
        self.model = SentenceTransformer(model_name)
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_length = max_chunk_length
        self.buffer_size = buffer_size
        self.min_chunk_length = min_chunk_length

    def _split_sentences(self, text):
        # Simple regex-based sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [{'sentence': sentence} for sentence in sentences if sentence]

    def _combine_sentences(self, sentences):
        for i in range(len(sentences)):
            combined = []
            range_start = max(0, i - self.buffer_size)
            range_end = min(len(sentences), i + 1 + self.buffer_size)
            for j in range(range_start, range_end):
                combined.append(sentences[j]['sentence'])
            sentences[i]['combined_sentence'] = ' '.join(combined)
        return sentences

    def _calculate_cosine_distances(self, sentences):
        embeddings = [s['combined_sentence_embedding'] for s in sentences]
        distances = []
        for i in range(len(embeddings) - 1):
            distance = cosine(embeddings[i], embeddings[i + 1])
            distances.append(distance)
            sentences[i]['distance_to_next'] = distance
        return distances, sentences

    def chunk_text(self, text):
        text_normalized = text_normalization_with_boundaries(text)
        text_clean = text_remove_stop_words_lemmanized(text_normalized)
        # text_block = coreference_resolution(text_clean)
        error, text_block = CoreferenceResolution.run(text_clean)
        if error:
            print(f"Warning: Error in CoreferenceResolution: {error}")
            text_block = text_clean  # Fallback to cleaned text if pronoun removal fails
        
        sentences = self._split_sentences(text_block)
        sentences = self._combine_sentences(sentences)
        
        # Generate embeddings for the combined sentences
        embeddings = self.model.encode([x['combined_sentence'] for x in sentences], show_progress_bar=False)
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]
        
        distances, sentences = self._calculate_cosine_distances(sentences)
        
        breakpoint_distance_threshold = np.percentile(distances, self.breakpoint_percentile)
        
        chunks = []
        current_chunk = []
        current_chunk_length = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence['sentence'])
            current_chunk_length += len(sentence['sentence'])
            
            if (i > 0 and distances[i-1] > breakpoint_distance_threshold and current_chunk_length >= self.min_chunk_length) or \
               current_chunk_length >= self.max_chunk_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_chunk_length = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

if __name__ == "__main__":
    text = """Babylon was an ancient city located on the lower Euphrates river in southern Mesopotamia, within modern-day Hillah, Iraq, about 85 kilometers (55 miles) south of modern day Baghdad. Babylon functioned as the main cultural and political centre of the Akkadian-speaking region of Babylonia. Its rulers established two important empires in antiquity, the 19th–16th century BC Old Babylonian Empire, and the 7th–6th century BC Neo-Babylonian Empire. Babylon was also used as a regional capital of other empires, such as the Achaemenid Empire. Babylon was one of the most important urban centres of the ancient Near East, until its decline during the Hellenistic period. Nearby ancient sites are Kish, Borsippa, Dilbat, and Kutha.[2] The earliest known mention of Babylon as a small town appears on a clay tablet from the reign of Shar-Kali-Sharri (2217–2193 BC), of the Akkadian Empire.[3] Babylon was merely a religious and cultural centre at this point and neither an independent state nor a large city, subject to the Akkadian Empire. After the collapse of the Akkadian Empire, the south Mesopotamian region was dominated by the Gutian Dynasty for a few decades, before the rise of the Third Dynasty of Ur, which encompassed the whole of Mesopotamia, including the town of Babylon. The town became part of a small independent city-state with the rise of the first Babylonian Empire, now known as the Old Babylonian Empire, in the 19th century BC. The Amorite king Hammurabi founded the short-lived Old Babylonian Empire in the 18th century BC. He built Babylon into a major city and declared himself its king. Southern Mesopotamia became known as Babylonia, and Babylon eclipsed Nippur as the region's holy city. The empire waned under Hammurabi's son Samsu-iluna, and Babylon spent long periods under Assyrian, Kassite and Elamite domination. After the Assyrians destroyed and then rebuilt it, Babylon became the capital of the short-lived Neo-Babylonian Empire, from 626 to 539 BC. The Hanging Gardens of Babylon were ranked as one of the Seven Wonders of the Ancient World, allegedly existing between approximately 600 BC and AD 1. However, there are questions about whether the Hanging Gardens of Babylon even existed, as there is no mention within any extant Babylonian texts of its existence.[4][5] After the fall of the Neo-Babylonian Empire, the city came under the rule of the Achaemenid, Seleucid, Parthian, Roman, Sassanid, and Muslim empires. The last known habitation of the town dates from the 11th century, when it was referred to as the "small village of Babel". It has been estimated that Babylon was the largest city in the world c. 1770 – c. 1670 BC, and again c. 612 – c. 320 BC. It was perhaps the first city to reach a population above 200,000.[6] Estimates for the maximum extent of its area range from 890 (3½ sq. mi.)[7] to 900 ha (2,200 acres).[8] The main sources of information about Babylon—excavation of the site itself, references in cuneiform texts found elsewhere in Mesopotamia, references in the Bible, descriptions in other classical writing, especially by Herodotus, and second-hand descriptions, citing the work of Ctesias and Berossus—present an incomplete and sometimes contradictory picture of the ancient city, even at its peak in the sixth century BC.[9] UNESCO inscribed Babylon as a World Heritage Site in 2019. The site receives thousands of visitors each year, almost all of whom are Iraqis.[10][11] Construction is rapidly increasing, which has caused encroachments upon the ruins.[12][13][14]"""
    chunker = SemanticChunker()
    chunks = chunker.chunk_text(text)

    for i, chunk in enumerate(chunks, 1):
        print_chunk = chunk.replace('\n', ' ')
        print(f"Chunk {i} (length: {len(chunk)})")
        print(f"{print_chunk[:100]}...")  # Print first 100 characters
        print()