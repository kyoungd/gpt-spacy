import re
import sys
import os
sys.path.insert(0, os.path.dirname(sys.path[0]))

from chunking.ichunker import IChunker
from chunking.semantic_chunker import SemanticChunker
from pre_text_normalization import text_normalization_with_boundaries, text_remove_stop_words_lemmatized
from ai import CoreferenceResolution
from coreference import coreference_resolution

def simple_sentence_tokenize(text):
    return re.findall(r'[^.!?]+[.!?]', text)

class ParagraphChunker(IChunker):
    def __init__(self):
        self.max_chunk_length = 2000
        self.min_chunk_length = 500
        self.semantic_chunker = SemanticChunker()

    def _split_paragraphs(self, text):
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_short_paragraphs(self, paragraphs):
        merged = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < self.min_chunk_length:
                current_chunk += " " + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged

    def _process_chunk(self, chunk):
        chunk_length = len(chunk)
        
        if chunk_length < self.min_chunk_length:
            # Short chunks are processed by semantic chunker
            return self.semantic_chunker.chunk_text(chunk)
        elif self.min_chunk_length <= chunk_length <= self.max_chunk_length:
            # Chunks within the desired range are kept as is
            return [chunk]
        else:
            # Chunks larger than max_chunk_length are processed by semantic chunker
            return self.semantic_chunker.chunk_text(chunk)

    def chunk_text(self, text):
        text_normalized = text_normalization_with_boundaries(text)
        text_clean = text_remove_stop_words_lemmatized(text_normalized)
        # text_block = coreference_resolution(text_clean)
        error, text_block = CoreferenceResolution.run(text_clean)
        if error:
            print(f"Coreference Resolution Error: {error}")
            text_block = text_clean

        paragraphs = self._split_paragraphs(text_block)
        merged_paragraphs = self._merge_short_paragraphs(paragraphs)
        
        chunks = []
        for merged_chunk in merged_paragraphs:
            chunks.extend(self._process_chunk(merged_chunk))

        return chunks

if __name__ == "__main__":
    TestText = """
    Babylon was an ancient city located on the lower Euphrates river in southern Mesopotamia, within modern-day Hillah, Iraq, about 85 kilometers (55 miles) south of modern day Baghdad. Babylon functioned as the main cultural and political centre of the Akkadian-speaking region of Babylonia. Its rulers established two important empires in antiquity, the 19th–16th century BC Old Babylonian Empire, and the 7th–6th century BC Neo-Babylonian Empire. Babylon was also used as a regional capital of other empires, such as the Achaemenid Empire. Babylon was one of the most important urban centres of the ancient Near East, until its decline during the Hellenistic period. Nearby ancient sites are Kish, Borsippa, Dilbat, and Kutha.[2]

    The earliest known mention of Babylon as a small town appears on a clay tablet from the reign of Shar-Kali-Sharri (2217–2193 BC), of the Akkadian Empire.[3] Babylon was merely a religious and cultural centre at this point and neither an independent state nor a large city, subject to the Akkadian Empire. After the collapse of the Akkadian Empire, the south Mesopotamian region was dominated by the Gutian Dynasty for a few decades, before the rise of the Third Dynasty of Ur, which encompassed the whole of Mesopotamia, including the town of Babylon.

    The town became part of a small independent city-state with the rise of the first Babylonian Empire, now known as the Old Babylonian Empire, in the 19th century BC. The Amorite king Hammurabi founded the short-lived Old Babylonian Empire in the 18th century BC. He built Babylon into a major city and declared himself its king. Southern Mesopotamia became known as Babylonia, and Babylon eclipsed Nippur as the region's holy city. The empire waned under Hammurabi's son Samsu-iluna, and Babylon spent long periods under Assyrian, Kassite and Elamite domination. After the Assyrians destroyed and then rebuilt it, Babylon became the capital of the short-lived Neo-Babylonian Empire, from 626 to 539 BC. The Hanging Gardens of Babylon were ranked as one of the Seven Wonders of the Ancient World, allegedly existing between approximately 600 BC and AD 1. However, there are questions about whether the Hanging Gardens of Babylon even existed, as there is no mention within any extant Babylonian texts of its existence.[4][5] After the fall of the Neo-Babylonian Empire, the city came under the rule of the Achaemenid, Seleucid, Parthian, Roman, Sassanid, and Muslim empires. The last known habitation of the town dates from the 11th century, when it was referred to as the "small village of Babel".

    It has been estimated that Babylon was the largest city in the world c. 1770 – c. 1670 BC, and again c. 612 – c. 320 BC. It was perhaps the first city to reach a population above 200,000.[6] Estimates for the maximum extent of its area range from 890 (3½ sq. mi.)[7] to 900 ha (2,200 acres).[8] The main sources of information about Babylon—excavation of the site itself, references in cuneiform texts found elsewhere in Mesopotamia, references in the Bible, descriptions in other classical writing, especially by Herodotus, and second-hand descriptions, citing the work of Ctesias and Berossus—present an incomplete and sometimes contradictory picture of the ancient city, even at its peak in the sixth century BC.[9] UNESCO inscribed Babylon as a World Heritage Site in 2019. The site receives thousands of visitors each year, almost all of whom are Iraqis.[10][11] Construction is rapidly increasing, which has caused encroachments upon the ruins.[12][13][14]
    """

    paragraph_chunker = ParagraphChunker()

    print("\nParagraph Chunking:")
    paragraph_chunks = paragraph_chunker.chunk_text(TestText)
    for i, chunk in enumerate(paragraph_chunks, 1):
        print(f"Chunk {i} (length: {len(chunk)})")
        print(f"{chunk[:100]}...")
        print()