from chunking.semantic_chunker import SemanticChunker
from chunking.paragraph_chunker import ParagraphChunker
from chunking.ichunker import IChunker

def get_chunker(text: str) -> IChunker:
    # Check if the text has paragraph separation
    if "\n\n" in text or "\r\n\r\n" in text:
        return ParagraphChunker()
    else:
        return SemanticChunker()

def review_text(text: str) -> str:
    if not text:
        return "The input text is empty."
    
    if "\n\n" in text or "\r\n\r\n" in text:
        return "The text contains paragraph separations. Using ParagraphChunker."
    elif "." in text or "!" in text or "?" in text:
        return "The text contains sentences but no paragraph separations. Using SemanticChunker."
    else:
        return "The text doesn't appear to have clear sentence or paragraph structure. Using SemanticChunker as default."