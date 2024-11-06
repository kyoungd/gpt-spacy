from typing import List, Set, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dataclasses import dataclass
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

@dataclass
class CleanedQuery:
    original_query: str
    cleaned_query: str
    removed_words: List[str]
    important_terms: List[str]
    query_structure: Dict[str, str]  # Maintains query structure information

class CigarStopWordsRemover:
    def __init__(self):
        # Get standard English stop words
        self.standard_stops = set(stopwords.words('english'))
        
        # Initialize custom stop words and preserved terms
        self.initialize_custom_words()
        
        # Question word handling
        self.question_words = {
            'what', 'which', 'who', 'where', 'when', 'how', 'why',
            'can', 'could', 'would', 'should'
        }
        
        # Compile regex patterns
        self.price_pattern = re.compile(r'\$?\d+(?:\.\d{2})?')
        self.size_pattern = re.compile(r'\d+\s*x\s*\d+')

    def initialize_custom_words(self):
        """Initialize custom stop words and preserved terms for cigar domain"""
        
        # Custom stop words specific to cigar queries
        self.custom_stops = {
            'show', 'find', 'get', 'give', 'look', 'search',
            'list', 'display', 'tell', 'want', 'need',
            'me', 'my', 'mine', 'please', 'thanks',
            'available', 'currently', 'existing'
        }
        
        # Terms to preserve (never remove these)
        self.preserved_terms = {
            # Strength terms
            'mild', 'medium', 'full', 'light', 'strong',
            
            # Size terms
            'robusto', 'toro', 'churchill', 'corona', 'torpedo',
            'petit', 'gordo', 'belicoso',
            
            # Wrapper types
            'maduro', 'claro', 'colorado', 'oscuro', 'natural',
            'connecticut', 'habano', 'corojo',
            
            # Important adjectives
            'similar', 'like', 'comparable',
            
            # Price-related
            'under', 'over', 'between', 'around',
            
            # Comparison terms
            'versus', 'vs', 'better', 'worse', 'stronger', 'lighter',
            
            # Origin terms
            'cuban', 'dominican', 'nicaraguan', 'honduran'
        }
        
        # Important verbs to preserve
        self.important_verbs = {
            'compare', 'recommend', 'suggest', 'prefer',
            'rate', 'review', 'match', 'pair'
        }
        
        # Relationship words to preserve
        self.relationship_words = {
            'with', 'from', 'by', 'of', 'in'
        }

    def is_price(self, token: str) -> bool:
        """Check if token is a price value"""
        return bool(self.price_pattern.match(token))

    def is_size(self, token: str) -> bool:
        """Check if token is a size specification"""
        return bool(self.size_pattern.match(token))

    def should_preserve(self, token: str, prev_token: str = None) -> bool:
        """Determine if a token should be preserved"""
        token_lower = token.lower()
        
        # Check various preservation conditions
        return (
            token_lower in self.preserved_terms or
            token_lower in self.important_verbs or
            token_lower in self.relationship_words or
            self.is_price(token) or
            self.is_size(token) or
            token.startswith('$') or
            # Preserve brand names (usually capitalized)
            token[0].isupper() or
            # Preserve numbers (ring gauges, lengths, etc.)
            token.isdigit() or
            # Preserve compound terms
            '-' in token or
            # Context-based preservation
            (prev_token and f"{prev_token} {token_lower}" in self.preserved_terms)
        )

    def extract_query_structure(self, tokens: List[str]) -> Dict[str, str]:
        """Extract and preserve query structure information"""
        structure = {}
        
        # Identify question type
        if tokens and tokens[0].lower() in self.question_words:
            structure['query_type'] = 'question'
            structure['question_word'] = tokens[0].lower()
        else:
            structure['query_type'] = 'statement'
        
        # Identify command words
        command_words = {'find', 'show', 'list', 'get', 'compare', 'recommend'}
        for token in tokens:
            if token.lower() in command_words:
                structure['command'] = token.lower()
                break
        
        return structure

    def remove_stop_words(self, query: str) -> CleanedQuery:
        """
        Remove stop words while preserving important cigar-related terms
        Returns CleanedQuery object with original and cleaned query
        """
        # Tokenize the query
        tokens = word_tokenize(query)
        
        # Extract query structure before removal
        query_structure = self.extract_query_structure(tokens)
        
        # Process tokens
        preserved_tokens = []
        removed_words = []
        important_terms = []
        
        for i, token in enumerate(tokens):
            prev_token = tokens[i-1].lower() if i > 0 else None
            token_lower = token.lower()
            
            # Check if token should be preserved
            if (self.should_preserve(token, prev_token) or 
                (query_structure['query_type'] == 'question' and i == 0)):  # Preserve first word of questions
                preserved_tokens.append(token)
                if token_lower not in self.relationship_words:
                    important_terms.append(token)
            elif token_lower not in self.standard_stops and token_lower not in self.custom_stops:
                preserved_tokens.append(token)
                important_terms.append(token)
            else:
                removed_words.append(token)
        
        # Reconstruct cleaned query
        cleaned_query = ' '.join(preserved_tokens)
        
        return CleanedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            removed_words=removed_words,
            important_terms=important_terms,
            query_structure=query_structure
        )

def main():
    # Initialize remover
    remover = CigarStopWordsRemover()
    
    # Test queries
    test_queries = [
        "Can you please show me some mild cigars that are under $15?",
        "I want to find cigars that are similar to Padron 1964",
        "What is the strength level of the Opus X cigar?",
        "Show me all available Nicaraguan cigars with maduro wrapper",
        "Could you tell me which cigars are comparable to Montecristo No. 2?",
        "I need to find some good medium-bodied robustos",
        "Please list all cigars between $20 and $30"
    ]
    
    print("Stop Words Removal Examples")
    print("==========================")
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        result = remover.remove_stop_words(query)
        
        print(f"Cleaned Query: {result.cleaned_query}")
        print(f"Removed Words: {', '.join(result.removed_words)}")
        print(f"Important Terms: {', '.join(result.important_terms)}")
        print(f"Query Structure: {result.query_structure}")
        print("-" * 60)

if __name__ == "__main__":
    main()