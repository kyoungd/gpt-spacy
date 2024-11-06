from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from enum import Enum

class QueryIntent(Enum):
    SEARCH = "search"                   # Basic search queries
    RECOMMENDATION = "recommendation"    # Asking for recommendations
    COMPARISON = "comparison"           # Compare different cigars
    FILTER = "filter"                   # Apply specific filters
    ATTRIBUTE_QUERY = "attribute"       # Ask about specific attributes
    PRICE_QUERY = "price"              # Price-related queries
    SIMILAR_ITEMS = "similar"          # Find similar items
    BRAND_QUERY = "brand"              # Brand-specific queries
    UNKNOWN = "unknown"                # Fallback for unclassified intents

@dataclass
class IntentClassification:
    primary_intent: QueryIntent
    confidence: float
    secondary_intent: Optional[QueryIntent] = None
    extracted_parameters: Dict = None
    query_type: str = None

class CigarIntentClassifier:
    def __init__(self):
        # Load spaCy model for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize intent patterns
        self.initialize_intent_patterns()
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        
        # Fit vectorizer on pattern examples
        self.fit_vectorizer()

    def initialize_intent_patterns(self):
        """Initialize patterns for each intent type"""
        self.intent_patterns = {
            QueryIntent.SEARCH: {
                "patterns": [
                    "find cigars",
                    "show me cigars",
                    "search for cigars",
                    "list cigars",
                    "display cigars"
                ],
                "keywords": ["find", "show", "search", "list", "display"],
                "query_type": "MATCH (c:Cigar)"
            },
            
            QueryIntent.RECOMMENDATION: {
                "patterns": [
                    "recommend cigars",
                    "suggest cigars",
                    "what cigars would I like",
                    "what should I try",
                    "give me recommendations"
                ],
                "keywords": ["recommend", "suggest", "should try", "would like"],
                "query_type": "MATCH (c:Cigar) WHERE ... RETURN c ORDER BY c.rating DESC"
            },
            
            QueryIntent.COMPARISON: {
                "patterns": [
                    "compare cigars",
                    "difference between",
                    "which is better",
                    "versus",
                    "vs"
                ],
                "keywords": ["compare", "difference", "better", "versus", "vs"],
                "query_type": "MATCH (c1:Cigar), (c2:Cigar) WHERE ..."
            },
            
            QueryIntent.FILTER: {
                "patterns": [
                    "with strength",
                    "by price",
                    "from country",
                    "by brand",
                    "under price",
                    "stronger than",
                    "milder than"
                ],
                "keywords": ["with", "by", "from", "under", "over", "than"],
                "query_type": "MATCH (c:Cigar) WHERE c.property ..."
            },
            
            QueryIntent.ATTRIBUTE_QUERY: {
                "patterns": [
                    "what is the strength",
                    "how strong is",
                    "what flavor",
                    "what kind of wrapper",
                    "what size"
                ],
                "keywords": ["what is", "how", "what kind"],
                "query_type": "MATCH (c:Cigar)-[:HAS_ATTRIBUTE]->(a)"
            },
            
            QueryIntent.SIMILAR_ITEMS: {
                "patterns": [
                    "similar to",
                    "like this one",
                    "cigars like",
                    "alternatives to",
                    "comparable to"
                ],
                "keywords": ["similar", "like", "alternatives", "comparable"],
                "query_type": "MATCH (c1:Cigar)-[:SIMILAR_TO]->(c2:Cigar)"
            }
        }

    def fit_vectorizer(self):
        """Fit TF-IDF vectorizer on all patterns"""
        all_patterns = []
        self.intent_vectors = {}
        
        for intent, data in self.intent_patterns.items():
            all_patterns.extend(data["patterns"])
        
        # Fit vectorizer
        self.vectorizer.fit(all_patterns)
        
        # Create vectors for each intent
        for intent, data in self.intent_patterns.items():
            patterns_vector = self.vectorizer.transform(data["patterns"])
            self.intent_vectors[intent] = patterns_vector.mean(axis=0)

    def get_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        vec1 = self.vectorizer.transform([text1])
        vec2 = self.vectorizer.transform([text2])
        return cosine_similarity(vec1, vec2)[0][0]

    def extract_parameters(self, query: str, intent: QueryIntent) -> Dict:
        """Extract relevant parameters based on intent"""
        doc = self.nlp(query.lower())
        params = {}
        
        if intent == QueryIntent.FILTER:
            # Extract filter conditions
            for token in doc:
                if token.text in ["under", "over", "above", "below"]:
                    params["price_operator"] = token.text
                elif token.like_num or token.text.startswith("$"):
                    params["price_value"] = token.text.replace("$", "")
                    
        elif intent == QueryIntent.COMPARISON:
            # Extract items being compared
            params["compare_items"] = []
            for chunk in doc.noun_chunks:
                if "cigar" in chunk.text.lower():
                    params["compare_items"].append(chunk.text)
                    
        return params

    def classify_intent(self, query: str) -> IntentClassification:
        """
        Classify the intent of a query
        Returns IntentClassification with primary and possibly secondary intent
        """
        # Convert query to vector
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calculate similarities with all intents
        similarities = {}
        for intent, vector in self.intent_vectors.items():
            similarity = cosine_similarity(query_vector, vector)[0][0]
            similarities[intent] = similarity
        
        # Get top 2 intents
        sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        primary_intent = sorted_intents[0][0]
        primary_confidence = sorted_intents[0][1]
        
        # Get secondary intent if confidence is close enough
        secondary_intent = None
        if len(sorted_intents) > 1 and sorted_intents[1][1] > primary_confidence * 0.8:
            secondary_intent = sorted_intents[1][0]
        
        # Extract parameters based on primary intent
        parameters = self.extract_parameters(query, primary_intent)
        
        # Get query type
        query_type = self.intent_patterns[primary_intent]["query_type"]
        
        return IntentClassification(
            primary_intent=primary_intent,
            confidence=primary_confidence,
            secondary_intent=secondary_intent,
            extracted_parameters=parameters,
            query_type=query_type
        )

def main():
    # Initialize classifier
    classifier = CigarIntentClassifier()
    
    # Test queries
    test_queries = [
        "Find mild cigars under $15",
        "Recommend cigars similar to Padron 1964",
        "Compare Churchill and Robusto sizes",
        "What is the strength of Opus X?",
        "Show me cigars from Nicaragua",
        "Which cigar is better, Cohiba or Montecristo?",
        "Suggest cigars with maduro wrapper"
    ]
    
    print("Intent Classification Examples")
    print("============================")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        classification = classifier.classify_intent(query)
        
        print(f"Primary Intent: {classification.primary_intent.value}")
        print(f"Confidence: {classification.confidence:.2f}")
        
        if classification.secondary_intent:
            print(f"Secondary Intent: {classification.secondary_intent.value}")
            
        if classification.extracted_parameters:
            print("Parameters:", classification.extracted_parameters)
            
        print(f"Query Type: {classification.query_type}")
        print("-" * 50)

if __name__ == "__main__":
    main()