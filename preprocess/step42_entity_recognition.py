import spacy
from typing import List, Dict, Set, Any
from dataclasses import dataclass
import json
import re
from collections import defaultdict

@dataclass
class RecognizedEntity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_value: str = None

class CigarEntityRecognizer:
    def __init__(self, vocabulary_file: str = 'enhanced_cigar_vocabulary.json'):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load domain vocabulary
        with open(vocabulary_file, 'r') as f:
            self.vocabulary = json.load(f)
            
        # Initialize entity dictionaries
        self.initialize_entity_dictionaries()
        
        # Add custom pipeline components
        self.add_custom_pipeline()

    def initialize_entity_dictionaries(self):
        """Initialize dictionaries for different entity types"""
        self.entities = {
            "BRAND": set(),
            "CIGAR_NAME": set(),
            "WRAPPER_TYPE": set(),
            "STRENGTH": set(),
            "FLAVOR": set(),
            "VITOLA": set(),
            "ORIGIN": set(),
            "PRICE_RANGE": set()
        }
        
        # Populate from vocabulary
        if "nodes" in self.vocabulary:
            for node_type, data in self.vocabulary["nodes"].items():
                if "examples" in data:
                    self.entities[node_type.upper()].update(
                        example.lower() for example in data["examples"]
                    )
        
        # Add common patterns
        self.price_patterns = [
            r'\$\d+',
            r'under \$\d+',
            r'less than \$\d+',
            r'more than \$\d+',
            r'\$\d+-\$\d+'
        ]
        
        # Add strength levels
        self.strength_levels = {
            "mild", "medium", "full", "mild to medium", 
            "medium to full", "mild-medium", "medium-full"
        }
        
        # Add wrapper types
        self.wrapper_types = {
            "maduro", "claro", "colorado", "oscuro", "natural",
            "connecticut", "habano", "corojo"
        }
        
        # Add vitolas (sizes)
        self.vitolas = {
            "robusto", "toro", "churchill", "corona", "torpedo",
            "belicoso", "perfecto", "petit corona", "gordo"
        }

    def add_custom_pipeline(self):
        """Add custom pipeline components to spaCy"""
        # Add entity ruler for custom entities
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        patterns = []
        
        # Add patterns for each entity type
        for entity_type, terms in self.entities.items():
            for term in terms:
                patterns.append({
                    "label": entity_type,
                    "pattern": term
                })
        
        ruler.add_patterns(patterns)

    def extract_price_mentions(self, text: str) -> List[RecognizedEntity]:
        """Extract price-related mentions from text"""
        price_entities = []
        
        for pattern in self.price_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                price_entities.append(
                    RecognizedEntity(
                        text=match.group(),
                        label="PRICE_RANGE",
                        start=match.start(),
                        end=match.end(),
                        confidence=1.0,
                        normalized_value=self.normalize_price(match.group())
                    )
                )
        
        return price_entities

    def normalize_price(self, price_text: str) -> str:
        """Normalize price mentions to a standard format"""
        # Extract numeric values
        numbers = re.findall(r'\d+', price_text)
        if 'under' in price_text or 'less than' in price_text:
            return f"<{numbers[0]}"
        elif 'more than' in price_text:
            return f">{numbers[0]}"
        elif len(numbers) == 2:
            return f"{numbers[0]}-{numbers[1]}"
        else:
            return numbers[0]

    def extract_strength_mentions(self, text: str) -> List[RecognizedEntity]:
        """Extract strength-related mentions from text"""
        strength_entities = []
        
        # Look for strength level mentions
        text_lower = text.lower()
        for strength in self.strength_levels:
            if strength in text_lower:
                start = text_lower.find(strength)
                strength_entities.append(
                    RecognizedEntity(
                        text=strength,
                        label="STRENGTH",
                        start=start,
                        end=start + len(strength),
                        confidence=1.0,
                        normalized_value=self.normalize_strength(strength)
                    )
                )
        
        return strength_entities

    def normalize_strength(self, strength: str) -> str:
        """Normalize strength levels to standard values"""
        strength = strength.lower().replace(' to ', '-')
        mapping = {
            "mild": "MILD",
            "mild-medium": "MILD_TO_MEDIUM",
            "medium": "MEDIUM",
            "medium-full": "MEDIUM_TO_FULL",
            "full": "FULL"
        }
        return mapping.get(strength, strength.upper())

    def recognize_entities(self, text: str) -> List[RecognizedEntity]:
        """
        Main method to recognize all entities in text
        Returns list of RecognizedEntity objects
        """
        entities = []
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Get entities from spaCy's NER
        for ent in doc.ents:
            entities.append(
                RecognizedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8  # Default confidence for spaCy entities
                )
            )
        
        # Add price mentions
        entities.extend(self.extract_price_mentions(text))
        
        # Add strength mentions
        entities.extend(self.extract_strength_mentions(text))
        
        # Extract wrapper types
        for wrapper in self.wrapper_types:
            if wrapper in text.lower():
                start = text.lower().find(wrapper)
                entities.append(
                    RecognizedEntity(
                        text=wrapper,
                        label="WRAPPER_TYPE",
                        start=start,
                        end=start + len(wrapper),
                        confidence=1.0
                    )
                )
        
        # Extract vitolas
        for vitola in self.vitolas:
            if vitola in text.lower():
                start = text.lower().find(vitola)
                entities.append(
                    RecognizedEntity(
                        text=vitola,
                        label="VITOLA",
                        start=start,
                        end=start + len(vitola),
                        confidence=1.0
                    )
                )
        
        return sorted(entities, key=lambda x: x.start)

def main():
    # Example usage
    recognizer = CigarEntityRecognizer()
    
    # Test queries
    test_queries = [
        "Find mild to medium Padron cigars under $15",
        "Show me maduro wrapped Nicaraguan robustos",
        "What full-bodied cigars are similar to Opus X?",
        "Compare Churchill and Toro sizes from Davidoff",
        "Find cigars between $20-$30 with Connecticut wrapper"
    ]
    
    print("Entity Recognition Examples")
    print("=========================")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        entities = recognizer.recognize_entities(query)
        
        print("Recognized Entities:")
        for entity in entities:
            print(f"- {entity.text} ({entity.label})")
            if entity.normalized_value:
                print(f"  Normalized: {entity.normalized_value}")
            
        print("-" * 50)

if __name__ == "__main__":
    main()