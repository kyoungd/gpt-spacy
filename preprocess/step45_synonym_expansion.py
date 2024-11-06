import logging
from typing import Dict, List, Tuple
import json
import spacy
from collections import defaultdict

class CigarSynonymExpander:
    def __init__(self, vocabulary_file: str = 'enhanced_cigar_vocabulary.json'):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Load vocabulary from our previously generated file
        try:
            with open(vocabulary_file, 'r') as f:
                self.vocabulary = json.load(f)
        except FileNotFoundError:
            self.vocabulary = {}
            logging.error(f"Vocabulary file '{vocabulary_file}' not found. Using empty vocabulary.")
        except json.JSONDecodeError as e:
            self.vocabulary = {}
            logging.error(f"Error parsing vocabulary file '{vocabulary_file}': {e}")
        
        # Create reversed mappings for quick lookup
        self.synonym_map = self._create_synonym_map()
        
        # Load spaCy for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logging.error(f"Failed to load spaCy model: {e}")
            raise e  # Cannot proceed without spaCy

    def _create_synonym_map(self) -> Dict[str, Tuple[str, str]]:
        """
        Creates a flat mapping of synonyms to their canonical terms
        Returns: Dict[synonym, (category, canonical_term)]
        """
        synonym_map = {}
        try:
            # Process node labels
            for label, data in self.vocabulary.get("nodes", {}).items():
                for synonym in data.get("synonyms", []):
                    synonym_map[synonym.lower()] = ("node", label)
            
            # Process properties
            for prop, data in self.vocabulary.get("properties", {}).items():
                for synonym in data.get("synonyms", []):
                    synonym_map[synonym.lower()] = ("property", prop)
            
            # Process domain concepts
            for category, data in self.vocabulary.get("domain_concepts", {}).items():
                for term in data.get("terms", []):
                    synonym_map[term.lower()] = ("concept", category)
        except Exception as e:
            logging.error(f"Error creating synonym map: {e}")
        return synonym_map

    def expand_query(self, query: str) -> Tuple[str, List[Dict]]:
        """
        Expands a query by replacing synonyms with canonical terms
        Returns: (expanded query, list of replacements made)
        """
        replacements = []
        expanded_terms = []
        try:
            doc = self.nlp(query.lower())
            
            i = 0
            while i < len(doc):
                term_found = False
                
                # Try multi-word phrases first
                if i < len(doc) - 1:
                    bigram = f"{doc[i].text} {doc[i + 1].text}"
                    if bigram in self.synonym_map:
                        category, canonical = self.synonym_map[bigram]
                        replacements.append({
                            "original": bigram,
                            "canonical": canonical,
                            "category": category
                        })
                        expanded_terms.append(canonical)
                        i += 2  # Skip the next token as it's part of the bigram
                        term_found = True
                        continue
                
                # Try single words
                token_text = doc[i].text
                if not term_found and token_text in self.synonym_map:
                    category, canonical = self.synonym_map[token_text]
                    replacements.append({
                        "original": token_text,
                        "canonical": canonical,
                        "category": category
                    })
                    expanded_terms.append(canonical)
                else:
                    expanded_terms.append(token_text)
                i += 1
        except Exception as e:
            logging.error(f"Error expanding query '{query}': {e}")
            # Fallback to original query if error occurs
            expanded_terms = query.split()
        
        expanded_query = " ".join(expanded_terms)
        return expanded_query, replacements

    def process_examples(self):
        """
        Process example queries to show how synonym expansion works
        """
        example_queries = [
            "Show me strong cigars from Nicaragua",
            "Find light smokes under $15",
            "What powerful stogies are made by Padron?",
            "Show me medium bodied cigars with coffee notes"
        ]
        
        results = []
        for query in example_queries:
            expanded, replacements = self.expand_query(query)
            results.append({
                "original": query,
                "expanded": expanded,
                "replacements": replacements
            })
        
        return results

def main():
    # Example usage
    expander = CigarSynonymExpander()
    
    print("Cigar Query Synonym Expansion Examples")
    print("=====================================")
    
    results = expander.process_examples()
    
    for result in results:
        print(f"\nOriginal Query: {result['original']}")
        print(f"Expanded Query: {result['expanded']}")
        print("Replacements made:")
        for replacement in result['replacements']:
            print(f"- '{replacement['original']}' → '{replacement['canonical']}' ({replacement['category']})")
        print("-" * 50)

if __name__ == "__main__":
    main()