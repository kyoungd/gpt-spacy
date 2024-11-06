from neo4j import GraphDatabase
import json
from typing import Dict, List, Set
from dotenv import load_dotenv
import os
from openai import OpenAI
from collections import defaultdict
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from neo4j.exceptions import Neo4jError
import openai
import logging

# Load environment variables
load_dotenv()

class OpenAIEnhancedVocabularyGenerator:
    def __init__(self):
        # Initialize Neo4j connection
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_user = os.getenv('NEO4J_USER')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize base mappings
        self.initialize_base_mappings()

        # Set up logging
        logging.basicConfig(level=logging.INFO)

    def initialize_base_mappings(self):
        """Initialize basic domain-specific terminology"""
        self.node_label_synonyms = {
            "Cigar": ["stogie", "smoke", "stick"],
            "Brand": ["manufacturer", "maker", "producer"],
            "Origin": ["country", "region", "source"],
        }

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def get_openai_response(self, prompt: str, system_message: str) -> Dict:
        """Get response from OpenAI with retry logic"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={ "type": "json_object" }
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from OpenAI response: {str(e)}")
            return {}
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise

    def enrich_with_llm(self, term: str, category: str) -> Dict:
        """Use OpenAI to enrich vocabulary for a given term"""
        system_message = """You are a cigar industry expert helping to build a comprehensive domain vocabulary. 
        Provide detailed, accurate information about cigar-related terms and concepts. 
        Always return valid JSON."""

        prompts = {
            "node": f"""
            For the node type '{term}' in a cigar database, create a JSON object with these keys:
            {{
                "synonyms": [list of synonyms and alternative phrases],
                "industry_terms": [list of technical and professional terms],
                "common_queries": [list of typical user questions],
                "related_concepts": [list of related terms and concepts],
                "description": [detailed description of the term],
                "examples": [list of example instances or use cases]
            }}
            """,
            
            "property": f"""
            For the property '{term}' in a cigar database, create a JSON object with these keys:
            {{
                "synonyms": [list of alternative terms],
                "common_values": [list of typical values or ranges],
                "measurement_terms": [list of related units or measures],
                "user_queries": [list of typical questions],
                "validation_rules": [list of common validation rules or constraints],
                "contextual_usage": [list of common usage contexts]
            }}
            """,
            
            "relationship": f"""
            For the relationship type '{term}' in a cigar database, create a JSON object with these keys:
            {{
                "alternative_phrases": [list of alternative ways to express this relationship],
                "common_expressions": [list of natural language expressions],
                "typical_queries": [list of common user questions],
                "related_relationships": [list of related relationship types],
                "business_rules": [list of common business rules or constraints],
                "use_cases": [list of typical use cases]
            }}
            """
        }

        return self.get_openai_response(prompts[category], system_message)

    def generate_query_patterns(self) -> Dict:
        """Generate common query patterns using OpenAI"""
        system_message = """You are a cigar database expert helping to create query patterns. 
        Focus on practical, real-world queries that users might want to make."""

        prompt = """
        Create a JSON object containing common cigar database query patterns with these categories:
        {
            "simple_lookups": [list of patterns for finding cigars by basic properties],
            "relationship_queries": [list of patterns involving relationships between entities],
            "comparison_queries": [list of patterns for comparing different cigars],
            "aggregation_queries": [list of patterns involving counts, averages, etc.],
            "complex_queries": [list of patterns combining multiple conditions],
            "natural_language_variations": {
                "strength": [variations of asking about cigar strength],
                "flavor": [variations of asking about flavor profiles],
                "origin": [variations of asking about cigar origin],
                "price": [variations of asking about price ranges]
            }
        }
        """

        return self.get_openai_response(prompt, system_message)

    def enrich_domain_concepts(self) -> Dict:
        """Enrich domain-specific concepts using OpenAI"""
        system_message = """You are a cigar industry expert providing comprehensive information about 
        cigar-related concepts and terminology."""

        prompt = """
        Create a JSON object containing comprehensive cigar domain concepts with these categories:
        {
            "wrapper_types": {
                "categories": [list of wrapper types],
                "descriptions": [descriptions of each type],
                "characteristics": [key characteristics]
            },
            "strength_levels": {
                "levels": [list of strength levels],
                "descriptions": [descriptions of each level],
                "indicators": [how to identify each level]
            },
            "flavor_profiles": {
                "primary_flavors": [list of primary flavor categories],
                "flavor_notes": [list of specific flavor notes],
                "flavor_combinations": [common flavor combinations]
            },
            "vitolas": {
                "sizes": [list of standard sizes],
                "shapes": [list of shapes],
                "measurements": [standard measurements]
            },
            "aging_process": {
                "stages": [list of aging stages],
                "terminology": [aging-related terms],
                "effects": [effects of aging]
            },
            "regional_variations": {
                "regions": [major cigar regions],
                "characteristics": [regional characteristics],
                "terminology": [regional terms]
            }
        }
        """

        return self.get_openai_response(prompt, system_message)

    def get_database_structure(self) -> Dict:
        """Extract database structure including labels, relationships, and properties"""
        try:
            with self.driver.session() as session:
                # Get node labels
                labels = session.run("CALL db.labels()").value()
                
                # Get relationship types
                relationships = session.run("CALL db.relationshipTypes()").value()
                
                # Get property keys and sample values
                result = session.run("""
                    MATCH (n)
                    UNWIND keys(n) AS key
                    WITH key, n[key] AS value
                    WHERE NOT key IN ['id', 'created_at', 'updated_at']
                    RETURN DISTINCT key, COLLECT(DISTINCT value)[..5] AS sample_values
                """)
                
                properties = {row['key']: row['sample_values'] for row in result}
                
                return {
                    "labels": labels,
                    "relationships": relationships,
                    "properties": properties
                }
        except Neo4jError as e:
            logging.error(f"Neo4j database error: {str(e)}")
            raise

    def generate_enhanced_vocabulary(self) -> Dict:
        """Generate comprehensive vocabulary with OpenAI enrichment"""
        db_structure = self.get_database_structure()
        
        vocabulary = {
            "metadata": {
                "purpose": "OpenAI-enhanced cigar domain vocabulary and synonym mapping",
                "source": "Combined database analysis and AI-enhanced domain knowledge",
                "generation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "nodes": {},
            "properties": {},
            "relationships": {},
            "query_patterns": {},
            "domain_concepts": {}
        }

        # Enrich node labels
        print("Enriching node labels...")
        for label in db_structure["labels"]:
            print(f"Processing node: {label}")
            vocabulary["nodes"][label] = self.enrich_with_llm(label, "node")

        # Enrich properties
        print("\nEnriching properties...")
        for prop in db_structure["properties"]:
            print(f"Processing property: {prop}")
            vocabulary["properties"][prop] = {
                "llm_enriched": self.enrich_with_llm(prop, "property"),
                "sample_values": db_structure["properties"][prop]
            }

        # Enrich relationships
        print("\nEnriching relationships...")
        for rel in db_structure["relationships"]:
            print(f"Processing relationship: {rel}")
            vocabulary["relationships"][rel] = self.enrich_with_llm(rel, "relationship")

        # Add query patterns
        print("\nGenerating query patterns...")
        vocabulary["query_patterns"] = self.generate_query_patterns()

        # Add domain concepts
        print("\nEnriching domain concepts...")
        vocabulary["domain_concepts"] = self.enrich_domain_concepts()

        return vocabulary

    def save_vocabulary(self, filename: str = "enhanced_cigar_vocabulary.json"):
        """Save enhanced vocabulary to JSON file"""
        try:
            vocabulary = self.generate_enhanced_vocabulary()
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(current_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Enhanced vocabulary saved to: {filepath}")
            return vocabulary
        except IOError as e:
            logging.error(f"File I/O error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error while saving vocabulary: {str(e)}")
            raise

    def close(self):
        # Ensure driver is closed properly
        if self.driver:
            self.driver.close()

def main():
    generator = OpenAIEnhancedVocabularyGenerator()
    try:
        vocabulary = generator.save_vocabulary()
        
        # Print sample of the enhanced vocabulary
        print("\nSample of generated vocabulary:")
        if vocabulary["nodes"]:
            print("\nSample Node Label:")
            first_node = next(iter(vocabulary["nodes"].items()))
            print(json.dumps(first_node, indent=2))
            
        if vocabulary["properties"]:
            print("\nSample Property:")
            first_prop = next(iter(vocabulary["properties"].items()))
            print(json.dumps(first_prop, indent=2))
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        generator.close()

if __name__ == "__main__":
    main()