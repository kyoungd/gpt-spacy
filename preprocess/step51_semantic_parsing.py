from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import spacy
from enum import Enum
import json

class SemanticRole(Enum):
    SUBJECT = "subject"           # The main entity being queried
    ACTION = "action"            # The action to perform
    ATTRIBUTE = "attribute"      # Properties or characteristics
    CONDITION = "condition"      # Filtering conditions
    MODIFIER = "modifier"        # Modifying terms
    COMPARISON = "comparison"    # Comparison terms
    VALUE = "value"             # Specific values
    CONTEXT = "context"         # Contextual information

@dataclass
class SemanticComponent:
    role: SemanticRole
    text: str
    normalized_text: str
    schema_mapping: Optional[Dict] = None
    cypher_template: Optional[str] = None

@dataclass
class SemanticParseResult:
    components: List[SemanticComponent]
    query_structure: Dict
    cypher_template: str
    parameters: Dict

class CigarSemanticParser:
    def __init__(self, schema_file: str = 'enhanced_cigar_vocabulary.json'):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load schema information
        with open(schema_file, 'r') as f:
            self.schema = json.load(f)
            
        # Initialize semantic patterns
        self.initialize_patterns()

    def initialize_patterns(self):
        """Initialize semantic patterns and mappings"""
        self.action_patterns = {
            "find": {
                "role": SemanticRole.ACTION,
                "cypher_template": "MATCH ({entity}:{label})",
                "type": "search"
            },
            "recommend": {
                "role": SemanticRole.ACTION,
                "cypher_template": "MATCH ({entity}:{label}) WHERE ... RETURN {entity} ORDER BY {entity}.rating DESC",
                "type": "recommendation"
            },
            "compare": {
                "role": SemanticRole.ACTION,
                "cypher_template": "MATCH ({entity1}:{label}), ({entity2}:{label})",
                "type": "comparison"
            }
        }

        self.attribute_patterns = {
            "strength": {
                "role": SemanticRole.ATTRIBUTE,
                "schema_mapping": {"property": "strength_level", "node": "Cigar"},
                "cypher_template": "{entity}.strength_level"
            },
            "price": {
                "role": SemanticRole.ATTRIBUTE,
                "schema_mapping": {"property": "price", "node": "Cigar"},
                "cypher_template": "{entity}.price"
            },
            "wrapper": {
                "role": SemanticRole.ATTRIBUTE,
                "schema_mapping": {"relationship": "HAS_WRAPPER", "node": "Wrapper"},
                "cypher_template": "()-[:HAS_WRAPPER]->(:Wrapper)"
            }
        }

        self.condition_patterns = {
            "under": {
                "role": SemanticRole.CONDITION,
                "cypher_template": "< {value}",
                "type": "comparison"
            },
            "over": {
                "role": SemanticRole.CONDITION,
                "cypher_template": "> {value}",
                "type": "comparison"
            },
            "between": {
                "role": SemanticRole.CONDITION,
                "cypher_template": "BETWEEN {value1} AND {value2}",
                "type": "range"
            }
        }

    def identify_semantic_roles(self, doc) -> List[SemanticComponent]:
        """Identify semantic roles in the parsed text"""
        components = []
        
        # Track processed tokens to avoid duplicates
        processed_tokens = set()
        
        # Identify main action
        action = None
        for token in doc:
            if token.text.lower() in self.action_patterns and token.i not in processed_tokens:
                action = SemanticComponent(
                    role=SemanticRole.ACTION,
                    text=token.text,
                    normalized_text=token.text.lower(),
                    cypher_template=self.action_patterns[token.text.lower()]["cypher_template"]
                )
                processed_tokens.add(token.i)
                break
        
        if action:
            components.append(action)
        
        # Process noun chunks for subjects and attributes
        for chunk in doc.noun_chunks:
            if any(i in processed_tokens for i in range(chunk.start, chunk.end)):
                continue
                
            # Check if chunk contains attribute
            for attr, pattern in self.attribute_patterns.items():
                if attr in chunk.text.lower():
                    components.append(SemanticComponent(
                        role=SemanticRole.ATTRIBUTE,
                        text=chunk.text,
                        normalized_text=attr,
                        schema_mapping=pattern["schema_mapping"],
                        cypher_template=pattern["cypher_template"]
                    ))
                    processed_tokens.update(range(chunk.start, chunk.end))
                    break
            
            # If not processed as attribute, check for subject
            if not any(i in processed_tokens for i in range(chunk.start, chunk.end)):
                if "cigar" in chunk.text.lower():
                    components.append(SemanticComponent(
                        role=SemanticRole.SUBJECT,
                        text=chunk.text,
                        normalized_text="Cigar",
                        schema_mapping={"node": "Cigar"}
                    ))
                    processed_tokens.update(range(chunk.start, chunk.end))

        # Process conditions
        for token in doc:
            if token.i in processed_tokens:
                continue
                
            if token.text.lower() in self.condition_patterns:
                # Look ahead for values
                value_token = None
                for right_token in token.rights:
                    if right_token.like_num or '$' in right_token.text:
                        value_token = right_token
                        break
                
                if value_token:
                    components.append(SemanticComponent(
                        role=SemanticRole.CONDITION,
                        text=f"{token.text} {value_token.text}",
                        normalized_text=token.text.lower(),
                        cypher_template=self.condition_patterns[token.text.lower()]["cypher_template"]
                    ))
                    processed_tokens.add(token.i)
                    processed_tokens.add(value_token.i)

        return components

    def generate_cypher_template(self, components: List[SemanticComponent]) -> Tuple[str, Dict]:
        """Generate Cypher template from semantic components"""
        # Start with base template from action
        action_component = next((c for c in components if c.role == SemanticRole.ACTION), None)
        if not action_component:
            return "MATCH (c:Cigar)", {}
            
        template = action_component.cypher_template
        parameters = {}
        
        # Add conditions
        conditions = []
        for component in components:
            if component.role == SemanticRole.CONDITION:
                if component.schema_mapping:
                    conditions.append(f"{component.cypher_template}")
                    # Extract values from text
                    value = ''.join(filter(str.isdigit, component.text))
                    parameters[f"value_{len(parameters)}"] = int(value)
                    
        if conditions:
            template += "\nWHERE " + " AND ".join(conditions)
            
        # Add return clause
        template += "\nRETURN c"
        
        return template, parameters

    def parse(self, query: str) -> SemanticParseResult:
        """
        Parse a natural language query into semantic components
        Returns structured semantic parse result
        """
        # Process with spaCy
        doc = self.nlp(query)
        
        # Identify semantic components
        components = self.identify_semantic_roles(doc)
        
        # Generate Cypher template
        cypher_template, parameters = self.generate_cypher_template(components)
        
        # Create query structure
        query_structure = {
            "type": next((c.normalized_text for c in components if c.role == SemanticRole.ACTION), "search"),
            "subject": next((c.normalized_text for c in components if c.role == SemanticRole.SUBJECT), "Cigar"),
            "attributes": [c.normalized_text for c in components if c.role == SemanticRole.ATTRIBUTE],
            "conditions": [c.normalized_text for c in components if c.role == SemanticRole.CONDITION]
        }
        
        return SemanticParseResult(
            components=components,
            query_structure=query_structure,
            cypher_template=cypher_template,
            parameters=parameters
        )

def main():
    # Initialize parser
    parser = CigarSemanticParser()
    
    # Test queries
    test_queries = [
        "Find mild cigars under $15",
        "Recommend cigars similar to Padron 1964",
        "Show me cigars with maduro wrapper",
        "Compare strength levels of Opus X and Padron 1964",
        "Find full-bodied cigars from Nicaragua"
    ]
    
    print("Semantic Parsing Examples")
    print("========================")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = parser.parse(query)
        
        print("\nSemantic Components:")
        for component in result.components:
            print(f"- Role: {component.role.value}")
            print(f"  Text: {component.text}")
            print(f"  Normalized: {component.normalized_text}")
            if component.schema_mapping:
                print(f"  Schema Mapping: {component.schema_mapping}")
                
        print("\nQuery Structure:", result.query_structure)
        print("\nCypher Template:", result.cypher_template)
        if result.parameters:
            print("Parameters:", result.parameters)
        
        print("-" * 60)

if __name__ == "__main__":
    main()
    