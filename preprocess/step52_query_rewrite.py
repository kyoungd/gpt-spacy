from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from enum import Enum
import spacy
import json

class RewriteType(Enum):
    ABBREVIATION = "abbreviation"
    COMPARATIVE = "comparative"
    SUPERLATIVE = "superlative"
    SYNONYM = "synonym"
    NEGATION = "negation"
    RANGE = "range"

@dataclass
class RewriteRule:
    original: str
    rewritten: str
    type: RewriteType
    cypher_template: Optional[str] = None

@dataclass
class RewrittenQuery:
    original_query: str
    rewritten_query: str
    applied_rules: List[RewriteRule]
    cypher_modifications: Dict[str, str]
    explanations: List[str]

class CigarQueryRewriter:
    def __init__(self, vocabulary_file: str = 'enhanced_cigar_vocabulary.json', rules_file: str = 'rewrite_rules.json'):
        # Load spaCy for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load vocabulary if available
        try:
            with open(vocabulary_file, 'r') as f:
                self.vocabulary = json.load(f)
        except FileNotFoundError:
            self.vocabulary = {}
            print(f"Warning: Vocabulary file '{vocabulary_file}' not found. Using empty vocabulary.")
        except json.JSONDecodeError as e:
            self.vocabulary = {}
            print(f"Error: Could not parse vocabulary file '{vocabulary_file}': {e}")
        
        # Load rewrite rules from configuration file
        try:
            with open(rules_file, 'r') as f:
                self.rules_config = json.load(f)
        except FileNotFoundError:
            self.rules_config = {}
            print(f"Warning: Rewrite rules file '{rules_file}' not found. Using empty rules.")
        except json.JSONDecodeError as e:
            self.rules_config = {}
            print(f"Error: Could not parse rewrite rules file '{rules_file}': {e}")

        # Initialize rewrite rules
        self.initialize_rules()

    def initialize_rules(self):
        """Initialize rewrite rules from configuration file"""
        # Load abbreviations
        self.abbreviations = self.rules_config.get('abbreviations', {})
        
        # Load comparative patterns
        self.comparative_patterns = self.rules_config.get('comparative_patterns', {})
        
        # Load superlative patterns
        self.superlative_patterns = self.rules_config.get('superlative_patterns', {})
        
        # Load range patterns
        self.range_patterns = self.rules_config.get('range_patterns', {})

    def expand_abbreviations(self, query: str) -> Tuple[str, List[RewriteRule]]:
        """Expand any known abbreviations in the query with error handling"""
        applied_rules = []
        try:
            words = query.split()
            for i, word in enumerate(words):
                word_lower = word.lower()
                if word_lower in self.abbreviations:
                    expanded = self.abbreviations[word_lower]
                    words[i] = expanded
                    applied_rules.append(RewriteRule(
                        original=word,
                        rewritten=expanded,
                        type=RewriteType.ABBREVIATION
                    ))
            rewritten_query = ' '.join(words)
            return rewritten_query, applied_rules
        except Exception as e:
            print(f"Error in expand_abbreviations: {e}")
            return query, applied_rules  # Return original query if error occurs

    def handle_comparatives(self, query: str) -> Tuple[str, List[RewriteRule], Dict[str, str]]:
        """Process comparative expressions with error handling"""
        applied_rules = []
        cypher_mods = {}
        try:
            for pattern, info in self.comparative_patterns.items():
                if pattern in query.lower():
                    parts = query.lower().split(pattern)
                    if len(parts) > 1:
                        target = parts[1].strip()
                        cypher_mods['ordering'] = info.get('ordering', '')
                        cypher_mods['comparison'] = info.get('cypher', '')
                        
                        applied_rules.append(RewriteRule(
                            original=f"{pattern} {target}",
                            rewritten=f"compared to {target}",
                            type=RewriteType.COMPARATIVE,
                            cypher_template=info.get('cypher', '')
                        ))
        except Exception as e:
            print(f"Error in handle_comparatives: {e}")
        return query, applied_rules, cypher_mods

    def handle_superlatives(self, query: str) -> Tuple[str, List[RewriteRule], Dict[str, str]]:
        """Process superlative expressions with error handling"""
        applied_rules = []
        cypher_mods = {}
        try:
            for pattern, info in self.superlative_patterns.items():
                if pattern in query.lower():
                    cypher_mods['ordering'] = info.get('cypher', '')
                    
                    applied_rules.append(RewriteRule(
                        original=pattern,
                        rewritten=f"ordered by {pattern}",
                        type=RewriteType.SUPERLATIVE,
                        cypher_template=info.get('cypher', '')
                    ))
        except Exception as e:
            print(f"Error in handle_superlatives: {e}")
        return query, applied_rules, cypher_mods

    def handle_ranges(self, query: str) -> Tuple[str, List[RewriteRule], Dict[str, str]]:
        """Process range expressions with error handling"""
        applied_rules = []
        cypher_mods = {}
        try:
            for pattern, info in self.range_patterns.items():
                matches = re.finditer(pattern, query.lower())
                for match in matches:
                    values = match.groups()
                    cypher_condition = info.get('cypher', '').format(*values)
                    cypher_mods['range_condition'] = cypher_condition
                    
                    applied_rules.append(RewriteRule(
                        original=match.group(),
                        rewritten=f"in range ({' to '.join(values)})",
                        type=RewriteType.RANGE,
                        cypher_template=cypher_condition
                    ))
        except Exception as e:
            print(f"Error in handle_ranges: {e}")
        return query, applied_rules, cypher_mods

    def rewrite_query(self, query: str) -> RewrittenQuery:
        """
        Apply all rewrite rules to the query with error handling.
        Returns RewrittenQuery with all modifications and explanations.
        """
        rewritten = query
        all_rules = []
        all_cypher_mods = {}
        explanations = []
        try:
            # Expand abbreviations
            rewritten, abbrev_rules = self.expand_abbreviations(rewritten)
            all_rules.extend(abbrev_rules)
            
            if abbrev_rules:
                explanations.append("Expanded abbreviations to their full forms")
            
            # Handle comparatives
            rewritten, comp_rules, comp_mods = self.handle_comparatives(rewritten)
            all_rules.extend(comp_rules)
            all_cypher_mods.update(comp_mods)
            
            if comp_rules:
                explanations.append("Processed comparative expressions for proper ordering")
            
            # Handle superlatives
            rewritten, super_rules, super_mods = self.handle_superlatives(rewritten)
            all_rules.extend(super_rules)
            all_cypher_mods.update(super_mods)
            
            if super_rules:
                explanations.append("Processed superlative expressions for result ordering")
            
            # Handle ranges
            rewritten, range_rules, range_mods = self.handle_ranges(rewritten)
            all_rules.extend(range_rules)
            all_cypher_mods.update(range_mods)
            
            if range_rules:
                explanations.append("Processed range expressions for proper filtering")
            
            if not all_rules:
                explanations.append("No rewrite rules were applied.")
            
            return RewrittenQuery(
                original_query=query,
                rewritten_query=rewritten,
                applied_rules=all_rules,
                cypher_modifications=all_cypher_mods,
                explanations=explanations
            )
        except Exception as e:
            print(f"Error in rewrite_query: {e}")
            return RewrittenQuery(
                original_query=query,
                rewritten_query=query,
                applied_rules=[],
                cypher_modifications={},
                explanations=["An error occurred during query rewriting."]
            )

def main():
    # Initialize rewriter
    rewriter = CigarQueryRewriter()
    
    # Test queries
    test_queries = [
        "Find med bod cigars from Nic",
        "Show me cigars stronger than Padron 1964",
        "What are the strongest maduro cigars?",
        "Find cigars between $10 and $20",
        "Show me cigars better rated than Cohiba",
        "What are the most expensive conn wrapper cigars?",
        "Find cigars between mild and medium strength"
    ]
    
    print("Query Rewriting Examples")
    print("=======================")
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        result = rewriter.rewrite_query(query)
        
        print(f"Rewritten Query: {result.rewritten_query}")
        
        if result.applied_rules:
            print("\nApplied Rules:")
            for rule in result.applied_rules:
                print(f"- {rule.type.value}: '{rule.original}' → '{rule.rewritten}'")
        
        if result.cypher_modifications:
            print("\nCypher Modifications:")
            for mod_type, mod in result.cypher_modifications.items():
                print(f"- {mod_type}: {mod}")
        
        if result.explanations:
            print("\nExplanations:")
            for explanation in result.explanations:
                print(f"- {explanation}")
        
        print("-" * 60)

if __name__ == "__main__":
    main()