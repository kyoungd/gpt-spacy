from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum
import spacy
import json

class ImplicitContext(Enum):
    POPULARITY = "popularity"
    QUALITY = "quality"
    VALUE = "value"
    AVAILABILITY = "availability"
    SEASONALITY = "seasonality"
    EXPERIENCE_LEVEL = "experience_level"
    TIME_COMMITMENT = "time_commitment"
    OCCASION = "occasion"

@dataclass
class ContextRule:
    trigger_terms: Set[str]
    implied_conditions: List[str]
    cypher_modifications: Dict[str, str]
    explanation: str

@dataclass
class ExpandedQuery:
    original_query: str
    expanded_query: str
    added_context: Dict[str, str]
    cypher_modifications: Dict[str, str]
    explanations: List[str]

class CigarContextExpander:
    def __init__(self, vocabulary_file: str = 'enhanced_cigar_vocabulary.json'):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load domain vocabulary if available
        try:
            with open(vocabulary_file, 'r') as f:
                self.vocabulary = json.load(f)
        except FileNotFoundError:
            self.vocabulary = {}
        
        # Initialize context rules
        self.initialize_context_rules()
        self.trigger_term_to_context = {}
        self.build_trigger_term_index()

    def build_trigger_term_index(self):
        """Build a reverse index of trigger terms to context types"""
        for context_type, rule in self.context_rules.items():
            for term in rule.trigger_terms:
                self.trigger_term_to_context[term.lower()] = context_type

    def initialize_context_rules(self):
        """Initialize context expansion rules"""
        self.context_rules = {
            ImplicitContext.POPULARITY: ContextRule(
                trigger_terms={'popular', 'best-selling', 'trending', 'top', 'favorite', 'well-known'},
                implied_conditions=[
                    'High user ratings',
                    'High sales volume',
                    'Frequently reviewed'
                ],
                cypher_modifications={
                    'ordering': 'ORDER BY c.rating DESC, c.review_count DESC',
                    'limit': 'LIMIT 10',
                    'conditions': 'c.rating >= 4.0 AND c.review_count >= 50'
                },
                explanation="Expanded popularity context to include highly rated and frequently reviewed cigars"
            ),

            ImplicitContext.QUALITY: ContextRule(
                trigger_terms={'premium', 'high-end', 'luxury', 'finest', 'top-rated', 'best'},
                implied_conditions=[
                    'High rating',
                    'Premium price range',
                    'Expert reviews'
                ],
                cypher_modifications={
                    'ordering': 'ORDER BY c.rating DESC',
                    'conditions': 'c.rating >= 4.5 AND c.price >= 15',
                    'joins': 'MATCH (c)-[:HAS_REVIEW]->(r:Review) WHERE r.expert = true'
                },
                explanation="Included quality indicators such as ratings and expert reviews"
            ),

            ImplicitContext.VALUE: ContextRule(
                trigger_terms={'good value', 'bang for buck', 'affordable', 'budget', 'worth', 'deal'},
                implied_conditions=[
                    'Price-to-rating ratio',
                    'Moderate price range',
                    'High user satisfaction'
                ],
                cypher_modifications={
                    'ordering': 'ORDER BY (c.rating / c.price) DESC',
                    'conditions': 'c.price <= 15 AND c.rating >= 4.0',
                    'limit': 'LIMIT 10'
                },
                explanation="Added value metrics considering price-to-rating ratio"
            ),

            ImplicitContext.EXPERIENCE_LEVEL: ContextRule(
                trigger_terms={'beginner', 'starter', 'new', 'first time', 'novice', 'starting'},
                implied_conditions=[
                    'Mild to medium strength',
                    'Moderate price',
                    'Good construction'
                ],
                cypher_modifications={
                    'conditions': """
                    c.strength_level IN ['Mild', 'Medium'] 
                    AND c.price <= 12 
                    AND c.construction_rating >= 4.0
                    """,
                    'ordering': 'ORDER BY c.strength_level ASC, c.rating DESC'
                },
                explanation="Included beginner-friendly characteristics"
            ),

            ImplicitContext.OCCASION: ContextRule(
                trigger_terms={'special', 'celebration', 'occasion', 'gift', 'birthday', 'holiday'},
                implied_conditions=[
                    'Premium quality',
                    'Gift packaging available',
                    'Highly regarded brands'
                ],
                cypher_modifications={
                    'conditions': 'c.rating >= 4.3 AND c.gift_box = true',
                    'joins': 'MATCH (c)-[:MADE_BY]->(b:Brand) WHERE b.prestige_level >= 4',
                    'ordering': 'ORDER BY c.rating DESC, c.price DESC'
                },
                explanation="Added special occasion considerations"
            ),

            ImplicitContext.TIME_COMMITMENT: ContextRule(
                trigger_terms={'quick', 'short', 'lunch break', 'brief', 'small'},
                implied_conditions=[
                    'Shorter length',
                    'Smaller ring gauge',
                    'Consistent burn'
                ],
                cypher_modifications={
                    'conditions': 'c.smoking_time <= 30 AND c.ring_gauge <= 50',
                    'ordering': 'ORDER BY c.smoking_time ASC'
                },
                explanation="Included smoking duration considerations"
            )
        }

    def identify_implicit_context(self, query: str) -> List[ImplicitContext]:
        """Identify implicit context from query terms using reverse index"""
        query_terms = query.lower().split()
        identified_contexts = set()
        for term in query_terms:
            context_type = self.trigger_term_to_context.get(term)
            if context_type:
                identified_contexts.add(context_type)
        return list(identified_contexts)

    def expand_query(self, query: str) -> ExpandedQuery:
        """
        Expand query with implicit context
        Returns ExpandedQuery with added context and modifications
        """
        # Identify contexts
        contexts = self.identify_implicit_context(query)
        
        expanded_query = query
        added_context = {}
        cypher_mods = {}
        explanations = []
        
        # Apply context rules
        for context in contexts:
            rule = self.context_rules[context]
            
            # Add context information
            if context.value not in added_context:
                added_context[context.value] = rule.implied_conditions
                explanations.append(rule.explanation)
            
            # Collect Cypher modifications
            for mod_type, mod_value in rule.cypher_modifications.items():
                if mod_type not in cypher_mods:
                    cypher_mods[mod_type] = []
                cypher_mods[mod_type].append(mod_value)
            
            # Expand query with implicit conditions
            if not any(term in query.lower() for term in rule.trigger_terms):
                expanded_query += f" (Implied: {', '.join(rule.implied_conditions)})"
        
        # Combine Cypher modifications if multiple contexts
        final_cypher_mods = {}
        for mod_type, mod_list in cypher_mods.items():
            if mod_type == 'conditions':
                final_cypher_mods[mod_type] = ' AND '.join(mod_list)
            elif mod_type == 'ordering':
                final_cypher_mods[mod_type] = mod_list[0]  # Use first ordering
            else:
                final_cypher_mods[mod_type] = ' '.join(mod_list)
        
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_query,
            added_context=added_context,
            cypher_modifications=final_cypher_mods,
            explanations=explanations
        )

def main():
    # Initialize expander
    expander = CigarContextExpander()
    
    # Test queries
    test_queries = [
        "Show me popular cigars",
        "What are some good beginner cigars?",
        "Find premium cigars for a special occasion",
        "Recommend good value cigars",
        "What's a quick smoke for lunch break?",
        "Show me the best-selling mild cigars",
        "Find affordable premium cigars"
    ]
    
    print("Context Expansion Examples")
    print("=========================")
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        result = expander.expand_query(query)
        
        print("\nExpanded Understanding:")
        for context_type, implications in result.added_context.items():
            print(f"\nContext: {context_type}")
            print("Implied Conditions:")
            for impl in implications:
                print(f"- {impl}")
        
        if result.cypher_modifications:
            print("\nCypher Modifications:")
            for mod_type, mod in result.cypher_modifications.items():
                print(f"- {mod_type}: {mod}")
        
        print("\nExplanations:")
        for explanation in result.explanations:
            print(f"- {explanation}")
        
        print("-" * 70)

if __name__ == "__main__":
    main()