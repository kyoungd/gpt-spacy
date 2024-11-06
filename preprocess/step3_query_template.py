import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class CigarQueryTemplates:
    def __init__(self):
        self.templates = {
            "single_property_search": {
                "description": "Search cigars by a single property",
                "patterns": [
                    "Find cigars with {property} {operator} {value}",
                    "Show cigars that have {property} {operator} {value}",
                    "What cigars are {property} {operator} {value}"
                ],
                "examples": [
                    {
                        "user_query": "Find cigars with price less than 15",
                        "cypher": """
                        MATCH (c:Cigar)
                        WHERE c.price < 15
                        RETURN c.name, c.price, c.strength_level
                        ORDER BY c.price ASC
                        """,
                        "explanation": "Matches cigars based on price property with less than operator"
                    },
                    {
                        "user_query": "Show me full-bodied cigars",
                        "cypher": """
                        MATCH (c:Cigar)
                        WHERE c.strength_level = 'Full'
                        RETURN c.name, c.strength_level, c.price
                        ORDER BY c.price ASC
                        """,
                        "explanation": "Filters cigars by strength level property with exact match"
                    }
                ]
            },
            "similarity_search": {
                "description": "Find similar cigars based on characteristics",
                "patterns": [
                    "Find cigars similar to {cigar_name}",
                    "What cigars are like {cigar_name}",
                    "Show me alternatives to {cigar_name}"
                ],
                "examples": [
                    {
                        "user_query": "Find cigars similar to Padron 1964",
                        "cypher": """
                        MATCH (c:Cigar {name: 'Padron 1964'})
                        MATCH (c)-[:HAS_STRENGTH]->(s:Strength)
                        MATCH (c)-[:HAS_FLAVOR]->(f:FlavorProfile)
                        WITH c, s, f
                        MATCH (similar:Cigar)-[:HAS_STRENGTH]->(s)
                        MATCH (similar)-[:HAS_FLAVOR]->(f)
                        WHERE similar.name <> c.name
                        RETURN similar.name, similar.price, similar.strength_level
                        ORDER BY similar.rating DESC
                        LIMIT 5
                        """,
                        "explanation": "Finds cigars sharing the same strength and flavor profile as the reference cigar"
                    }
                ]
            },
            "recommendation": {
                "description": "Get personalized recommendations",
                "patterns": [
                    "Recommend cigars based on {preference}",
                    "What cigars would I like if I enjoy {preference}",
                    "Suggest cigars similar to {preference} but {modifier}"
                ],
                "examples": [
                    {
                        "user_query": "Recommend cigars like Padron but cheaper",
                        "cypher": """
                        MATCH (ref:Cigar)-[:HAS_FLAVOR]->(f:FlavorProfile)
                        WHERE ref.name CONTAINS 'Padron'
                        WITH ref, collect(f) as flavors
                        MATCH (c:Cigar)-[:HAS_FLAVOR]->(f2:FlavorProfile)
                        WHERE c.price < ref.price
                        AND f2 IN flavors
                        RETURN c.name, c.price, c.strength_level,
                               collect(DISTINCT f2.name) as flavor_profiles
                        ORDER BY c.rating DESC
                        LIMIT 5
                        """,
                        "explanation": "Finds cigars with similar flavor profiles but lower price point"
                    }
                ]
            }
        }

    def generate_text_output(self) -> str:
        """Generate formatted text output for templates"""
        output = "CIGAR DATABASE QUERY TEMPLATES\n"
        output += "============================\n\n"
        output += "This document provides templates and examples for translating natural language queries to Cypher for a cigar database.\n\n"

        for template_name, template in self.templates.items():
            # Template section header
            output += f"\n{template_name.upper().replace('_', ' ')}\n"
            output += "=" * len(template_name) + "\n\n"
            
            # Description
            output += f"Description: {template['description']}\n\n"
            
            # Common patterns
            output += "Common Query Patterns:\n"
            for pattern in template["patterns"]:
                output += f"- {pattern}\n"
            output += "\n"
            
            # Examples
            output += "Examples:\n"
            for i, example in enumerate(template["examples"], 1):
                output += f"\nExample {i}:\n"
                output += f"User Query: {example['user_query']}\n"
                output += f"Explanation: {example['explanation']}\n"
                output += "Cypher Query:\n"
                output += f"{example['cypher'].strip()}\n"
                output += "-" * 50 + "\n"

        output += "\nUSAGE NOTES:\n"
        output += "============\n\n"
        output += "1. Variables in patterns are shown in {curly_braces}\n"
        output += "2. All Cypher queries should be properly parameterized in production\n"
        output += "3. Order and limit clauses can be adjusted based on needs\n"
        output += "4. Property names and relationship types should match your database schema\n"

        return output

def main():
    templates = CigarQueryTemplates()
    output = templates.generate_text_output()
    filepath = 'cigar_query_templates.txt'
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(output)
        logging.info(f"Template file generated: {filepath}")
    except IOError as e:
        logging.error(f"File I/O error while writing template file: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        raise

    # Print first few lines as preview
    preview_lines = output.split('\n')[:10]
    print("\nPreview of generated file:")
    print("\n".join(preview_lines))
    print("...")

if __name__ == "__main__":
    main()