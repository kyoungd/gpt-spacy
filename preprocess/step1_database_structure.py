from neo4j import GraphDatabase
import json
from typing import Dict, List
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

class DatabaseStructureExtractor:
    def __init__(self):
        uri = os.getenv('NEO4J_URI')
        username = os.getenv('NEO4J_USER')
        password = os.getenv('NEO4J_PASSWORD')
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def get_complete_node_info(self) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
            MATCH (n)
            WITH DISTINCT labels(n)[0] as label, 
                 COLLECT(DISTINCT keys(n)) as all_properties,
                 COLLECT(DISTINCT properties(n))[0] as sample_properties,
                 COUNT(n) as node_count
            RETURN {
                label: label,
                all_possible_properties: apoc.coll.flatten(all_properties),
                sample: sample_properties,
                count: node_count
            } as nodeInfo
            """)
            return [record["nodeInfo"] for record in result]

    def get_relationship_structure(self) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
            MATCH ()-[r]->()
            WITH DISTINCT type(r) AS relType,
                 startNode(r) AS start,
                 endNode(r) AS end,
                 keys(r) AS property_keys,
                 r AS sample_rel,
                 count(r) as frequency
            RETURN {
                type: relType,
                fromLabels: labels(start),
                toLabels: labels(end),
                properties: [k IN property_keys | {
                    name: k,
                    sampleValue: sample_rel[k]
                }],
                frequency: frequency
            } AS relStructure
            """)
            return [record["relStructure"] for record in result]

    def get_sample_patterns(self) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run("""
            MATCH (n)-[r]->(m)
            WITH DISTINCT type(r) as relType, 
                 labels(n)[0] as fromLabel, 
                 labels(m)[0] as toLabel,
                 n as fromNode,
                 m as toNode,
                 r as rel
            LIMIT 5
            RETURN {
                pattern: {
                    fromNode: {
                        label: fromLabel,
                        properties: properties(fromNode)
                    },
                    relationship: {
                        type: relType,
                        properties: properties(rel)
                    },
                    toNode: {
                        label: toLabel,
                        properties: properties(toNode)
                    }
                }
            } AS samplePattern
            """)
            return [record["samplePattern"] for record in result]

    def get_constraints_and_indexes(self) -> Dict:
        with self.driver.session() as session:
            constraints = session.run("SHOW CONSTRAINTS")
            indexes = session.run("SHOW INDEXES")
            
            return {
                "constraints": [self._clean_dict(dict(record)) for record in constraints],
                "indexes": [self._clean_dict(dict(record)) for record in indexes]
            }

    def _clean_dict(self, d: Dict) -> Dict:
        """Clean dictionary by converting non-serializable objects to strings"""
        return {k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) 
                else self._clean_dict(v) if isinstance(v, dict)
                else [self._clean_dict(i) if isinstance(i, dict) else str(i) if not isinstance(i, (list, str, int, float, bool, type(None))) else i for i in v] if isinstance(v, list)
                else v
                for k, v in d.items()}

    def _process_properties(self, props: Dict) -> List[Dict]:
        """Convert properties dict to structured list with types"""
        return [
            {
                "name": key,
                "type": type(value).__name__,
                "sample_value": value
            }
            for key, value in props.items()
        ] if props else []

    def generate_schema(self) -> Dict:
        """Generate comprehensive schema as structured JSON"""
        nodes = self.get_complete_node_info()
        relationships = self.get_relationship_structure()
        patterns = self.get_sample_patterns()
        constraints_indexes = self.get_constraints_and_indexes()

        schema = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "database_purpose": "Cigar recommendation system with accessories and brand information"
            },
            "nodes": {},
            "relationships": {},
            "sample_patterns": [],
            "constraints_and_indexes": constraints_indexes
        }

        # Process nodes
        for node in nodes:
            properties = []
            if node['sample']:
                properties = [
                    {
                        "name": prop,
                        "type": type(node['sample'].get(prop)).__name__ if prop in node['sample'] else "unknown",
                        "sample_value": node['sample'].get(prop)
                    }
                    for prop in node['all_possible_properties']
                ]

            schema["nodes"][node['label']] = {
                "count": node.get('count', 0),
                "properties": properties,
                "sample": node['sample']
            }

        # Process relationships
        for rel in relationships:
            rel_type = rel['type']
            schema["relationships"][rel_type] = {
                "frequency": rel['frequency'],
                "source_labels": rel['fromLabels'],
                "target_labels": rel['toLabels'],
                "properties": [
                    {
                        "name": prop['name'],
                        "type": type(prop['sampleValue']).__name__,
                        "sample_value": prop['sampleValue']
                    }
                    for prop in rel['properties']
                ]
            }

        # Process sample patterns
        schema["sample_patterns"] = [
            {
                "source": {
                    "label": pattern['pattern']['fromNode']['label'],
                    "properties": self._process_properties(pattern['pattern']['fromNode']['properties'])
                },
                "relationship": {
                    "type": pattern['pattern']['relationship']['type'],
                    "properties": self._process_properties(pattern['pattern']['relationship']['properties'])
                },
                "target": {
                    "label": pattern['pattern']['toNode']['label'],
                    "properties": self._process_properties(pattern['pattern']['toNode']['properties'])
                }
            }
            for pattern in patterns
        ]

        return schema

    def save_schema(self, output_format='json'):
        """Save schema to file in specified format"""
        schema = self.generate_schema()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Save JSON version
        json_file = os.path.join(current_dir, 'db_structure.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        print(f"\nDatabase structure has been saved to: {json_file}")
        return schema

def main():
    extractor = DatabaseStructureExtractor()
    try:
        # Generate and save schema
        schema = extractor.save_schema()
        
        # Print sample of the JSON structure
        print("\nSample of generated schema:")
        print(json.dumps(schema["metadata"], indent=2))
        print("\nFirst node label found:", next(iter(schema["nodes"].keys())))
        print("\nFirst relationship type found:", next(iter(schema["relationships"].keys())))
            
    finally:
        extractor.close()

if __name__ == "__main__":
    main()