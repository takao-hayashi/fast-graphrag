import os
import sys
from src.extractor import EntityExtractor, RelationExtractor
from src.graph import DynamicGraphIndex
from src.search import GraphSearcher

def test_workflow():
    print("--- Starting Logic Test ---")
    
    # 1. Initialize components
    print("Initializing components...")
    extractor = EntityExtractor()
    rel_extractor = RelationExtractor()
    index = DynamicGraphIndex()
    searcher = GraphSearcher(index)
    
    # 2. Sample data
    text = "Elon Musk is the CEO of Tesla. Tesla produces electric cars. SpaceX was founded by Elon Musk."
    print(f"Processing text: {text}")
    
    # 3. Extraction
    entities = extractor.extract_entities(text)
    print(f"Extracted Entities: {[e['name'] for e in entities]}")
    
    relations = rel_extractor.extract_relations(text, entities)
    print(f"Extracted Relations: {relations}")
    
    # 4. Indexing
    index.add_document(entities, relations)
    print(f"Graph Nodes: {list(index.graph.nodes)}")
    print(f"Graph Edges: {list(index.graph.edges)}")
    
    # 5. Search (PPR)
    query = "Who is the CEO of Tesla?"
    print(f"Query: {query}")
    important_nodes = searcher.personalized_pagerank(query, top_k=5)
    print(f"Important Nodes (PPR): {important_nodes}")
    
    context = searcher.get_context_from_nodes(important_nodes)
    print(f"Generated Context:\n{context}")
    
    if len(important_nodes) > 0:
        print("\n--- Logic Test Passed! ---")
    else:
        print("\n--- Logic Test Failed: No nodes found ---")

if __name__ == "__main__":
    test_workflow()
