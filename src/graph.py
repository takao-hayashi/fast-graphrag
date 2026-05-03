import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class DynamicGraphIndex:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.graph = nx.Graph()
        self.model = SentenceTransformer(embedding_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # HNSWインデックスの初期化
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.node_id_to_name = {}
        self.name_to_node_id = {}

    def add_document(self, entities: List[Dict], relations: List[Any]):
        for ent in entities:
            name = ent["name"]
            if name not in self.graph:
                self.graph.add_node(name, label=ent["label"])
                # ベクトル化とインデックス追加
                embedding = self.model.encode([name])[0]
                node_id = self.index.ntotal
                self.index.add(np.array([embedding]).astype('float32'))
                self.node_id_to_name[node_id] = name
                self.name_to_node_id[name] = node_id

        for source, rel, target in relations:
            if source in self.graph and target in self.graph:
                if self.graph.has_edge(source, target):
                    self.graph[source][target]['weight'] = self.graph[source][target].get('weight', 1) + 1
                else:
                    self.graph.add_edge(source, target, relation=rel, weight=1)

    def search_similar_nodes(self, query: str, top_k: int = 5) -> List[str]:
        query_vector = self.model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx in self.node_id_to_name:
                results.append(self.node_id_to_name[idx])
        return results

    def get_subgraph(self, nodes: List[str], radius: int = 1) -> nx.Graph:
        all_nodes = set(nodes)
        for node in nodes:
            if node in self.graph:
                neighbors = nx.single_source_shortest_path_length(self.graph, node, cutoff=radius)
                all_nodes.update(neighbors.keys())
        
        return self.graph.subgraph(all_nodes)
