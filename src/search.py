import networkx as nx
from typing import List, Dict

class GraphSearcher:
    def __init__(self, graph_index):
        self.graph_index = graph_index

    def personalized_pagerank(self, query: str, top_k: int = 10) -> List[str]:
        # 1. クエリに関連するシードノードを見つける
        seed_nodes = self.graph_index.search_similar_nodes(query, top_k=3)
        if not seed_nodes:
            return []

        # 2. シードノードに重みを置いたパーソナライズドPageRankを実行
        personalization = {node: 1.0 for node in seed_nodes if node in self.graph_index.graph}
        
        if not personalization:
            return seed_nodes

        try:
            pagerank_scores = nx.pagerank(
                self.graph_index.graph, 
                alpha=0.85, 
                personalization=personalization,
                weight='weight'
            )
            # スコア順にソート
            sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            return [node for node, score in sorted_nodes[:top_k]]
        except Exception:
            # グラフが切断されている場合などのフォールバック
            return seed_nodes

    def get_context_from_nodes(self, nodes: List[str]) -> str:
        context_parts = []
        graph = self.graph_index.graph
        
        for node in nodes:
            if node in graph:
                neighbors = list(graph.neighbors(node))
                rels = []
                for neighbor in neighbors[:5]: # 各ノード最大5つの関係を表示
                    edge_data = graph.get_edge_data(node, neighbor)
                    rel_type = edge_data.get('relation', 'related_to')
                    rels.append(f"{node} --({rel_type})--> {neighbor}")
                
                if rels:
                    context_parts.append("\n".join(rels))
        
        return "\n".join(context_parts)
