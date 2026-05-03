import os
import argparse
from src.extractor import EntityExtractor, RelationExtractor
from src.graph import DynamicGraphIndex
from src.search import GraphSearcher
from src.llm import LLMInterface
from tqdm import tqdm

class FastGraphRAG:
    def __init__(self, model_name="en_core_web_sm", llm_model="llama3"):
        self.extractor = EntityExtractor(model_name)
        self.rel_extractor = RelationExtractor()
        self.index = DynamicGraphIndex()
        self.searcher = GraphSearcher(self.index)
        self.llm = LLMInterface(model=llm_model)

    def ingest_directory(self, directory_path: str):
        if not os.path.exists(directory_path):
            print(f"Error: Directory {directory_path} does not exist.")
            return
            
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        if not files:
            print(f"No .txt files found in {directory_path}")
            return

        print(f"Indexing {len(files)} files from {directory_path}...")
        
        for filename in tqdm(files):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                entities = self.extractor.extract_entities(text)
                relations = self.rel_extractor.extract_relations(text, entities)
                self.index.add_document(entities, relations)

    def query(self, user_query: str):
        # 1. PPRで重要ノードを特定
        important_nodes = self.searcher.personalized_pagerank(user_query)
        # 2. コンテキスト生成
        context = self.searcher.get_context_from_nodes(important_nodes)
        # 3. LLMで回答生成
        answer = self.llm.generate_answer(user_query, context)
        return answer, important_nodes

def main():
    parser = argparse.ArgumentParser(description="FastGraphRAG: Lightweight & Fast GraphRAG Implementation")
    parser.add_argument("--dir", type=str, help="Directory containing text files to index")
    parser.add_argument("--query", type=str, help="Query to ask the GraphRAG")
    args = parser.parse_args()

    rag = FastGraphRAG()

    if args.dir:
        rag.ingest_directory(args.dir)
        
    if args.query:
        answer, nodes = rag.query(args.query)
        print(f"\n--- Relevant Nodes ---\n{', '.join(nodes)}")
        print(f"\n--- Answer ---\n{answer}")
    elif not args.dir:
        parser.print_help()

if __name__ == "__main__":
    main()
