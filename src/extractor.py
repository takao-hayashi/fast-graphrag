import spacy
from typing import List, Tuple, Dict
import json

class EntityExtractor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            import os
            os.system(f"python3 -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "name": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities

class RelationExtractor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract_relations(self, text: str, entities: List[Dict]) -> List[Tuple[str, str, str]]:
        # 軽量化のため、まずはルールベースまたは小型LLMへのプロンプトを想定
        # ここではSLM（小型LLM）をシミュレートするか、シンプルな共起関係をベースにする
        relations = []
        entity_names = list(set([e["name"] for e in entities]))
        
        if len(entity_names) < 2:
            return relations

        # 簡易的な共起ベースの関係抽出（MVP用）
        # 実際にはLLMを使用して関係性を確定させる
        for i in range(len(entity_names)):
            for j in range(i + 1, min(i + 3, len(entity_names))):
                relations.append((entity_names[i], "related_to", entity_names[j]))
        
        return relations
