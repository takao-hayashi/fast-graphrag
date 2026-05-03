import os
from openai import OpenAI
from typing import List, Dict

class LLMInterface:
    def __init__(self, api_key: str = None, base_url: str = "http://localhost:11434/v1", model: str = "llama3"):
        # デフォルトでOllamaを想定
        self.client = OpenAI(
            api_key=api_key or "ollama",
            base_url=base_url
        )
        self.model = model

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""
以下のコンテキスト（知識グラフから抽出された関係性）を使用して、質問に答えてください。
コンテキストにない情報は、一般的な知識として補足しても構いませんが、コンテキストを優先してください。

### コンテキスト:
{context}

### 質問:
{query}

### 回答:
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは優秀なアシスタントです。提供された知識グラフの情報を活用して正確に答えてください。"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
