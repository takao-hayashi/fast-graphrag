# FastGraphRAG: High-Performance Lightweight GraphRAG

**FastGraphRAG** is a next-generation, local-first Retrieval-Augmented Generation (RAG) framework that combines the power of Knowledge Graphs with the speed of classical algorithms. Designed for 2026 standards, it prioritizes efficiency, low latency, and privacy by leveraging a hybrid approach of Small Language Models (SLMs) and high-performance graph processing.

---

## 🚀 Key Features

### 1. Hybrid Extraction Engine
Instead of relying solely on heavy LLMs for entity extraction, FastGraphRAG uses a hybrid pipeline:
- **NER (Named Entity Recognition):** Powered by high-speed NLP libraries (Spacy/GritLM) for near-instant entity identification.
- **RE (Relation Extraction):** Targeted relationship determination using lightweight SLMs (e.g., Llama-3-8B, DeepSeek-R1-Distill).
- **Result:** 10x faster indexing compared to traditional LLM-only GraphRAG implementations.

### 2. Dynamic Graph Indexing
- **Streaming Updates:** Unlike static indexes, FastGraphRAG supports incremental updates. Add new documents without rebuilding the entire graph.
- **HNSW Integration:** Combines Hierarchical Navigable Small World (HNSW) vector search with graph topology for unified semantic and structural retrieval.

### 3. Intelligent Retrieval via PPR
- **Personalized PageRank (PPR):** Uses the user's query to identify "knowledge clusters" within the graph.
- **Contextual Relevance:** Ranks nodes based on their structural importance relative to the query, ensuring the LLM receives the most pertinent information.

### 4. Local-First & Zero-Config
- **Privacy by Design:** Optimized for local execution via Ollama or similar providers.
- **Ease of Use:** Point to a directory and start querying. No complex configuration files required.

---

## 🛠️ Tech Stack

- **Core Logic:** Python (with Rust-backed libraries for performance)
- **Graph Engine:** NetworkX / Petgraph
- **Vector Search:** Faiss (HNSW)
- **NLP:** Spacy
- **Data Processing:** Polars (High-speed DataFrame processing)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/fast-graphrag.git
cd fast-graphrag

# Install dependencies
pip install -r requirements.txt

# Download NLP model
python3 -m spacy download en_core_web_sm
```

---

## 📖 Usage

### Quick Start (Indexing & Querying)
Simply specify the directory containing your text files and your question:

```bash
python3 main.py --dir ./your_documents --query "How does the dynamic graph algorithm improve performance?"
```

### Integration with Local LLMs
FastGraphRAG is pre-configured to work with **Ollama**. Ensure your local server is running:

```bash
# Default model: llama3
ollama run llama3
```

---

## 📂 Project Structure

- `main.py`: Unified CLI entry point.
- `src/extractor.py`: Hybrid NER and Relation Extraction logic.
- `src/graph.py`: Dynamic graph management and vector indexing.
- `src/search.py`: PPR-based ranking and context synthesis.
- `src/llm.py`: Interface for local and cloud-based LLMs.

---

## ⚖️ License
This project is licensed under the MIT License.
