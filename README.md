# Capstone Project: Retrieval-Augmented Generation (RAG) System for Academic Papers

This repository contains the code and resources for the capstone project on the Retrieval-Augmented Generation (RAG) System for Academic Papers, focused on developing a new RAG strategy for academic paper retrieval and answer generation. This project is part of the Data Science program at the American University of Armenia.


## Installation and Usage

1) Install required dependencies:

```
pip install -r requirements.txt
```
2) Run start_rag.py, writing your question in the input window

```
python start_rag.py
```

## Repository Structure

This repository is organized into the following directories:

### 1. `dataset`
- **`link_to_arxiv_dataset.txt`**: Contains a link to the dataset source.
- **`json_to_csv.py`**: Converts the dataset from JSON format to CSV.
- **`arxiv_metadata.csv`**: The output CSV file containing metadata of the papers.
- **`vectorization.py`**: Script for vectorizing abstracts from `arxiv_metadata.csv`.
- **`embedding.pkl`**: Output file containing the embeddings from the vectorization script.

### 2. `search_methods`
- **`faiss_search.ipynb`**: Demonstrates the use of FAISS for efficient similarity search.
- **`hnsw_search.ipynb`**: Implementation of Hierarchical Navigable Small World (HNSW) graphs for search.
- **`hybrid_search.ipynb`**: A hybrid search approach combining multiple methods.
- **`key_word_search.py`**: Basic keyword search implementation.
- **`sbert_search.ipynb`**: Search using Sentence-BERT for semantic similarity.

### 3. `pdf_manipulations`
- **`pdf_downloader.py`**: Script for downloading PDFs.
- **`pdf_db`**: Folder containing 200 downloaded PDFs.
- **`pdf_to_text.py`**: Converts PDF documents to text.
- **`pdfminer`**: Python package used for PDF text conversion.
- **`pdf_vectorization.py`**: Script for vectorizing text extracted from PDFs.
- **`pdf_paragraphs_embeddings.pkl`**: Output file containing embeddings of PDF paragraphs.

### 4. `rag`
- **`hnsw_cosine.py`**: Implements HNSW with cosine similarity for vector searches at both abstract and PDF levels.
- **`query_to_gpt.py`**: Integrates RAG with GPT for query handling.
- **`query_to_gemini.py`**: Uses Gemini model for RAG.
- **`query_to_llama3.ipynb`**: Incorporates LLaMA-3 for query processing.
- **`query_to_t5.ipynb`**: T5 integration for handling queries.

### 5. `analysis`
- **`analysis.ipynb`**: Notebook containing detailed analysis of the project outcomes, including evaluations of different LLMs, sample question generation, and more.
- **`analysed_data`** and **`plots`**: Directories containing output data and visualizations from the analysis.


