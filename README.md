# CAX-BiHierGraphs

# Hierarchical Legal Graph (HierLegalGraph) Construction

This directory contains the scripts necessary to construct the Hierarchical Legal Graph (`HierLegalGraph`) from the ECtHR violation dataset and the ECHR articles.

## Overview
The hierarchical graph represents relations between cases, facts, terms (nouns), and articles. It consists of:
* **Nodes**: Articles, Terms, Facts, and Cases.
* **Edges**: 
  - `('articles', 'has', 'terms')`
  - `('facts', 'has', 'terms')`
  - `('cases', 'violate', 'articles')`
  - `('facts', 'part_of', 'cases')`
  - Self-loops for each node type.
  - Undirected conversions for all relations.

---

## Requirements

### Python Libraries
Ensure you have the following packages installed:
* `torch` and `torch_geometric`
* `transformers` (Hugging Face)
* `datasets` (Hugging Face)
* `spacy` (with `en_core_web_sm` model)
* `pandas`
* `numpy`
* `tqdm`

To install the required spaCy model, run:
```bash
python -m spacy download en_core_web_sm
```

### Datasets
1. **ECtHR Dataset**: Loaded via Hugging Face (`load_dataset('ecthr_cases')`).
2. **ECHR Articles**: A CSV file containing article descriptions. By default, it expects `/home/gokul/Hier-Legal-Graph/datasets/ECHR_Articles_new.csv` (delimited by `\t`).

---

## Graph Construction Steps

### 1. Extract Terms & Map Article-Term Edges
Run the term extraction script to extract nouns from the ECHR Article descriptions and generate the article-to-term connections:
```bash
python legal_exp/graph/construction/term_extraction.py
```
* **Input**: `ECHR_Articles_new.csv`
* **Output**: `datasets/article_term_graph.pkl` (contains `term_index`, `term_reverse_index`, and `article_term_edges`).
* **Note**: In `term_extraction.py`, check that the hardcoded `article_file_path` matches your local dataset path (currently points to `/home/gokul/LegalGraph/code/Article-Data/ECHR_Articles_new.csv`).

### 2. Extract Facts & Map Fact-Term Edges
Map the facts in the ECtHR training dataset to the terms extracted from the articles:
```bash
python legal_exp/graph/construction/fact_term_graph.py
```
* **Inputs**:
  - `datasets/article_term_graph.pkl`
  - Hugging Face `ecthr_cases` (train split)
* **Output**: `datasets/fact_term_graph_new.pkl` (contains `fact_term_edges`).

### 3. Map Case-Fact Edges
Run the case-fact graph builder to link facts to their parent cases:
```bash
python legal_exp/graph/construction/case_fact_graph.py
```
* **Input**: Hugging Face `ecthr_cases` (`alleged-violation-prediction` split)
* **Output**: `datasets/case_fact_graph.pkl`

### 4. Map Case-Article Edges
Run the case-article graph builder to link cases to the ECHR articles they violate:
```bash
python legal_exp/graph/construction/case_article_graph.py
```
* **Inputs**:
  - `datasets/ECHR_Articles_new.csv`
  - Hugging Face `ecthr_cases` (`violation-prediction` split)
* **Output**: `datasets/case_article_graph.pkl`

### 5. Generate Node Embeddings
Before constructing the hierarchical graph, run the embedding generation scripts located under `legal_exp/graph/embeddings/`:

* **Article Embeddings**:
  ```bash
  python legal_exp/graph/embeddings/article_embedding.py
  ```
  - **Output**: `datasets/article_embeddings_all.pkl`

* **Term Embeddings**:
  ```bash
  python legal_exp/graph/embeddings/term_embeddings.py
  ```
  - **Output**: `datasets/term_embeddings_all.pkl`
  - **Note**: Ensure you uncomment lines 64-65 in `term_embeddings.py` to enable saving the embeddings pickle file:
    ```python
    with open('./datasets/term_embeddings_all.pkl', 'wb') as file:
        pkl.dump(term_embeddings_all, file)
    ```

* **Fact Embeddings**:
  ```bash
  python legal_exp/graph/embeddings/fact_embeddings.py
  ```
  - **Output**: `datasets/fact_embeddings1.pkl`, `datasets/fact_embeddings2.pkl`, `datasets/fact_embeddings3.pkl` (representing splits 0-3000, 3000-6000, and 6000-9000 of the train set).

### 6. Construct the Hierarchical Graph
With all edge lists and node embeddings successfully generated, run the final integration script:
```bash
python legal_exp/graph/construction/construct_hier_graph.py
```
* **Inputs**:
  - `./datasets/article_term_graph.pkl`
  - `./datasets/fact_term_graph_new.pkl`
  - `./datasets/case_fact_graph.pkl`
  - `./datasets/case_article_graph.pkl`
  - `./datasets/article_embeddings_all.pkl`
  - `./datasets/term_embeddings_all.pkl`
  - `./datasets/fact_embeddings1.pkl`, `./datasets/fact_embeddings2.pkl`, `./datasets/fact_embeddings3.pkl`
* **Output**: `./datasets/Hetero_Data_With_Self_Loops.pkl` (a PyTorch Geometric `HeteroData` undirected graph object including self-loops).
* **Note**: Case node embeddings (`cases`) are initialized as a zero tensor of size `[9000, 768]`.
