# CAFE

# Bifacet LegalGraph Construction

This directory contains the scripts necessary to construct the Bifacet LegalGraph from the ECtHR violation dataset and the ECHR articles.

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


# Appendix

---

## A. More Related Work

### A.1 Explainability in Legal AI

**Kapoor et al. (2022)** employs BERT-based transformers followed by SVM and XGBoost classifiers for bail prediction and salient sentence recognition, respectively. A multitask learning framework is utilized, where improved salient sentence detection contributes to more accurate predictions, and the salient sentences serve as explanations.

**Malik et al. (2021)** adopts a hierarchical XLNet-based architecture with BiGRU units for context regularization in the judgment prediction task. An occlusion-based method is applied wherein each complete chunk of the embedding is masked individually, and the resulting output probability is compared against the unmasked output. A positive difference indicates that the masked chunk contributes to the model's decision and is thus considered explanatory.

**Chalkidis et al. (2021)** learns a mask over case facts, compressing the representation to retain only the relevant facts, and performs alleged article violation prediction. While these methods address explainability, they do not incorporate legal precedents into their prediction processes.

**Wu et al. (2023)** introduces a precedent-enhanced approach that leverages domain models to identify similar cases—referred to as precedents—from a case database and employs Retrieval-Augmented Generation (RAG) with a large language model (LLM) to improve judgment prediction. Although this approach incorporates precedents, LLMs inherently lack explainability, and precedents can only serve as explanations after manual filtering of their factual content. Moreover, the approach requires the case database to conform to a specific format, limiting its applicability to real-world, unstandardized data.

**Paul et al. (2022)** utilizes both attribute and structural encoders to learn case embeddings. The structural encoder captures a citation network organized as a hierarchical graph, where nodes represent statutes, their chapters, topics, and sections, with links indicating hierarchical relationships. Case facts are also represented as nodes and connected to sections they cite. Inter- and intra-metapath aggregation are applied to derive the structural encoding, while a Hierarchical Attention Network is used to jointly encode the attributes of facts and sections. Formulated as an inductive link prediction task between a newly introduced fact and a section, this Legal Statute Identification method focuses on individual case facts rather than entire cases and does not provide explainability.

---

### A.2 Explainability in Medical AI

**Amann et al. (2020)** provides a multidisciplinary assessment of explainability in medical AI, emphasizing that it extends beyond technical transparency to encompass ethical and legal accountability in clinical decision support. While they note that post-hoc techniques such as LIME and SHAP provide interpretability, these methods remain detached from the structural logic of clinical reasoning and fail to capture how individual medical elements interact. Our proposed Bifacet Graph framework addresses this by embedding medical cases within a hierarchical structure that encodes the relationships among findings, diagnoses, and procedures. This allows explanations to emerge as subgraphs that reflect clinical dependencies, maintaining transparency while preserving contextual meaning.

**Huang et al. (2024)** conducts a comprehensive review of explainable and interpretable deep learning methods in healthcare NLP, identifying attention mechanisms and knowledge-graph-based reasoning as dominant strategies. However, most of these approaches operate at a local level and cannot generalize to global, system-level reasoning across complex medical data. By training a BiGNN for transductive link prediction over the hierarchical case graph, our framework extends explainability beyond local feature importance to capture relational patterns between medical entities, enabling global structural interpretability while retaining post-hoc flexibility.

**Healthcare (2025)** presents a meta-analysis of explainable AI in clinical decision support systems, highlighting the prevalence of model-agnostic methods such as Grad-CAM, SHAP, and counterfactuals across domains like radiology and oncology. They point out that current XAI techniques struggle with fragmented data representation, low explanation fidelity, and limited usability in clinical workflows. In contrast, our method provides a unified, graph-based representation of the medical case database, from which subgraph explanations are extracted and translated into natural language using an LLM. This produces faithful, relationally grounded, and human-readable explanations that integrate seamlessly with clinical reasoning, overcoming the opacity and fragmentation inherent in existing XAI methods.

---

## B. Construction of Bifacet Graphs

### B.1 Legal Domain — Bifacet LegalGraph

The Bifacet LegalGraph is constructed using the **European Court of Human Rights (ECtHR)** dataset. This dataset comprises 11,000 legal cases, each providing structured information including detailed facts, alleged and adjudicated violations of the European Convention on Human Rights (ECHR), and expert- or court-identified rationales (termed silver or gold rationales) that justify the judgments. This rich structure makes it ideal for tasks like legal judgment prediction, rationale extraction, and prior case retrieval.

The knowledge code nodes represent the allegedly violated ECtHR articles; the fundamental nodes comprise law-specific terms tokenized from the article descriptions. The document nodes include Level 1 nodes for individual paragraphs (facts) and Level 2 nodes for complete legal case texts. All node attributes leverage **LegalBERT** embeddings. Specifically, `F` nodes are embedded as the legal term, `K` nodes as the full article description, and `D₁` nodes as the individual paragraph text. For the high-level `D₂` case nodes, where concatenated paragraphs often exceed LegalBERT's context limit, a **sliding window** is used: the final attribute is the mean of the LegalBERT encodings across all resulting windows.


The structure of the Bifacet LegalGraph supports two core tasks:

1. **Transductive Edge Prediction:** predicting links between legal cases (`D₂`) and ECtHR articles (`K`), corresponding to the Legal Judgment Prediction problem.
2. **Explanation Generation:** for each positively predicted case–article link, a justification is produced via Rationale Extraction, drawing on relevant precedential case pathways in the graph. This implicitly addresses the Prior Case Retrieval problem by identifying relevant evidential paths.

---

### B.2 Medical Domain — Bifacet MedGraph

For the medical domain, we construct the **Bifacet MedGraph** using the **MIMIC-IV** dataset. This comprehensive dataset contains de-identified electronic health records (EHRs) from the Beth Israel Deaconess Medical Center to support **Medical Procedure Necessity Prediction (MPNP)**. The task is to predict the necessity of specific medical procedures (ICD-10-PCS codes) based on the clinical context provided in discharge summaries.

Our preprocessing pipeline transforms the raw data into the Bifacet Graph structure, involving three primary stages: Knowledge Code (ICD) node generation, Document (Discharge Summary) node generation, and Foundation (Medical Entity) node extraction.

**Knowledge Code Nodes (`K`)** are represented by ICD-10 procedure codes. We extract official titles and descriptions from standard ICD-10-PCS code files published by CMS, then use **Clinical T5** (trained on MIMIC III and IV) to enrich these formal definitions into multi-sentence descriptions, which serve as the primary text attribute for each knowledge code node.

**Document Nodes** are constructed from MIMIC-IV discharge summaries, each tied to a specific `subject_id` and hospital admission (`hadm_id`), forming the high-level Level 2 (`D₂`) nodes. To create fine-grained Level 1 (`D₁`) nodes, we parse the unstructured `text` field using a regex-based method to extract content following predefined section headers:

- `Chief Complaint`
- `Major Surgical or Invasive Procedure`
- `History of Present Illness`
- `Past Medical History`
- `Social History`
- `Family History`
- `Physical Exam`
- `Brief Hospital Course`
- `Pertinent Results`
- `Medications on Admission`
- `Discharge Medications`
- `Discharge Instructions`
- `Discharge Diagnosis`

Ground-truth edges (`E_DK`) link each `hadm_id` (Level 2 document) to its corresponding ICD procedure codes.

**Foundation Nodes (`F`)** are extracted via Named Entity Recognition (NER) using the `Clinical-AI-Apollo/Medical-NER` model, applied to both ICD code descriptions and parsed clinical notes. Entities are filtered to the following semantic groups:

- `BIOLOGICAL_STRUCTURE`
- `DIAGNOSTIC_PROCEDURE`
- `DISEASE_DISORDER`
- `HISTORY`
- `MEDICATION`
- `SEVERITY`
- `SIGN_SYMPTOM`
- `THERAPEUTIC_PROCEDURE`

Edges are then established as follows:
- An edge in `E_FK` is created between a foundation node `f` (e.g., *"hydrocephalus"*) and a knowledge code `k` if `f` is extracted from `k`'s detailed description.
- An edge in `E_FD` is created between a foundation node `f` (e.g., *"ascites"*) and a Level 1 document node `d` (e.g., the `Chief Complaint` section for admission `hadm_id` 22595853) if `f` is extracted from the text of `d`.

---

## C. Implementation Details

We implement **3-layer BiGNN** variants, since from an input document node `d`, potential precedents are 2 hops away, and articles sharing fundamental nodes are 3 hops away. `GATConv` layers are used for node feature aggregation with `ReLU` activations. Link prediction logits are computed as the dot product of learned document and knowledge code node embeddings. This same architecture is used for Bifacet LegalGraph, Bifacet MedGraph, HeteroLegalGraph, and HeteroMedGraph, though only **2-layer** variants are used for HeteroLegalGraph and HeteroMedGraph to ensure comparable information aggregation.

2-layer GNNs with `GATConv` and dropout are used for LegalGraph and MedGraph, with a fully connected layer for the node classification task (reformulated as `|K|` binary classifications using Binary Cross Entropy).

For **TabularLegal**, we reimplement `HIERBERT-HA`, a hierarchical BERT with hard attention, using Distribution Loss instead of Binary Cross Entropy to account for label imbalance (each case violates 1.35 articles on average out of 40 total; maximum is 8). **TabularMed** is implemented as a 10-layer deep neural network with `LayerNorm`, `ReLU`, and dropout, trained with Asymmetric Loss.

As the Graph Explainer `Φ`, we develop **HeteroGNNExplainer**, a PyTorch Geometric-compatible heterogeneous graph variant of GNNExplainer (see Appendix D for motivation). HeteroGNNExplainer is trained per instance for 50 epochs with:
- Learning rate: `0.01`
- Edge mask sparsity regularizer (`α₁`): `0.005`
- Edge mask entropy regularizer (`α₂`): `1.0`

Beam search with `m=2` and `m=3` is applied for comparative analysis, selecting the top `m` neighbours of each node type. Bidirectional Dijkstra's algorithm extracts the top `p=5` paths from `d` to `k`, with edges from previously extracted paths dropped iteratively.

LLMs used:
- **Bifacet LegalGraph:** SaulLM-7B-Instruct
- **Bifacet MedGraph:** BioMedical LLaMA-3-8B

**Training hyperparameters:**

| Model | Optimizer | LR | Epochs | Batch Size | Other |
|---|---|---|---|---|---|
| BiGNN (Bifacet/Hetero Graphs) | Adam | 5×10⁻⁴ | 500 | — | — |
| GNN (LegalGraph) | Adam | 1×10⁻³ | 100 | — | — |
| TabularLegal | AdamW | 1×10⁻³ | 100 | 64 | — |
| TabularMed | AdamW | 5×10⁻⁴ | 150 | 128 | L2 weight decay: 1×10⁻⁴ |

Asymmetric Loss hyperparameters for TabularMed: `γ_neg=4`, `γ_pos=4`, `clip=0.05`, `ε=0.1`.

## D. Likert Scale Study

For the legal domain, a qualified legal expert evaluated **20 comparative explanation sets**, each comprising a baseline, silver rationale, and BiFacetExplainer output for a unique case–article pair. For the medical domain, **two medical professionals** assessed 11 comparative sets.

> Due to the data privacy and redistribution clauses mandated by PhysioNet, qualitative explanation sets for MIMIC-IV are omitted to maintain full regulatory compliance with the dataset's access requirements.

---

## E. Ablation Studies

### E.1 Beam Search (SEARCH) vs. Threshold-Based (MASK) Subgraph Extraction

We compare our beam search strategy (**BFE-SEARCH**) against a threshold-based approach (**BFE-MASK**), which applies a fixed threshold `τ_mask = 0.385` to edge weights produced by the graph explainer to prune the subgraph.

Unlike BFE-SEARCH, BFE-MASK requires manual tuning of a subgraph-specific hyperparameter and may produce highly sparse subgraphs that sever all valid paths between the document node `d` and knowledge code node `k`. We measure the average percentage of test samples retaining at least one `d → k` path in `G_dk`.

**Key finding:** The Bifacet MedGraph achieves 100% valid path retention under BFE-SEARCH (with `m=3`), while BFE-MASK yields a slightly lower percentage of valid paths in both domains, demonstrating that path-guided extraction produces more consistently grounded explanations.

---

### E.2 Effect of Duplicate Edges in Bifacet MedGraph

During Bifacet MedGraph construction, the same fundamental medical term often appeared multiple times within a single medical note, resulting in multiple parallel edges between a fundamental node and a knowledge code node. To assess the impact, we constructed **NoDupBifacetMedGraph** by retaining only a single instance of each duplicated edge.

**NoDupBifacetMedGraph Statistics:**

| Component | Count |
|-----------|-------|
| **Nodes** | |
| Notes | 42,755 |
| ICDs | 8,367 |
| Terms | 5,759 |
| **Edges** | |
| Notes ↔ Terms | 1,735,382 |
| Notes ↔ ICDs | 125,653 |
| ICDs ↔ Terms | 44,559 |

**GEF scores (HeteroGNNExplainer):**

| Graph | m | GEF ↓ |
|---|---|---|
| Bifacet MedGraph | 2 | 0.4037 |
| Bifacet MedGraph | 3 | 0.3870 |
| NoDupBifacetMedGraph | 2 | **0.3734** |
| NoDupBifacetMedGraph | 3 | **0.3090** |

Although duplicate edges provide a modest improvement in AUROC and F1, NoDupBifacetMedGraph yields explanation subgraphs with substantially higher graph faithfulness, suggesting that removing duplicate edges leads to explanations that more accurately reflect the model's true decision-making pathways.

---

## F. Additional Benefits of Bifacet Graphs

### F.1 Scaling Bifacet Graphs

The Bifacet Graph is designed for efficient, dynamic updates:

- **Adding a new knowledge code node:** `O(L_k)` time, where `L_k` is the length of the knowledge code text — iterates over each token to query a hash table and establish edges with existing fundamental nodes.
- **Adding a new document node:** a two-step process — (1) build a subgraph for the document and its hierarchical levels; (2) apply the same `O(L_d)` algorithm to connect its lowest-level nodes `d ∈ D₁` to fundamental nodes, followed by an `O(|K|)` operation to add edges between the new document node and all knowledge codes. Overall complexity: **`O(L_d + |K|)`**.

Furthermore, due to the `l`-layer architecture of the BiGNN, a node's final embedding is influenced exclusively by its `l`-hop neighbourhood. When new nodes are introduced, re-training the entire model is unnecessary — an **incremental training strategy** updates only the embeddings of newly added nodes and their `l`-hop subgraphs.

---

### F.2 Model-Agnostic Design

BiFacetExplainer is fundamentally **model-agnostic**. It does not depend on the internal architecture, training objective, or message passing formulation of the underlying graph model. Instead, it operates on structural and representational outputs — such as node embeddings, edge weights, or attention maps — produced by any graph-based predictor, whether homogeneous or heterogeneous. This allows BiFacetExplainer to interface seamlessly with existing GNNs, relational models, or symbolic graph frameworks without requiring model retraining or modification, acting as a unifying interpretability layer across diverse graph learning approaches.

---

## G. Limitations

**Hierarchy construction.** Bifacet Graphs require careful selection of fundamental nodes and hierarchical structures. Without domain expertise or robust extraction heuristics, the graph may misrepresent relationships or omit key concepts, limiting the reliability of generated explanations.

**Path extraction limitations.** The path extraction process may miss reasoning patterns that are non-linear or distributed, such as aggregated diffusion signals. This can limit the completeness of explanations generated from the graph.

**Residual LLM hallucinations.** Deterministic extraction reduces hallucinations but cannot fully prevent them during the synthesis step. Careful prompt design, confidence scoring, and verification are still needed to maintain reliability.

**Restriction to true positives.** BiFacetExplainer currently applies only to true positive predictions, as existing graph explanation models are designed for correctly classified instances. This limits analysis of false positives or false negatives; future work could integrate graph counterfactual explainers that generalize to all prediction outcomes.

**One-to-one document–knowledge code explanations.** The explanation generation pipeline is designed for one document node to one knowledge code prediction. Generating explanations for a document associated with multiple knowledge codes requires running the pipeline separately for each pair, limiting efficiency in multi-label or dense knowledge code assignment settings.

---

## H. Prompt Templates

### H.1 SaulLM — BiFacetExplainer Prompt

```
You're a lawyer working on a case on Human Rights. You have to generate an argument
to prove that the defendant has violated a European Court of Human Rights article
using the information that you're given.

The following is the case against the plaintiff:
<caseText>

(For each node in the path, preface the node text with the following template)
The following is a <nodeType> (violated/mentioned/contained) by the previously
mentioned <nodeType>:
<nodeText>

The following is the article you want to argue that the defendant violated:
<articleText>

Using the cases mentioned as precedents, build an argument to prove that the
defendant is guilty of violating the article just mentioned.
Citing relevant precedents using the cases I mentioned, give me only the argument
in the following format: Introduction, Violation(s), Precedent(s), Conclusion.
```

---

### H.2 SaulLM — Baseline Prompt

```
You're a lawyer working on a case on Human Rights. You have to generate an argument
to prove that the defendant has violated a European Court of Human Rights article
using the information that you're given.

The following is the case against the plaintiff:
<caseText>

The following is the article you want to argue that the defendant violated:
<articleText>

Using the cases mentioned as precedents, build an argument to prove that the
defendant is guilty of violating the article just mentioned.
Citing relevant precedents using the cases I mentioned, give me only the argument
in the following format: Introduction, Violation(s), Precedent(s), Conclusion.
```

---

### H.3 SaulLM — Silver Rationale Prompt

```
You're a lawyer working on a case on Human Rights. You have to generate an argument
to prove that the defendant has violated a European Court of Human Rights article
using the information that you're given.

The following is the case against the plaintiff:
<caseText>

The following is a list of very important facts, each mentioned inside a <sr></sr>
block, of the case that you must focus on while generating an argument.
(Add each silver rationale in a separate <sr></sr> block)
<sr>{silverRationale}</sr>

The following is the article you want to argue that the defendant violated:
<articleText>

Using the cases mentioned as precedents, build an argument to prove that the
defendant is guilty of violating the article just mentioned.
Citing relevant precedents using the cases I mentioned, give me only the argument
in the following format: Introduction, Violation(s), Precedent(s), Conclusion.
```

---

### H.4 BioMedical-LLaMA — BiFacetExplainer Prompt

```
You are now a medical insurance agent working on a case on medical necessities.
You have to generate arguments to prove that the medical procedures administered
by the hospital are deemed a medical necessity using the information that you're given.

The following is the medical case of the patient:
<caseText>

(For each node in the path, preface the node text with the following template)
The following is a <nodeType> that the previous <nodeType> (required/contained):
<nodeText>

The following are the procedures you want to argue that the patient needs:
<icdText>

Build an argument to prove that the patient needed the procedure mentioned earlier.
Citing relevant parallels/precedents, give me ONLY the argument in the following
format: Introduction, Procedures, Similar Cases, Conclusion.
```

---

### H.5 BioMedical-LLaMA — Baseline Prompt

```
You are now a medical insurance agent working on a case on medical necessities.
You have to generate arguments to prove that the medical procedures administered
by the hospital are deemed a medical necessity using the information that you're given.

The following is the medical case of the patient:
<caseText>

The following are the procedures you want to argue that the patient needs:
<icdText>

Build an argument to prove that the patient needed the procedure mentioned earlier.
Citing relevant parallels/precedents, give me ONLY the argument in the following
format: Introduction, Procedures, Similar Cases, Conclusion.
```
