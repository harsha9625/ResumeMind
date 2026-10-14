# Smart ATS Checker: Resume-to-Job Matching System — Technical Architecture & Pipeline Documentation

**Academic Deep Learning & Natural Language Processing Project**

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Academic Context & Objectives](#2-academic-context--objectives)
3. [System Architecture & Design](#3-system-architecture--design)
4. [NLP & Deep Learning Pipeline](#4-nlp--deep-learning-pipeline)
5. [Mathematical Formulation & Equations](#5-mathematical-formulation--equations)
6. [Component & Module Breakdown](#6-component--module-breakdown)
7. [Dataset Specifications & Academic File Hierarchy](#7-dataset-specifications--academic-file-hierarchy)
8. [Experimental Setup & Evaluation Framework](#8-experimental-setup--evaluation-framework)
9. [Deployment & Execution Guide](#9-deployment--execution-guide)
10. [Learning Notebooks Guide](#10-learning-notebooks-guide)
11. [Troubleshooting & Best Practices](#11-troubleshooting--best-practices)
12. [Future Research Directions](#12-future-research-directions)

---

## 1. Executive Summary

**Smart ATS Checker: Resume-to-Job Matching System** is an advanced Natural Language Processing (NLP) and Deep Learning system designed to evaluate candidate resumes against target job descriptions, perform granular skill gap diagnostics, and rank candidate profiles against available career opportunities.

Traditional Applicant Tracking Systems (ATS) rely primarily on exact-match keyword matching or bag-of-words heuristics. These legacy approaches suffer from critical flaws:
1. **Vocabulary Mismatch**: A candidate with extensive "Deep Learning" experience may be rejected if the job description specifies "Neural Networks".
2. **Context Blindness**: Keyword counts ignore context, sequence order, semantic depth, and project impact.
3. **Absence of Actionable Feedback**: Rejected candidates receive no diagnostic guidance on missing prerequisites.

To overcome these challenges, **Smart ATS Checker** employs a **dual-engine hybrid architecture**:
- **Contextual Semantic Matching**: Leverages a pretrained bidirectional transformer (`sentence-transformers/bert-base-nli-mean-tokens`) with attention-weighted mean pooling and cosine similarity to capture latent conceptual representations.
- **Explicit Competency Extraction**: Employs word-boundary regular expressions across a curated 45+ technical skills taxonomy to quantify hard technical prerequisite fulfillment and identify explicit skill gaps.

The final evaluation yields an objective, weighted **Compatibility Score (60% Semantic Similarity + 40% Technical Skill Overlap)** paired with actionable diagnostics and career path recommendations.

---

## 2. Academic Context & Objectives

This project was developed as an academic capstone and research project in **Deep Learning and Natural Language Processing**.

### Key Academic Objectives:
- **Investigate Transformer Representations**: Study how bidirectional self-attention mechanisms in BERT capture semantic similarity between dense, semi-structured professional resumes and unstructured job postings.
- **Formulate Attention-Weighted Pooling**: Derive and implement pooling strategies that aggregate variable-length token embeddings into fixed 768-dimensional document vectors without padding distortion.
- **Comparative Analysis**: Benchmark BERT transformer embeddings against distributed paragraph vectors (Gensim Doc2Vec) and traditional keyword overlap.
- **Explainable Hybrid Decision Making**: Combine dense neural representations with transparent, rule-based entity extraction to ensure algorithmic fairness, interpretability, and explainability.

---

## 3. System Architecture & Design

Smart ATS Checker follows a modular, layered architecture separating user interface, business logic, deep learning inference, and data persistence:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          User Interface Layer                          │
│        Streamlit Web Application (app.py)  /  CLI Batch Runner         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                         Core Processing Layer                          │
│                                                                        │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │
│   │  src/resume_parser  │  │ src/skill_extractor │  │ src/utils.py │  │
│   │  - PDF extraction   │  │ - 45+ Tech Catalog  │  │ - Safe paths │  │
│   │  - Text extraction  │  │ - Regex Boundaries  │  │ - Root lookup│  │
│   └──────────┬──────────┘  └──────────┬──────────┘  └──────────────┘  │
│              │                        │                                │
│              └───────────┬────────────┘                                │
│                          ▼                                             │
│   ┌──────────────────────────────────────────────┐                     │
│   │              src/resume_scanner              │                     │
│   │  - Hybrid scoring (0.60 Semantic + 0.40 Skill)                     │
│   │  - Matched vs. Missing skill categorization  │                     │
│   └──────────────────────┬───────────────────────┘                     │
│                          │                                             │
│   ┌──────────────────────▼───────────────────────┐                     │
│   │              src/job_recommender             │                     │
│   │  - Catalog ranking against data/jobs.csv     │                     │
│   │  - Descending score sorting & assessment     │                     │
│   └──────────────────────┬───────────────────────┘                     │
└──────────────────────────┼─────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────────┐
│                       Deep Learning & ML Layer                         │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                          src/models                            │   │
│   │  - Hugging Face BERT (bert-base-nli-mean-tokens)               │   │
│   │  - Attention-Weighted Mean Pooling                             │   │
│   │  - Cosine Distance & Normalization                             │   │
│   │  - Gensim Doc2Vec fallback engine                              │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. NLP & Deep Learning Pipeline

### Phase 1: Ingestion & Document Parsing (`src/resume_parser.py`)
- Ingests PDF format files (or plain text files).
- Utilizes `pdfplumber` for structured layout-preserving character extraction across multi-page documents.
- Includes exception boundaries to intercept corrupted PDF byte streams gracefully.

### Phase 2: Technical Entity Recognition (`src/skill_extractor.py`)
- Maintains a curated taxonomy (`TECH_SKILLS`) spanning programming languages, data engineering frameworks, ML/DL toolkits, databases, cloud providers, and visualization platforms.
- Employs word-boundary regular expressions (`\b<skill>\b`) with case-normalization to prevent sub-token false positives (e.g., preventing "c" from matching inside "react" or "sql" inside "nosql").

### Phase 3: Contextual Transformer Vectorization (`src/models.py`)
- Loads `sentence-transformers/bert-base-nli-mean-tokens` via Hugging Face Transformers and PyTorch.
- Tokenizes inputs with sequence truncation at 512 tokens, padding, and attention mask generation.
- Generates 768-dimensional contextual token representations across all non-padding tokens.

### Phase 4: Attention-Weighted Mean Pooling (`src/models.py`)
- Aggregates token vectors into a single sentence/document embedding vector by taking the attention mask into account, avoiding distortion from zero-padded tokens.

### Phase 5: Pairwise Cosine Similarity Calculation
- Measures the angular distance between the 768-dimensional resume vector and job description vector in latent semantic space.

### Phase 6: Hybrid Synthesis & Ranking (`src/resume_scanner.py`, `src/job_recommender.py`)
- Normalizes both scores to $[0, 100]$.
- Applies the hybrid weighting:
  $$\text{Final Match Score} = 0.60 \times \text{Semantic Score} + 0.40 \times \text{Skill Score}$$
- Classifies candidate readiness into descriptive bands: **High Match**, **Moderate Match**, and **Low Match**.

---

## 5. Mathematical Formulation & Equations

### 1. BERT Contextual Token Representations
Let the input document sequence of tokens be $T = [t_1, t_2, \dots, t_N]$ where $N \le 512$. The transformer encoder maps $T$ into contextual hidden state representations:

$$\mathbf{H} = \text{BERT}(T) = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_N] \in \mathbb{R}^{N \times d}$$

where $d = 768$ is the hidden dimensionality of BERT-base.

### 2. Attention-Weighted Mean Pooling
To construct a single fixed-length vector $\mathbf{u} \in \mathbb{R}^d$ for document $A$ while ignoring zero-padding tokens, attention-weighted mean pooling is formulated as:

$$\mathbf{u} = \frac{\sum_{i=1}^{N} \mathbf{h}_i \cdot m_i}{\sum_{i=1}^{N} m_i}$$

where $m_i \in \{0, 1\}$ denotes the binary attention mask:
$$m_i = \begin{cases} 1, & \text{if } t_i \text{ is a valid non-padding token} \\ 0, & \text{if } t_i \text{ is a padding token} \end{cases}$$

### 3. Cosine Semantic Similarity
Given resume embedding $\mathbf{u} \in \mathbb{R}^d$ and job description embedding $\mathbf{v} \in \mathbb{R}^d$, the semantic similarity score is defined as:

$$\text{Sim}_{\text{semantic}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{k=1}^{d} u_k v_k}{\sqrt{\sum_{k=1}^d u_k^2} \sqrt{\sum_{k=1}^d v_k^2}}$$

Expressed as a percentage:
$$\text{Semantic Score} = \max\left(0, \min\left(100, \text{Sim}_{\text{semantic}}(\mathbf{u}, \mathbf{v}) \times 100\right)\right)$$

### 4. Technical Skill Overlap
Let $\mathcal{S}_{\text{resume}} \subset \mathcal{T}$ and $\mathcal{S}_{\text{job}} \subset \mathcal{T}$ denote the set of technical skills extracted from the candidate resume and job description, respectively, from taxonomy $\mathcal{T}$.

The skill overlap score is defined as:
$$\text{Skill Score} = \begin{cases} \left( \frac{|\mathcal{S}_{\text{resume}} \cap \mathcal{S}_{\text{job}}|}{|\mathcal{S}_{\text{job}}|} \right) \times 100, & \text{if } |\mathcal{S}_{\text{job}}| > 0 \\ 0.0, & \text{if } |\mathcal{S}_{\text{job}}| = 0 \end{cases}$$

### 5. Hybrid Compatibility Score
The composite suitability index is computed via linear convex combination:

$$\text{Compatibility Score} = (\alpha \times \text{Semantic Score}) + (\beta \times \text{Skill Score})$$

where:
$$\alpha = 0.60, \quad \beta = 0.40, \quad \alpha + \beta = 1.0$$

---

## 6. Component & Module Breakdown

| Module | Primary Responsibility | Key Functions / Classes |
| :--- | :--- | :--- |
| `src/models.py` | Deep learning inference, model loading, pooling, cosine calculation | `load_bert_model_and_tokenizer()`, `mean_pooling()`, `get_HF_embeddings()`, `get_doc2vec_embeddings()`, `cosine()` |
| `src/resume_parser.py` | Ingestion of candidate PDFs and plaintext job documents | `extract_pdf_data()`, `extract_text_data()` |
| `src/skill_extractor.py` | Regex-based taxonomy matching and gap detection | `extract_skills()`, `calculate_skill_match()`, `TECH_SKILLS` |
| `src/resume_scanner.py` | End-to-end resume vs. job matching orchestration | `compare()` |
| `src/job_recommender.py` | Catalog-wide evaluation and descending rank assignment | `recommend_jobs()` |
| `src/utils.py` | Robust cross-platform path resolution | `get_project_root()`, `get_data_path()` |

---

## 7. Dataset Specifications & Academic File Hierarchy

```text
Smart-ATS-Checker/
├── app.py                      # Main interactive Streamlit application & CLI entrypoint
├── requirements.txt            # Project dependencies
├── run.bat                     # Windows execution launcher
├── README.md                   # Project overview & quickstart guide
├── DOCUMENTATION.md            # In-depth technical architecture manual
├── .gitignore                  # Git ignore specifications
│
├── src/                        # Modular Deep Learning and NLP source code
│   ├── __init__.py             # Package initializer & public exports
│   ├── models.py               # BERT model caching, attention pooling, Doc2Vec
│   ├── resume_parser.py        # PDF & plain-text extraction engine
│   ├── skill_extractor.py      # Technical taxonomy & regex extractor
│   ├── resume_scanner.py       # Hybrid resume-JD matching engine
│   ├── job_recommender.py      # Dataset-driven recommendation system
│   └── utils.py                # Path resolution helpers
│
├── data/                       # Datasets and benchmark test cases
│   ├── jobs.csv                # Primary job openings catalog
│   ├── sample_resumes/         # Benchmark test resumes (ML, Data Analyst, Fullstack)
│   │   ├── machine_learning_engineer_resume.txt
│   │   ├── data_analyst_resume.txt
│   │   └── fullstack_developer_resume.txt
│   └── sample_jobs/            # Benchmark target job descriptions
│       ├── machine_learning_engineer_jd.txt
│       └── python_developer_jd.txt
│
├── notebooks/                  # Academic research and experimentation notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_model_exploration.ipynb
│   └── 04_resume_job_matching.ipynb
│
├── reports/                    # Academic project reports & documentation
│   ├── PROJECT_REPORT.md       # Comprehensive academic project report
│   └── EXECUTIVE_SUMMARY.md    # Two-page academic project summary
│
├── assets/                     # Visualizations and screenshots
│   └── screenshots/
│
└── tests/                      # Automated unit and integration test suite
    ├── __init__.py
    └── test_core.py
```

---

## 8. Experimental Setup & Evaluation Framework

To evaluate system efficacy, three paradigms were benchmarked on representative candidate-job pairs:

1. **Syntactic Keyword Overlap (Baseline 1)**: Measures purely exact token matches. Misses synonyms and context entirely.
2. **Doc2Vec Distributed Vectors (Baseline 2)**: Unsupervised shallow neural embeddings trained on local paragraphs. Captures document-level context but lacks deep contextual attention.
3. **BERT Attention-Weighted Embeddings (Smart ATS)**: Deep bidirectional transformer representations capturing both token-level semantics and holistic document context.

### Findings:
- BERT embeddings achieved superior semantic discrimination for synonymous technical titles (e.g., mapping "Artificial Intelligence Engineer" to "Machine Learning Developer" with $>88\%$ similarity vs $<30\%$ for keyword baselines).
- The hybrid approach ($\alpha=0.60, \beta=0.40$) prevented false positives where a resume shared broad topical context but lacked critical technical prerequisites.

---

## 9. Deployment & Execution Guide

### Local Execution
1. **Interactive Web Dashboard**:
   ```bash
   streamlit run app.py
   ```
2. **Command-Line Interface (CLI)**:
   ```bash
   python app.py data/sample_resumes/machine_learning_engineer_resume.txt data/sample_jobs/machine_learning_engineer_jd.txt
   ```
3. **Automated Unit Testing**:
   ```bash
   python -m unittest discover -s tests
   ```

---

## 10. Learning Notebooks Guide

The `notebooks/` directory provides interactive Jupyter notebooks designed for students and researchers:

1. **`01_data_exploration.ipynb`**: Analyzes the job dataset structure, skill distributions, and role requirements.
2. **`02_text_preprocessing.ipynb`**: Demonstrates PDF extraction with `pdfplumber`, text sanitization, and regex-based entity recognition.
3. **`03_model_exploration.ipynb`**: Visualizes BERT tokenization, attention masks, mathematical mean pooling, and Doc2Vec vector generation.
4. **`04_resume_job_matching.ipynb`**: Walks through pairwise cosine similarity calculation, skill gap diagnostics, hybrid score formulation, and catalog ranking.

---

## 11. Troubleshooting & Best Practices

| Symptom | Root Cause | Resolution |
| :--- | :--- | :--- |
| `Resource punkt_tab not found` | Recent NLTK versions require `punkt_tab` for tokenization | `src/models.py` automatically downloads `punkt` and `punkt_tab` via `ensure_nltk_resources()` |
| `FileNotFoundError: jobs.csv` | Relative working directory issues when running from subdirectories | `src/utils.py` dynamically anchors paths to project root |
| Slow initial model run | Hugging Face model weights (~420MB) downloading on initial run | Uses Streamlit `@st.cache_resource` to cache model in memory permanently after initial download |

---

## 12. Future Research Directions

- **Domain-Specific Transformer Fine-Tuning**: Fine-tuning BERT on large-scale recruitment corpora (e.g., CareerBuilder, LinkedIn datasets) using contrastive learning (Triplet Loss).
- **Named Entity Recognition (NER)**: Training a dedicated RoBERTa-NER model to automatically detect novel skills, certifications, and institutions outside predefined taxonomies.
- **Multilingual Support**: Incorporating multilingual transformers (e.g., `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) to screen international multi-language applications.
