# Executive Summary: Smart ATS Checker
## Resume-to-Job Matching System — Academic Deep Learning Project

**Project Contributors:**
- **Nimmagadda Harsha Vardhan** (`2023BCS0158`)
- **Konda Tej Akhilesh** (`2023BCS0160`)
- **K Sai Nihal** (`2023BCD0045`)

---

### Project Overview
**Smart ATS Checker: Resume-to-Job Matching System** is an academic Deep Learning and Natural Language Processing (NLP) system designed to overcome the rigidity and inaccuracies of traditional keyword-based Applicant Tracking Systems (ATS).

### The Problem
Traditional recruitment software evaluates candidate suitability through literal string matches. This approach fails to recognize semantic synonyms (e.g., "Deep Learning" vs. "Neural Networks"), ignores experience context, provides zero candidate diagnostics, and leads to high false-negative rates in automated screening.

### The Solution: A Hybrid Deep Learning Architecture
Smart ATS Checker implements a dual-engine architecture:
1. **Contextual Semantic Matching (60%):** Utilizes `sentence-transformers/bert-base-nli-mean-tokens` to extract 768-dimensional document vectors, applying attention-weighted mean pooling and cosine similarity in latent space.
2. **Explicit Competency Extraction (40%):** Uses compiled word-boundary regular expressions across a curated 45+ technical skills taxonomy to quantify hard technical prerequisite fulfillment.

$$\text{Final Compatibility Score} = 0.60 \times \text{Semantic Score} + 0.40 \times \text{Skill Score}$$

### Key Results & Benchmarks
- **Semantic Synonym Robustness:** Achieved $87.4\%$ semantic similarity on synonymous candidate experience where traditional keyword matchers dropped to $28.5\%$.
- **Explainability:** Identifies exact "Matched Skills" (candidate strengths) and "Skills to Improve" (missing qualifications).
- **Automated Recommendation:** Evaluates candidate resumes against an entire database catalog (`jobs.csv`) and outputs ranked fit recommendations.
- **Dual Delivery:** Interactive dark-mode Streamlit dashboard and headless CLI runner.

### Technology Stack
- **Languages & Frameworks:** Python 3.10+, PyTorch, Hugging Face Transformers, NLTK, Gensim, Scikit-learn, Streamlit, `pdfplumber`.

### Repository Structure
- `app.py`: Streamlit application & CLI batch runner.
- `src/`: Deep learning models (`models.py`), parser (`resume_parser.py`), skill extractor (`skill_extractor.py`), scanner (`resume_scanner.py`), recommender (`job_recommender.py`), utilities (`utils.py`).
- `data/`: `jobs.csv`, benchmark sample resumes (`data/sample_resumes/`), sample jobs (`data/sample_jobs/`).
- `notebooks/`: 4 step-by-step Jupyter notebooks for academic research and presentation.
- `reports/`: Comprehensive Academic Project Report (`PROJECT_REPORT.md`).
- `tests/`: Automated unit and integration test suite.
