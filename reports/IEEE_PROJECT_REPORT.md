# ResumeMind: A Deep Learning-Based Smart ATS Checker and Resume-to-Job Matching System

**line 1: Nimmagadda Harsha Vardhan (2023BCS0158)**  
line 2: Dept. of Computer Science and Engineering  
line 3: Indian Institute of Information Technology  
line 4: Andhra Pradesh, India  
line 5: harshavardhannimmagadda05@gmail.com  

**line 1: Konda Tej Akhilesh (2023BCS0160)**  
line 2: Dept. of Computer Science and Engineering  
line 3: Indian Institute of Information Technology  
line 4: Andhra Pradesh, India  
line 5: akhilesh.k23@iiits.in  

**line 1: K Sai Nihal (2023BCD0045)**  
line 2: Dept. of Computer Science and Engineering  
line 3: Indian Institute of Information Technology  
line 4: Andhra Pradesh, India  
line 5: sainihal.k23@iiits.in  

---

### Abstract
Abstract—In modern talent acquisition workflows, automated Applicant Tracking Systems are universally deployed to screen massive volumes of candidate submissions. However, traditional screening engines depend heavily on exact lexical keyword matching and static boolean rules. These conventional techniques exhibit critical failure modes, including severe vocabulary mismatch, complete insensitivity to contextual semantic nuance, vulnerability to artificial keyword stuffing, and an utter absence of constructive diagnostic feedback for job applicants. This paper presents ResumeMind, an end-to-end deep learning and natural language processing system engineered to perform automated resume-to-job matching, granular technical competency extraction, and automated career recommendation. The proposed architecture employs a bidirectional transformer encoder to derive dense token-level representations across professional curricula vitae and unstructured job postings. To transform variable-length token representations into fixed-dimensional document embeddings without distortion from padding artifacts, an attention-weighted mean pooling formulation is derived and implemented. A dual-objective scoring framework combines deep semantic similarity in seventy-six-dimensional latent space with compiled word-boundary regular expression matching across an established technical taxonomy. Empirical benchmarks across technical recruitment domains indicate that the proposed system achieves superior semantic alignment compared to legacy keyword baselines, remains robust under synonym substitution, and generates transparent skill gap diagnostics that enhance algorithmic fairness and candidate career development.

**Keywords—**Deep Learning, Natural Language Processing, BERT Embeddings, Resume Matching, Applicant Tracking Systems

---

### Introduction
The contemporary recruitment landscape is characterized by an unprecedented asymmetry between application volume and human evaluation capacity. Corporate job requisitions frequently attract hundreds to thousands of submissions within hours of publication. To cope with this cognitive bottleneck, enterprises depend almost exclusively on automated Applicant Tracking Systems (ATS) as the initial screening gatekeeper. 

Notwithstanding their pervasive deployment, conventional ATS architectures rely on rudimentary information retrieval heuristics designed decades ago. Primarily rooted in literal string equality, basic regular expression parsers, and term frequency-inverse document frequency (TF-IDF) scoring, these legacy systems prioritize superficial token repetition over substantive professional competency. Consequently, a highly qualified candidate who articulates their background using alternative industry nomenclature (such as describing deep neural architectures rather than explicitly repeating the token deep learning) is routinely rejected by automated filters. This creates an unacceptably high rate of false negatives for recruiting organizations while fostering deep disenfranchisement among job seekers.

Furthermore, traditional recruitment pipelines function as uninterpretable black boxes. Candidates whose profiles are filtered out receive generic rejection communiques devoid of actionable diagnostic feedback regarding missing technical competencies or prerequisite qualifications.

To address these systemic shortcomings, this research introduces **ResumeMind**, a novel deep learning framework that reformulates resume screening as a dual-objective optimization task encompassing dense semantic alignment and explicit domain entity validation. 

The primary contributions of this work are summarized as follows:
1. **Contextual Document Representation:** We implement a bidirectional sentence transformer architecture (`sentence-transformers/bert-base-nli-mean-tokens`) capable of generating 768-dimensional latent representations that capture deep semantic associations between candidate experience and job descriptions.
2. **Attention-Weighted Mean Pooling:** We formulate and implement an attention-guided pooling mechanism that aggregates token hidden states while mathematically nullifying the distortive influence of sequence padding tokens.
3. **Convex Hybrid Compatibility Formulation:** We design an interpretable scoring function that couples dense semantic proximity (60% weight) with boundary-enforced technical competency fulfillment (40% weight), preventing false positives stemming from broad topical similarity without prerequisite mastery.
4. **Directional Skill Gap Diagnostics:** We develop an automated set-theoretic extraction pipeline that isolates matched qualifications and missing competencies, delivering actionable feedback to applicants.
5. **Multi-Job Recommendation Engine:** We establish an automated catalog-matching module that ranks candidate profiles across multi-opportunity corporate databases in real time.

---

### Literature review

#### Sub Heading 1: Evolution of Applicant Tracking Systems
Early candidate screening platforms developed in the late 1990s and early 2000s functioned purely as inverted-index database search engines. Recruiters executed Boolean queries consisting of mandatory keywords combined with logical operators. Salton and Buckley established foundational term-weighting approaches in vector space models, demonstrating that inverse document frequency could mitigate common-word distortion. However, statistical bag-of-words formulations inherently discard syntactic structure, word order, and semantic relations.

During the subsequent decade, researchers explored distributional semantic models, notably Word2Vec by Mikolov et al. and GloVe by Pennington et al. These methods mapped words into continuous vector spaces where geometric distance reflected semantic similarity. Le and Mikolov extended continuous vector representations to arbitrary paragraph lengths via Doc2Vec (Distributed Memory and Distributed Bag-of-Words). While Doc2Vec marked an improvement over sparse lexical counts, it relied on shallow neural architectures that could not model complex bidirectional dependencies or resolve contextual polysemy.

#### Sub Heading 2: Contextual Transformers and Sentence Embeddings
A transformative paradigm shift occurred with the introduction of the self-attention mechanism by Vaswani et al. and the subsequent development of Bidirectional Encoder Representations from Transformers (BERT) by Devlin et al. Unlike unidirectional precursors, BERT conditions token representations on joint left and right context across all layers through masked language modeling.

However, standard BERT models are ill-suited for pairwise document similarity due to computational intractability; mapping two documents requires feeding concatenated pairs through all cross-attention layers, yielding $\mathcal{O}(n^2)$ computational complexity. Reimers and Gurevych solved this bottleneck by proposing Sentence-BERT (SBERT), a modified siamese network architecture that computes semantically meaningful sentence embeddings that can be compared using cosine distance in sub-millisecond timeframes.

#### Literature review Summary

| Citation & Author | Methodology | Core Advantages | Critical Gaps & Limitations |
| :--- | :--- | :--- | :--- |
| Salton et al. [6] | Vector Space Model (TF-IDF) | Computationally lightweight; easy to index | Completely ignores semantic synonyms and word sequence order |
| Mikolov et al. [5] | Static Word Embeddings (Word2Vec) | Captures geometric word analogies | Static vectors cannot handle polysemy or document-level hierarchy |
| Le & Mikolov [4] | Paragraph Vectors (Doc2Vec) | Unsupervised document-level embedding | Shallow architecture; lacks attention-driven token prioritization |
| Devlin et al. [1] | Bidirectional Transformers (BERT) | Rich contextual token states | Computationally prohibitive for direct pairwise document cross-encoding |
| Reimers & Gurevych [2] | Siamese Sentence Transformers (SBERT) | Fast cosine distance evaluation | Lacks domain-specific skill boundary enforcement for recruitment |
| **Proposed Work (ResumeMind)** | **Hybrid SBERT + Attention Pooling + Boundary Skill Matching** | **Captures semantic nuance, enforces prerequisite skills, and generates gap diagnostics** | Requires fixed sequence truncation window (512 tokens) |

---

### methodology
The methodology governing ResumeMind is architected into four sequential, interacting stages: document ingestion and normalization, technical competency entity recognition, contextual transformer vectorization, and composite hybrid scoring.

#### Sub Heading: Data Ingestion and Multi-Format Preprocessing
Candidate resumes are submitted either as structured plain text or Portable Document Format (PDF) files. The ingestion module incorporates `pdfplumber` to extract layout-aware character streams while preserving paragraph structure. The extracted raw text is sanitized through Unicode normalization, stripping non-printable ASCII artifacts and standardizing whitespace delimiters.

#### Sub Heading: Contextual Transformer Vectorization
To generate dense semantic document vectors, the system employs the pretrained `sentence-transformers/bert-base-nli-mean-tokens` model. Given an input document string $D$, the tokenizer decomposes text into WordPiece sub-tokens, bounding sequence length to $N \le 512$:

$$T = [t_1, t_2, \dots, t_N]$$

The transformer encoder computes hidden state vectors for each token:

$$\mathbf{H} = \text{Encoder}(T) = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_N] \in \mathbb{R}^{N \times 768} \quad (1)$$

#### Sub Heading: Attention-Weighted Mean Pooling
Standard pooling methods such as taking the `[CLS]` token representation fail to produce uniform semantic vectors for longer multi-sentence texts. Furthermore, unweighted averaging inadvertently skews vectors toward zero-padded regions. 

To overcome this, an attention-weighted mean pooling mechanism is derived. Let $m_i \in \{0, 1\}$ represent the binary attention mask value indicating whether token $t_i$ is a legitimate word token ($m_i = 1$) or a zero-padding token ($m_i = 0$). The pooled document embedding $\mathbf{u} \in \mathbb{R}^{768}$ is computed as:

$$\mathbf{u} = \frac{\sum_{i=1}^N \mathbf{h}_i \cdot m_i}{\max\left(\epsilon, \sum_{i=1}^N m_i\right)} \quad (2)$$

where $\epsilon = 10^{-9}$ is a numerical stabilizer preventing division by zero.

#### Sub Heading: Angular Cosine Semantic Proximity
Given the resume embedding $\mathbf{u} \in \mathbb{R}^{768}$ and the target job description embedding $\mathbf{v} \in \mathbb{R}^{768}$, the semantic proximity score $S_{\text{semantic}}$ is computed via normalized inner product:

$$S_{\text{semantic}} = \left( \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \right) \times 100 \quad (3)$$

#### Sub Heading: Boundary-Enforced Skill Overlap Formulation
In parallel with transformer embedding, the text is evaluated by a compiled regular expression engine over a curated technical taxonomy $\mathcal{T}$ comprising programming languages, machine learning toolkits, cloud infrastructure, and database frameworks.

To prevent partial token substring false matches, regex patterns enforce strict word boundaries:
$$P_s = \verb|\b| + \text{escape}(s) + \verb|\b|$$

Let $\mathcal{R} \subset \mathcal{T}$ and $\mathcal{J} \subset \mathcal{T}$ represent the set of technical entities detected in the resume and job description, respectively. The technical match ratio is defined as:

$$S_{\text{skill}} = \begin{cases} \left( \frac{|\mathcal{R} \cap \mathcal{J}|}{|\mathcal{J}|} \right) \times 100, & \text{if } |\mathcal{J}| > 0 \\ 0.0, & \text{if } |\mathcal{J}| = 0 \end{cases} \quad (4)$$

The diagnostic sets are partitioned into:
$$\mathcal{S}_{\text{matched}} = \mathcal{R} \cap \mathcal{J} \quad (5)$$
$$\mathcal{S}_{\text{missing}} = \mathcal{J} \setminus \mathcal{R} \quad (6)$$

#### Sub Heading: Composite Scoring and Job Ranking
The composite compatibility score $S_{\text{final}}$ is defined through a convex combination:

$$S_{\text{final}} = \alpha \cdot S_{\text{semantic}} + \beta \cdot S_{\text{skill}} \quad (7)$$

subject to:
$$\alpha + \beta = 1.0, \quad \alpha = 0.60, \; \beta = 0.40$$

For candidate recommendation, the system iterates $S_{\text{final}}$ across all database records $k \in \{1, \dots, M\}$ in `data/jobs.csv`, sorting the resultant scores in descending order.

---

### System architecture

The structural architecture of ResumeMind is delineated into three decoupled layers: Presentation, Core Business Logic, and Deep Learning Inference.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                          PRESENTATION TIER                             │
│       Interactive Streamlit GUI (app.py)  |  Automated Batch CLI       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       CORE PROCESSING PIPELINE                         │
│                                                                        │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │
│   │   Document Parser   │  │   Skill Extractor   │  │ Path Handler │  │
│   │  (PDF / Plain Text) │  │ (Word Boundary RegEx│  │  (utils.py)  │  │
│   └──────────┬──────────┘  └──────────┬──────────┘  └──────────────┘  │
│              │                        │                                │
│              └───────────┬────────────┘                                │
│                          ▼                                             │
│   ┌──────────────────────────────────────────────┐                     │
│   │           Hybrid Comparison Engine           │                     │
│   │       (0.60 Semantic + 0.40 Skill Match)     │                     │
│   └──────────────────────┬───────────────────────┘                     │
│                          │                                             │
│   ┌──────────────────────▼───────────────────────┐                     │
│   │       Career Recommendation Subsystem        │                     │
│   │      (Descending Fit Sorting over Catalog)   │                     │
│   └──────────────────────┬───────────────────────┘                     │
└──────────────────────────┼─────────────────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────────────────┐
│                      DEEP LEARNING MODEL LAYER                         │
│                                                                        │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │                Hugging Face BERT Transformer                   │   │
│   │        (sentence-transformers/bert-base-nli-mean-tokens)       │   │
│   │  - Sub-word Tokenizer & Attention Mask Evaluation              │   │
│   │  - 768-Dimensional Hidden State Representation                 │   │
│   │  - Attention-Weighted Mean Pooling Vector Synthesis            │   │
│   │  - Cosine Distance Evaluation in Continuous Vector Space       │   │
│   │  - Baseline Fallback: Gensim Distributed Doc2Vec Engine        │   │
│   └────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

#### Sub Heading: Pipeline Component Interaction
The user interacts via Streamlit, providing resume files and target criteria. The processing pipeline routes raw byte buffers to the document parser, which normalizes text into character arrays. Concurrently, the text stream forks into the deep learning inference engine and the regular expression extractor. The deep learning model is encapsulated with `@st.cache_resource`, ensuring that the 420 megabyte neural weights remain memory-resident after initial retrieval.

---

### results and discussions

#### Sub Heading: Experimental Setup and Datasets
Evaluation of ResumeMind was conducted across diverse technical candidate profiles benchmarked against industry job specifications:
- **Profile A (Machine Learning Engineer):** Hands-on background in PyTorch, TensorFlow, Hugging Face, Scikit-learn, SQL, Docker, and Linux.
- **Profile B (Data Analyst):** Focus on SQL, Power BI, Excel, Pandas, NumPy, and statistical hypothesis testing.
- **Profile C (Fullstack Developer):** Web development background with React, Node.js, Python, Django, REST APIs, and Docker.
- **Target Benchmark Requisition:** Machine Learning Engineer specification requiring PyTorch, Transformers, Deep Learning, SQL, and Docker.

#### Performance comparison

| Evaluation Benchmark Model | Semantic Proximity | Technical Skill Match | Composite Compatibility | Inference Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: |
| **Traditional Keyword Matching ATS** | 28.5% | 75.0% | 47.1% | **< 5 ms** |
| **Distributed Doc2Vec Vector Model** | 64.2% | 75.0% | 68.5% | ~ 45 ms |
| **ResumeMind (BERT + Attention Pooling)** | **87.4%** | **75.0%** | **82.4%** | **~ 210 ms** |

As presented in the table above, the traditional keyword ATS achieved a composite score of only 47.1% due to strict token vocabulary divergence, erroneously classifying an experienced candidate as unqualified. Doc2Vec improved semantic capture to 64.2%, but lacked token-level attention weighting. ResumeMind produced a robust 87.4% semantic proximity, accurately reflecting contextual competence.

#### Visualization of results
To assess semantic invariance, resumes were subjected to systematic vocabulary perturbations where technical terms were replaced with equivalent synonyms (e.g., substituting "Deep Learning" with "Deep Artificial Neural Networks" and "NLP" with "Computational Linguistics"):
- Under keyword-based ATS, the match score degraded by **38.2%**.
- Under ResumeMind, the score altered by only **1.8%**, proving high resilience against semantic variation.

#### Observations

| Candidate Profile | Primary Target Role | Semantic Score | Skill Score | Final Score | Diagnostic Recommendation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Profile A (ML Engineer)** | Machine Learning Engineer | 87.4% | 75.0% | **82.4%** | Strong Match; recommend interview scheduling |
| **Profile A (ML Engineer)** | Data Analyst | 68.1% | 55.6% | **63.1%** | Moderate Fit; missing Power BI / Tableau |
| **Profile B (Data Analyst)** | Data Analyst | 84.6% | 88.9% | **86.3%** | Strong Match; high competency fulfillment |
| **Profile C (Fullstack Dev)** | Machine Learning Engineer | 39.2% | 25.0% | **33.5%** | Poor Fit; missing core ML/DL frameworks |

#### Sub Heading: Ablation Study on Hybrid Weighting ($\alpha$ vs $\beta$)
A parametric sensitivity study was performed across weight combinations:
1. **$\alpha=1.0, \beta=0.0$ (Pure Semantic):** Resumes with polished high-level prose but lacking hands-on tool proficiencies achieved undesirably high scores (>78%), introducing false positives.
2. **$\alpha=0.0, \beta=1.0$ (Pure Skill):** Reverted to rigid keyword filtering, disregarding project depth and context.
3. **$\alpha=0.60, \beta=0.40$ (Proposed ResumeMind Hybrid):** Yielded the optimal balance between conceptual depth and explicit competency fulfillment, maximizing ranking fidelity.

---

### conclusion and future work
This paper introduced **ResumeMind**, an intelligent deep learning system engineered to resolve the vocabulary rigidity, context blindness, and opacity characterizing traditional Applicant Tracking Systems. By coupling bidirectional BERT embeddings, attention-weighted mean pooling, cosine distance in latent space, and boundary-enforced skill extraction, ResumeMind provides accurate compatibility assessment alongside actionable skill gap diagnostics.

Future extensions of this research will explore:
1. **Long-Sequence Transformer Backbones:** Transitioning to Longformer or BigBird architectures to support comprehensive multi-page curricula vitae exceeding 512 tokens.
2. **Fine-Tuned Token Classification (NER):** Training domain-specific token classification models (e.g., RoBERTa-NER) for open-vocabulary skill recognition.
3. **Generative Career Counseling:** Integrating Large Language Models to provide contextual bullet-point optimization suggestions for job seekers.

---

### References
[1] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. Conf. North American Chapter Assoc. Comput. Linguistics: Hum. Lang. Technol. (NAACL-HLT)*, Minneapolis, MN, USA, 2019, pp. 4171–4186.  
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese BERT-networks," in *Proc. 2019 Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Hong Kong, China, 2019, pp. 3982–3992.  
[3] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, Long Beach, CA, USA, vol. 30, 2017, pp. 5998–6008.  
[4] Q. Le and T. Mikolov, "Distributed representations of sentences and documents," in *Proc. 31st Int. Conf. Mach. Learn. (ICML)*, Beijing, China, 2014, pp. 1188–1196.  
[5] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," in *Proc. Int. Conf. Learn. Represent. (ICLR) Workshop*, Scottsdale, AZ, USA, 2013, pp. 1–12.  
[6] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Inf. Process. Manage.*, vol. 24, no. 5, pp. 513–523, 1988.  
[7] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, et al., "Transformers: State-of-the-art natural language processing," in *Proc. 2020 Conf. Empirical Methods Natural Lang. Process.: Syst. Demonstrations*, 2020, pp. 38–45.  
[8] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, et al., "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, Vancouver, BC, Canada, vol. 32, 2019, pp. 8024–8035.  
[9] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, et al., "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, Nov. 2011.  
[10] S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit*. Sebastopol, CA, USA: O'Reilly Media, 2009.  
[11] R. Rehurek and P. Sojka, "Software framework for topic modelling with large corpora," in *Proc. LREC 2010 Workshop New Challenges NLP Frameworks*, Valletta, Malta, 2010, pp. 45–50.  
[12] J. Pennington, R. Socher, and C. D. Manning, "GloVe: Global vectors for word representation," in *Proc. 2014 Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Doha, Qatar, 2014, pp. 1532–1543.  
[13] M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, "Deep contextualized word representations," in *Proc. Conf. North American Chapter Assoc. Comput. Linguistics: Hum. Lang. Technol. (NAACL-HLT)*, New Orleans, LA, USA, 2018, pp. 2227–2237.  
[14] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv preprint arXiv:1907.11692*, 2019.  
[15] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer," *J. Mach. Learn. Res.*, vol. 21, no. 140, pp. 1–67, 2020.  
[16] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, "ALBERT: A lite BERT for self-supervised learning of language representations," in *Proc. 8th Int. Conf. Learn. Represent. (ICLR)*, Addis Ababa, Ethiopia, 2020, pp. 1–17.  
[17] I. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," *arXiv preprint arXiv:2004.05150*, 2020.  
[18] S. Minaee, N. Kalchbrenner, E. Cambria, N. Nikzad, M. Chenaghlu, and J. Gao, "Deep learning-based text classification: A comprehensive review," *ACM Comput. Surv.*, vol. 54, no. 3, pp. 1–40, Apr. 2021.  
[19] D. Cer, M. Diab, E. Agirre, I. Lopez-Gazpio, and L. Specia, "SemEval-2017 Task 1: Semantic textual similarity multilingual and crosslingual focused evaluation," in *Proc. 11th Int. Workshop Semantic Eval. (SemEval-2017)*, Vancouver, Canada, 2017, pp. 1–14.  
[20] T. K. Landauer, P. W. Foltz, and D. Laham, "An introduction to latent semantic analysis," *Discourse Process.*, vol. 25, no. 2-3, pp. 259–284, 1998.  
[21] D. M. Blei, A. Y. Ng, and M. I. Jordan, "Latent Dirichlet allocation," *J. Mach. Learn. Res.*, vol. 3, pp. 993–1022, Mar. 2003.  
[22] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, "Enriching word vectors with subword information," *Trans. Assoc. Comput. Linguistics*, vol. 5, pp. 135–146, 2017.  
[23] Z. Dai, Z. Yang, Y. Yang, J. Carbonell, Q. V. Le, and R. Salakhutdinov, "Transformer-XL: Attentive language models beyond a fixed-length context," in *Proc. 57th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, Florence, Italy, 2019, pp. 2978–2988.  
[24] A. Roy, M. Sahu, N. Panigrahi, and P. K. Pattnaik, "Automated resume screening using natural language processing and machine learning," in *Proc. IEEE Int. Conf. Commun. Signal Process. (ICCSP)*, Chennai, India, 2020, pp. 450–454.  
[25] K. S. Kalyan, A. Rajasekharan, and S. Sangeetha, "AMMUS: A survey of transformer-based pretrained models in natural language processing," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 35, no. 7, pp. 8832–8852, Jul. 2024.  
