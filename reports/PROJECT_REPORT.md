# ResumeMind — Smart ATS Checker
# ResumeMind: A Deep Learning-Based Smart ATS Checker and Resume-to-Job Matching System

**Nimmagadda Harsha Vardhan (2023BCS0158)**  
Dept. of Computer Science and Engineering  
Indian Institute of Information Technology  
Kottayam, Kerala, India  
`nimmaga23bcs158@iiitkottayam.ac.in`  
GitHub: [https://github.com/harsha9625](https://github.com/harsha9625)  

**Konda Tej Akhilesh (2023BCS0160)**  
Dept. of Computer Science and Engineering  
Indian Institute of Information Technology  
Kottayam, Kerala, India  
`konda23bcs160@iiitkottayam.ac.in`  
GitHub: [https://github.com/tejakhileshkonda-ops](https://github.com/tejakhileshkonda-ops)  

**K Sai Nihal (2023BCD0045)**  
Dept. of Computer Science and Engineering  
Indian Institute of Information Technology  
Kottayam, Kerala, India  
`kandima23bcd45@iiitkottayam.ac.in`  

---

### Abstract
Abstract— The increasing volume of online job applications has created a significant challenge for organizations seeking to identify suitable candidates efficiently. Applicant Tracking Systems (ATS) are widely used to automate preliminary resume screening; however, conventional approaches that depend primarily on keyword frequency and exact textual matches may fail to recognize relevant qualifications expressed through alternative terminology or different linguistic structures. To address these limitations, this paper proposes ResumeMind, a deep learning-based resume analysis and job recommendation framework that applies Natural Language Processing (NLP) techniques to estimate the compatibility between candidate profiles and job requirements. The proposed system processes resumes and job descriptions through a text extraction and preprocessing pipeline before generating contextual representations using a pretrained BERT-based sentence-transformer model. An attention-mask-aware pooling strategy is used to obtain fixed-dimensional document embeddings while minimizing the influence of padding tokens. Semantic similarity between candidate profiles and job descriptions is then evaluated using cosine similarity and integrated with explicit technical skill matching to produce an interpretable compatibility score. In addition to ranking relevant job opportunities, the system identifies skills shared by the candidate and employer requirements, highlights potentially missing competencies, and provides structured feedback to support resume improvement. The framework is intended to overcome the limitations of purely lexical matching by considering contextual relationships between textual expressions. Its effectiveness can be evaluated against conventional keyword-based and document-embedding baselines using ranking metrics, skill extraction measures, and inference time. The proposed approach offers a practical foundation for intelligent recruitment assistance while retaining transparent matching evidence for users. Further validation on representative, manually reviewed resume–job datasets is necessary to establish its generalization ability, reliability, and suitability for real-world recruitment environments.

---

### I. INTRODUCTION
Recruitment platforms frequently receive a large number of applications for individual job openings, making manual resume evaluation time-consuming and difficult to manage. Applicant Tracking Systems (ATS) help recruiters organize applications and identify potential candidates based on job requirements. However, traditional screening methods often depend on exact keyword matching and predefined rules, which may overlook qualified candidates whose experience is described using synonyms, alternative terminology, or different sentence structures. These limitations can also cause systems to prioritize frequently repeated keywords rather than the actual relevance of a candidate's skills and experience.

To overcome these challenges, ResumeMind proposes a deep learning-based approach that combines Natural Language Processing (NLP) with transformer-based language representations. The system processes resumes and job descriptions to understand their textual content and identify relationships between candidate qualifications and employer requirements. A pretrained BERT-based sentence-transformer generates contextual embeddings, which are compared using cosine similarity to estimate semantic relevance. This approach enables the system to identify potentially related professional experiences even when the wording differs between the resume and the job description.

In addition to semantic analysis, ResumeMind incorporates a technical skill extraction component that identifies relevant competencies from both documents. The extracted skills are compared to determine matching and potentially missing requirements. These results are combined with semantic similarity through a weighted scoring mechanism to produce an interpretable compatibility score. The system can also rank multiple job opportunities according to their estimated relevance and provide skill-gap information to help candidates understand where their profiles may require improvement.

The main contributions of this work include contextual document representation using transformer embeddings, attention-mask-aware mean pooling, hybrid semantic and skill-based scoring, and automated skill-gap analysis. The proposed framework aims to provide a more informative alternative to purely keyword-based resume screening. Its effectiveness can be assessed by comparing it with conventional matching techniques using appropriate evaluation metrics and representative resume–job datasets. The resulting system is intended to support recruitment analysis and career planning rather than replace human judgment in hiring decisions.

---

### II. LITERATURE REVIEW
Traditional information retrieval techniques, such as Term Frequency–Inverse Document Frequency (TF-IDF), measure document relevance using the frequency and importance of individual words. Although these methods are computationally efficient, they have limited ability to recognize semantic relationships between different expressions. Word2Vec and GloVe improved text representation by mapping words into dense vector spaces, while Doc2Vec extended distributed representations to entire documents. However, these approaches cannot dynamically capture contextual meaning in the same way as modern transformer-based language models. Consequently, resumes describing relevant skills using alternative terminology may receive lower matching scores when compared with job descriptions that use different vocabulary.

The Transformer architecture introduced self-attention mechanisms that enable models to learn relationships between words based on their surrounding context. BERT uses bidirectional contextual representations, while Sentence-BERT generates fixed-dimensional embeddings that can be efficiently compared using cosine similarity. These capabilities make transformer-based models suitable for identifying semantic relationships between resumes and job requirements. ResumeMind builds on this approach by combining contextual document embeddings with explicit technical skill matching. The semantic component estimates the overall relevance between documents, while a predefined skill taxonomy identifies matching and potentially missing competencies. This hybrid strategy aims to reduce the limitations of purely keyword-based screening and provide more interpretable compatibility estimates. Its effectiveness must be evaluated against traditional methods using representative datasets and appropriate performance metrics.

#### TABLE I. SUMMARY OF RELATED METHODS

| Method | Strength | Limitation |
| :--- | :--- | :--- |
| **TF-IDF / vector-space model** | Lightweight and easy to index | Weak synonym and context handling |
| **Word2Vec / GloVe** | Dense word-level representations | Static representations; limited document context |
| **Doc2Vec** | Document-level vectors | Less contextual than transformer encoders |
| **BERT** | Context-sensitive token representations | Direct pairwise cross-encoding can be costly |
| **Sentence-BERT** | Efficient semantic comparison | May need domain adaptation for recruitment |
| **ResumeMind hybrid** | **Semantic similarity plus explicit skill overlap** | Taxonomy coverage and sequence length remain limitations |

---

### III. METHODOLOGY

#### A. Document ingestion and preprocessing
The system accepts resume text or PDF documents and a target job description. Text is extracted using a PDF parsing library, normalized using Unicode normalization, and cleaned to standardize whitespace and remove non-printable artifacts. The cleaned text is passed to the tokenizer, which converts it into WordPiece tokens and applies a maximum sequence length of 512 tokens for the encoder configuration described in the supplied draft.

#### B. Transformer representations and pooling
The pretrained `sentence-transformers/bert-base-nli-mean-tokens` encoder produces a contextual hidden-state vector for each token. Let $H = [h_1, h_2, \dots, h_n]$ be the hidden states and $m_i$ be the attention-mask value for token $i$. The document vector is calculated by masked mean pooling:

$$u = \frac{\sum_i h_i m_i}{\max(\epsilon, \sum_i m_i)} \quad (1)$$

where $\epsilon$ is a small numerical constant that prevents division by zero. The mask excludes padding tokens from the pooled representation. The same procedure is applied to the job description to obtain vector $v$.

#### C. Semantic similarity
The semantic score is the cosine similarity between resume vector $u$ and job vector $v$, scaled to a percentage:

$$S_{\text{semantic}} = 100 \times \frac{u \cdot v}{\|u\|_2 \|v\|_2} \quad (2)$$

#### D. Technical skill extraction
A curated taxonomy $T$ contains technical terms such as programming languages, ML frameworks, cloud tools, and databases. Compiled regular expressions use word boundaries and escaped terms to reduce partial-substring matches. Let $R$ be the skills detected in the resume and $J$ the skills identified in the job description. Skill coverage is:

$$S_{\text{skill}} = 100 \times \frac{|R \cap J|}{|J|}, \quad \text{when } |J| > 0; \quad \text{otherwise } 0. \quad (3)$$

The matched-skill set is $R \cap J$, and the missing-skill set is $J \setminus R$. This component is transparent but limited by the completeness of the taxonomy and does not by itself verify proficiency.

#### E. Hybrid scoring and ranking
The final compatibility score is a convex combination of semantic proximity and explicit skill coverage:

$$S_{\text{final}} = 0.60 \times S_{\text{semantic}} + 0.40 \times S_{\text{skill}} \quad (4)$$

The system computes this score for each job in the available catalogue and sorts jobs in descending order. Scores should be treated as estimated compatibility measures, not as predictions of an employer's actual hiring decision.

#### F. System architecture

```text
┌────────────────────────────────────────────────────────┐
│                    INPUT DOCUMENTS                     │
│           Resume (PDF / text)  +  Job description      │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                     PREPROCESSING                      │
│     Text extraction • Unicode normalization • Tokenization│
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
┌─────────────▼───────────────┐ ┌─────────▼──────────────┐
│     NLP SKILL EXTRACTION    │ │   TRANSFORMER ENCODER  │
│  Boundary-aware taxonomy    │ │ BERT token states +    │
│                             │ │ masked mean pooling    │
└─────────────┬───────────────┘ └─────────┬──────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│                 HYBRID MATCHING ENGINE                 │
│         0.60 * semantic similarity + 0.40 * skill overlap│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                         OUTPUT                         │
│   Match score • matched/missing skills • ranked jobs   │
└────────────────────────────────────────────────────────┘
```
*Fig. 1. ResumeMind system architecture, showing document ingestion, parallel skill extraction and transformer encoding, hybrid scoring, and outputs.*

---

### IV. RESULTS AND DISCUSSION

#### A. Evaluation setup
The supplied draft describes three illustrative candidate profiles: a machine-learning engineer, a data analyst, and a full-stack developer. The target requisition is a machine-learning engineer role requiring PyTorch, Transformers, deep learning, SQL, and Docker. The values below are transcribed from the supplied draft and are included to organize the proposed results section. They are not independently reproduced or verified in this document.

#### TABLE II. PERFORMANCE VALUES REPORTED IN THE SUPPLIED DRAFT

| Model | Semantic proximity | Skill match | Composite score | CPU latency |
| :--- | :---: | :---: | :---: | :---: |
| **Traditional keyword ATS** | 28.5% | 75.0% | 47.1% | < 5 ms |
| **Doc2Vec** | 64.2% | 75.0% | 68.5% | ≈ 45 ms |
| **ResumeMind (BERT + pooling)** | **87.4%** | **75.0%** | **82.4%** | **≈ 210 ms** |

*Fig. 2. Comparison of reported semantic proximity and composite compatibility values. Validate against a reproducible test set before submission.*

#### B. Candidate-to-job observations

#### TABLE III. CANDIDATE-ROLE SCORES REPORTED IN THE SUPPLIED DRAFT

| Candidate profile | Target role | Semantic | Skill | Final | Draft interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Profile A: ML Engineer** | ML Engineer | 87.4% | 75.0% | 82.4% | Strong match |
| **Profile A: ML Engineer** | Data Analyst | 68.1% | 55.6% | 63.1% | Moderate fit; check BI skills |
| **Profile B: Data Analyst** | Data Analyst | 84.6% | 88.9% | 86.3% | Strong competency overlap |
| **Profile C: Full-stack Developer** | ML Engineer | 39.2% | 25.0% | 33.5% | Low overlap with ML requirements |

#### C. Vocabulary variation
The robustness of ResumeMind can be evaluated by examining how synonym substitutions affect resume–job compatibility scores. Traditional keyword-based systems may produce lower scores when candidates describe the same skills using different terms, whereas transformer-based models can capture contextual relationships between words and phrases. The supplied draft reports a score change of 38.2% for the keyword-based ATS and 1.8% for ResumeMind, suggesting improved resistance to terminology variations. However, these results require validation through controlled experiments using a clearly defined set of synonym substitutions and representative test examples. Comparing score changes and job-ranking consistency under identical conditions would help determine whether ResumeMind maintains reliable matching performance when the wording of resumes or job descriptions changes without altering their intended meaning.

*Fig. 3. Reported score change under synonym substitution (values from the supplied draft).*

#### D. Discussion and limitations
The hybrid design of ResumeMind combines contextual semantic similarity with explicit technical skill coverage. A purely semantic approach may overrate generally relevant text, while a purely keyword-based approach may miss synonyms and contextual relationships. The selected weighting should therefore be optimized using labeled validation data. Other limitations include the 512-token sequence limit, incomplete skill taxonomies, variations in resume formatting, and potential biases in pretrained models. A practical system should provide evidence for detected and missing skills and avoid making hiring decisions autonomously.

Before claiming model superiority, ResumeMind should be evaluated using documented datasets, consistent baselines, fixed data splits, and repeated experiments. Suitable metrics, such as precision, recall, F1-score, and ranking measures, can help assess matching performance. Robustness testing should examine the effects of terminology and formatting changes. Latency comparisons should also specify hardware, batch size, model-loading time, and warm-up procedures to ensure reproducible and reliable results.

---

### V. CONCLUSION AND FUTURE WORK
ResumeMind proposes a hybrid deep learning framework for resume-to-job matching by integrating BERT-based contextual document embeddings with explicit technical skill matching. The system aims to overcome the limitations of traditional keyword-based screening by identifying semantic relationships between resumes and job descriptions while highlighting relevant and missing skills. It generates compatibility scores and ranked job recommendations to help candidates identify suitable employment opportunities and understand areas where their skills may not meet job requirements. Future work will focus on fine-tuning token-classification models for more flexible skill extraction, exploring long-document transformer architectures to process lengthy resumes, and evaluating performance using larger, annotated resume–job datasets. Additional improvements may include score calibration to make compatibility estimates more interpretable and reliable, along with carefully constrained career-writing assistance to help candidates improve resume clarity and relevance without introducing inaccurate information.

---

### REFERENCES
[1] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.  
[2] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using siamese BERT-networks," in *Proc. EMNLP-IJCNLP*, 2019, pp. 3982–3992.  
[3] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.  
[4] Q. Le and T. Mikolov, "Distributed representations of sentences and documents," in *Proc. ICML*, 2014, pp. 1188–1196.  
[5] T. Mikolov et al., "Efficient estimation of word representations in vector space," in *Proc. ICLR Workshop*, 2013.  
[6] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, vol. 24, no. 5, pp. 513–523, 1988.  
[7] T. Wolf et al., "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38–45.  
[8] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems*, vol. 32, 2019.  
[9] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.  
[10] S. Bird, E. Klein, and E. Loper, *Natural Language Processing with Python*. Sebastopol, CA: O'Reilly Media, 2009.  
[11] J. Pennington, R. Socher, and C. D. Manning, "GloVe: Global vectors for word representation," in *Proc. EMNLP*, 2014, pp. 1532–1543.  
[12] M. E. Peters et al., "Deep contextualized word representations," in *Proc. NAACL-HLT*, 2018, pp. 2227–2237.  
[13] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv:1907.11692*, 2019.  
[14] Z. Beltagy, M. E. Peters, and A. Cohan, "Longformer: The long-document transformer," *arXiv:2004.05150*, 2020.  
[15] S. Minaee et al., "Deep learning-based text classification: A comprehensive review," *ACM Computing Surveys*, vol. 54, no. 3, 2021.  
[16] Z. Lan et al., "ALBERT: A lite BERT for self-supervised learning of language representations," in *Proc. ICLR*, 2020.  
[17] F. Bojanowski et al., "Enriching word vectors with subword information," *Transactions of the Association for Computational Linguistics*, vol. 5, 2017.  
[18] Z. Dai et al., "Transformer-XL: Attentive language models beyond a fixed-length context," in *Proc. ACL*, 2019.  
[19] Y. Goldberg, *Neural Network Methods for Natural Language Processing*. Morgan & Claypool, 2017.  
[20] D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. draft, 2023.  
[21] K. S. Kalyan, A. Rajasekharan, and S. Sangeetha, "AMMUS: A survey of transformer-based pretrained models in NLP," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 35, no. 7, 2024.  
[22] A. Roy et al., "Automated resume screening using natural language processing and machine learning," in *Proc. IEEE ICCSP*, 2020, pp. 450–454.  
[23] S. Guo, F. Alamudun, and T. Hammond, "RésuMatcher: A personalized résumé-job matching system," *Expert Systems with Applications*, vol. 60, pp. 169–182, 2016. doi: 10.1016/j.eswa.2016.04.013.  
[24] A. Conneau et al., "Supervised learning of universal sentence representations from natural language inference data," in *Proc. EMNLP*, 2017, pp. 670–680.  
[25] "An Advanced Real-Time Job Recommendation System and Resume Analyser," in *Proc. International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS)*, 2023. doi: 10.1109/ICSSAS57918.2023.10331788.  

---

**Github Link :**  
[https://github.com/harsha9625/ResumeMind](https://github.com/harsha9625/ResumeMind)
