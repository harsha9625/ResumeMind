"""
Smart ATS Checker: Resume-to-Job Matching System.
An academic Deep Learning and NLP system for resume-to-job compatibility analysis.
"""

from src.resume_parser import extract_pdf_data, extract_text_data
from src.skill_extractor import TECH_SKILLS, calculate_skill_match, extract_skills
from src.utils import get_data_path, get_project_root

try:
    from src.models import cosine, get_HF_embeddings, get_doc2vec_embeddings, mean_pooling
    from src.resume_scanner import compare
    from src.job_recommender import recommend_jobs
except ImportError:
    # Allow lightweight utilities and text parsers to be imported
    # even when heavy deep learning dependencies are not yet installed.
    cosine = None
    get_HF_embeddings = None
    get_doc2vec_embeddings = None
    mean_pooling = None
    compare = None
    recommend_jobs = None

__version__ = "1.0.0"
__all__ = [
    "get_HF_embeddings",
    "get_doc2vec_embeddings",
    "cosine",
    "mean_pooling",
    "extract_pdf_data",
    "extract_text_data",
    "TECH_SKILLS",
    "extract_skills",
    "calculate_skill_match",
    "compare",
    "recommend_jobs",
    "get_project_root",
    "get_data_path",
]
