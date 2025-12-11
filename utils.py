# utils.py
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Load spaCy model lazily (call ensure_spacy before use)
NLP = None

def ensure_spacy(model="en_core_web_sm"):
    global NLP
    if NLP is None:
        try:
            NLP = spacy.load(model)
        except Exception as e:
            raise RuntimeError(f"spaCy model '{model}' not found. Install with: python -m spacy download {model}") from e
    return NLP

def read_text_from_file(path_or_fileobj):
    # Accept Path or file-like
    if hasattr(path_or_fileobj, "read"):
        text = path_or_fileobj.read()
        if isinstance(text, bytes):
            text = text.decode(errors="ignore")
        return text
    else:
        return Path(path_or_fileobj).read_text(encoding="utf-8", errors="ignore")

def clean_text(text):
    # Basic cleaning: remove emails/URLs, extra whitespace
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\-\s+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_keywords_spacy(text, top_n=20):
    nlp = ensure_spacy()
    doc = nlp(text)
    cand = []
    for token in doc:
        # consider nouns, proper nouns and compounds/adjectives as candidate keywords
        if token.pos_ in ("NOUN", "PROPN", "ADJ"):
            # normalize
            cand.append(token.lemma_.lower())
    # frequency-based top candidates
    if not cand:
        return []
    freq = pd.Series(cand).value_counts()
    return list(freq.index[:top_n])

def compute_similarity_score(texts):
    # texts: list of strings [resume_text, jd_text] or vice versa
    vect = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    sim = cosine_similarity(X[0:1], X[1:2])[0][0]
    return float(sim), vect

def match_keywords(jd_keywords, resume_keywords):
    # return present and missing lists (simple substring match)
    resume_set = set(resume_keywords)
    present = [k for k in jd_keywords if any(k in r for r in resume_set)]
    missing = [k for k in jd_keywords if k not in present]
    return present, missing
