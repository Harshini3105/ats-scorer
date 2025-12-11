# ATS Scorer (Resume ↔ Job Description matcher)

Small Flask app that computes a similarity score between a resume and a job description using TF-IDF cosine similarity, extracts keywords (spaCy), and highlights missing keywords.

## Quick install & run (macOS / Linux / Windows WSL)
1. Create venv & activate:
```bash
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
