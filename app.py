# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from utils import read_text_from_file, clean_text, extract_keywords_spacy, compute_similarity_score, match_keywords, ensure_spacy

APP = Flask(__name__)
APP.secret_key = os.environ.get("FLASK_SECRET", "devsecret")  # change for production

@APP.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        resume_f = request.files.get("resume")
        jd_f = request.files.get("jd")
        if not resume_f or not jd_f:
            flash("Please upload both Resume and Job Description files (txt, pdf text or .docx not supported directly).")
            return redirect(url_for("index"))
        # read text
        resume_text = read_text_from_file(resume_f)
        jd_text = read_text_from_file(jd_f)
        # clean
        resume_clean = clean_text(resume_text)
        jd_clean = clean_text(jd_text)
        # ensure spaCy
        try:
            ensure_spacy()
        except Exception as e:
            flash(str(e))
            return redirect(url_for("index"))

        # compute similarity
        score, vect = compute_similarity_score([resume_clean, jd_clean])
        score_percent = round(score * 100, 1)

        # extract keywords
        jd_keys = extract_keywords_spacy(jd_clean, top_n=40)
        resume_keys = extract_keywords_spacy(resume_clean, top_n=200)

        present, missing = match_keywords(jd_keys, resume_keys)

        # simple counts
        result = {
            "score": score_percent,
            "jd_keywords": jd_keys,
            "resume_keywords_sample": resume_keys[:60],
            "present_keywords": present,
            "missing_keywords": missing
        }
        return render_template("index.html", result=result)
    return render_template("index.html", result=None)


if __name__ == "__main__":
    # CLI fallback: quick scoring from two text files
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", "-r", help="Path to resume text file")
    parser.add_argument("--jd", "-j", help="Path to job description text file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.resume and args.jd:
        # CLI mode: print results
        from utils import read_text_from_file, clean_text, extract_keywords_spacy, compute_similarity_score, match_keywords, ensure_spacy
        ensure_spacy()
        r = clean_text(read_text_from_file(args.resume))
        j = clean_text(read_text_from_file(args.jd))
        score, _ = compute_similarity_score([r, j])
        jd_keys = extract_keywords_spacy(j, top_n=40)
        resume_keys = extract_keywords_spacy(r, top_n=200)
        present, missing = match_keywords(jd_keys, resume_keys)
        print(f"Similarity score: {score*100:.2f}%")
        print("Top JD keywords:", jd_keys[:20])
        print("Present:", present[:20])
        print("Missing:", missing[:20])
    else:
        APP.run(host=args.host, port=args.port, debug=True)
