# AI Resume Analyzer - Complete Phase Breakdown (Pure NLP)

## Overview
- **Total Phases:** 8 (6 core + 2 optional)
- **Approach:** Traditional NLP pipeline (no neural networks)
- **Dev Environment:** Jupyter notebooks (step-by-step like Kloris)

---

## PHASE 1: Data Preparation & Exploration
**Goal:** Download, explore, and extract resume text from raw datasets

### Tasks:
1. Download Kaggle datasets (CareerCorpus, AI-Powered Screening Dataset, 2,400 Resume Dataset)
2. Explore dataset format (CSV? PDF? Structure?)
3. Extract all resume text → create unified text corpus
4. Create `/data/raw/` and `/data/processed/` folders
5. Save extracted resumes as CSV: `id | text | source`

### Deliverables:
- ✅ Datasets downloaded to `/data/raw/`
- ✅ `resumes_text.csv` (all resumes extracted as text)
- ✅ Data exploration notebook: `01_data_exploration.ipynb`
- ✅ Statistics: total resumes, avg length, data quality report

### Key Decisions:
- How many resumes to use? (all, or sample?)
- How to handle PDFs if present? (pdfplumber, PyPDF2)
- Clean text or keep raw?

### Notebook: `notebooks/01_data_exploration.ipynb`
```
- Load datasets
- Explore structure
- Extract text from PDFs/CSVs
- Save to resumes_text.csv
- Basic statistics & quality checks
```

---

## PHASE 2: NER Training Data Preparation & Model Training
**Goal:** Train a custom spaCy NER model to extract skills, experience, education, certifications

### Tasks:
1. Select ~500-1000 resume samples for labeling
2. Manually annotate entities (or use semi-automatic labeling):
   - SKILL: Python, Java, Machine Learning, etc.
   - JOB_TITLE: Software Engineer, Data Scientist, etc.
   - COMPANY: Google, Microsoft, etc.
   - EXPERIENCE_DURATION: 2019-2023, 5 years, etc.
   - DEGREE: B.Tech, MBA, etc.
   - INSTITUTION: IIT Bombay, Stanford, etc.
   - CERTIFICATION: AWS, IELTS, etc.
3. Convert labeled data to spaCy training format (.spacy files)
4. Train spaCy NER model
5. Evaluate NER accuracy on test set

### Deliverables:
- ✅ `resumes_labeled.csv` (annotated training data)
- ✅ Trained NER model saved to `/models/ner_model/`
- ✅ NER training notebook: `02_ner_training.ipynb`
- ✅ Evaluation metrics: Precision, Recall, F1-score

### Key Decisions:
- Manual labeling tool? (Prodigy, LabelImg, or manual CSV?)
- How many samples to label?
- Entity categories to extract?

### Notebook: `notebooks/02_ner_training.ipynb`
```
- Load resumes_text.csv
- Manual entity annotation (or semi-auto)
- Convert to spaCy training format
- Train NER model with spaCy
- Evaluate on test set (precision, recall, F1)
- Save best model to /models/ner_model/
```

---

## PHASE 3: Entity Extraction & Feature Creation
**Goal:** Use trained NER to extract entities from all resumes, create structured feature vectors

### Tasks:
1. Load trained NER model
2. Run inference on all resumes → extract entities
3. Aggregate entities per category:
   - Skills list (deduplicated)
   - Experience years + job titles + companies
   - Education level + field + institution
   - Certifications list
4. Create `resumes_entities.csv` with structured data
5. Create feature vectors for each resume

### Deliverables:
- ✅ `resumes_entities.csv` (structured extracted entities)
- ✅ Entity extraction notebook: `03_entity_extraction.ipynb`
- ✅ Python module: `src/entity_extractor.py` (reusable)
- ✅ Feature vector format defined

### Key Decisions:
- How to aggregate skills? (list, set, count?)
- How to normalize experience years? (handle "5 years", "2019-2023", etc.)
- What features matter most?

### Notebook: `notebooks/03_entity_extraction.ipynb`
```
- Load trained NER model from /models/ner_model/
- Run inference on all resumes
- Extract and aggregate entities
- Handle edge cases (missing fields, duplicates, etc.)
- Save to resumes_entities.csv
- Create feature vector representation
```

---

## PHASE 4: Vectorization (TF-IDF + Word Embeddings)
**Goal:** Convert text and entities into numerical vectors for similarity comparison

### Tasks:
1. Implement TF-IDF vectorizer for skills, education, experience text
2. Implement word embeddings (Word2Vec or GloVe or sentence-transformers)
3. Create composite feature vectors:
   - Skills vector (TF-IDF)
   - Experience vector (embeddings + years normalized)
   - Education vector (TF-IDF)
   - Overall resume vector (concatenation)
4. Save vectorizer models for inference phase
5. Test vectorization on sample resumes & JDs

### Deliverables:
- ✅ TF-IDF vectorizer saved to `/models/vectorizer/tfidf_model.pkl`
- ✅ Embeddings model saved to `/models/vectorizer/embeddings_model.pkl`
- ✅ Vectorization notebook: `04_vectorization.ipynb`
- ✅ Python module: `src/vectorizer.py` (reusable)
- ✅ Vector dimensionality & format defined

### Key Decisions:
- TF-IDF vs embeddings? (use both for robustness)
- Vector dimensions? (TF-IDF: top N features, embeddings: 300-dim)
- How to normalize experience years numerically?

### Notebook: `notebooks/04_vectorization.ipynb`
```
- Load resumes_entities.csv
- Implement TF-IDF vectorizer (skills, education)
- Implement embeddings vectorizer (experience, overall)
- Create composite vectors
- Save models to /models/vectorizer/
- Test on sample data
```

---

## PHASE 5: Similarity Scoring & Category Scores
**Goal:** Calculate cosine similarity between resume and JD vectors, output category scores

### Tasks:
1. Implement cosine similarity function
2. Create scoring logic for each category:
   - **Skills Score:** Cosine similarity(resume_skills_vector, jd_skills_vector) * 100
   - **Experience Score:** Match job titles + years + companies * 100
   - **Education Score:** Match degree + field * 100
   - **Overall Score:** Weighted average of above 3 scores
3. Extract JD entities (run NER + vectorization on job description)
4. Compare resume vs JD vectors → generate scores
5. Add actionable feedback (missing skills, experience gaps, etc.)
6. Test on sample resume-JD pairs

### Deliverables:
- ✅ Similarity scoring notebook: `05_similarity_scoring.ipynb`
- ✅ Python module: `src/similarity_scorer.py` (reusable)
- ✅ Python module: `src/text_extractor.py` (PDF/docx parsing)
- ✅ Output format defined: `{skills_score, experience_score, education_score, overall_score, feedback}`

### Key Decisions:
- Weights for category scores? (e.g., 40% skills, 40% experience, 20% education)
- What constitutes a "match"? (exact, partial, semantic?)
- Feedback generation logic?

### Notebook: `notebooks/05_similarity_scoring.ipynb`
```
- Load vectorizer models
- Implement cosine similarity
- Extract entities from sample JD
- Vectorize JD
- Compare resume vs JD vectors
- Calculate category scores
- Generate feedback
- Test on multiple resume-JD pairs
```

---

## PHASE 6: Testing, Evaluation & Pipeline Integration
**Goal:** Test entire pipeline end-to-end, evaluate accuracy, create reusable modules

### Tasks:
1. Create test dataset (10-20 resume-JD pairs with ground truth)
2. Run full pipeline: Resume → NER → Entity extraction → Vectorization → Scoring
3. Evaluate output quality:
   - Do extracted entities make sense?
   - Are similarity scores reasonable?
   - Is feedback actionable?
4. Benchmark pipeline:
   - Processing time per resume
   - Memory usage
   - Accuracy metrics
5. Create unified Python pipeline module (`src/resume_analyzer.py`)
6. Write unit tests for each module
7. Document all code with docstrings

### Deliverables:
- ✅ Evaluation notebook: `06_evaluation.ipynb`
- ✅ Unified pipeline module: `src/resume_analyzer.py`
- ✅ Unit tests: `tests/test_*.py`
- ✅ Benchmark report (speed, accuracy, results)
- ✅ README.md updated with usage instructions

### Key Decisions:
- Success metrics? (accuracy, speed, user satisfaction)
- How to evaluate without ground truth?

### Notebook: `notebooks/06_evaluation.ipynb`
```
- Load all pipeline components
- Test on sample resume-JD pairs
- Evaluate entity extraction quality
- Evaluate scoring reasonableness
- Benchmark performance
- Generate evaluation report
```

---

## PHASE 7 (OPTIONAL): Frontend / Demo Interface
**Goal:** Build browser-based UI for easy testing (optional, only if time allows)

### Tasks:
1. Create HTML/CSS/JavaScript frontend
2. Implement PDF upload functionality (PDF.js)
3. Implement JD text input (textarea)
4. Call backend API or local Python server
5. Display category scores, feedback, best-fit roles
6. Style UI for competition demo

### Deliverables:
- ✅ `/frontend/` folder with HTML/CSS/JS
- ✅ Local Python server to serve frontend + run pipeline
- ✅ Working demo (upload resume → get scores)

**Note:** This is optional and only if you have time after Phase 6.

---

## PHASE 8 (OPTIONAL): FastAPI Backend
**Goal:** Wrap pipeline in REST API for potential deployment (optional)

### Tasks:
1. Create FastAPI application
2. Endpoints:
   - `/analyze` (POST: resume file + JD text → returns scores)
   - `/health` (GET: check if service is running)
3. Error handling
4. Documentation

**Note:** This is optional. Judges focus on model quality, not deployment.

---

## Summary Table

| Phase | Title | Key Output | Status |
|-------|-------|-----------|--------|
| 1 | Data Prep & Exploration | `resumes_text.csv` | ⬜ Todo |
| 2 | NER Training | Trained NER model | ⬜ Todo |
| 3 | Entity Extraction | `resumes_entities.csv` | ⬜ Todo |
| 4 | Vectorization | TF-IDF + embeddings models | ⬜ Todo |
| 5 | Similarity Scoring | Category scores + feedback | ⬜ Todo |
| 6 | Testing & Evaluation | Unified pipeline module | ⬜ Todo |
| 7 | Frontend (Optional) | Browser UI demo | ⬜ Todo |
| 8 | FastAPI (Optional) | REST API | ⬜ Todo |

---

## Key Technologies

**Data & NLP:**
- spaCy (NER training & inference)
- scikit-learn (TF-IDF vectorization)
- gensim or sentence-transformers (embeddings)
- pdfplumber or PyPDF2 (PDF parsing)
- pandas (data manipulation)

**Development:**
- Jupyter notebooks (interactive development)
- Python 3.10+
- Virtual environment (venv in `/notebooks/`)

**Optional:**
- FastAPI (backend)
- HTML/CSS/JavaScript (frontend)
- PDF.js (browser PDF handling)

---

## Success Metrics

By end of Phase 6, you should have:
- ✅ Working NLP pipeline (resume → entities → vectors → scores)
- ✅ Category scores for skills, experience, education
- ✅ Actionable feedback for users
- ✅ Tested on 10+ resume-JD pairs
- ✅ Clean, documented Python code
- ✅ Benchmark report (speed, accuracy)

This demonstrates:
- **NLP expertise** (custom NER, entity extraction, vectorization)
- **Feature engineering** (creating meaningful vectors from text)
- **Software engineering** (modular, testable code)
- **Problem-solving** (end-to-end pipeline)

Perfect for an ML competition! 🚀

---

## Next Steps

**Ready to start Phase 1?**

Let me know and we'll begin:
1. Setting up project folder structure
2. Downloading Kaggle datasets
3. Exploring data format
