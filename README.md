# AI Resume Analyzer

A machine learning-based resume analyzer that evaluates how well a candidate's resume matches a job description using NLP techniques. **Now deployed as a production web app on Railway.**

## 🎯 Overview

This project implements a **traditional NLP pipeline** (no neural networks) to:
- Extract structured information from resumes using Named Entity Recognition (NER)
- Vectorize resume and job description text using TF-IDF
- Calculate similarity scores across multiple categories
- Provide actionable feedback to job seekers
- Serve results via a Flask web interface with Coral Sunset theme

## 🚀 Live Demo

**Public URL:** `https://ai-resume-analyzer-production-d77b.up.railway.app`

Upload any job posting PDF to see:
- Profile match score (0–100%)
- Required skills extracted
- Skills you already have
- Skills to develop by priority
- Related job openings from OpenWeb Ninja JSearch API

## ✨ Features

- **Custom NER Model**: Trained spaCy model for extracting skills, experience, education, and certifications
- **Multi-Category Scoring**:
  - Skills Match Score (40% weight)
  - Job Title Match Score (25% weight)
  - Education Match Score (20% weight)
  - Experience Match Score (10% weight)
  - Institution Match Score (5% weight)
  - Overall Match Score (0–100)
- **Actionable Feedback**: Identify missing skills by priority (HIGH/MEDIUM/LOW)
- **Entity Extraction**: Automatically extract and categorize resume entities
- **Web Interface**: Drag-drop PDF upload with animated score visualization
- **Job Search Integration**: Find relevant job postings by extracted skills
- **Cloud Deployment**: Production-ready on Railway with auto-deploy from GitHub

## 🏗️ Architecture

```
Job Posting PDF (User Upload)
    ↓
[Extract Text with pdfplumber]
    ↓
[NER Model (spaCy v3) - Entity Extraction]
    ↓
[Entity Categories: SKILL, JOB_TITLE, DEGREE, INSTITUTION, CERTIFICATION, EXPERIENCE_DURATION]
    ↓
[Load Pre-Computed Vectorizers & Resume Vectors]
    ↓
[Vectorize Job Requirements (TF-IDF)]
    ↓
[Calculate Cosine Similarity vs. 2,716 Resumes]
    ↓
[Rank Resumes Per Category (Top 50)]
    ↓
[Generate Improvement Recommendations]
    ↓
[Web UI Response (JSON)] → Score Ring Animation + Skill Cards
```

## 🛠️ Tech Stack

**NLP/ML:**
- spaCy 3.7.2 (NER model, en_core_web_sm fallback)
- scikit-learn 1.5.0 (TF-IDF vectorization, cosine similarity)
- pandas 2.1.0 (data processing)
- numpy (numerical operations)

**PDF Processing:**
- pdfplumber 0.11.0
- PyPDF2 (optional)

**Backend:**
- Flask 3.0.0
- flask-cors 4.0.0
- requests (API integration)

**Frontend:**
- Vanilla HTML/CSS/JS (no frameworks)
- SVG animations (progress ring, counters)
- Coral Sunset color scheme (#1f1010 bg, #f97316 accent)

**Deployment:**
- Railway.app (free tier, auto-deploy from GitHub)
- Python 3.9.18
- Git/GitHub for version control

**Development:**
- Jupyter Notebook (analysis & training)
- pytest (unit tests)

## 📁 Project Structure

```
AI-Resume-Analyzer/
├── app.py                              # Flask backend (main entry point)
├── requirements.txt                    # Python dependencies
├── Procfile                            # Railway deployment config
├── runtime.txt                         # Python version (3.9.18)
├── README.md                           # This file
├── .gitignore
│
├── frontend/                           # Web UI
│   ├── index.html                     # 3-step form (upload → report → search)
│   ├── style.css                      # Coral Sunset theme + animations
│   └── script.js                      # Vanilla JS (drag-drop, API calls)
│
├── models/
│   └── ner_model_v3/                  # Fine-tuned spaCy NER model
│
├── data/
│   ├── raw/                           # Original datasets from Kaggle
│   │   └── archive/                   # 2,484 resumes (24 professions)
│   │
│   ├── processed/                     # Phase 4-5 outputs
│   │   ├── resumes_with_entities.csv  # Extracted entity data
│   │   ├── vectorizer_skill.pkl       # TF-IDF vectorizers
│   │   ├── vectorizer_job_title.pkl
│   │   ├── resume_vectors_skill.npz   # Pre-computed vectors
│   │   ├── resume_vectors_job_title.npz
│   │   └── ...
│   │
│   └── input/
│       └── web_uploads/               # User-uploaded PDFs (transient)
│
├── notebooks/                         # Jupyter notebooks (Phases 1–6)
│   ├── 01_data_exploration.ipynb
│   ├── 02_ner_training.ipynb
│   ├── 03_entity_extraction.ipynb
│   ├── 04_vectorization.ipynb
│   ├── 05_similarity_scoring.ipynb
│   └── 06_evaluation.ipynb
│
├── src/                               # Python modules (utilities)
│   └── (helper functions)
│
└── tests/                             # Unit tests
```

## 💻 Installation & Setup

### Local Development

#### 1. Clone Repository
```bash
git clone https://github.com/advaitgawai-crypto/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### 4. Prepare Data
Ensure these files exist locally:
- `models/ner_model_v3/` (trained NER model)
- `data/processed/resumes_with_entities.csv` (2,716 resumes with extracted entities)
- `data/processed/vectorizer_*.pkl` (TF-IDF vectorizers for each entity type)
- `data/processed/resume_vectors_*.npz` (pre-computed vector matrices)

#### 5. Run Locally
```bash
python app.py
# Visit http://localhost:5000
```

### Cloud Deployment (Railway)

#### Prerequisites
- GitHub account with repo pushed
- Railway account (free tier available at railway.app)

#### Deploy Steps

1. **Connect GitHub to Railway:**
   - Go to railway.app
   - Click "New Project" → "Deploy from GitHub"
   - Authorize and select your repo

2. **Verify deployment files:**
   ```
   ✓ Procfile (web: python app.py)
   ✓ runtime.txt (python-3.9.18)
   ✓ requirements.txt (dependencies listed)
   ```

3. **Commit and push to main branch:**
   ```bash
   git add .
   git commit -m "Deploy to Railway"
   git push origin main
   ```

4. **Railway auto-builds and deploys** (2–3 minutes)

5. **Get live URL** from Railway dashboard (in Settings → Domains)

#### Optional: Environment Variables
Set in Railway dashboard → Variables:
```
FLASK_ENV=production
PYTHONUNBUFFERED=1
```

## 🎮 Usage

### Web Interface (Recommended)

1. **Visit:** `https://ai-resume-analyzer-production-d77b.up.railway.app` (or local `http://localhost:5000`)

2. **Step 1: Upload Job Posting PDF**
   - Drag-drop or browse for a PDF
   - Supports job descriptions in any format

3. **Step 2: Review Analysis Report**
   - **Profile Score (0–100%)**: Overall match strength
   - **Required Skills**: Skills extracted from job posting
   - **Your Strengths**: Skills you have that match
   - **Skills to Develop**: Missing skills by priority (HIGH/MEDIUM/LOW)
   - **Action Plan**: Recommendations for skill development

4. **Step 3: Search Related Jobs**
   - Select location and country
   - Jobs pre-filtered by extracted skills
   - View job descriptions and apply links

### Python Notebooks (Local Analysis)

Run Jupyter notebooks for detailed analysis:

```bash
# Phase 1: Explore resume dataset
jupyter notebook notebooks/01_data_exploration.ipynb

# Phase 2: Train NER model
jupyter notebook notebooks/02_ner_training.ipynb

# Phase 3: Extract entities from all resumes
jupyter notebook notebooks/03_entity_extraction.ipynb

# Phase 4: Create vectorizers and resume vectors
jupyter notebook notebooks/04_vectorization.ipynb

# Phase 5: Similarity scoring and matching
jupyter notebook notebooks/05_similarity_scoring.ipynb

# Phase 6: Evaluation and testing
jupyter notebook notebooks/06_evaluation.ipynb
```

## 📊 API Response Format

### `/api/analyze` (POST)
Upload job posting PDF → returns analysis JSON:

```json
{
  "job_posting_name": "Senior_Engineer.pdf",
  "analyzed_at": "2026-08-18 10:20:50",
  "profile_score": 66,
  "required_experience": 5,
  "core_skills": ["Python", "ML", "TensorFlow", "AWS"],
  "job_titles": ["Senior Engineer", "Tech Lead"],
  "required_skills": ["Python", "ML", "TensorFlow", "AWS", ...],
  "strengths": ["Python", "Leadership"],
  "strengths_note": "You already have 8 of the top required skills! Great foundation!",
  "skills_to_develop": [
    {"skill": "HTML", "priority": "MEDIUM"},
    {"skill": "Kubernetes", "priority": "LOW"}
  ],
  "action_plan": "Learn 2 MEDIUM priority skill(s) + 1 LOW priority skill(s) to match 3 more of the top candidate profiles."
}
```

### `/api/search-jobs` (POST)
Search for jobs by skills and location:

```json
{
  "jobs": [
    {
      "job_title": "Senior Python Engineer",
      "company": "TechCorp",
      "location": "San Francisco, USA",
      "salary_min": 150000,
      "salary_max": 200000,
      "salary_currency": "USD",
      "description": "Lead backend infrastructure...",
      "posting_date": "2026-08-10",
      "application_url": "https://...",
      "match_score": 95,
      "matching_skills": 5
    },
    ...
  ],
  "country_name": "United States"
}
```

## 🔍 Entity Types & Scoring

### Named Entity Categories

| Entity | Examples | Weight |
|--------|----------|--------|
| **SKILL** | Python, AWS, Docker, Leadership | 40% |
| **JOB_TITLE** | Senior Engineer, Data Scientist | 25% |
| **DEGREE** | Bachelor's, Master's, PhD | 20% |
| **EXPERIENCE_DURATION** | 5+ years, 2-3 years | 10% |
| **INSTITUTION** | Stanford, MIT, IIT | 5% |
| **CERTIFICATION** | AWS Solutions Architect, CKAD | — |

### Scoring Logic

1. **Extract entities** from job posting using NER
2. **Vectorize** each entity category using pre-trained TF-IDF vectorizers
3. **Calculate cosine similarity** between job requirements and each resume
4. **Rank resumes** per category (top 50 each)
5. **Compute weighted overall score**:
   ```
   Overall Score = (40% × skill_score) + (25% × job_title_score) + 
                   (20% × degree_score) + (10% × experience_score) + 
                   (5% × institution_score)
   ```

## ⚡ Performance Metrics

| Task | Time | Notes |
|------|------|-------|
| PDF text extraction | ~0.2s | pdfplumber |
| NER entity extraction | ~0.5s | spaCy model |
| TF-IDF vectorization | ~0.1s | scikit-learn |
| Cosine similarity (2,716 resumes) | ~1.0s | scipy sparse |
| Output generation | ~0.5s | JSON serialization |
| **Total per request** | **~2–3s** | End-to-end |
| **Memory usage** | ~500 MB | Vectorizers + vectors in memory |

## 🐛 Troubleshooting

### "NER model unavailable" (Web UI error)
**Cause:** `models/ner_model_v3/` not committed to GitHub  
**Solution:**
```bash
git add -f models/ner_model_v3/
git commit -m "Add NER model"
git push origin main
```
Railway will redeploy with the model.

### "Resume database not loaded" (Web UI error)
**Cause:** `data/processed/resumes_with_entities.csv` missing from deployment  
**Solution:**
```bash
git add -f data/processed/resumes_with_entities.csv
git commit -m "Add resume database"
git push origin main
```

### 502 Bad Gateway on Railway
**Cause:** App crash during startup  
**Debug:**
1. Check Railway console logs (Deployments → View logs)
2. Look for Python import errors or missing files
3. Test locally: `python app.py`
4. Common issues: missing spacy model, missing data files

### Slow PDF uploads
**Cause:** Large file size or slow network  
**Fix:** Max upload limit is 16 MB (configurable in `app.py`)

### Vectorizer/Vector loading errors
**Cause:** Corrupted pickle or NPZ files  
**Solution:** Regenerate from `data/processed/resumes_with_entities.csv` locally, then commit

## 📚 Development Phases

| Phase | Title | Status | Deliverable |
|-------|-------|--------|------------|
| 1 | Data Exploration | ✅ Complete | Resume corpus analysis |
| 2 | NER Training | ✅ Complete | ner_model_v3 (F1=0.6638) |
| 3 | Entity Extraction | ✅ Complete | resumes_with_entities.csv |
| 4 | Vectorization | ✅ Complete | TF-IDF vectorizers + vectors |
| 5 | Similarity Scoring | ✅ Complete | Ranking system |
| 6 | Flask Web App | ✅ Complete | Local server + frontend |
| 7 | Cloud Deployment | ✅ Complete | Railway production URL |

## 🎓 Datasets Used

- **CareerCorpus** (Kaggle)
- **AI-Powered Screening Dataset** (Kaggle)
- **2,400 Resume Dataset** (Kaggle)
- **Custom Profession Dataset**: 2,484 resumes across 24 professions

## 🔑 Key Concepts

### Named Entity Recognition (NER)
Identifies and classifies structured information in unstructured text using a trained spaCy model.

### TF-IDF Vectorization
Converts text into numerical vectors emphasizing important terms in context (term frequency × inverse document frequency).

### Cosine Similarity
Measures angle between two vectors in vector space (0 = orthogonal, 1 = identical):
```
similarity = dot_product(v1, v2) / (||v1|| × ||v2||)
Scaled to 0–100 for readability.
```

## 🚀 Future Enhancements

- **Multi-profession models**: Separate NER per industry (Finance, Healthcare, Tech, etc.)
- **LLM integration**: Use GPT-based models for insights
- **Database persistence**: PostgreSQL for job matches history
- **User accounts**: Save analysis history, bookmarks
- **Resume upload**: Analyze user's own resume against jobs
- **Salary prediction**: Estimate salary based on skills
- **Career path recommendations**: Suggest skill progression

## 📄 Contributing

This is an educational project for ML competition. Contributions welcome:
- Improve NER accuracy (add training data, fine-tune)
- Enhance vectorization (add embeddings like Word2Vec, BERT)
- Optimize similarity calculation
- Add new entity categories
- Improve frontend UI/UX

## 📝 License

MIT License

## 👤 Author

- Advait Gawai
- BTech Mathematics and Scientific Computing

## 🔗 Links

- **GitHub Repo:** https://github.com/advaitgawai-crypto/AI-Resume-Analyzer
- **Live Demo:** https://ai-resume-analyzer-production-d77b.up.railway.app
- **LinkedIn:** (optional)
- **Email:** (optional)

## 📖 References

- spaCy Documentation: https://spacy.io/
- scikit-learn ML: https://scikit-learn.org/
- pdfplumber: https://github.com/jsvine/pdfplumber
- Flask: https://flask.palletsprojects.com/
- Railway Docs: https://docs.railway.app/

---

**Status:** Production-Ready ✅  
**Last Updated:** August 18, 2026  
**Deployment:** Railway (auto-deploy from GitHub main branch)