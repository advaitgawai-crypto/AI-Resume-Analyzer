# AI Resume Analyzer

A machine learning-based resume analyzer that evaluates how well a candidate's resume matches a job description using NLP techniques.

## Overview

This project implements a **traditional NLP pipeline** (no neural networks) to:
- Extract structured information from resumes using Named Entity Recognition (NER)
- Vectorize resume and job description text
- Calculate similarity scores across multiple categories
- Provide actionable feedback to job seekers

## Features

- **Custom NER Model**: Trained spaCy model for extracting skills, experience, education, and certifications
- **Multi-Category Scoring**:
  - Skills Match Score
  - Experience Match Score
  - Education Match Score
  - Overall Match Score (0-100)
- **Actionable Feedback**: Identify missing skills, experience gaps, and education mismatches
- **Entity Extraction**: Automatically extract and categorize resume entities

## Architecture

```
Resume (PDF/docx) + Job Description (text)
    ↓
Extract Text + Clean
    ↓
NER Model (Custom spaCy)
    ↓
Entity Extraction (Skills, Experience, Education, Certifications)
    ↓
Vectorization (TF-IDF + Word Embeddings)
    ↓
Cosine Similarity Scoring
    ↓
Category Scores + Feedback
```

## Tech Stack

- **NLP**: spaCy, scikit-learn, gensim/sentence-transformers
- **Data Processing**: pandas, numpy
- **PDF Parsing**: pdfplumber, PyPDF2
- **Development**: Jupyter Notebook, Python 3.10+
- **Version Control**: Git

## Project Structure

```
AI-Resume-Analyzer/
├── data/
│   ├── raw/                 # Original datasets from Kaggle
│   ├── processed/           # Cleaned, extracted data
│   └── test/                # Test resumes & JDs
├── models/
│   ├── ner_model/           # Trained spaCy NER
│   └── vectorizer/          # TF-IDF + embeddings models
├── notebooks/               # Jupyter notebooks (01-07)
├── src/                     # Python modules
├── tests/                   # Unit tests
├── README.md
├── .gitignore
├── requirements.txt
└── PROJECT_LOG.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Phase 1: Data Preparation
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Phase 2: NER Training
```bash
jupyter notebook notebooks/02_ner_training.ipynb
```

### Phase 3: Entity Extraction
```bash
jupyter notebook notebooks/03_entity_extraction.ipynb
```

### Phase 4: Vectorization
```bash
jupyter notebook notebooks/04_vectorization.ipynb
```

### Phase 5: Similarity Scoring
```bash
jupyter notebook notebooks/05_similarity_scoring.ipynb
```

### Phase 6: Testing & Evaluation
```bash
jupyter notebook notebooks/06_evaluation.ipynb
```

## Development Phases

| Phase | Title | Deliverable |
|-------|-------|------------|
| 1 | Data Prep & Exploration | `resumes_text.csv` |
| 2 | NER Training | Trained NER model |
| 3 | Entity Extraction | `resumes_entities.csv` |
| 4 | Vectorization | TF-IDF + embeddings models |
| 5 | Similarity Scoring | Category scores + feedback |
| 6 | Testing & Evaluation | Unified pipeline module |
| 7 | Frontend (Optional) | Browser UI demo |
| 8 | FastAPI (Optional) | REST API |

## Datasets Used

- CareerCorpus (Kaggle)
- AI-Powered Screening Dataset (Kaggle)
- 2,400 Resume Dataset (Kaggle)

## Key Concepts

### Named Entity Recognition (NER)
Automatically identifies and classifies entities in resume text:
- **SKILL**: Programming languages, tools, frameworks
- **JOB_TITLE**: Position titles (Software Engineer, Data Scientist, etc.)
- **COMPANY**: Organizations/employers
- **EXPERIENCE_DURATION**: Years of experience or date ranges
- **DEGREE**: Educational qualifications (B.Tech, MBA, etc.)
- **INSTITUTION**: Universities/colleges
- **CERTIFICATION**: Professional certifications (AWS, IELTS, etc.)

### Vectorization
Converts text into numerical vectors for comparison:
- **TF-IDF**: Represents importance of words in documents
- **Word Embeddings**: Captures semantic meaning of text

### Cosine Similarity
Measures similarity between two vectors (0-1 scale, converted to 0-100):
- 1.0 = perfect match
- 0.0 = no similarity

## Results Format

```json
{
  "skills_score": 75,
  "experience_score": 82,
  "education_score": 70,
  "overall_score": 76,
  "feedback": {
    "missing_skills": ["Python", "Docker"],
    "experience_gaps": "Need 2+ more years of experience",
    "education_match": "Bachelor's degree matches requirement"
  }
}
```

## Contributing

This is an educational project for ML competition. Feel free to:
- Improve NER accuracy
- Enhance vectorization methods
- Add new entity categories
- Optimize scoring logic

## Future Enhancements (Optional)

- Multi-profession specific models (separate NER per profession)
- FastAPI backend for deployment
- PostgreSQL/S3 database integration
- LLM-based insights (LongChain)
- Browser UI frontend
- Streamlit dashboard

## License

MIT License

## Author

ThunderLord (Educational ML Project)

## References

- spaCy NLP Library: https://spacy.io/
- scikit-learn: https://scikit-learn.org/
- Codebasics KLP Course
- Stanford NLP Playlist (Prof. Christopher Manning)
