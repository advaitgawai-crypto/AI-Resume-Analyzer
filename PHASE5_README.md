# Phase 5: Matching Engine & Resume Advisor

## Overview

Phase 5 transforms the AI Resume Analyzer into a **resume advisor system** that:

1. **Parses job posting PDFs** (using `pdfplumber`)
2. **Extracts entities** (SKILL, JOB_TITLE, DEGREE, INSTITUTION, CERTIFICATION) using the trained NER model
3. **Extracts experience duration** (regex + NLP fallback)
4. **Ranks all 2,716 resumes** per category using TF-IDF similarity
5. **Provides improvement recommendations** showing users what skills to learn to compete (Options A & B)
6. **Outputs 8 CSV files** (7 categories + 1 overall score)

---

## Architecture

```
Job Posting PDF
    ↓
[Extract Text (pdfplumber)]
    ↓
[Extract Entities (NER model)]
[Extract Experience Duration (regex + NLP)]
    ↓
[Load Phase 4 Vectorizers & Resume Vectors]
    ↓
[Vectorize Job Posting Entities]
    ↓
[Calculate Cosine Similarity to All Resumes]
    ↓
[Rank Resumes Per Category (top 50)]
[Calculate Overall Weighted Score]
[Generate Improvement Recommendations]
    ↓
8 Output CSVs + Summary Report
```

---

## Prerequisites

### 1. Complete Phase 4
Ensure Phase 4 outputs exist:
- `resumes_with_entities.csv` (2,716 resumes with extracted entities)
- `similarity_matrix.npz` (optional, Phase 5 doesn't use it)
- `category_scores.csv` (optional, Phase 5 doesn't use it)
- `ner_model_v2/` (trained spaCy model)

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn scipy spacy pdfplumber
```

### 3. Prepare Phase 4 Vectorizers
Run this **once** to save vectorizers and resume vectors:
```bash
python prepare_phase4_for_phase5.py
```

This creates:
- `vectorizer_skill.pkl`, `vectorizer_job_title.pkl`, etc.
- `resume_vectors_skill.npz`, `resume_vectors_job_title.npz`, etc.

---

## Usage

### Basic Usage
```bash
python phase5_matching_engine.py path/to/job_posting.pdf
```

### Example
```bash
python phase5_matching_engine.py "C:\Users\Advait Gawai\Desktop\senior_python_engineer.pdf"
```

### From Python Code
```python
from pathlib import Path
from phase5_matching_engine import run_phase5

job_pdf = Path("job_posting.pdf")
run_phase5(job_pdf)
```

---

## Outputs (8 Files)

All outputs saved to: `data/output/phase5/`

### 1. `skill_rankings_TIMESTAMP.csv`
Top 50 resumes ranked by skill match.
```
rank,resume_id,SKILL_score
1,12345678,0.876
2,87654321,0.834
...
50,99999999,0.125
```

### 2. `job_title_rankings_TIMESTAMP.csv`
Top 50 resumes ranked by job title relevance.
```
rank,resume_id,JOB_TITLE_score
1,54321098,0.789
...
```

### 3. `degree_rankings_TIMESTAMP.csv`
Top 50 resumes ranked by educational background.

### 4. `institution_rankings_TIMESTAMP.csv`
Top 50 resumes ranked by university/institution match.

### 5. `certification_rankings_TIMESTAMP.csv`
Top 50 resumes ranked by certification match.

### 6. `experience_rankings_TIMESTAMP.csv`
Top 50 resumes ranked by years of experience.
- Perfect match (within ±1 year): 1.0
- Overqualified (more experience): 0.9
- Underqualified: 0.3–0.7 (scaled by gap)
- No data: 0.5 (neutral)

### 7. `overall_rankings_TIMESTAMP.csv`
All 2,716 resumes ranked by weighted overall score.
```
rank,resume_id,overall_score,skill_score,job_title_score,degree_score,institution_score,certification_score,experience_score
1,12345678,0.876,0.92,0.78,0.91,0.00,0.05,0.85
2,87654321,0.834,0.87,0.73,0.88,0.02,0.00,0.80
...
```

**Weight breakdown:**
- SKILL: 40%
- JOB_TITLE: 25%
- DEGREE: 20%
- EXPERIENCE_DURATION: 10%
- INSTITUTION: 5%

### 8. `improvement_recommendations_TIMESTAMP.csv`
Skill gaps and recommendations using **both Option A & B**:

**Option A: Top Resume Analysis**
- Compare job requirements to top-ranked resume
- Show missing skills/certifications that top candidate has
- Example: "Top candidate has AWS, Docker, Kubernetes — you're missing these"

**Option B: Aggregate Priority**
- Analyze top-10 matched resumes
- Identify most-common missing skills (frequency count)
- Assign priority: HIGH (9-10/10 have it), MEDIUM (5-8/10), LOW (1-4/10)
- Example: "Learn AWS (HIGH priority: 10/10 top resumes have it)"

```
entity_type,category,missing_item,frequency,priority,recommendation
SKILL,Option B: Aggregate Priority,AWS,10/10,HIGH,Learn AWS (HIGH priority: 10/10 top resumes have it)
SKILL,Option B: Aggregate Priority,Docker,8/10,MEDIUM,Learn Docker (MEDIUM priority: 8/10 top resumes have it)
CERTIFICATION,Option A: Top Resume Skills,AWS Solutions Architect,,,"Top candidate has AWS Solutions Architect — you're missing this"
```

### 9. `phase5_summary_TIMESTAMP.txt`
Human-readable summary:
- Job posting requirements (extracted entities)
- Top 10 matched resumes with scores
- Full improvement recommendations

---

## Example Workflow

### Scenario
A job seeker wants to apply for a "Senior Python Engineer" role and wants to know:
1. How well their resume matches
2. What skills they're missing
3. Which resumes are the best matches

### Step 1: Prepare Job Posting
Create a PDF: `senior_python_engineer.pdf`
```
Senior Python Engineer - San Francisco, CA

Requirements:
- 5+ years Python experience
- Expert in AWS (EC2, S3, Lambda)
- Docker, Kubernetes, CI/CD
- Master's degree or equivalent
- Certifications: AWS Solutions Architect, Kubernetes CKAD

Responsibilities:
- Lead backend infrastructure
- Design scalable microservices
- Mentor junior engineers
```

### Step 2: Run Phase 5
```bash
python phase5_matching_engine.py senior_python_engineer.pdf
```

### Step 3: Review Outputs
1. **overall_rankings.csv** → Find top-50 candidates
2. **skill_rankings.csv** → See who has the best skill match
3. **improvement_recommendations.csv** → Learn what skills to develop

Example top recommendations:
```
SKILL: Learn Kubernetes (HIGH: 10/10 top resumes have it)
SKILL: Learn Docker (HIGH: 9/10 top resumes have it)
SKILL: Learn Terraform (MEDIUM: 7/10 top resumes have it)
CERTIFICATION: AWS Solutions Architect (appears in 8/10 top resumes)
```

---

## Configuration

Edit `phase5_matching_engine.py` to adjust:

```python
# Output configuration
TOP_N = 50  # Top 50 resumes per category (Phase 5 output)
IMPROVEMENT_TOP_N = 10  # Top 10 resumes for improvement suggestions

# Weights for overall score
WEIGHTS = {
    'SKILL': 0.40,
    'JOB_TITLE': 0.25,
    'DEGREE': 0.20,
    'EXPERIENCE_DURATION': 0.10,
    'INSTITUTION': 0.05,
    'CERTIFICATION': 0.00,  # Adjust if needed
}
```

---

## Performance

**Execution Time:** ~2–3 seconds per job posting
- PDF parsing: 0.2s
- NER extraction: 0.5s
- Vectorization: 0.1s
- Similarity calculation: 1.0s
- Output generation: 0.5s

**Memory Usage:** ~500 MB (vectorizers + vectors in memory)

**Scalability:**
- Can handle 1000s of job postings (batch mode)
- Vectorizers are loaded once, reused for each job

---

## Troubleshooting

### Error: "Vectorizer not found for SKILL"
**Cause:** `prepare_phase4_for_phase5.py` not run yet.
**Solution:** Run preparation script first.
```bash
python prepare_phase4_for_phase5.py
```

### Error: "NER model not found"
**Cause:** Phase 2.5 `ner_model_v2` missing.
**Solution:** Ensure Phase 2.5 is complete and model saved at `models/ner_model_v2/`.

### Error: "resumes_with_entities.csv not found"
**Cause:** Phase 3 not complete.
**Solution:** Run Phase 3 to generate entity extraction output.

### No experience duration extracted
**Cause:** Job posting doesn't follow standard "X years" format.
**Solution:** Manual extraction or refine regex patterns in `extract_experience_duration()`.

---

## Next Steps

After Phase 5:

### Phase 6: API Development
- FastAPI endpoints for resume upload, job posting upload
- Real-time ranking and recommendations
- Batch processing for multiple job postings

### Phase 7: UI/Frontend
- React web interface
- Upload job posting PDF
- View rankings + recommendations
- Export results as PDF/CSV

### Phase 8: Deployment
- Docker containerization
- Cloud deployment (AWS/GCP/Azure)
- Database storage (PostgreSQL)
- Caching (Redis)

---

## Files

- **phase5_matching_engine.py** - Main script (200 lines)
- **prepare_phase4_for_phase5.py** - Vectorizer preparation (150 lines)
- **PHASE5_README.md** - This file
- **data/output/phase5/** - All outputs

---

## Author

ThunderLord  
AI Resume Analyzer - Phase 5  
July 15, 2026

---

## License

Private project. For AI Resume Analyzer use only.
