"""
Semi-Automated Entity Extraction for Resume NER
Phase 2.5: Expand from 203 → 500 labeled resumes

Strategy:
1. Auto-extract SKILL entities using tech keyword lists
2. Auto-extract DEGREE using regex patterns
3. Auto-extract EXPERIENCE_DURATION using date/year patterns
4. Manual extraction for: JOB_TITLE, INSTITUTION, CERTIFICATION
5. Output: CSV for review + labeling
"""

import re
import csv
import json
from collections import defaultdict

# ============================================================================
# REGEX PATTERNS & KEYWORD LISTS
# ============================================================================

# Common tech skills (auto-extract)
TECH_SKILLS = {
    # Languages
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'go', 'rust', 
    'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css',
    'perl', 'groovy', 'haskell', 'clojure', 'elixir', 'erlang',
    
    # Frameworks & Libraries
    'django', 'flask', 'fastapi', 'react', 'vue', 'angular', 'next.js', 'svelte',
    'spring', 'springboot', 'hibernate', 'express', 'node.js', 'nodejs', 'laravel',
    'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
    'matplotlib', 'seaborn', 'plotly', 'bokeh', 'dash',
    
    # Databases
    'mysql', 'postgresql', 'mongodb', 'cassandra', 'redis', 'elasticsearch',
    'dynamodb', 'firestore', 'oracle', 'sqlite', 'mariadb', 'neo4j',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins',
    'gitlab', 'github', 'bitbucket', 'terraform', 'ansible', 'vagrant',
    'cloudformation', 'openstack', 'heroku',
    
    # Data & Big Data
    'hadoop', 'spark', 'hive', 'pig', 'kafka', 'airflow', 'etl', 'bigquery',
    'snowflake', 'redshift', 'dbt', 'databricks',
    
    # ML/AI
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'cv',
    'bert', 'gpt', 'transformers', 'lstm', 'cnn', 'rnn', 'gan',
    
    # Other Tools
    'git', 'jira', 'confluence', 'slack', 'linux', 'unix', 'windows',
    'macos', 'rest api', 'graphql', 'soap', 'grpc', 'agile', 'scrum',
    'junit', 'pytest', 'mocha', 'jest', 'selenium', 'cypress',
}

# Degree patterns (auto-extract)
DEGREE_PATTERNS = [
    r'\bb\.?(?:tech|a|s|sc|com|tech|eng)\.?\b',  # B.Tech, B.A, B.Sc, etc.
    r'\bm\.?(?:tech|a|s|sc|com|ba|btech|eng)\.?\b',  # M.Tech, M.A, M.Sc, etc.
    r'\bphd\.?\b',
    r'\bm\.?\.?b\.?\.?a\.?\b',  # MBA
    r'\bbachelor(?:\'s)?(?:\s+(?:of|in))?\b',
    r'\bmaster(?:\'s)?(?:\s+(?:of|in))?\b',
    r'\bpostgraduate\b',
    r'\bdiplomate?\b',
    r'\bcertificate?\b',
]

# Experience duration patterns (auto-extract)
DURATION_PATTERNS = [
    r'\b\d{4}\s*[-–]\s*\d{4}\b',  # 2019-2023
    r'\b\d{1,2}\s*(?:years?|yrs?|months?|mons?)\b',  # 5 years, 3 months
    r'\b(?:present|current|ongoing)\b',
]

# ============================================================================
# AUTO-EXTRACTION FUNCTIONS
# ============================================================================

def extract_skills(text):
    """Extract SKILL entities using keyword matching (case-insensitive)"""
    text_lower = text.lower()
    found_skills = set()
    
    for skill in TECH_SKILLS:
        # Word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill.title())
    
    return '|'.join(sorted(found_skills)) if found_skills else ''

def extract_degrees(text):
    """Extract DEGREE entities using regex patterns"""
    found_degrees = set()
    
    for pattern in DEGREE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            degree = match.group(0).strip()
            found_degrees.add(degree)
    
    return '|'.join(sorted(found_degrees)) if found_degrees else ''

def extract_experience_duration(text):
    """Extract EXPERIENCE_DURATION entities using regex patterns"""
    found_durations = set()
    
    for pattern in DURATION_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            duration = match.group(0).strip()
            found_durations.add(duration)
    
    return '|'.join(sorted(found_durations)) if found_durations else ''

# ============================================================================
# SAMPLE RESUME DATA (For demonstration — replace with your actual data)
# ============================================================================

SAMPLE_RESUMES = [
    {
        "id": "resume_001",
        "text": """
        John Doe
        Senior Software Engineer | 5 years experience
        
        SKILLS: Python, Java, AWS, Machine Learning, Docker, PostgreSQL
        EDUCATION: B.Tech Computer Science, IIT Bombay (2019-2023)
        CERTIFICATIONS: AWS Certified Solutions Architect, Google Cloud Associate
        
        EXPERIENCE:
        Tech Company (2021-2023) - Senior Engineer
        Previous Corp (2019-2021) - Junior Developer
        """
    },
    {
        "id": "resume_002",
        "text": """
        Jane Smith
        Data Scientist | 3 years
        
        SKILLS: Python, R, TensorFlow, Spark, SQL, Tableau
        EDUCATION: Master's in Data Science, Stanford University (2021)
        WORK: Analytics Firm (2021-Present), StartupXYZ (2018-2021)
        """
    },
    {
        "id": "resume_003",
        "text": """
        Bob Johnson
        Full Stack Developer
        
        SKILLS: JavaScript, React, Node.js, MongoDB, Git, Linux
        DEGREE: B.Sc Computer Science, University of Toronto (2020)
        CERTS: AWS Certified Developer Associate
        EXP: 2 years at TechCorp, 1 year freelance
        """
    }
]

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_resumes_for_labeling(resumes, output_file='resumes_for_review.csv'):
    """
    Process resumes with semi-automated extraction.
    Output CSV with auto-extracted entities for manual review.
    """
    
    results = []
    
    for resume in resumes:
        resume_id = resume['id']
        text = resume['text']
        
        # Auto-extract entities
        skills = extract_skills(text)
        degrees = extract_degrees(text)
        durations = extract_experience_duration(text)
        
        # Manual extraction placeholders (user must fill in)
        job_title = ''  # TO BE FILLED MANUALLY
        institution = ''  # TO BE FILLED MANUALLY
        certification = ''  # TO BE FILLED MANUALLY
        
        results.append({
            'resume_id': resume_id,
            'SKILL': skills,
            'DEGREE': degrees,
            'EXPERIENCE_DURATION': durations,
            'JOB_TITLE': job_title,  # [MANUAL]
            'INSTITUTION': institution,  # [MANUAL]
            'CERTIFICATION': certification,  # [MANUAL]
            'review_notes': 'CHECK auto-extracted entities | FILL manual fields'
        })
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'resume_id', 'SKILL', 'DEGREE', 'EXPERIENCE_DURATION',
            'JOB_TITLE', 'INSTITUTION', 'CERTIFICATION', 'review_notes'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Processed {len(results)} resumes")
    print(f"✓ Output saved to: {output_file}")
    print(f"\nNext steps:")
    print("1. Review auto-extracted entities (SKILL, DEGREE, EXPERIENCE_DURATION)")
    print("2. Fix any errors or false positives")
    print("3. Fill in manual fields (JOB_TITLE, INSTITUTION, CERTIFICATION)")
    print("4. Save and convert to spaCy format for training")
    
    return results

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SEMI-AUTOMATED ENTITY EXTRACTION FOR RESUME NER")
    print("=" * 70)
    
    # Process sample resumes
    results = process_resumes_for_labeling(SAMPLE_RESUMES, output_file='resumes_for_review.csv')
    
    # Print sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (First resume):")
    print("=" * 70)
    for key, value in results[0].items():
        print(f"{key:25} : {value}")
