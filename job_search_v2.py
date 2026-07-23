#!/usr/bin/env python3
"""
================================================================================
JOB SEARCH - Find jobs matching extracted skills (International)
================================================================================
Usage:
  python job_search.py <location> <country_code>

Examples:
  python job_search.py "Bangalore" "in"      # India
  python job_search.py "New York" "us"       # USA
  python job_search.py "London" "gb"         # UK
  python job_search.py "Toronto" "ca"        # Canada
  python job_search.py "Sydney" "au"         # Australia
  python job_search.py "Berlin" "de"         # Germany
  python job_search.py "Paris" "fr"          # France

This script:
1. Reads the most recent improvement_recommendations.csv (from resume_analyzer.py)
2. Extracts required skills from the report
3. Searches Adzuna API for matching jobs (international)
4. Ranks jobs by skill match
5. Displays top jobs with apply links
================================================================================
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(r"C:\Users\Advait Gawai\OneDrive\Desktop\AI resume Analyzer")
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "phase5"

# Adzuna API credentials
ADZUNA_APP_ID = "aacca97d"
ADZUNA_APP_KEY = "315d396533cd22919c1d827810601d77"

# Supported countries
COUNTRY_MAP = {
    "in": "India",
    "us": "USA",
    "gb": "UK",
    "ca": "Canada",
    "au": "Australia",
    "de": "Germany",
    "fr": "France",
    "nl": "Netherlands",
    "it": "Italy",
    "es": "Spain",
    "se": "Sweden",
    "nz": "New Zealand",
    "sg": "Singapore",
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def get_latest_recommendations_file():
    """Get the most recent improvement_recommendations.csv file"""
    try:
        csv_files = sorted(OUTPUT_DIR.glob("improvement_recommendations_*.csv"), 
                          key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not csv_files:
            print(f"✗ No recommendation files found in {OUTPUT_DIR}")
            return None
        
        latest_file = csv_files[0]
        print(f"✓ Found latest report: {latest_file.name}")
        return latest_file
    
    except Exception as e:
        print(f"✗ Error finding report: {e}")
        return None


def extract_skills_from_report(csv_file):
    """Extract skills from the improvement_recommendations.csv"""
    try:
        df = pd.read_csv(csv_file)
        
        # Get required skills (Section 1)
        required_skills_rows = df[df['section'] == '1. REQUIRED SKILLS FOR THIS ROLE']
        if not required_skills_rows.empty:
            skills_str = required_skills_rows.iloc[0]['item']
            required_skills = [s.strip() for s in skills_str.split('|')]
        else:
            required_skills = []
        
        print(f"\n✓ Required skills for this job:")
        for skill in required_skills:
            print(f"  • {skill}")
        
        return required_skills
    
    except Exception as e:
        print(f"✗ Error extracting skills: {e}")
        return []


def search_jobs_adzuna(keywords, location, country_code="in", results_per_page=30):
    """
    Search for jobs using Adzuna API (International)
    
    Args:
        keywords: List of skills to search for
        location: Job location (e.g., "Bangalore")
        country_code: Country code (in, us, gb, ca, au, de, fr, nl, it, es, se, nz, sg)
        results_per_page: Number of results (1-50)
    
    Returns:
        List of job dictionaries
    """
    
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code.lower()}/search/1"
    query = " ".join(keywords)
    country_name = COUNTRY_MAP.get(country_code.lower(), "Unknown")
    
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "what": query,
        "where": location,
        "results_per_page": results_per_page
    }
    
    try:
        print(f"\n🔍 Searching Adzuna for: '{query}' in {location}, {country_name}...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'results' not in data:
            print(f"✗ No results in API response")
            return []
        
        jobs = []
        for job in data['results']:
            salary_currency = job.get('salary_currency_code', 'USD')
            jobs.append({
                'job_title': job.get('title', ''),
                'company': job.get('company', {}).get('display_name', ''),
                'location': job.get('location', {}).get('display_name', ''),
                'salary_min': job.get('salary_min', None),
                'salary_max': job.get('salary_max', None),
                'salary_currency': salary_currency,
                'description': job.get('description', ''),
                'posting_date': job.get('created', ''),
                'application_url': job.get('redirect_url', '')
            })
        
        print(f"✓ Found {len(jobs)} jobs on Adzuna ({country_name})")
        return jobs
    
    except requests.exceptions.Timeout:
        print("✗ Request timeout - Adzuna server not responding")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP Error: {response.status_code} - {e}")
        return []
    except Exception as e:
        print(f"✗ Error searching jobs: {e}")
        return []


def score_jobs(jobs, required_skills):
    """
    Score each job based on skill match
    
    Args:
        jobs: List of job dictionaries
        required_skills: List of skills required for the job
    
    Returns:
        List of jobs with match scores
    """
    
    print(f"\n📊 Scoring {len(jobs)} jobs based on skill match...")
    
    for job in jobs:
        description_lower = job['description'].lower()
        match_count = 0
        
        # Count how many required skills appear in job description
        for skill in required_skills:
            if skill.lower() in description_lower:
                match_count += 1
        
        # Calculate match percentage (0-100%)
        match_score = (match_count / max(len(required_skills), 1)) * 100
        job['match_score'] = match_score
        job['matching_skills'] = match_count
    
    return jobs


def display_jobs(jobs, top_n=15):
    """Display top matching jobs"""
    
    # Sort by match score
    jobs = sorted(jobs, key=lambda x: x['match_score'], reverse=True)
    
    print("\n" + "="*100)
    print("TOP JOBS FOR YOU")
    print("="*100 + "\n")
    
    for i, job in enumerate(jobs[:top_n], 1):
        print(f"{i}. {job['job_title']}")
        print(f"   Company: {job['company']}")
        print(f"   Location: {job['location']}")
        print(f"   Match: {job['match_score']:.0f}% ({job['matching_skills']} skills matched) ⭐")
        
        if job['salary_min'] and job['salary_max']:
            print(f"   Salary: {job['salary_currency']} {job['salary_min']:,} - {job['salary_max']:,}")
        
        print(f"   Posted: {job['posting_date']}")
        print(f"   Apply: {job['application_url']}")
        print()
    
    print("="*100)
    print(f"Displayed top {min(top_n, len(jobs))} of {len(jobs)} jobs")
    print("="*100 + "\n")
    
    return jobs[:top_n]


def save_results(jobs, location, country_code):
    """Save results to CSV"""
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country_name = COUNTRY_MAP.get(country_code.lower(), country_code)
        output_file = PROJECT_ROOT / "data" / "output" / f"job_search_results_{country_name}_{location}_{timestamp}.csv"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(jobs)
        df = df[['job_title', 'company', 'location', 'salary_min', 'salary_max', 'salary_currency',
                 'match_score', 'matching_skills', 'posting_date', 'application_url']]
        
        df.to_csv(output_file, index=False)
        print(f"✓ Results saved to: {output_file.name}")
        
        return output_file
    
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def run(location, country_code="in"):
    """Main pipeline"""
    
    print("\n" + "="*100)
    print("JOB SEARCH - Find jobs matching your skills")
    print("="*100)
    
    country_name = COUNTRY_MAP.get(country_code.lower(), country_code.upper())
    
    # Step 1: Get latest report
    print("\n[1/5] Loading your latest report...")
    recommendations_file = get_latest_recommendations_file()
    
    if not recommendations_file:
        print("✗ No report found. Run resume_analyzer.py first!")
        return
    
    # Step 2: Extract skills
    print("\n[2/5] Extracting required skills from report...")
    required_skills = extract_skills_from_report(recommendations_file)
    
    if not required_skills:
        print("✗ No skills found in report")
        return
    
    # Step 3: Search Adzuna
    print(f"\n[3/5] Searching Adzuna API...")
    jobs = search_jobs_adzuna(required_skills, location, country_code=country_code, results_per_page=30)
    
    if not jobs:
        print("✗ No jobs found")
        return
    
    # Step 4: Score jobs
    print(f"\n[4/5] Scoring jobs...")
    jobs = score_jobs(jobs, required_skills)
    
    # Step 5: Display and save
    print(f"\n[5/5] Displaying results...")
    top_jobs = display_jobs(jobs, top_n=15)
    
    # Save results
    output_file = save_results(jobs, location, country_code)
    
    print("\n" + "="*100)
    print("JOB SEARCH COMPLETE ✓")
    print("="*100 + "\n")
    
    return jobs


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python job_search.py <location> <country_code>")
        print("\nExamples:")
        print("  python job_search.py \"Bangalore\" \"in\"    # India")
        print("  python job_search.py \"New York\" \"us\"     # USA")
        print("  python job_search.py \"London\" \"gb\"       # UK")
        print("  python job_search.py \"Toronto\" \"ca\"      # Canada")
        print("  python job_search.py \"Sydney\" \"au\"       # Australia")
        print("\nSupported countries: in, us, gb, ca, au, de, fr, nl, it, es, se, nz, sg")
        sys.exit(1)
    
    location = sys.argv[1]
    country_code = sys.argv[2] if len(sys.argv) > 2 else "in"
    
    run(location, country_code)
