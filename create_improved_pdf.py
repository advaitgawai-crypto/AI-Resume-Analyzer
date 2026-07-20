#!/usr/bin/env python3
"""
Create an IMPROVED test job posting PDF with better structured requirements
This will extract more entities (job titles, degrees, certifications)
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from pathlib import Path

output_path = Path("data/input/senior_python_engineer.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Creating improved PDF at: {output_path}")

doc = SimpleDocTemplate(str(output_path), pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'Title', parent=styles['Heading1'], fontSize=18, spaceAfter=0.15*inch,
    alignment=TA_CENTER, fontName='Helvetica-Bold'
)
heading_style = ParagraphStyle(
    'Heading', parent=styles['Heading2'], fontSize=11, spaceAfter=0.08*inch,
    fontName='Helvetica-Bold'
)
body_style = ParagraphStyle(
    'Body', parent=styles['Normal'], fontSize=9.5, spaceAfter=0.04*inch, leading=12
)

content = []

# Title
content.append(Paragraph("Senior Python Engineer", title_style))
content.append(Paragraph("San Francisco, CA | Full-Time | $180K-$220K", body_style))
content.append(Spacer(1, 0.1*inch))

# ========================================================================
content.append(Paragraph("ABOUT THE ROLE", heading_style))
content.append(Paragraph(
    "Join our engineering team as a Senior Python Engineer. You will design and "
    "lead our backend infrastructure, mentor junior developers, and drive architectural "
    "decisions across our platform.",
    body_style
))
content.append(Spacer(1, 0.05*inch))

# ========================================================================
content.append(Paragraph("REQUIRED QUALIFICATIONS", heading_style))

content.append(Paragraph("<b>Experience & Skills:</b>", body_style))
requirements = [
    "7+ years of professional software development experience",
    "5+ years specifically with Python programming",
    "Expert proficiency in Python, especially async/await frameworks",
    "Advanced knowledge of AWS cloud services: EC2, S3, Lambda, RDS, CloudFormation",
    "Strong expertise in Docker containerization and orchestration",
    "Kubernetes administration and deployment experience",
    "CI/CD pipeline design and implementation: Jenkins, GitLab CI, GitHub Actions",
    "SQL database design and optimization: PostgreSQL, MySQL",
    "NoSQL databases: MongoDB, Redis, DynamoDB",
    "RESTful API design and implementation using FastAPI or Django",
    "Microservices architecture and design patterns",
    "Git version control and workflow management",
    "Linux/Unix system administration and shell scripting",
]
for req in requirements:
    content.append(Paragraph(f"• {req}", body_style))

content.append(Spacer(1, 0.06*inch))

content.append(Paragraph("<b>Education:</b>", body_style))
education = [
    "Bachelor of Science in Computer Science",
    "Bachelor of Science in Software Engineering",
    "Master of Science in Computer Science (preferred)",
    "Master of Business Administration (preferred)",
]
for edu in education:
    content.append(Paragraph(f"• {edu}", body_style))

content.append(Spacer(1, 0.06*inch))

content.append(Paragraph("<b>Certifications (Preferred):</b>", body_style))
certs = [
    "AWS Certified Solutions Architect - Professional",
    "AWS Certified Solutions Architect - Associate",
    "Certified Kubernetes Administrator (CKA)",
    "AWS Certified Developer - Associate",
]
for cert in certs:
    content.append(Paragraph(f"• {cert}", body_style))

content.append(Spacer(1, 0.08*inch))

# ========================================================================
content.append(Paragraph("KEY RESPONSIBILITIES", heading_style))

responsibilities = [
    "Lead design and implementation of scalable backend systems",
    "Drive architectural decisions and lead code reviews",
    "Mentor and develop junior engineers on the team",
    "Optimize system performance, reliability, and cost efficiency",
    "Collaborate with DevOps engineers on infrastructure automation",
    "Participate in on-call rotation for production support",
    "Champion best practices and engineering standards",
]
for resp in responsibilities:
    content.append(Paragraph(f"• {resp}", body_style))

content.append(Spacer(1, 0.08*inch))

# ========================================================================
content.append(Paragraph("NICE TO HAVE", heading_style))

nice_to_have = [
    "Experience with data engineering or machine learning pipelines",
    "Contributions to open source projects",
    "Experience with Terraform or Infrastructure as Code",
    "Knowledge of system design and scalability",
    "Experience mentoring or leading engineering teams",
]
for item in nice_to_have:
    content.append(Paragraph(f"• {item}", body_style))

content.append(Spacer(1, 0.08*inch))

# ========================================================================
content.append(Paragraph("BENEFITS & PERKS", heading_style))

benefits = [
    "Competitive salary: $180,000 - $220,000 per year",
    "Equity options with 4-year vesting",
    "Flexible remote work policy (40% remote, 60% in-office)",
    "Comprehensive health insurance: medical, dental, vision, life",
    "401(k) matching up to 5%",
    "Generous PTO: 20 days vacation, 10 holidays, unlimited sick days",
    "Professional development budget: $2,000 per year",
    "Learning stipend for courses and certifications",
    "Home office equipment stipend",
]
for benefit in benefits:
    content.append(Paragraph(f"• {benefit}", body_style))

doc.build(content)
print(f"✓ Improved PDF created successfully!")
print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")
print(f"\nNow run Phase 5 with the improved PDF:")
print(f'  python phase5_matching_engine.py "data\\input\\senior_python_engineer.pdf"')
