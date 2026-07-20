#!/usr/bin/env python3
"""
Create test job posting PDF for Phase 5
Save this file as 'create_pdf.py' in your project root
Then run: python create_pdf.py
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
from pathlib import Path

output_path = Path("data/input/test_job_posting.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Creating PDF at: {output_path}")

doc = SimpleDocTemplate(str(output_path), pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=18,
    spaceAfter=0.2*inch,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

heading_style = ParagraphStyle(
    'CustomHeading',
    parent=styles['Heading2'],
    fontSize=12,
    spaceAfter=0.1*inch,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontSize=10,
    spaceAfter=0.05*inch,
)

content = [
    Paragraph("SENIOR PYTHON ENGINEER - SAN FRANCISCO, CA", title_style),
    Spacer(1, 0.1*inch),
    Paragraph("<b>Job Title:</b> Senior Python Engineer", body_style),
    Paragraph("<b>Company:</b> TechCorp Inc.", body_style),
    Paragraph("<b>Location:</b> San Francisco, CA", body_style),
    Paragraph("<b>Employment Type:</b> Full-Time", body_style),
    Paragraph("<b>Salary:</b> $180,000 - $220,000", body_style),
    Spacer(1, 0.1*inch),
]

content.append(Paragraph("JOB DESCRIPTION", heading_style))
content.append(Paragraph(
    "We are looking for an experienced Senior Python Engineer to lead our backend "
    "infrastructure and cloud platform initiatives.",
    body_style
))
content.append(Spacer(1, 0.05*inch))

content.append(Paragraph("REQUIREMENTS", heading_style))
content.append(Paragraph("<b>Required Skills & Experience:</b>", body_style))

requirements = [
    "7+ years of professional Python development experience",
    "Expert-level proficiency in Python, particularly with async frameworks",
    "Deep knowledge of AWS (EC2, S3, Lambda, RDS, CloudFormation)",
    "Strong expertise in Docker and Kubernetes containerization",
    "Experience with CI/CD pipelines (Jenkins, GitLab CI, GitHub Actions)",
    "SQL and NoSQL database design and optimization",
    "RESTful API design and implementation",
    "Microservices architecture",
    "Git version control",
    "Linux/Unix system administration"
]

for req in requirements:
    content.append(Paragraph(f"• {req}", body_style))

content.append(Spacer(1, 0.05*inch))
content.append(Paragraph("<b>Education:</b>", body_style))
content.append(Paragraph("• Bachelor's degree in Computer Science or related field", body_style))
content.append(Paragraph("• Master's degree preferred", body_style))

content.append(Spacer(1, 0.05*inch))
content.append(Paragraph("<b>Certifications:</b>", body_style))
content.append(Paragraph("• AWS Solutions Architect Associate or Professional certification preferred", body_style))
content.append(Paragraph("• Kubernetes CKA certification a plus", body_style))

content.append(Spacer(1, 0.1*inch))
content.append(Paragraph("RESPONSIBILITIES", heading_style))

responsibilities = [
    "Design and implement scalable backend systems",
    "Lead architectural decisions and code reviews",
    "Mentor junior software engineers",
    "Optimize system performance and reliability",
    "Collaborate with DevOps team on infrastructure automation",
    "Participate in on-call rotation for production support"
]

for resp in responsibilities:
    content.append(Paragraph(f"• {resp}", body_style))

doc.build(content)
print(f"✓ PDF created successfully!")
print(f"✓ File exists: {output_path.exists()}")
print(f"\nNext step, run Phase 5:")
print(f'  python phase5_matching_engine.py "data\\input\\test_job_posting.pdf"')
