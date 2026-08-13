import spacy
from spacy.pipeline import EntityRuler
import shutil
from pathlib import Path

def enhance_model():
    print("Loading original model...")
    nlp = spacy.load("models/ner_model_v2")
    
    # Add EntityRuler
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Massive comprehensive dictionary of skills across 24 professions
    skills = [
        # ACCOUNTANT / FINANCE / BANKING
        "GAAP", "Tax", "Audit", "Accounts Payable", "Accounts Receivable", "Financial Reporting", 
        "Reconciliation", "Payroll", "Accounting", "Financial Analysis", "Balance Sheets", 
        "Tax Returns", "Ledger", "Invoicing", "Budgeting", "Forecasting", "Risk Management",
        "Wealth Management", "Credit Analysis", "Investment Banking", "Corporate Finance",
        "Financial Modeling", "Portfolio Management", "SEC Filings", "Internal Controls",
        "QuickBooks", "SAP", "Oracle Financials", "Excel", "VLOOKUP", "Pivot Tables",
        
        # ADVOCATE / LEGAL
        "Litigation", "Legal Research", "Contract Drafting", "Mediation", "Corporate Law",
        "Intellectual Property", "Compliance", "Legal Briefs", "Depositions", "Trial Preparation",
        "Case Management", "Employment Law", "Family Law", "Real Estate Law", "Torts",
        "LexisNexis", "Westlaw", "Drafting", "Negotiation", "Due Diligence",
        
        # CHEF / FOOD / AVIATION / HOSPITALITY
        "Culinary Arts", "Food Safety", "Menu Planning", "Baking", "Pastry", "Grilling",
        "HACCP", "Inventory Management", "Food Preparation", "Catering", "Kitchen Management",
        "Recipe Development", "Sanitation", "Sous Chef", "Fine Dining", "Customer Service",
        "Aviation", "Flight Operations", "Cabin Crew", "Safety Procedures", "First Aid",
        "Aircraft Maintenance", "Logistics", "Scheduling", "Hospitality Management",
        
        # HEALTHCARE / FITNESS
        "Patient Care", "EMR", "EHR", "Clinical Research", "Nursing", "Vital Signs",
        "CPR", "BLS", "ACLS", "HIPAA", "Medical Billing", "Phlebotomy", "Triage",
        "Personal Training", "Nutrition", "Kinesiology", "Strength Training", "Yoga",
        "Rehabilitation", "Sports Medicine", "Group Fitness", "Wellness Coaching",
        
        # ENGINEERING / CONSTRUCTION / AUTOMOBILE
        "AutoCAD", "SolidWorks", "MATLAB", "CAD", "Structural Analysis", "MEP",
        "Civil Engineering", "Mechanical Engineering", "Electrical Engineering",
        "Blueprint Reading", "Project Estimation", "Site Supervision", "OSHA",
        "Safety Compliance", "Quality Control", "Lean Manufacturing", "Six Sigma",
        "HVAC", "Plumbing", "Welding", "Machining", "Automotive Repair", "Diagnostics",
        
        # AGRICULTURE
        "Crop Management", "Irrigation", "Pest Control", "Soil Analysis", "Agribusiness",
        "Harvesting", "Greenhouse Management", "Horticulture", "Animal Husbandry",
        "Farm Machinery", "Sustainable Farming", "Agronomy", "Fertilization",
        
        # HR / SALES / BUSINESS DEVELOPMENT / BPO / PUBLIC RELATIONS
        "Recruitment", "Onboarding", "Employee Relations", "Performance Management",
        "Talent Acquisition", "HRIS", "Compensation", "Benefits Administration",
        "Salesforce", "B2B Sales", "B2C Sales", "Lead Generation", "Cold Calling",
        "Account Management", "CRM", "Business Strategy", "Market Analysis",
        "Client Retention", "Telemarketing", "Customer Support", "SLA", "Zendesk",
        "Press Releases", "Media Relations", "Crisis Management", "Brand Management",
        
        # DESIGNER / ARTS / APPAREL
        "Adobe Creative Suite", "Photoshop", "Illustrator", "InDesign", "UI/UX",
        "Figma", "Sketch", "Graphic Design", "Typography", "Color Theory",
        "Fashion Design", "Pattern Making", "Textiles", "Merchandising", "Styling",
        "Sewing", "Visual Merchandising", "Fine Arts", "Illustration", "Photography",
        
        # DIGITAL MEDIA / TEACHER / CONSULTANT
        "SEO", "SEM", "Content Marketing", "Social Media Management", "Google Analytics",
        "Copywriting", "Video Editing", "Premiere Pro", "After Effects", "Digital Marketing",
        "Curriculum Development", "Lesson Planning", "Classroom Management", "Special Education",
        "E-learning", "Instructional Design", "Tutoring", "Student Assessment",
        "Management Consulting", "Change Management", "Process Improvement", "Strategic Planning",
        
        # INFORMATION TECHNOLOGY (Keeping original good IT skills)
        "Python", "Java", "JavaScript", "HTML", "CSS", "SQL", "AWS", "Agile", "Scrum",
        "Linux", "Machine Learning", "Data Analysis", "React", "Node.js", "Docker",
        "Kubernetes", "Git", "C++", "C#", "PHP", "Ruby", "Swift", "Azure", "GCP"
    ]
    
    # Additional generic keywords often missed by NER
    generic_skills = [
        "Communication", "Leadership", "Teamwork", "Problem Solving", "Time Management",
        "Project Management", "Data Entry", "Public Speaking", "Event Planning", "Operations"
    ]
    
    all_skills = list(set(skills + generic_skills))
    
    print(f"Adding {len(all_skills)} diverse skills to EntityRuler...")
    
    # Create case-insensitive patterns
    patterns = []
    for skill in all_skills:
        # Split into tokens for pattern matching
        tokens = skill.split()
        pattern = [{"LOWER": token.lower()} for token in tokens]
        patterns.append({"label": "SKILL", "pattern": pattern})
        
    ruler.add_patterns(patterns)
    
    # Save the enhanced model
    output_dir = "models/ner_model_v3"
    print(f"Saving enhanced model to {output_dir}...")
    
    # Make sure output directory is clean
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
        
    nlp.to_disk(output_dir)
    print("Done! Model upgraded successfully.")

if __name__ == "__main__":
    enhance_model()
