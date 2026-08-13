import spacy
from spacy.pipeline import EntityRuler
import pdfplumber

def test_improved_ner():
    nlp = spacy.load('models/ner_model_v2')
    
    # Add EntityRuler for skills
    if not nlp.has_pipe("entity_ruler"):
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")
        
    skills = [
        "GAAP", "Tax", "Audit", "Accounts Payable", "Accounts Receivable", 
        "QuickBooks", "Financial Reporting", "Reconciliation", "Payroll",
        "Accounting", "Financial Analysis", "Excel", "Balance Sheets"
    ]
    
    patterns = [{"label": "SKILL", "pattern": [{"LOWER": word.lower()} for word in skill.split()]} for skill in skills]
    ruler.add_patterns(patterns)
    
    # Test on PDF
    text = ""
    with pdfplumber.open('01_ACCOUNTANT.pdf') as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
            
    doc = nlp(text)
    extracted_skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    print("Extracted Skills:", set(extracted_skills))

if __name__ == "__main__":
    test_improved_ner()
