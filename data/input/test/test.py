import spacy
nlp = spacy.load("models/ner_model_v2")
text = "Experienced with GAAP, IFRS, SAP, Oracle ERP, Financial analysis, Tax compliance"
doc = nlp(text)
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")