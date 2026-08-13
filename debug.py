import pdfplumber
import spacy

print("=" * 60)
print("DIAGNOSTIC 1: PDF TEXT EXTRACTION")
print("=" * 60)
try:
    text = pdfplumber.open("01_ACCOUNTANT.pdf").pages[0].extract_text()
    print(text)
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC 2: NER MODEL SKILL EXTRACTION")
print("=" * 60)
try:
    nlp = spacy.load("models/ner_model_v2")
    test_text = "Experienced with GAAP, IFRS, SAP, Oracle ERP, Financial analysis, Tax compliance"
    doc = nlp(test_text)
    print(f"Test text: {test_text}\n")
    print("Extracted entities:")
    if len(doc.ents) == 0:
        print("  (No entities found!)")
    else:
        for ent in doc.ents:
            print(f"  {ent.text} -> {ent.label_}")
except Exception as e:
    print(f"ERROR: {e}")

test_text2 = "I have experience with Python, Java, and Machine Learning"
doc2 = nlp(test_text2)
print(f"\nTest text 2: {test_text2}")
print("Extracted entities:")
if len(doc2.ents) == 0:
    print("  (No entities found!)")
else:
    for ent in doc2.ents:
        print(f"  {ent.text} -> {ent.label_}")