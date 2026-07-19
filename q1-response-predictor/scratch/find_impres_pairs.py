import pypdf
import sys

# Configure output encoding to prevent Unicode errors
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = "../docs/literature/Auslander 2018.pdf"
reader = pypdf.PdfReader(pdf_path)

with open("scratch/extracted_pages.txt", "w", encoding="utf-8") as out:
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        out.write(f"=== PAGE {i+1} ===\n")
        out.write(text)
        out.write("\n\n")

print("PDF text extracted successfully to scratch/extracted_pages.txt")
