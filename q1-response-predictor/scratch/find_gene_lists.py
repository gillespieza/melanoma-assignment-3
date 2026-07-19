import sys

# Configure stdout encoding to prevent Unicode errors
sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/extracted_pages.txt", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split("\n")
with open("scratch/gene_contexts.txt", "w", encoding="utf-8") as out:
    for i, line in enumerate(lines):
        # Clean line to ensure it is printed cleanly
        cleaned_line = line.encode('utf-8', errors='ignore').decode('utf-8')
        if any(g in line for g in ["CD40", "TNFRSF14", "CD274", "VSIR", "CD28", "CD276", "PDCD1"]):
            out.write(f"Line {i+1}: {cleaned_line}\n")

print("Gene contexts written to scratch/gene_contexts.txt")
