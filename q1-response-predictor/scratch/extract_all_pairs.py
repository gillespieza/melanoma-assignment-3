import re

with open("scratch/extracted_pages.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Find patterns like Gene/Gene
pattern = r"\b[A-Z0-9a-z\-]+/[A-Z0-9a-z\-]+\b"
matches = re.findall(pattern, text)

# Print unique matches sorted
unique_matches = sorted(list(set(matches)))
print("--- ALL GENE PAIRS FOUND IN PDF ---")
for m in unique_matches:
    print(m)
