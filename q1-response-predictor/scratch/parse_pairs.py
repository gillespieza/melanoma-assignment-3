with open("scratch/extracted_pages.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for the sentence detailing the pairs
# Usually they are listed in a table or figure caption.
print("--- SEARCH FOR PAIRS ---")
lines = text.split("\n")
for i, line in enumerate(lines):
    if any(p in line for p in ["Fig. 3", "Fig. 1", "IMPRES features", "pairwise features"]):
        # print 5 lines before and after
        start = max(0, i - 4)
        end = min(len(lines), i + 6)
        print(f"\n--- Context around line {i+1} ---")
        for j in range(start, end):
            print(f"{j+1}: {lines[j]}")
