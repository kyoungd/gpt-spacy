from fuzzywuzzy import fuzz, process

sentence1 = "South Sutton Street"
available_addresses = [
    "14810 Sutton St",
    "594 Maple St",
    "11978 Mariposa Bay Ln"
]

# Use the process.extractOne() method to find the most similar string
best_match = process.extractOne(sentence1, available_addresses)
print(f"Best match: {best_match[0]}, Score: {best_match[1]}")

