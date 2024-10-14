import json

# Load the JSON data
with open("augmented_kelsen_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize empty lists for each dataset
strings_dataset = []
assets_dataset = []
clauses_dataset = []

# Loop over each entry in the translations
for entry in data['translations']:
    # Always include the article number, original, input text, and target text
    base_entry = {
        'article_number': entry['article_number'],
        'original': entry['original']
    }

    # If strings are present, create a string-only version of the entry
    if 'strings' in entry and entry['strings']:
        string_entry = base_entry.copy()  # Copy the base entry
        string_entry['strings'] = entry['strings']
        strings_dataset.append(string_entry)
        
    # If assets are present, create an asset-only version of the entry
    if 'assets' in entry and entry['assets']:
        asset_entry = base_entry.copy()  # Copy the base entry
        asset_entry['assets'] = entry['assets']
        assets_dataset.append(asset_entry)
        
    # If clauses are present, create a clause-only version of the entry
    if 'clauses' in entry and entry['clauses']:
        clause_entry = base_entry.copy()  # Copy the base entry
        clause_entry['clauses'] = entry['clauses']
        clauses_dataset.append(clause_entry)

# Write the datasets into new files
with open("./kelsen_strings_only_dataset.json", "w", encoding="utf-8") as f:
    json.dump(strings_dataset, f, ensure_ascii=False, indent=4)

with open("./kelsen_assets_only_dataset.json", "w", encoding="utf-8") as f:
    json.dump(assets_dataset, f, ensure_ascii=False, indent=4)

with open("./kelsen_clauses_only_dataset.json", "w", encoding="utf-8") as f:
    json.dump(clauses_dataset, f, ensure_ascii=False, indent=4)

print("Datasets successfully split into strings, assets, and clauses with article_number, original, input_text, and target_text.")

