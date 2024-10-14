import json
import os

class GlobalRecords:
    def __init__(self):
        self.strings = {}
        self.assets = {}

    def add_string(self, key, value):
        self.strings[key] = value

    def add_asset(self, key, value):
        self.assets[key] = value

    def to_dict(self):
        return {
            "strings": self.strings,
            "assets": self.assets
        }

    def from_dict(self, data):
        self.strings = data.get("strings", {})
        self.assets = data.get("assets", {})

global_records = GlobalRecords()

def create_string():
    key = input("Enter string key: ")
    value = input(f"Enter value for {key}: ")
    global_records.add_string(key, value)
    return key, value

def create_asset(subjects):
    asset_key = input("Enter asset key (variable name for this asset): ")
    
    asset_types = ['Service', 'Property']
    service_subtypes = ['+', '-']
    property_subtypes = ['M', 'NM']

    asset_type = input(f"Enter asset type ({'/'.join(asset_types)}): ")
    while asset_type not in asset_types:
        print("Invalid asset type. Please try again.")
        asset_type = input(f"Enter asset type ({'/'.join(asset_types)}): ")

    if asset_type == 'Service':
        subtype = input(f"Enter service subtype ({'/'.join(service_subtypes)}): ")
        while subtype not in service_subtypes:
            print("Invalid subtype. Please try again.")
            subtype = input(f"Enter service subtype ({'/'.join(service_subtypes)}): ")
    else:  # Property
        subtype = input(f"Enter property subtype ({'/'.join(property_subtypes)}): ")
        while subtype not in property_subtypes:
            print("Invalid subtype. Please try again.")
            subtype = input(f"Enter property subtype ({'/'.join(property_subtypes)}): ")

    print("Available subjects:", ", ".join(subjects))
    holder = input("Enter holder (must be a predefined subject): ")
    while holder not in subjects:
        print("Invalid subject. Please try again.")
        holder = input("Enter holder (must be a predefined subject): ")

    print("Available strings:", ", ".join(global_records.strings.keys()))
    string_key = input("Enter string key (must be a previously defined string): ")
    while string_key not in global_records.strings:
        print("Invalid string key. Please try again.")
        string_key = input("Enter string key (must be a previously defined string): ")

    bearer = input("Enter bearer (must be a predefined subject): ")
    while bearer not in subjects:
        print("Invalid subject. Please try again.")
        bearer = input("Enter bearer (must be a predefined subject): ")

    asset_value = f"asset {asset_key} = {asset_type}, {subtype}, {holder}, {string_key}, {bearer};"
    global_records.add_asset(asset_key, asset_value)
    return asset_key, asset_value

def create_clause():
    clause_key = input("Enter clause key (variable name for this clause): ")
    
    print("Available assets:", ", ".join(global_records.assets.keys()))
    condition = input("Enter condition (single asset or two assets with AND): ")
    # Basic validation for condition
    condition_parts = condition.split(' AND ')
    if len(condition_parts) > 2 or not all(part in global_records.assets for part in condition_parts):
        print("Invalid condition. Please use valid asset keys and AND operator if needed.")
        return None

    operators = ['PR', 'OB', 'CR', 'PVG']
    print(f"Available operators: {', '.join(operators)}")
    operator = input("Enter legal operator: ")
    while operator not in operators:
        print("Invalid operator. Please try again.")
        operator = input("Enter legal operator: ")

    operated_asset = input("Enter operated asset key (must be a previously defined asset): ")
    while operated_asset not in global_records.assets:
        print("Invalid asset key. Please try again.")
        operated_asset = input("Enter operated asset key (must be a previously defined asset): ")

    return clause_key, f"clause {clause_key} = {{{condition},{operator}({operated_asset})}};"

def create_kelsen_translation(subjects):
    translation = {
        "article_number": input("Enter article number: "),
        "original": input("Enter original text: "),
        "strings": {},
        "assets": {},
        "clauses": {}
    }

    # Strings
    while True:
        key, value = create_string()
        translation["strings"][key] = value
        if input("Add another string? (y/n): ").lower() != 'y':
            break

    # Assets
    while True:
        asset_key, asset = create_asset(subjects)
        translation["assets"][asset_key] = asset
        if input("Add another asset? (y/n): ").lower() != 'y':
            break

    # Clauses
    while True:
        clause_result = create_clause()
        if clause_result:
            clause_key, clause = clause_result
            translation["clauses"][clause_key] = clause
        if input("Add another clause? (y/n): ").lower() != 'y':
            break

    return translation

def load_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the data is in the new format (dict with 'global_records' and 'translations')
        if isinstance(data, dict) and "global_records" in data and "translations" in data:
            global_records.from_dict(data.get("global_records", {}))
            return data.get("translations", [])
        
        # If it's not in the new format, assume it's the old format (list of translations)
        elif isinstance(data, list):
            # Populate global records from the existing translations
            for translation in data:
                global_records.strings.update(translation.get('strings', {}))
                global_records.assets.update(translation.get('assets', {}))
            return data
        
        # If it's neither, return an empty list
        else:
            print("Warning: Unrecognized data format in the JSON file. Starting with empty translations.")
            return []
    
    # If the file doesn't exist, return an empty list
    return []

# Update the save_data function to always use the new format
def save_data(translations, filename):
    data = {
        "global_records": global_records.to_dict(),
        "translations": translations
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
def main():
    filename = "kelsen_translations.json"
    translations = load_data(filename)

    # Predefined subjects (in a real scenario, these might be loaded from a file)
    subjects = ["COMPRADOR", "VENDEDOR", "DEUDOR", "ACREEDOR", "OFERENTE", "MUTUANTE", "MUTUARIO", "PROPIETARIO", "ARRENDADOR", "ARRENDATARIO", "PRESTADOR", "ACREDITADO"]

    while True:
        print("\n1. Add new translation")
        print("2. View all translations")
        print("3. Edit a translation")
        print("4. Delete a translation")
        print("5. View global records")
        print("6. Save and exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            new_translation = create_kelsen_translation(subjects)
            translations.append(new_translation)
            print("Translation added successfully.")

        elif choice == '2':
            for t in translations:
                print(f"\nArtículo {t['article_number']}:")
                print(f"Original: {t['original']}")
                print("Strings:", t['strings'])
                print("Assets:", t['assets'])
                print("Clauses:", t['clauses'])

        elif choice == '3':
            article_number = input("Enter article number to edit: ")
            for i, t in enumerate(translations):
                if t['article_number'] == article_number:
                    print("Current translation:")
                    print(json.dumps(t, indent=2))
                    new_translation = create_kelsen_translation(subjects)
                    translations[i] = new_translation
                    print("Translation updated successfully.")
                    break
            else:
                print("Article not found.")

        elif choice == '4':
            article_number = input("Enter article number to delete: ")
            translations = [t for t in translations if t['article_number'] != article_number]
            print("Translation deleted if it existed.")

        elif choice == '5':
            print("\nGlobal Records:")
            print("Strings:", global_records.strings)
            print("Assets:", global_records.assets)

        elif choice == '6':
            save_data(translations, filename)
            print(f"Data saved to {filename}")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
