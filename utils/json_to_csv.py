import csv
import json

# Define a function to convert JSON to CSV
def json_to_csv(json_file_path, csv_file_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    
    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(['input_text', 'target_text'])
        
        # Loop through each entry in the JSON data
        for entry in json_data:
            article_number = entry["article_number"]
            original_text = entry["original"]
            strings = entry["strings"]
            assets = entry["assets"]
            clauses = entry["clauses"]
            
            # Build the input_text
            input_text = f'Artículo {article_number} - {original_text}'
            
            # Build the target_text
            target_strings = ' '.join([f'string {key} = "{value}"' for key, value in strings.items()])
            target_assets = ' '.join([f'{value}' for key, value in assets.items()])
            target_clauses = ' '.join([f'{value}' for key, value in clauses.items()])
            
            # Combine all parts into the final target_text
            target_text = f'{target_strings} {target_assets} {target_clauses}'
            
            # Ensure proper formatting of quotation marks
            target_text = target_text.replace('""', '"')
            
            # Write the formatted data to CSV
            writer.writerow([input_text, target_text])

# File paths
json_file_path = 'augmented_kelsen_dataset.json'
csv_file_path = 'kelsen_data.csv'

# Convert the JSON file to CSV
json_to_csv(json_file_path, csv_file_path)


# Define a function to convert JSON with only strings to CSV
def json_to_csv_strings_only(json_file_path, csv_file_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    
    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(['input_text', 'target_text'])
        
        # Loop through each entry in the JSON data
        for entry in json_data:
            article_number = entry["article_number"]
            original_text = entry["original"]
            strings = entry["strings"]
            
            # Build the input_text
            input_text = f'Artículo {article_number} - {original_text}'
            
            # Build the target_text (only strings)
            target_strings = ' '.join([f'string {key} = "{value}"' for key, value in strings.items()])
            
            # Ensure proper formatting of quotation marks
            target_strings = target_strings.replace('""', '"')
            

            # Write the formatted data to CSV
            writer.writerow([input_text, target_strings])

# File paths
json_file_path = 'augmented_kelsen_dataset.json'
csv_file_path = 'processed_kelsen_string_data.csv'

# Convert the JSON file to CSV (strings only)
json_to_csv_strings_only(json_file_path, csv_file_path)


# Define a function to convert JSON with only assets to CSV
def json_to_csv_assets_only(json_file_path, csv_file_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    
    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(['input_text', 'target_text'])
        
        # Loop through each entry in the JSON data
        for entry in json_data:
            article_number = entry["article_number"]
            original_text = entry["original"]
            assets = entry["assets"]
            
            # Build the input_text
            input_text = f'Artículo {article_number} - {original_text}'
            
            # Build the target_text (only assets)
            target_assets = ' '.join([f'{value}' for key, value in assets.items()])
            
            # Write the formatted data to CSV
            writer.writerow([input_text, target_assets])

# File paths
json_file_path = 'augmented_kelsen_dataset.json'
csv_file_path = 'processed_kelsen_assets_data.csv'

# Convert the JSON file to CSV (assets only)
json_to_csv_assets_only(json_file_path, csv_file_path)


# Define a function to convert JSON with only clauses to CSV
def json_to_csv_clauses_only(json_file_path, csv_file_path):
    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    
    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the CSV header
        writer.writerow(['input_text', 'target_text'])
        
        # Loop through each entry in the JSON data
        for entry in json_data:
            article_number = entry["article_number"]
            original_text = entry["original"]
            clauses = entry["clauses"]
            
            # Build the input_text
            input_text = f'Artículo {article_number} - {original_text}'
            
            # Build the target_text (only clauses)
            target_clauses = ' '.join([f'{value}' for key, value in clauses.items()])
            
            # Write the formatted data to CSV
            writer.writerow([input_text, target_clauses])

# File paths
json_file_path = 'augmented_kelsen_dataset.json'
csv_file_path = 'processed_kelsen_clauses_data.csv'

# Convert the JSON file to CSV (clauses only)
json_to_csv_clauses_only(json_file_path, csv_file_path)

