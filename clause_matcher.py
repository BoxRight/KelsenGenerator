import sys
import json
import time
from typing import Dict, List, Any
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to load model with memory constraints
def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        return tokenizer, model, classifier
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None

# Load the model (using a smaller model for speed)
model_name = "distilbert-base-multilingual-cased"
tokenizer, model, classifier = load_model(model_name)

if not (tokenizer and model and classifier):
    print("Failed to load model. Exiting.")
    sys.exit(1)

# Move model to the selected device
model = model.to(device)

def load_json_to_dict(json_file_path: str) -> Dict[str, Any]:
    with open(json_file_path, 'r') as file:
        return json.load(file)
        
def extract_descriptions(contract_data: Dict[str, Any]) -> Dict[str, str]:
    descriptions = {}
    node = contract_data
    while node:
        if (node.get("NodeType") == "AST_DECLARATION" and 
            node.get("Type", {}).get("Type") == "string"):
            identifier_info = node.get("Identifier", {})
            identifier = identifier_info.get("Identifier")
            expression = node.get("Expression", {})
            if identifier:
                description_value = expression.get("String", "").strip('"')
                descriptions[identifier] = description_value
        node = node.get("Next")
    return descriptions

def extract_assets(contract_data: Dict[str, Any], descriptions: Dict[str, str]) -> Dict[str, str]:
    assets = {}
    node = contract_data
    while node:
        if (node.get("NodeType") == "AST_DECLARATION" and 
            node.get("Type", {}).get("Type") == "asset"):
            identifier_obj = node.get("Identifier")
            expression = node.get("Expression")
            if identifier_obj and expression:
                identifier = identifier_obj.get("Identifier")
                type_of_asset = expression.get("Type", {}).get("Type")
                subject1 = expression.get("Subject1", {}).get("Identifier")
                description_key = expression.get("Description", {}).get("Identifier")
                subject2 = expression.get("Subject2", {}).get("Identifier")
                description = descriptions.get(description_key, description_key)
                if type_of_asset in ["Service", "Property"]:
                    assets[identifier] = f"El {subject1} {description} al {subject2}"
        node = node.get("Next")
    return assets

def handle_condition(condition: Dict[str, Any], descriptions: Dict[str, str], assets: Dict[str, str]) -> str:
    node_type = condition.get("NodeType")
    if node_type == "AST_AND":
        left_condition = handle_condition(condition.get("Left"), descriptions, assets)
        right_condition = handle_condition(condition.get("Right"), descriptions, assets)
        return f"{left_condition} y {right_condition}"
    elif node_type == "AST_CONDITION":
        return handle_condition(condition.get("Left"), descriptions, assets)
    elif node_type == "AST_IDENTIFIER":
        identifier = condition.get("Identifier")
        return assets.get(identifier, identifier)
    else:
        return "Unknown Condition"

def handle_consequence(consequence: Dict[str, Any], descriptions: Dict[str, str], assets: Dict[str, str]) -> str:
    node_type = consequence.get("NodeType")
    if node_type in ["AST_CR", "AST_OB"]:
        identifier = consequence.get("Left", {}).get("Identifier")
        asset_description = assets.get(identifier, identifier)
        return f"se tiene derecho de exigir que {asset_description}"
    elif node_type == "AST_PVG":
        identifier = consequence.get("Left", {}).get("Identifier")
        asset_description = assets.get(identifier, identifier)
        return f"se permite que {asset_description}"
    elif node_type == "AST_PR":
        identifier = consequence.get("Left", {}).get("Identifier")
        asset_description = assets.get(identifier, identifier)
        return f"se prohibe que {asset_description}"
    else:
        return "Unknown Consequence"

def parse_kelsen_statutes(json_file_path: str) -> List[Dict[str, Any]]:
    contract_data = load_json_to_dict(json_file_path)
    descriptions = extract_descriptions(contract_data)
    assets = extract_assets(contract_data, descriptions)
    
    parsed_statutes = []
    node = contract_data
    clause_number = 1
    
    while node:
        if node.get("NodeType") == "AST_DECLARATION" and node.get("Type", {}).get("Type") == "clause":
            expression = node.get("Expression")
            if expression:
                condition = expression.get("Condition")
                consequence = expression.get("Consequence")
                
                formatted_condition = handle_condition(condition, descriptions, assets)
                formatted_consequence = handle_consequence(consequence, descriptions, assets)
                
                parsed_statutes.append({
                    'id': f"CLÁUSULA {clause_number}",
                    'condition': formatted_condition,
                    'consequence': formatted_consequence
                })
                clause_number += 1
        node = node.get("Next")
    
    return parsed_statutes

@torch.no_grad()
def get_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

def semantic_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    return torch.cosine_similarity(emb1, emb2, dim=0).item()

def parse_contract(contract_text: str) -> List[str]:
    return [clause.strip() for clause in contract_text.split('\n') if clause.strip()]

def analyze_single_clause(clause_text: str, kelsen_statutes: List[Dict[str, Any]], 
                          statute_embeddings: Dict[str, torch.Tensor],
                          similarity_threshold: float = 0.8) -> Dict[str, Any]:
    clause_emb = get_embedding(clause_text)
    matches = []
    near_matches = []

    for statute in kelsen_statutes:
        condition_similarity = semantic_similarity(clause_emb, statute_embeddings[statute['id'] + '_condition'])
        consequence_similarity = semantic_similarity(clause_emb, statute_embeddings[statute['id'] + '_consequence'])
        max_similarity = max(condition_similarity, consequence_similarity)
        
        result = {
            'statute_id': statute['id'],
            'statute_condition': statute['condition'],
            'statute_consequence': statute['consequence'],
            'max_similarity': max_similarity,
        }
        
        if max_similarity > similarity_threshold:
            matches.append(result)
        else:
            near_matches.append(result)

    near_matches.sort(key=lambda x: x['max_similarity'], reverse=True)
    
    return {
        'clause_text': clause_text,
        'matches': matches,
        'near_matches': near_matches[:5],  # Top 5 near matches
    }

def analyze_full_contract(contract_text: str, kelsen_statutes: List[Dict[str, Any]], 
                          similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    contract_clauses = parse_contract(contract_text)
    
    # Precompute embeddings for Kelsen statutes
    statute_embeddings = {}
    for statute in tqdm(kelsen_statutes, desc="Computing statute embeddings"):
        statute_embeddings[statute['id'] + '_condition'] = get_embedding(statute['condition'])
        statute_embeddings[statute['id'] + '_consequence'] = get_embedding(statute['consequence'])

    # Process clauses sequentially
    results = []
    for clause in tqdm(contract_clauses, desc="Analyzing clauses"):
        result = analyze_single_clause(clause, kelsen_statutes, statute_embeddings, similarity_threshold)
        results.append(result)

    return results

def print_analysis_results(analysis_results: List[Dict[str, Any]], similarity_threshold: float):
    print(f"Contract Analysis Results (Similarity Threshold: {similarity_threshold}):")
    
    for i, clause_result in enumerate(analysis_results, 1):
        print(f"\nClause {i}: {clause_result['clause_text'][:50]}...")  # Print first 50 chars of clause
        
        if clause_result['matches']:
            print("  Matches:")
            for match in clause_result['matches'][:3]:  # Show top 3 matches
                print(f"    Matched Statute: {match['statute_id']}")
                print(f"    Statute Condition: {match['statute_condition']}")
                print(f"    Statute Consequence: {match['statute_consequence']}")
                print(f"    Max Similarity: {match['max_similarity']:.2f}")
        else:
            print("  No matches found above the threshold.")
        
        if clause_result['near_matches']:
            print("  Top Near Matches:")
            for near_match in clause_result['near_matches'][:3]:  # Show top 3 near matches
                print(f"    Near Match Statute: {near_match['statute_id']}")
                print(f"    Statute Condition: {near_match['statute_condition']}")
                print(f"    Statute Consequence: {near_match['statute_consequence']}")
                print(f"    Max Similarity: {near_match['max_similarity']:.2f}")

def main():
    kelsen_statutes_path = './ast_output.json'
    contract_file_path = './1794-2022.txt'
    
    start_time = time.time()

    try:
        kelsen_statutes = parse_kelsen_statutes(kelsen_statutes_path)
        print(f"Loaded {len(kelsen_statutes)} Kelsen statutes.")
    except Exception as e:
        print(f"Error parsing Kelsen statutes: {e}")
        sys.exit(1)
    
    try:
        with open(contract_file_path, 'r') as file:
            full_contract_text = file.read()
        print(f"Loaded contract: {len(full_contract_text)} characters.")
    except Exception as e:
        print(f"Error reading contract file: {e}")
        sys.exit(1)
    
    similarity_threshold = 0.8
    
    try:
        analysis_results = analyze_full_contract(full_contract_text, kelsen_statutes, similarity_threshold)
        print_analysis_results(analysis_results, similarity_threshold)
    except Exception as e:
        print(f"Error analyzing contract: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
