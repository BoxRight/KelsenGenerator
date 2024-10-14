import psutil
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig
from datasets import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from clause_matcher import get_embedding, semantic_similarity
import evaluate
import re
from torch.utils.data import DataLoader, Dataset
from difflib import SequenceMatcher
import re
import gc

class KelsenDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=256):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_encoding = self.tokenizer(item['input_text'], 
                                        max_length=self.max_length, 
                                        padding='max_length', 
                                        return_tensors='pt')
        
        target_encoding = self.tokenizer(item['target_text'], 
                                         max_length=self.max_length, 
                                         padding='max_length', 
                                         return_tensors='pt')

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'target_text': item['target_text']  # Keep this as a string, not a tensor
        }

def collate_fn(batch):
    # Separate tensor data from non-tensor data
    tensor_keys = ['input_ids', 'attention_mask', 'labels']
    non_tensor_keys = ['target_text']
    
    # Collate tensor data
    tensor_data = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    
    # Collate non-tensor data
    non_tensor_data = {key: [item[key] for item in batch] for key in non_tensor_keys}
    
    # Combine tensor and non-tensor data
    return {**tensor_data, **non_tensor_data}

def create_dataloader(dataset, batch_size=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def compute_partial_match_score(prediction, reference):
    matcher = SequenceMatcher(None, prediction, reference)
    return matcher.ratio()

def compute_metrics(predictions, references):
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
    partial_match_scores = [compute_partial_match_score(pred, ref) for pred, ref in zip(predictions, references)]
    avg_partial_match = sum(partial_match_scores) / len(partial_match_scores)
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "partial_match": avg_partial_match
    }


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_memory_usage(step):
    memory = get_memory_usage()
    print(f"Memory usage after {step}: {memory:.2f} MB")

# Load the ROUGE metric
rouge_metric = evaluate.load("rouge")

# Define the special tokens (same as in the training script)
extra_id_to_special_token = {
    "<extra_id_0>": "string",
    "<extra_id_1>": "asset",
    "<extra_id_2>": "subject",
    "<extra_id_3>": "clause",
    "<extra_id_4>": "=",
    "<extra_id_5>": ")",
    "<extra_id_6>": "CR(",
    "<extra_id_7>": "PVG(",
    "<extra_id_8>": "OB(",
    "<extra_id_9>": "PR(",
    "<extra_id_10>": "Service",
    "<extra_id_11>": "Property",
    "<extra_id_12>": "N",
    "<extra_id_13>": "NM",
    "<extra_id_14>": "+",
    "<extra_id_15>": "-",
    "<extra_id_16>": "COMPRADOR",
    "<extra_id_17>": "VENDEDOR",
    "<extra_id_18>": "PROPIETARIO",
    "<extra_id_19>": "ACREEDOR",
    "<extra_id_20>": "DEUDOR",
    "<extra_id_21>": "ADQUIRENTE",
    "<extra_id_22>": "{",
    "<extra_id_23>": "}",
    "<extra_id_24>": ";",
    "<extra_id_25>": "AND",
    "<extra_id_26>": "OFERENTE",
    "<extra_id_27>": "MUTUANTE",
    "<extra_id_28>": "MUTUARIO",
    "<extra_id_29>": "ARRENDADOR",
    "<extra_id_30>": "ARRENDATARIO",
    "<extra_id_31>":"PERMUTANTE1",
    "<extra_id_32>":"PERMUTANTE2",
    "<extra_id_33>":"DONANTE",
    "<extra_id_34>":"DONATARIO",       
    "<extra_id_35>":"PRESTADOR",
    "<extra_id_36>":"ACREDITADO",
}

def load_model_and_tokenizer(model_path, device):
    peft_config = PeftConfig.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    
    special_tokens_dict = {'additional_special_tokens': list(extra_id_to_special_token.values())}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    base_model = T5ForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=True,
        device_map=None  # Disable device map
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)  # Explicitly move model to the specified device
    
    log_memory_usage("loading model and tokenizer")
    return model, tokenizer

def load_test_data(test_data_path, tokenizer):
    dataset = KelsenDataset(test_data_path, tokenizer)
    log_memory_usage("after loading test data")
    return dataset

def preprocess_function(examples, tokenizer, max_length=256):
    inputs = [example['input_text'] for example in examples]
    targets = [example['target_text'] for example in examples]
    
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding='max_length')

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def data_generator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_predictions(model, tokenizer, dataloader, device, max_length=512, min_length=10):
    model.eval()
    predictions = []
    references = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="Generating predictions")):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logger.info(f"Processing batch {i+1}")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=1,  # Use greedy decoding
                    do_sample=False,  # Disable sampling
                    no_repeat_ngram_size=0,  # Disable n-gram repetition prevention
                    repetition_penalty=1.0,  # Disable repetition penalty
                    length_penalty=1.0,  # Neutral length penalty
                    early_stopping=False,  # Don't stop early
                    use_cache=True,  # Use caching for efficiency
                    num_return_sequences=1
                )
            
            logger.info(f"Generated output for batch {i+1}")
            
            # Use tokenizer's decoder directly
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]
            logger.info(f"Decoded output for batch {i+1} (first 100 chars): {decoded_outputs[0][:100]}...")
            
            # Apply post-processing
            post_processed_outputs = [post_process_output(output) for output in decoded_outputs]
            logger.info(f"Post-processed output for batch {i+1} (first 100 chars): {post_processed_outputs[0][:100]}...")
            
            predictions.extend(post_processed_outputs)
            references.extend(batch['target_text'])
            
            logger.info(f"Processed batch {i+1}")
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {str(e)}")
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    logger.info(f"Total predictions generated: {len(predictions)}")
    logger.info(f"Total references: {len(references)}")
    
    return predictions, references
    
def replace_special_tokens(text):
    for extra_id, special_token in extra_id_to_special_token.items():
        text = text.replace(extra_id, special_token)
    return text

def compute_metrics(predictions, references):
    # Replace special tokens in predictions and references
    predictions = [replace_special_tokens(pred) for pred in predictions]
    references = [replace_special_tokens(ref) for ref in references]

    rouge_scores = rouge_metric.compute(predictions=predictions, references=[[r] for r in references])
    
    similarities = [semantic_similarity(get_embedding(pred), get_embedding(ref)) 
                    for pred, ref in zip(predictions, references)]
    avg_similarity = np.mean(similarities)
    
    exact_matches = [pred == ref for pred, ref in zip(predictions, references)]
    exact_match_ratio = np.mean(exact_matches)
    
    kelsen_metrics = compute_kelsen_specific_metrics(predictions, references)
    
    return {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "semantic_similarity": avg_similarity,
        "exact_match_ratio": exact_match_ratio,
        **kelsen_metrics
    }

def compute_kelsen_specific_metrics(predictions, references):
    def count_elements(text):
        return {
            'string': len(re.findall(r'string\s+\w+\s*=', text)),
            'asset': len(re.findall(r'asset\s+\w+\s*=', text)),
            'clause': len(re.findall(r'clause\s+\w+\s*=', text)),
        }
    
    pred_counts = [count_elements(pred) for pred in predictions]
    ref_counts = [count_elements(ref) for ref in references]
    
    accuracies = {
        'string_accuracy': np.mean([p['string'] == r['string'] for p, r in zip(pred_counts, ref_counts)]),
        'asset_accuracy': np.mean([p['asset'] == r['asset'] for p, r in zip(pred_counts, ref_counts)]),
        'clause_accuracy': np.mean([p['clause'] == r['clause'] for p, r in zip(pred_counts, ref_counts)]),
    }
    
    legal_concepts = ['CR(', 'PVG(', 'OB(', 'PR(']
    legal_concept_accuracy = np.mean([
        any(concept in pred for concept in legal_concepts) == 
        any(concept in ref for concept in legal_concepts)
        for pred, ref in zip(predictions, references)
    ])
    
    accuracies['legal_concept_accuracy'] = legal_concept_accuracy
    
    return accuracies


def optimize_model(model):
    model.half()  # Convert to half precision
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    return model

def improved_detokenize(tokenizer, token_ids):
    # Decode the tokens
    raw_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # Clean up extra spaces and line breaks
    cleaned_text = ' '.join(raw_text.split())
    
    # Ensure proper spacing around key tokens
    for token in ['string', 'asset', 'clause']:
        cleaned_text = re.sub(rf'({token})\s*', f'{token} ', cleaned_text)
        cleaned_text = re.sub(rf'\s*({token})', f' {token}', cleaned_text)
    
    # Ensure proper spacing around punctuation
    cleaned_text = re.sub(r'\s*([,;])\s*', r'\1 ', cleaned_text)
    cleaned_text = re.sub(r'\s*(=)\s*', r' \1 ', cleaned_text)
    
    # Ensure proper spacing around quotes
    cleaned_text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', cleaned_text)
    
    return cleaned_text.strip()

def post_process_output(text):
    # Remove unwanted tokens
    unwanted_tokens = ['<pad>', '<s>', '</s>']
    for token in unwanted_tokens:
        text = text.replace(token, '')
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Ensure proper spacing around key tokens
    kelsen_tokens = ['string', 'asset', 'clause', 'CR(', 'PVG(', 'OB(', 'PR(', 'Service', 'Property', 'M', 'NM']
    for token in kelsen_tokens:
        text = re.sub(rf'\b{re.escape(token)}\b', f' {token} ', text)
    
    # Ensure proper spacing around punctuation
    text = re.sub(r'\s*([,;={}()])\s*', r' \1 ', text)
    
    # Ensure proper spacing around quotes
    text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
    
    # Handle potential artifacts from tokenization
    text = text.replace('Ġ', '')  # Remove any 'Ġ' characters (often used by some tokenizers)
    
    # Ensure proper structure for assets
    text = re.sub(r'(asset\s+\w+\s*=\s*)([^,;]+)([,;])', lambda m: f"{m.group(1)}{m.group(2).replace(' ', ', ')}{m.group(3)}", text)
    
    # Ensure proper structure for clauses
    text = re.sub(r'(clause\s+\w+\s*=\s*\{)([^}]+)(\})', lambda m: f"{m.group(1)}{m.group(2).replace(' ', ',')}{m.group(3)}", text)
    
    # Remove any extra spaces
    text = ' '.join(text.split())
    
    return text

def analyze_errors(predictions, references):
    error_types = {
        "incomplete_string": 0,
        "incomplete_asset": 0,
        "incomplete_clause": 0,
        "missing_string": 0,
        "missing_asset": 0,
        "missing_clause": 0,
        "incorrect_format": 0
    }
    
    for pred, ref in zip(predictions, references):
        pred_strings = re.findall(r'string\s+\w+\s*=\s*"[^"]*"\s*;', pred)
        pred_assets = re.findall(r'asset\s+\w+\s*=\s*\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s*;', pred)
        pred_clauses = re.findall(r'clause\s+\w+\s*=\s*\{[^}]*\}\s*;', pred)
        
        ref_strings = re.findall(r'string\s+\w+\s*=\s*"[^"]*"\s*;', ref)
        ref_assets = re.findall(r'asset\s+\w+\s*=\s*\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s*;', ref)
        ref_clauses = re.findall(r'clause\s+\w+\s*=\s*\{[^}]*\}\s*;', ref)
        
        error_types["incomplete_string"] += sum(1 for s in re.findall(r'string\s+[^;]*;', pred) if s not in pred_strings)
        error_types["incomplete_asset"] += sum(1 for a in re.findall(r'asset\s+[^;]*;', pred) if a not in pred_assets)
        error_types["incomplete_clause"] += sum(1 for c in re.findall(r'clause\s+[^;]*;', pred) if c not in pred_clauses)
        
        error_types["missing_string"] += max(0, len(ref_strings) - len(pred_strings))
        error_types["missing_asset"] += max(0, len(ref_assets) - len(pred_assets))
        error_types["missing_clause"] += max(0, len(ref_clauses) - len(pred_clauses))
        
        error_types["incorrect_format"] += 1 if not (pred_strings and pred_assets and pred_clauses) else 0
    
    total = len(predictions)
    error_report = {k: f"{v} ({v/total*100:.2f}%)" for k, v in error_types.items()}
    return error_report

def merge_outputs(full_prediction, clause_prediction):
    # Extract components from full prediction
    strings = re.findall(r'string\s+\w+\s*=\s*"[^"]*"\s*;', full_prediction)
    assets = re.findall(r'asset\s+\w+\s*=\s*[^;]+;', full_prediction)
    
    # Prefer clauses from the clause-specific model, but fall back to full if none found
    clauses = re.findall(r'clause\s+\w+\s*=\s*\{[^}]*\}\s*;', clause_prediction)
    if not clauses:
        clauses = re.findall(r'clause\s+\w+\s*=\s*\{[^}]*\}\s*;', full_prediction)
    
    # Combine the components
    merged = strings + assets + clauses
    return ' '.join(merged)

def run_full_model(test_dataset, device, chunk_size=100):
    model_path = "./results_Full/checkpoint-8370"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    
    logger.info("Running Full model...")
    all_predictions = []
    all_references = []
    
    for i in range(0, len(test_dataset), chunk_size):
        chunk = torch.utils.data.Subset(test_dataset, range(i, min(i + chunk_size, len(test_dataset))))
        dataloader = create_dataloader(chunk, batch_size=1)
        chunk_predictions, chunk_references = generate_predictions(
            model, 
            tokenizer, 
            dataloader, 
            device, 
            max_length=512,
            min_length=10
        )
        all_predictions.extend(chunk_predictions)
        all_references.extend(chunk_references)
        
        logger.info(f"Processed chunk {i//chunk_size + 1}, total predictions: {len(all_predictions)}")
        log_memory_usage(f"after processing chunk {i//chunk_size + 1}")
    
    return all_predictions, all_references

def run_clauses_model(test_dataset, device, chunk_size=100):
    model_path = "./results_Clauses/checkpoint-5190"
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    
    logger.info("Running Clauses model...")
    all_predictions = []
    all_references = []
    
    for i in range(0, len(test_dataset), chunk_size):
        chunk = torch.utils.data.Subset(test_dataset, range(i, min(i + chunk_size, len(test_dataset))))
        dataloader = create_dataloader(chunk, batch_size=1)
        chunk_predictions, chunk_references = generate_predictions(
            model, 
            tokenizer, 
            dataloader, 
            device, 
            max_length=512,
            min_length=10
        )
        all_predictions.extend(chunk_predictions)
        all_references.extend(chunk_references)
        
        logger.info(f"Processed chunk {i//chunk_size + 1}, total predictions: {len(all_predictions)}")
        log_memory_usage(f"after processing chunk {i//chunk_size + 1}")
    
    return all_predictions, all_references

def post_process_output(prediction):
    # Remove unwanted tokens
    prediction = prediction.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
    
    # Ensure proper spacing around key tokens
    kelsen_tokens = ['string', 'asset', 'clause', 'CR(', 'PVG(', 'OB(', 'PR(', 'Service', 'Property', 'M', 'NM']
    for token in kelsen_tokens:
        prediction = re.sub(rf'\b{re.escape(token)}\b', f' {token} ', prediction)
    
    # Ensure proper spacing around punctuation
    prediction = re.sub(r'\s*([,;={}()])\s*', r' \1 ', prediction)
    
    # Ensure proper spacing around quotes
    prediction = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', prediction)
    
    # Remove any extra spaces
    prediction = ' '.join(prediction.split())
    
    return prediction

def log_constructs(prediction):
    logger.info("Final Kelsen constructs:")
    
    # Log strings
    strings = re.findall(r'string\s+(\w+)\s*=\s*"([^"]*)"', prediction)
    for name, value in strings:
        logger.info(f"String: {name} = \"{value}\"")
    
    # Log assets
    assets = re.findall(r'asset\s+(\w+)\s*=\s*([^;]+)', prediction)
    for name, definition in assets:
        logger.info(f"Asset: {name} = {definition}")
    
    # Log clauses
    clauses = re.findall(r'clause\s+(\w+)\s*=\s*\{([^}]+)\}', prediction)
    for name, definition in clauses:
        logger.info(f"Clause: {name} = {{{definition}}}")

def main():
    log_memory_usage("start of main")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Full model and its tokenizer
    logger.info("Loading Full model...")
    full_model_path = "./results_Full/checkpoint-15990"
    full_model, full_tokenizer = load_model_and_tokenizer(full_model_path, device)
    
    # Load test data using the full model's tokenizer
    test_data_path = "kelsen_test_data.csv"
    test_dataset = load_test_data(test_data_path, full_tokenizer)
    
    logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
    
    chunk_size = 100
    all_predictions = []
    all_references = []
    
    # Process Full model
    for i in range(0, len(test_dataset), chunk_size):
        chunk = torch.utils.data.Subset(test_dataset, range(i, min(i + chunk_size, len(test_dataset))))
        dataloader = create_dataloader(chunk, batch_size=1)
        chunk_predictions, chunk_references = generate_predictions(full_model, full_tokenizer, dataloader, device)
        all_predictions.extend(chunk_predictions)
        all_references.extend(chunk_references)
    
    # Clear memory
    del full_model, full_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Load and process Clauses model
    logger.info("Loading and processing Clauses model...")
    clauses_model_path = "./results_Clauses/checkpoint-7995"
    clauses_model, clauses_tokenizer = load_model_and_tokenizer(clauses_model_path, device)
    
    clauses_predictions = []
    
    for i in range(0, len(test_dataset), chunk_size):
        chunk = torch.utils.data.Subset(test_dataset, range(i, min(i + chunk_size, len(test_dataset))))
        dataloader = create_dataloader(chunk, batch_size=1)
        chunk_predictions, _ = generate_predictions(clauses_model, clauses_tokenizer, dataloader, device)
        clauses_predictions.extend(chunk_predictions)
    
    # Combine predictions and post-process
    combined_predictions = []
    for full, clause in zip(all_predictions, clauses_predictions):
        combined = merge_outputs(full, clause)
        post_processed = post_process_output(combined)
        combined_predictions.append(post_processed)
    
    # Compute scores and keep only the best prediction
    scores = [compute_partial_match_score(pred, ref) for pred, ref in zip(combined_predictions, all_references)]
    best_index = scores.index(max(scores))
    
    best_prediction = combined_predictions[best_index]
    best_reference = all_references[best_index]
    best_score = scores[best_index]
    
    logger.info(f"Best prediction score: {best_score}")
    logger.info(f"Best prediction:\n{best_prediction}")
    logger.info(f"Reference:\n{best_reference}")
    
    # Log the final constructs of the best prediction
    log_constructs(best_prediction)
    
    # Save detailed results for the best prediction
    results_df = pd.DataFrame({
        "original_input": [test_dataset[best_index]['input_ids'].tolist()],
        "reference": [best_reference],
        "prediction": [best_prediction],
        "score": [best_score]
    })
    
    results_df.to_csv("best_prediction_result.csv", index=False)
    logger.info("Saved best prediction result to best_prediction_result.csv")
    
    log_memory_usage("end of main")
        
if __name__ == "__main__":
    main()
