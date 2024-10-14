# Standard library imports
import re
from collections import Counter
import gc
# Third-party library imports
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
import evaluate
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Tokenizer,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
# Local module imports
from clause_matcher import get_embedding, semantic_similarity
import os
from torch import nn
import hashlib
import random
# Load the ROUGE metric
rouge_metric = evaluate.load("rouge")

def group_by_target_text(dataset):
    target_text_to_samples = {}
    for sample in dataset:
        target_text = sample['target_text']
        if target_text not in target_text_to_samples:
            target_text_to_samples[target_text] = []
        target_text_to_samples[target_text].append(sample)
    return target_text_to_samples
    
def compute_sample_hash(input_text, target_text):
    sample_string = input_text + target_text
    return hashlib.md5(sample_string.encode('utf-8')).hexdigest()
    
def remove_duplicates(dataset):
    # Compute hashes for each sample
    hashes = []
    for sample in dataset:
        sample_hash = compute_sample_hash(sample['input_text'], sample['target_text'])
        hashes.append(sample_hash)
    
    # Add hashes to the dataset
    dataset = dataset.add_column('sample_hash', hashes)
    
    # Convert the dataset to a pandas DataFrame
    df = dataset.to_pandas()
    
    # Use pandas to drop duplicates based on 'sample_hash'
    df_unique = df.drop_duplicates(subset='sample_hash', keep='first').reset_index(drop=True)
    
    # Remove the 'sample_hash' column
    df_unique = df_unique.drop(columns=['sample_hash'])
    
    # Convert back to a Dataset
    unique_dataset = Dataset.from_pandas(df_unique)
    
    return unique_dataset



class CustomSeq2SeqTrainerWithHumanInput(Seq2SeqTrainer):
    def __init__(self, *args, similarity_weight=0.05, rouge_weight=0.8, human_feedback_weight=0, exact_match_weight=0.01, human_feedback_start_step=100, enable_human_input=True, log_texts=True, current_stage="full", **kwargs):
    

        self.similarity_weight = similarity_weight  # Weight for semantic similarity
        self.rouge_weight = rouge_weight  # Weight for ROUGE-L score
        self.human_feedback_weight = human_feedback_weight  # Weight for human feedback
        self.exact_match_weight = exact_match_weight  # Weight for exact match criterion
        self.human_feedback_start_step = human_feedback_start_step
        self.global_step = 0  # Track the step for human feedback
        self.enable_human_input = enable_human_input
        self.log_texts = log_texts
        self.current_stage = current_stage
        super().__init__(*args, **kwargs)
        
        self.extra_id_to_special_token = {
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
            "<extra_id_27>":"MUTUANTE",
            "<extra_id_28>":"MUTUARIO",
            "<extra_id_29>":"ARRENDADOR",
            "<extra_id_30>":"ARRENDATARIO",
            "<extra_id_31>":"PERMUTANTE1",
            "<extra_id_32>":"PERMUTANTE2",
            "<extra_id_33>":"DONANTE",
            "<extra_id_34>":"DONATARIO",       
            "<extra_id_35>":"PRESTADOR",
            "<extra_id_36>":"ACREDITADO",
        }

    def get_human_feedback(self, generated_texts, target_texts, input_texts):
        """
        Simulate human feedback. In practice, this would involve actual human input.
        """
        human_scores = []
        for input_text, generated_text, target_text in zip(input_texts, generated_texts, target_texts):
            print(f"Input: {input_text}")
            print(f"Generated: {generated_text}")
            print(f"Target: {target_text}")
            try:
                score = float(input("Provide feedback score (0 to 1): "))
            except ValueError:
                score = 0.0  # Default to 0 if invalid input
            human_scores.append(score)
        return np.mean(human_scores)

    def replace_extra_ids_with_special_tokens(self, text):
        for extra_id, special_token in self.extra_id_to_special_token.items():
            text = text.replace(extra_id, special_token)
        return text

    # Example modification to log_texts_to_console
    def log_texts_to_console(self, input_texts, generated_texts, target_texts):
        for input_text, generated_text, target_text in zip(input_texts, generated_texts, target_texts):
        # Replace <extra_id_*> with your actual special tokens in the generated text
            generated_text = self.replace_extra_ids_with_special_tokens(generated_text)
            print(f"Input: {input_text}")
            print(f"Generated: {generated_text}")
            print(f"Target: {target_text}")

    def compute_missing_string_penalty(self, generated_texts):
        penalty = 0
        for text in generated_texts:
            if not ('string' in text and '=' in text and '"' in text):
                penalty += 1
        return penalty / len(generated_texts)

    def compute_missing_asset_penalty(self, generated_texts):
        penalty = 0
        for text in generated_texts:
            if not ('asset' in text and 
                ((('Service' in text) and ('+' in text or '-' in text)) or
                 (('Property' in text) and ('NM' in text or 'M' in text)))):
                penalty += 1
        return penalty / len(generated_texts)

    def compute_missing_clause_penalty(self, generated_texts):
        penalty = 0
        for text in generated_texts:
            if not ('clause' in text and '{' in text and '}' in text and ')' in text and
                ('PVG(' in text or 'CR(' in text or 'PR(' in text or 'OB(' in text)):

                penalty += 1
        return penalty / len(generated_texts)

    def compute_missing_declaration_penalty(self, generated_texts):
        penalty = 0
        for text in generated_texts:
            string_check = 'string' in text and '=' in text and '"' in text
            asset_check = ('asset' in text and 
                       ((('Service' in text) and ('+' in text or '-' in text)) or
                        (('Property' in text) and ('NM' in text or 'M' in text))))
            clause_check = ('clause' in text and '{' in text and '}' in text and ')' in text and
                        ('PVG(' in text or 'CR(' in text or 'PR(' in text or 'OB(' in text))
            if not (string_check and asset_check and clause_check):
                penalty += 1
        return penalty / len(generated_texts)

    def compute_loss(self, model, inputs, return_outputs=False):
        self.global_step += 1

        # Step 1: Extract input IDs and labels
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        # Step 2: Forward pass to get model outputs
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        # Step 3: Calculate Cross-Entropy Loss
        cross_entropy_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss_ce = cross_entropy_loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

       # Step 4: Decode input, generated, and target texts
       # Use `skip_special_tokens=False` to keep task-specific tokens (e.g., <extra_id_*>), but we will filter irrelevant ones manually.
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)  
        generated_ids = model.generate(input_ids=input_ids, attention_mask=inputs["attention_mask"], repetition_penalty=1.5, num_beams=5, temperature=0.7)
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)  
        target_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=False)  

        # Filter out only irrelevant tokens (like <pad>) from the generated text
        def filter_irrelevant_tokens(text):
            return text.replace('<pad>', '').replace('<s>', '').replace('</s>', '')

        input_texts = [filter_irrelevant_tokens(text) for text in input_texts]
        generated_texts = [filter_irrelevant_tokens(text) for text in generated_texts]
        target_texts = [filter_irrelevant_tokens(text) for text in target_texts]

        # **New logic to remove the trailing double quote after ;**
        def remove_trailing_double_quotes(text):
            return text.replace('";"', '";').strip()

        # Apply the fix to target texts
        target_texts = [remove_trailing_double_quotes(text) for text in target_texts]

        # Join tokens back into sentences, ensuring spaces between tokens
        generated_texts = [' '.join(generated_text.split()) for generated_text in generated_texts]
        target_texts = [' '.join(target_text.split()) for target_text in target_texts]
        input_texts = [' '.join(input_text.split()) for input_text in input_texts]

        # Replace <extra_id_*> with your actual special tokens in the generated text
        generated_texts = [self.replace_extra_ids_with_special_tokens(text) for text in generated_texts]

        # Log texts if logging is enabled
        if self.log_texts:
            self.log_texts_to_console(input_texts, generated_texts, target_texts)

        # Step 5: Compute Semantic Similarity Loss
        total_similarity = 0
        for generated_text, target_text in zip(generated_texts, target_texts):
            similarity = semantic_similarity(get_embedding(generated_text), get_embedding(target_text))
            total_similarity += similarity
        average_similarity = total_similarity / len(generated_texts)
        similarity_loss = 1 - average_similarity

        # Step 6: Compute ROUGE-L Loss
        rouge_result = rouge_metric.compute(predictions=generated_texts, references=[[t] for t in target_texts])
        rouge_l_f1_score = rouge_result.get('rougeL', 0)
        rouge_loss = 1 - rouge_l_f1_score

        # Step 7: Compute Exact Match
        exact_matches = sum([1 for gen, target in zip(generated_texts, target_texts) if gen == target])
        exact_match_ratio = exact_matches / len(generated_texts)
        exact_match_loss = 1 - exact_match_ratio

        # Step 8: Optional Human Feedback after a certain number of steps
        human_feedback_loss = 0
        if self.global_step >= self.human_feedback_start_step and self.enable_human_input:
            human_feedback_loss = self.get_human_feedback(generated_texts, target_texts, input_texts)

        # Step 9: Combine Cross-Entropy, Similarity, ROUGE-L, Exact Match, and Human Feedback Loss
        total_loss = (
            loss_ce + 
            self.similarity_weight * similarity_loss + 
            self.rouge_weight * rouge_loss + 
            self.exact_match_weight * exact_match_loss + 
            self.human_feedback_weight * human_feedback_loss
    )


        if self.current_stage == "strings":
            missing_string_penalty = self.compute_missing_string_penalty(generated_texts)
            total_loss += 0.1 * missing_string_penalty  # Adjust weight as needed
        elif self.current_stage == "assets":
            missing_asset_penalty = self.compute_missing_asset_penalty(generated_texts)
            total_loss += 0.1 * missing_asset_penalty  # Adjust weight as needed
        elif self.current_stage == "clauses":
            missing_clause_penalty = self.compute_missing_clause_penalty(generated_texts)
            total_loss += 0.1 * missing_clause_penalty  # Adjust weight as needed
        elif self.current_stage == "full":
            missing_declaration_penalty = self.compute_missing_declaration_penalty(generated_texts)
            total_loss += 0.1 * missing_declaration_penalty  # Adjust weight as needed

        return (total_loss, outputs) if return_outputs else total_loss


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
def tokenize(text):
    # Split by spaces, punctuation, or keep meaningful word chunks
    return re.findall(r'\w+|\S', text)

# 1. Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    memory_efficient_offload=True
)


# 2. Load the tokenizer and model with 8-bit quantization
model_name = 'Salesforce/codet5-large'  # Change to CodeT5
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto',
    offload_folder="offload"
)

# 3. Configure LoRA
target_modules = ['q', 'v']  # May need adjustment based on T5 architecture
lora_config = LoraConfig(
    r=2,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

# 4. Apply LoRA to the model
model = get_peft_model(model, lora_config)


def verify_no_target_text_overlap(train_dataset, val_dataset):
    train_targets = set(sample['target_text'] for sample in train_dataset)
    val_targets = set(sample['target_text'] for sample in val_dataset)
    
    overlap_train_val = train_targets.intersection(val_targets)
    
    if overlap_train_val:
        print("Data leakage detected in target_texts across datasets.")
        print(f"Overlap between train and validation sets: {len(overlap_train_val)}")
    else:
        print("No target_text overlap detected across datasets.")


def validate_model_outputs(model, tokenizer, eval_dataset):
    clear_memory()
    model.eval()
    total_similarity = 0
    total_samples = 0

    for batch in eval_dataset:
        # Convert the lists to tensors
        input_ids = torch.tensor(batch['input_ids']).unsqueeze(0).to("cuda")  # Add batch dimension
        attention_mask = torch.tensor(batch['attention_mask']).unsqueeze(0).to("cuda")
        target_texts = torch.tensor(batch['labels']).unsqueeze(0).to("cuda")  # Add batch dimension

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256)

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        target_texts = tokenizer.batch_decode(target_texts.cpu(), skip_special_tokens=False)  # Decoding on CPU

        # Filter out irrelevant tokens (<pad>, <s>, </s>) and remove trailing double quotes after semicolons
        def filter_and_clean_text(text):
            text = text.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
            return text.replace('";"', '";').strip()  # Clean up any trailing quotes

        generated_texts = [filter_and_clean_text(text) for text in generated_texts]
        target_texts = [filter_and_clean_text(text) for text in target_texts]

        # Compute similarity between the generated and target texts
        for generated_text, target_text in zip(generated_texts, target_texts):
            similarity = semantic_similarity(get_embedding(generated_text).to("cuda"), 
                                             get_embedding(target_text).to("cuda"))  # Ensure embedding operations on GPU
            total_similarity += similarity
            total_samples += 1

    average_similarity = total_similarity / total_samples
    return average_similarity

SIMILARITY_THRESHOLD = 0.8

def compute_metrics(eval_preds, model, tokenizer, eval_dataset):
    preds, labels = eval_preds.predictions, eval_preds.label_ids

    # If preds is a tuple, extract logits (usually the first element)
    if isinstance(preds, tuple):
        preds = preds[0]

    # Ensure preds and labels are tensors
    preds = torch.as_tensor(preds) if not isinstance(preds, torch.Tensor) else preds
    labels = torch.as_tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    # If preds is in logits format, convert it to token IDs by taking argmax over the vocabulary dimension
    if len(preds.shape) == 3:  # Shape: [batch_size, sequence_length, vocab_size]
        preds = torch.argmax(preds, dim=-1)  # Take the index of the highest logit (argmax) to get token IDs
    
    # Move preds and labels to CPU before converting to numpy
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode both predictions and labels to human-readable text
    generated_texts = tokenizer.batch_decode(preds, skip_special_tokens=False)
    target_texts = tokenizer.batch_decode(labels, skip_special_tokens=False)

    # Filter out irrelevant tokens and clean trailing quotes
    def filter_and_clean_text(text):
        text = text.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
        return text.replace('";"', '";').strip()

    generated_texts = [filter_and_clean_text(text) for text in generated_texts]
    target_texts = [filter_and_clean_text(text) for text in target_texts]

    # Debugging: Print a sample of generated texts and target texts
    print(f"Generated texts: {generated_texts[:2][0]}")
    print(f"Target texts: {target_texts[:2][0]}")

    # Compute average similarity using your validation function
    average_similarity = validate_model_outputs(model, tokenizer, eval_dataset)
    above_threshold = average_similarity > SIMILARITY_THRESHOLD
    thresholded_similarity = average_similarity - SIMILARITY_THRESHOLD if above_threshold else -1
    
    return {
        'semantic_similarity': average_similarity,
        'above_threshold': above_threshold,
        'thresholded_similarity': thresholded_similarity
    }



# Ensure cache is disabled if gradient checkpointing is enabled
model.config.use_cache = False

# Ensure all parameters that should be trained have requires_grad=True
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Warning: {name} does not require gradients.")





def split_dataset_by_target_text(dataset, train_ratio=0.9, seed=42):
    random.seed(seed)
    # Group samples by target_text
    target_text_to_samples = group_by_target_text(dataset)
    all_target_texts = list(target_text_to_samples.keys())
    random.shuffle(all_target_texts)
    
    # Compute split sizes
    total_targets = len(all_target_texts)
    train_size = int(total_targets * train_ratio)
    
    # Split target_texts
    train_target_texts = all_target_texts[:train_size]
    val_target_texts = all_target_texts[train_size:]
    
    # Collect samples for each split
    train_samples = []
    val_samples = []
    
    for target_text in train_target_texts:
        train_samples.extend(target_text_to_samples[target_text])
    for target_text in val_target_texts:
        val_samples.extend(target_text_to_samples[target_text])
    
    # Convert to Datasets using from_list()
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    
    return train_dataset, val_dataset


# 9. Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
    return_tensors="pt",
)


# Subset evaluation dynamically based on memory availability
def dynamic_subset_eval(eval_dataset, training_args, max_eval_steps=7):
    eval_subset_size = min(max_eval_steps * training_args.per_device_eval_batch_size, len(eval_dataset))
    return eval_dataset.select(range(0, eval_subset_size))

def reverse_input_targets(dataset):
    # Swap the input_text and target_text fields
    reversed_dataset = dataset.map(lambda example: {'input_text': example['target_text'], 'target_text': example['input_text']})
    return reversed_dataset  # Add this line to return the dataset


def run_curriculum_training(string_data_file, asset_data_file, clause_data_file, full_data_file, model, tokenizer, data_collator):

    # Training arguments
    def get_training_args(phase):
        epochs = {
           "Strings":8,
           "Assets":20,
           "Clauses":25,
           "Full": 40
        }
        return Seq2SeqTrainingArguments(
            output_dir=f'./results_{phase}',
            eval_strategy='steps',
            eval_steps=100,
            learning_rate=3e-4,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            max_grad_norm=1,
            num_train_epochs=epochs[phase],
            weight_decay=0.01,
            save_total_limit=3,
            logging_steps=10,
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model='thresholded_similarity',
            greater_is_better=True,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            fp16=True,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=2,
            dataloader_pin_memory=True
        )


    # Load datasets
    print("Loading datasets...")
    string_data = pd.read_csv(string_data_file)
    asset_data = pd.read_csv(asset_data_file)
    clause_data = pd.read_csv(clause_data_file)
    full_data = pd.read_csv(full_data_file)

    def add_special_tokens_and_prepare_dataset(dataset):
        # Define the special tokens to be added
        special_tokens = {
        'additional_special_tokens': [
            "string", "asset", "subject", "clause", "=", ")", "CR(", "PVG(", "OB(", "PR(", "Service", "Property", "N", "NM", "+", "-", "COMPRADOR", "VENDEDOR", "PROPIETARIO", "ACREEDOR", "DEUDOR", "ADQUIRENTE", "{", "}", ";", "AND", "OFERENTE", "MUTUANTE", "MUTUARIO", "ARRENDADOR", "ARRENDATARIO", "PERMUTANTE1", "PERMUTANTE2", "DONANTE", "DONATARIO", "PRESTADOR", "ACREDITADO"
            ]
        }
    
        # Add special tokens to tokenizer
        tokenizer.add_special_tokens(special_tokens)
    
        # Resize model embeddings to accommodate the added special tokens
        model.resize_token_embeddings(len(tokenizer))
        
        print(f"Added special tokens: {special_tokens['additional_special_tokens']}")    
        # Convert the dataset if it's a pandas DataFrame
        if isinstance(dataset, pd.DataFrame):
            return Dataset.from_pandas(dataset)
    
        return dataset  # Return as-is if it's already a Dataset

    # Add special tokens and convert to Dataset for each phase
    string_data = add_special_tokens_and_prepare_dataset(string_data)
    asset_data = add_special_tokens_and_prepare_dataset(asset_data)
    clause_data = add_special_tokens_and_prepare_dataset(clause_data)
    full_data = add_special_tokens_and_prepare_dataset(full_data)

    # Remove duplicates from each dataset
    string_data = remove_duplicates(string_data)
    asset_data = remove_duplicates(asset_data)
    clause_data = remove_duplicates(clause_data)
    full_data = remove_duplicates(full_data)


    # Split each dataset for training and evaluation
    print("Splitting datasets for training and evaluation...")
    train_strings, eval_strings = split_dataset_by_target_text(string_data)
    train_assets, eval_assets = split_dataset_by_target_text(asset_data)
    train_clauses, eval_clauses = split_dataset_by_target_text(clause_data)
    train_full, eval_full = split_dataset_by_target_text(full_data)


    verify_no_target_text_overlap(train_strings, eval_strings)
    verify_no_target_text_overlap(train_assets, eval_assets)
    verify_no_target_text_overlap(train_clauses, eval_clauses)
    verify_no_target_text_overlap(train_full, eval_full)
    
    # Tokenization and preprocessing logic
    MAX_LENGTH = 126

    def preprocess_function_truncate(examples):
        inputs = examples['input_text']
        targets = examples['target_text']

        # Tokenize the inputs (source text) and targets (target text)
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True, padding='max_length')

        # Assign the tokenized target labels to the 'labels' key in the model inputs
        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    # Preprocess the datasets: tokenize and format them correctly for the model
    def preprocess_and_tokenize(train_dataset, eval_dataset):
        train_dataset = train_dataset.map(preprocess_function_truncate, batched=True, remove_columns=['input_text', 'target_text'])
        eval_dataset = eval_dataset.map(preprocess_function_truncate, batched=True, remove_columns=['input_text', 'target_text'])
        return train_dataset, eval_dataset

    # Apply preprocessing to each dataset
    print("Preprocessing datasets...")
    train_strings, eval_strings = preprocess_and_tokenize(train_strings, eval_strings)
    train_assets, eval_assets = preprocess_and_tokenize(train_assets, eval_assets)
    train_clauses, eval_clauses = preprocess_and_tokenize(train_clauses, eval_clauses)
    train_full, eval_full = preprocess_and_tokenize(train_full, eval_full)


    # Subset evaluation dynamically based on memory availability
    def dynamic_subset_eval(eval_dataset, training_args, max_eval_steps=7):
        eval_subset_size = min(max_eval_steps * training_args.per_device_eval_batch_size, len(eval_dataset))
        return eval_dataset.select(range(0, eval_subset_size))

    # Helper function to run training phases
    def train_phase(train_dataset, eval_dataset, phase_name, training_args, current_stage):
        # Initialize optimizer and scheduler dynamically for each phase
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        total_training_steps = (
            len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        ) * training_args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_training_steps
        )

        # Dynamically subset the evaluation dataset
        eval_dataset = dynamic_subset_eval(eval_dataset, training_args)

        print(f"Training {phase_name} phase...")
        trainer = CustomSeq2SeqTrainerWithHumanInput(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, model, tokenizer, eval_dataset),
            optimizers=(optimizer, scheduler),  # Provide optimizer and scheduler
            similarity_weight=0.8,  # Adjust weights as needed
            enable_human_input=False,
            current_stage=current_stage  # Pass the current stage
        )
        trainer.train()
        eval_result = trainer.evaluate(eval_dataset=eval_dataset, device=torch.device("cuda"))
        print(f"{phase_name} phase evaluation results: {eval_result}")

    # Phase 1: Train on strings
    training_args = get_training_args("Strings")
    train_phase(train_strings, eval_strings, "Strings", training_args, current_stage="strings")

    # Phase 2: Train on assets
    training_args = get_training_args("Assets")
    train_phase(train_assets, eval_assets, "Assets", training_args, current_stage="assets")

    # Phase 3: Train on clauses
    training_args = get_training_args("Clauses")
    train_phase(train_clauses, eval_clauses, "Clauses", training_args, current_stage="clauses")


    print("Final training with full dataset...")
    # Final training loop with full dataset
    training_args = get_training_args("Full")
    train_phase(train_full, eval_full, "Full", training_args, current_stage="full")

    reverse_train_dataset = reverse_input_targets(train_dataset)
    reverse_eval_dataset = reverse_input_targets(eval_dataset)
    
    reverse_trainer = CustomSeq2SeqTrainerWithHumanInput(
        model=model,
        args=training_args,
        train_dataset=reverse_train_dataset,
        eval_dataset=reverse_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),  # Provide optimizer and scheduler
        similarity_weight=0.8,  # Adjust weights as needed
        enable_human_input=False,
        current_stage=current_stage
    )
    reverse_trainer.train()
    
    # 16. Save the LoRA adapters and model
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')



# Example usage
run_curriculum_training(
    string_data_file="processed_kelsen_string_data.csv",
    asset_data_file="processed_kelsen_assets_data.csv",
    clause_data_file="processed_kelsen_clauses_data.csv",
    full_data_file="kelsen_data.csv",
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator
)


