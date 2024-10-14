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

# Load the BLEU metric
bleu_metric = evaluate.load("bleu")

class CustomSeq2SeqTrainerWithHumanInput(Seq2SeqTrainer):
    def __init__(self, *args, similarity_weight=0.3, bleu_weight=0.3, human_feedback_weight=0.3, human_feedback_start_step=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_weight = similarity_weight  # Weight for semantic similarity
        self.bleu_weight = bleu_weight  # Weight for BLEU score
        self.human_feedback_weight = human_feedback_weight  # Weight for human feedback
        self.human_feedback_start_step = human_feedback_start_step
        self.global_step = 0  # Track the step for human feedback

    def get_human_feedback(self, generated_texts, target_texts):
        """
        Simulate human feedback. In practice, this would involve actual human input.
        """
        human_scores = []
        for generated_text, target_text in zip(generated_texts, target_texts):
            print(f"Generated: {generated_text}")
            print(f"Target: {target_text}")
            try:
                score = float(input("Provide feedback score (0 to 1): "))
            except ValueError:
                score = 0.0  # Default to 0 if invalid input
            human_scores.append(score)
        return np.mean(human_scores)

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

        # Step 4: Generate predictions for BLEU and similarity computation
        generated_ids = model.generate(input_ids=input_ids, attention_mask=inputs["attention_mask"])
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        target_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Join tokens back into sentences
        generated_texts = [''.join(generated_text.split()) for generated_text in generated_texts]
        target_texts = [''.join(target_text.split()) for target_text in target_texts]

        # Step 5: Compute Semantic Similarity Loss
        total_similarity = 0
        for generated_text, target_text in zip(generated_texts, target_texts):
            similarity = semantic_similarity(get_embedding(generated_text), get_embedding(target_text))
            total_similarity += similarity
        average_similarity = total_similarity / len(generated_texts)
        similarity_loss = 1 - average_similarity

        # Step 6: Compute BLEU Loss
        bleu_score = bleu_metric.compute(predictions=generated_texts, references=[[t] for t in target_texts])['bleu']
        bleu_loss = 1 - bleu_score

        # Step 7: Optional Human Feedback after a certain number of steps
        human_feedback_loss = 0
        if self.global_step >= self.human_feedback_start_step:
            human_feedback_loss = self.get_human_feedback(generated_texts, target_texts)

        # Step 8: Combine Cross-Entropy, Similarity, BLEU, and Human Feedback Loss
        total_loss = (loss_ce + 
                      self.similarity_weight * similarity_loss + 
                      self.bleu_weight * bleu_loss + 
                      self.human_feedback_weight * human_feedback_loss)

        return (total_loss, outputs) if return_outputs else total_loss



def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
    
def tokenize(text):
    # Use a simple regex to split by space, punctuation, or special characters
    return re.findall(r"\w+|[{}()\[\];=+-]", text)

# 1. Configure 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    memory_efficient_offload=True  # Enable memory-efficient offloading
)


# 2. Load the tokenizer and model with 8-bit quantization
model_name = 't5-base'  # Options: 't5-small', 't5-base', 't5-large'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto'
)

# 3. Configure LoRA
target_modules = ['q', 'v']  # May need adjustment based on T5 architecture
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

# 4. Apply LoRA to the model
model = get_peft_model(model, lora_config)


# Load the dataset once
file_path = 'kelsen_data.csv'
data = pd.read_csv(file_path)
clear_memory()

# Extract all 'target_text' from the dataset
target_texts = data['target_text'].tolist()

# 5. Tokenize the 'target_text' by splitting on spaces and punctuation
def tokenize_texts(texts):
    tokens = []
    for text in tqdm(texts):
        tokens.extend(tokenize(text))
    return tokens

all_tokens = tokenize_texts(target_texts)
flat_tokens = [token for sublist in all_tokens for token in sublist]

# Count the frequency of each token
token_counts = Counter(flat_tokens)

# Display a sample of unique tokens
special_tokens = list(set(flat_tokens))

# Add special tokens to tokenizer
tokenizer.add_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# 6. Prepare your dataset
dataset = Dataset.from_pandas(data)
clear_memory()  # Clear memory after conversion

# Split into train, validation, and test sets
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
temp_dataset = dataset['test'].train_test_split(test_size=0.5)
eval_dataset = temp_dataset['train']
test_dataset = temp_dataset['test']

# Clear memory after dataset preparation
clear_memory()
# 7. Tokenization function
# Set the max length for truncation
MAX_LENGTH = 512

def preprocess_function_truncate(examples):
    inputs = examples['input_text']
    targets = examples['target_text']

    # Tokenize the inputs and targets, truncating them to the specified max length
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding='max_length')

    # Tokenize the target/label text, also applying truncation
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True, padding='max_length')

    # Store the tokenized labels
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# 8. Preprocess the datasets
train_dataset = train_dataset.map(preprocess_function_truncate, batched=True, remove_columns=['input_text', 'target_text'])
eval_dataset = eval_dataset.map(preprocess_function_truncate, batched=True, remove_columns=['input_text', 'target_text'])

# Clear memory after preprocessing
clear_memory()

# Subset evaluation dynamically based on memory availability
def dynamic_subset_eval(eval_dataset, max_eval_steps=7):
    eval_subset_size = min(max_eval_steps * training_args.per_device_eval_batch_size, len(eval_dataset))
    return eval_dataset.select(range(0, eval_subset_size))

# 9. Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=tokenizer.pad_token_id,
    return_tensors="pt",
)

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

        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        target_texts = tokenizer.batch_decode(target_texts.cpu(), skip_special_tokens=True)  # Decoding on CPU

        for generated_text, target_text in zip(generated_texts, target_texts):
            similarity = semantic_similarity(get_embedding(generated_text).to("cuda"), 
                                             get_embedding(target_text).to("cuda"))  # Ensure embedding operations on GPU
            total_similarity += similarity
            total_samples += 1

    average_similarity = total_similarity / total_samples
    return average_similarity




SIMILARITY_THRESHOLD = 0.8

def compute_metrics(eval_preds):
    preds, labels = eval_preds.predictions, eval_preds.label_ids

    # Debugging print to ensure preds is what we expect
    print(f"Preds type: {type(preds)}")
    
    # If preds is a tuple, extract logits (usually the first element)
    if isinstance(preds, tuple):
        print(f"Preds tuple length: {len(preds)}")  # Debugging
        preds = preds[0]

    # Ensure preds and labels are tensors
    preds = torch.as_tensor(preds) if not isinstance(preds, torch.Tensor) else preds
    labels = torch.as_tensor(labels) if not isinstance(labels, torch.Tensor) else labels

    # Debugging: Check shapes of preds and labels
    print(f"Preds shape: {preds.shape}, Labels shape: {labels.shape}")
    
    # If preds is in logits format, convert it to token IDs by taking argmax over the vocabulary dimension
    if len(preds.shape) == 3:  # Shape: [batch_size, sequence_length, vocab_size]
        preds = torch.argmax(preds, dim=-1)  # Take the index of the highest logit (argmax) to get token IDs
    
    # Move preds and labels to CPU before converting to numpy
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    # Debugging: Check if conversion to numpy works fine
    print(f"Preds numpy shape: {preds.shape}, Labels numpy shape: {labels.shape}")

    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode both predictions and labels to human-readable text
    generated_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    target_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Debugging: Print a sample of generated texts and target texts
    print(f"Generated texts: {generated_texts[:2]}")
    print(f"Target texts: {target_texts[:2]}")

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




# 11. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    eval_strategy='steps',
    eval_steps=100,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_grad_norm=0.5,
    num_train_epochs=5,
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
    eval_accumulation_steps=1,  # Accumulate evaluation steps to avoid high memory usage
    gradient_accumulation_steps=2,
    dataloader_pin_memory=True

)


# 12. Calculate total training steps
total_training_steps = (
    len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
) * training_args.num_train_epochs

# Dynamically adjust eval dataset size before passing to the trainer
eval_subset = dynamic_subset_eval(eval_dataset)


# 13. Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=total_training_steps
)



# Helper function to split dataset for training and evaluation
def split_dataset(dataframe, test_size=0.2):
    dataset = Dataset.from_pandas(dataframe)
    train_test_split = dataset.train_test_split(test_size=test_size)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    return train_dataset, eval_dataset


def reverse_input_targets(dataset):
    #Swap the input_text and target_text fields
    reversed_dataset = dataset.map(lambda example: {'input_text': example['target_text'], 'target_text': example['input_text']})

# Main function to load the datasets and run curriculum learning
def run_curriculum_training(string_data_file, asset_data_file, clause_data_file, full_data_file, model, tokenizer, data_collator):
    # Load datasets
    print("Loading datasets...")
    string_data = pd.read_csv(string_data_file)
    asset_data = pd.read_csv(asset_data_file)
    clause_data = pd.read_csv(clause_data_file)
    full_data = pd.read_csv(full_data_file)

    # Split each dataset for training and evaluation
    print("Splitting datasets for training and evaluation...")
    train_strings, eval_strings = split_dataset(string_data)
    train_assets, eval_assets = split_dataset(asset_data)
    train_clauses, eval_clauses = split_dataset(clause_data)
    train_full, eval_full = split_dataset(full_data)
    eval_full = dynamic_subset_eval(eval_full)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        eval_strategy='steps',
        eval_steps=100,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_grad_norm=0.5,
        num_train_epochs=5,  # 5 epochs for each phase
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

    # Training phases for strings, assets, and clauses
    def train_phase(train_dataset, eval_dataset, phase_name):
        print(f"Training {phase_name} phase...")
        trainer = CustomSeq2SeqTrainerWithHumanInput(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, scheduler),  # Provide optimizer and scheduler
            similarity_weight=0.8,  # Adjust weights as needed
        )
        trainer.train()
        eval_result = trainer.evaluate(eval_dataset=eval_dataset, device=torch.device("cuda"))
        print(f"{phase_name} phase evaluation results: {eval_result}")

    # Phase 1: Train on strings
    train_phase(train_strings, eval_strings, "Strings")

    # Phase 2: Train on assets
    train_phase(train_assets, eval_assets, "Assets")

    # Phase 3: Train on clauses
    train_phase(train_clauses, eval_clauses, "Clauses")

    # Final training loop with full dataset
    print("Final training with full dataset...")
    trainer = CustomSeq2SeqTrainerWithHumanInput(
        model=model,
        args=training_args,
        train_dataset=train_full,
        eval_dataset=eval_full,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        similarity_weight=0.8,  # Adjust weights as needed
    )
    trainer.train()
    clear_memory()
    
    reverse_train_dataset = reverse_inputs_targets(train_dataset)
    reverse_eval_dataset = reverse_inputs_targets(eval_dataset)
    
    reverse_trainer = CustomSeq2SeqTrainerWithHumanInput(
        model=model,
        args=training_args,
        train_dataset=reverse_train_dataset,
        eval_dataset=reverse_eval_dataset
        tokenizer=tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),  # Provide optimizer and scheduler
        similarity_weight=0.8,  # Adjust weights as needed
        enable_human_input=False,
        current_stage=current_stage
    )
    
    reverse_trainer.train()
    eval_result = trainer.evaluate(eval_dataset=eval_full, device=torch.device("cuda"))
    clear_memory()
    best_metric = trainer.state.best_metric
    
    if best_metric <= 0:
        print(f"Warning: No model exceeded the similarity threshold of {SIMILARITY_THRESHOLD}")
    else:
        print(f"Best model exceeded threshold with a similarity of {best_metric + SIMILARITY_THRESHOLD}")
    print(f"Evaluation results: {eval_result}")

    # 16. Save the LoRA adapters and model
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')



    print(f"Final training evaluation results: {eval_result}")

# Example usage
run_curriculum_training(
    string_data_file="processed_kelsen_string_data.csv",
    asset_data_file="processed_kelsen_assets_data.csv",
    clause_data_file="processed_kelsen_clauses_data.csv",
    full_data_file="kelsen_data.csv",
    model=model,  # Assuming model and tokenizer are already initialized
    tokenizer=tokenizer,
    data_collator=data_collator
)





