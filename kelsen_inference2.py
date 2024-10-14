import torch
import gc
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from test_kelsen_model import (
    load_model_and_tokenizer,
    generate_predictions,
    compute_kelsen_specific_metrics,
    merge_outputs
)
from datasets import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collate_fn_inference(batch):
    tensor_keys = ['input_ids', 'attention_mask']
    tensor_data = {}
    for key in tensor_keys:
        tensor_data[key] = torch.stack([torch.tensor(item[key]) for item in batch])
    return tensor_data

def load_test_text(text, tokenizer):
    data = {"input_text": [text], "target_text": [""]}
    dataset = Dataset.from_pandas(pd.DataFrame(data))

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=256,
            truncation=True,
            padding='max_length'
        )
        return model_inputs

    dataset = dataset.map(preprocess_function, batched=True)
    return dataset

def infer_kelsen_translation(legal_text, assets_model_path, clauses_model_path, full_model_path, device):
    # Load the tokenizer once (assuming all models use the same tokenizer)
    _, tokenizer = load_model_and_tokenizer(full_model_path, device)

    # Preprocess the input legal text
    logger.info(f"Preprocessing legal text...")
    test_dataset = load_test_text(legal_text, tokenizer)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_inference)

    # Process Assets Model
    logger.info("Processing Assets Model...")
    assets_model, _ = load_model_and_tokenizer(assets_model_path, device)
    with torch.no_grad():
        assets_predictions, _ = generate_predictions(assets_model, tokenizer, dataloader, device)
    del assets_model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Assets Model processing complete and memory cleared.")

    # Process Clauses Model
    logger.info("Processing Clauses Model...")
    clauses_model, _ = load_model_and_tokenizer(clauses_model_path, device)
    with torch.no_grad():
        clauses_predictions, _ = generate_predictions(clauses_model, tokenizer, dataloader, device)
    del clauses_model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Clauses Model processing complete and memory cleared.")

    # Combine the outputs
    logger.info("Combining outputs from Assets and Clauses Models...")
    combined_output = merge_outputs(assets_predictions[0], clauses_predictions[0])

    # Prepare combined output for Full Model
    combined_dataset = load_test_text(combined_output, tokenizer)
    combined_dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=1, collate_fn=collate_fn_inference)

    # Process Full Model
    logger.info("Processing Full Model...")
    full_model, _ = load_model_and_tokenizer(full_model_path, device)
    with torch.no_grad():
        full_predictions, _ = generate_predictions(full_model, tokenizer, combined_dataloader, device)
    del full_model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Full Model processing complete and memory cleared.")

    final_prediction = full_predictions[0]

    # Compute Kelsen metrics (if desired)
    logger.info("Computing Kelsen metrics...")
    kelsen_metrics = compute_kelsen_specific_metrics([final_prediction], [""])

    # Save the final prediction and metrics
    logger.info("Saving final combined prediction and metrics...")
    save_predictions([final_prediction], [""], "final_combined_prediction_with_metrics.csv", metrics=kelsen_metrics)

    # Log final result
    logger.info(f"Final Kelsen translation: {final_prediction[:500]}...")

    return final_prediction

def save_predictions(predictions, references, file_name, metrics=None):
    logger.info(f"Saving predictions to {file_name}...")
    data = {
        "Predictions": predictions,
        "References": references
    }
    if metrics:
        for metric, value in metrics.items():
            data[metric] = [value] * len(predictions)
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    logger.info(f"Predictions and metrics saved to {file_name}")

if __name__ == "__main__":
    legal_text = "El vendedor acuerda transferir la propiedad de un bien inmueble al comprador."

    # Paths to the trained models
    assets_model_path = "./results_Assets/checkpoint-10660"
    clauses_model_path = "./results_Clauses/checkpoint-7995"
    full_model_path = "./results_Full/checkpoint-15990"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Perform the inference
    translated_text = infer_kelsen_translation(
        legal_text,
        assets_model_path,
        clauses_model_path,
        full_model_path,
        device
    )

    print("Kelsen Translation:", translated_text)

