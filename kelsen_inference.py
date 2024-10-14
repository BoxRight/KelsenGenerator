import torch
import gc
import pandas as pd
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from peft import PeftModel
import logging
from test_kelsen_model import load_model_and_tokenizer, generate_predictions, create_dataloader, post_process_output, compute_kelsen_specific_metrics, merge_outputs
from datasets import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn_inference(batch):
    """Handles inference-time collation by ensuring correct tensor conversion."""
    tensor_keys = ['input_ids', 'attention_mask']
    non_tensor_keys = ['target_text']
    
    # Convert to tensors if necessary (handle lists gracefully)
    tensor_data = {}
    for key in tensor_keys:
        if isinstance(batch[0][key], list):
            tensor_data[key] = torch.tensor([item[key] for item in batch])  # Convert list to tensor
        else:
            tensor_data[key] = torch.stack([item[key] for item in batch])  # Already tensors
    
    # Keep non-tensor data as is
    non_tensor_data = {key: [item[key] for item in batch] for key in non_tensor_keys}
    
    return {**tensor_data, **non_tensor_data}

def load_test_text(text, tokenizer):
    """Creates a test dataset from the input legal text."""
    data = {"input_text": [text], "target_text": [""]}  # Empty target_text, since we're inferring
    dataset = Dataset.from_pandas(pd.DataFrame(data))
    
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True, padding='max_length')
        return model_inputs
    
    dataset = dataset.map(preprocess_function, batched=True)
    return dataset

def save_predictions(predictions, references, file_name, metrics=None):
    """
    Save predictions, references, and optionally metrics to a CSV file.
    Args:
        predictions (list): List of model predictions.
        references (list): List of reference texts (ground truth).
        file_name (str): Output file name for saving predictions.
        metrics (dict, optional): Dictionary of Kelsen metrics to include in the saved file.
    """
    logger.info(f"Saving predictions to {file_name}...")
    
    # Prepare data for saving
    data = {
        "Predictions": predictions,
        "References": references
    }
    
    # Include metrics if provided
    if metrics:
        for metric, value in metrics.items():
            data[metric] = [value] * len(predictions)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    logger.info(f"Predictions and metrics saved to {file_name}")

def infer_kelsen_translation(legal_text, full_model_path, clause_model_path, device):
    # Load full model and tokenizer
    
    
    full_model, full_tokenizer = load_model_and_tokenizer(full_model_path, device)
    
    # Preprocess the input legal text
    logger.info(f"Preprocessing legal text: {legal_text[:100]}...")
    test_dataset = load_test_text(legal_text, full_tokenizer)
    
    # Create a dataloader for batch processing with modified collate_fn for inference
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_inference)

    # Generate predictions from the full model
    logger.info("Running Full model inference...")
    full_predictions, _ = generate_predictions(full_model, full_tokenizer, dataloader, device)
    
    # Save partial predictions after full model inference
    logger.info("Saving partial predictions from Full model...")
    save_predictions(full_predictions, [""] * len(full_predictions), "partial_full_model_predictions.csv")

    # Clear memory after using the full model
    del full_model  # Don't delete the tokenizer, keep it for the clause model
    torch.cuda.empty_cache()  # Clear GPU memory
    gc.collect()  # Run Python's garbage collection to free up memory
    logger.info("Full model memory cleared.")
    
    # Generate predictions from the clauses model (reusing tokenizer)
    logger.info("Loading Clauses model for inference...")
    clauses_model, _ = load_model_and_tokenizer(clause_model_path, device)

    logger.info("Running Clauses model inference...")
    clauses_predictions, _ = generate_predictions(clauses_model, full_tokenizer, dataloader, device)

    # Save partial predictions from the clauses model
    logger.info("Saving partial predictions from Clauses model...")
    save_predictions(clauses_predictions, [""] * len(clauses_predictions), "partial_clauses_model_predictions.csv")
    
    # Combine the predictions using the merge_outputs function
    logger.info("Combining outputs from both models using the merge_outputs function...")
    final_combined_prediction = merge_outputs(full_predictions[0], clauses_predictions[0])

    # Compute Kelsen metrics
    logger.info("Computing Kelsen metrics...")
    kelsen_metrics = compute_kelsen_specific_metrics([final_combined_prediction], [""])  # Assuming reference is empty
    
    # Save the final combined prediction and metrics
    logger.info("Saving final combined prediction and metrics...")
    save_predictions([final_combined_prediction], [""], "final_combined_prediction_with_metrics.csv", metrics=kelsen_metrics)
    
    # Log final result
    logger.info(f"Final Kelsen translation: {final_combined_prediction[:500]}...")  # Log first 500 characters for brevity

    # Return the final combined prediction
    return final_combined_prediction
    
def infer_kelsen_translation_single_model(legal_text, full_model_path, device):
    
    full_model, full_tokenizer = load_model_and_tokenizer(full_model_path, device)
    
    # Preprocess the input legal text
    logger.info(f"Preprocessing legal text: {legal_text[:100]}...")
    test_dataset = load_test_text(legal_text, full_tokenizer)
    
    # Create a dataloader for batch processing with modified collate_fn for inference
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn_inference)

    # Generate predictions from the full model
    logger.info("Running Full model inference...")
    full_predictions, _ = generate_predictions(full_model, full_tokenizer, dataloader, device)
    
    # Save partial predictions after full model inference
    logger.info("Saving partial predictions from Full model...")
    save_predictions(full_predictions, [""] * len(full_predictions), "partial_full_model_predictions.csv")

    return full_predictions
    
    
    
if __name__ == "__main__":
    # Example usage
    legal_text = """El vendedor acuerda transferir la propiedad de un bien inmueble al comprador."""
    
    # Paths to the trained models

    assets_model_path = "./results_Assets/checkpoint-10660"
    clause_model_path = "./results_Clauses/checkpoint-7995"
    full_model_path = "./results_Full/checkpoint-15990"
    
    # Choose the device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Perform the inference
    translated_text = infer_kelsen_translation(legal_text, assets_model_path, clause_model_path, device)
    
    print("Kelsen Translation:", translated_text)
    
        # Perform the inference
    translated_text = infer_kelsen_translation_single_model(translated_text, full_model_path, device)
    
    print("Kelsen Translation:", translated_text)
    


