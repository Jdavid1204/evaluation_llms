# Import necessary libraries
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
from dotenv import load_dotenv

import torch
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
huggin_face_token = os.getenv("HUGGING_FACE_TOKEN")


# Step 1: Authenticate

login(huggin_face_token)

# Step 2: Load the Llama 3.2 Model and Tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.float16  # Use FP16
).to("mps")  # Move to Apple GPU (MPS)

# Set pad_token_id to eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Step 3: Load the Evaluation Dataset
# Here, we use a sample dataset, but you can replace it with your custom dataset as needed
dataset = load_dataset("squad_v2", split="validation[:10]")  # Process only the first 10 items

# Step 4: Load Metrics
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

# Step 5: Define the Evaluation Function
def evaluate_model(model, tokenizer, dataset):
    references, predictions = [], []
    latencies = []  # To store latency times for each response
    for item in dataset:
        # Start the latency timer
        start_time = time.time()

        # Tokenize the input
        # Tokenize the input with attention mask and padding
        for idx, item in enumerate(dataset):
            print(f"Processing item {idx + 1}/{len(dataset)}")
            ...
        inputs = tokenizer(item["question"], return_tensors="pt", padding=True, truncation=True).to("mps")
        
        # Generate a response from the model
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=20)

        # End the latency timer and calculate elapsed time  
        latency = time.time() - start_time
        latencies.append(latency)

        # Decode the model's output
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store predictions and references for metric calculation
        predictions.append(prediction)
        references.append(item["answers"]["text"][0] if item["answers"]["text"] else "")

        
    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies)

    # Calculate metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {"BLEU": bleu_score, "ROUGE": rouge_score, "BERTScore": bertscore_score, "Average Latency (seconds)": avg_latency}

# Step 6: Run the Evaluation
results = evaluate_model(model, tokenizer, dataset)
print(results)
