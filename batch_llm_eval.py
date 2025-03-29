from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
from dotenv import load_dotenv
import torch
import time
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Step 1: Authenticate (if required)
# Uncomment the following if you need to log in to Hugging Face
# from huggingface_hub import login

load_dotenv()

huggin_face_token = os.getenv("HUGGING_FACE_TOKEN")
# Step 1: Authenticate

login(huggin_face_token)

# Step 2: Load the Llama 3.2 Model and Tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("mps")

# Set pad_token_id to eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Step 3: Load the Evaluation Dataset
# Here, we use a sample dataset, but you can replace it with your custom dataset as needed
dataset = load_dataset("squad_v2", split="validation[:10]")  # Process only the first 10 items

# Step 4: Load Metrics
ter = load("ter")
meteor = load("meteor")
sacrebleu = load("sacrebleu")
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

def evaluate_model(model, tokenizer, dataset, batch_size=4):
    references, predictions = [], []
    latencies = []  # To store latency times for each response
    num_batches = len(dataset) // batch_size + 1  # Calculate the number of batches

    for batch_idx in range(num_batches):
        # Start the latency timer
        start_time = time.time()

        # Create a batch of items
        batch = dataset[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # Tokenize the batch of questions
        questions = [item["question"] for item in batch]
        inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to("mps")

        # Generate responses for the batch
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=20)

        # End the latency timer and calculate elapsed time  
        latency = time.time() - start_time
        latencies.append(latency)

        # Decode the model's output for the batch
        batch_predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Store predictions and references for metric calculation
        predictions.extend(batch_predictions)
        for item, prediction in zip(batch, batch_predictions):
            references.append(item["answers"]["text"][0] if item["answers"]["text"] else "")

    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies)

    # Calculate metrics
    meteor_score = meteor.compute(predictions=predictions, references=references)
    ter_score = ter.compute(predictions=predictions, references=references)
    sacrebleu_score = sacrebleu.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {"METEOR": meteor_score, "SACREBLEU": sacrebleu_score, "TER": ter_score, "BLEU": bleu_score, "ROUGE": rouge_score, "BERTScore": bertscore_score, "Average Latency (seconds)": avg_latency}


results = evaluate_model(model, tokenizer, dataset, batch_size=8)  # Adjust batch size as needed
print(results)
