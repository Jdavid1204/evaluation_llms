from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from optimum.quanto import QuantizedModelForCausalLM, qint8

import torch
import time
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

huggin_face_token = os.getenv("HUGGING_FACE_TOKEN")
if not huggin_face_token:
    raise ValueError("HUGGING_FACE_TOKEN is missing. Ensure it's set in .env.")

login(huggin_face_token)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = QuantizedModelForCausalLM.from_pretrained("quantized-qwen").to("mps")

# # ✅ Apply Dynamic Quantization (Linear layers → INT8)
# quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )

# # ✅ Move to CPU for inference
# quantized_model.to("cpu")
# print("Quantized Model Loaded on CPU")

# Set pad_token_id to eos_token_id if not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dataset = Dataset.from_json("custom_dataset.json")

# Metrics
ter = load("ter")
meteor = load("meteor")
sacrebleu = load("sacrebleu")
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")


def retrieve_courses(query, courses_data):
    # Simple retrieval logic to find relevant courses for a query
    relevant_courses = [course for course in courses_data if any(keyword in query for keyword in course['title'])]
    return relevant_courses


def evaluate_model(model, tokenizer, dataset):
    references, predictions = [], []
    latencies = []
    for item in dataset:
        start_time = time.time()

        # Check progress
        for idx, item in enumerate(dataset):
            print(f"Processing item {idx + 1}/{len(dataset)}")
            ...
 
        relevant_courses = retrieve_courses(item["query"], item["courses"])
        
        # Create a combined prompt with the query and course recommendations
        course_recommendations = "Recommended Courses:\n" + "\n".join([f"{course['title']}: {course['description']}" for course in relevant_courses])
        prompt = item["query"] + "\n\n" + course_recommendations
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("mps")
        
        with torch.no_grad():
            try:
                outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=50)
            except RuntimeError as e:
                print(f"Error during generation: {e}")
                continue


        latency = time.time() - start_time
        latencies.append(latency)

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store predictions and references for metric calculation
        predictions.append(prediction)
        references.append(item["additional_info"])

        
    avg_latency = sum(latencies) / len(latencies)

    # Calculate metrics
    meteor_score = meteor.compute(predictions=predictions, references=references)
    ter_score = ter.compute(predictions=predictions, references=references)
    sacrebleu_score = sacrebleu.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)
    bertscore_score = bertscore.compute(predictions=predictions, references=references, lang="en")

    return {"METEOR": meteor_score, "SACREBLEU": sacrebleu_score, "TER": ter_score, "BLEU": bleu_score, "ROUGE": rouge_score, "BERTScore": bertscore_score,"Average Latency (seconds)": avg_latency}


results = evaluate_model(model, tokenizer, dataset)
print(results)