# This script runs the Gemma2-9B model on every training instance from "dataset_playground/instruction_data/no_robots_cs.jsonl"
# and saves the logits to a file.
"""
# Example inference:
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    device_map="auto",
)

input_text = "Hořkost drogy je nepříjemná, ale měli byste dávat pozor"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b",
    device_map="auto",
)

with open("dataset-playground/instruction_data/no_robots_cs_logits.jsonl", "a") as f_write:
    with open("dataset-playground/instruction_data/no_robots_cs.jsonl", "r") as f:
        data = f.readlines()
        for i, line in enumerate(data):
            obj = json.loads(line)
            try:
                input_text = obj["Human"]
                input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

                # outputs = model.generate(**input_ids, max_new_tokens=100)
                logits = model(**input_ids, return_dict=True).logits
                
                # decode:
                # output_text = tokenizer.decode(logits[0].argmax(dim=-1))
                # print(output_text)

                # save logits to another file
                f_write.write(json.dumps({"logits": logits.tolist()}) + "\n")

            except Exception as e:
                print(e)
                print("Error at line", i)


