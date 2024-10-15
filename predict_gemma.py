from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", attn_implementation='eager').to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

input_text = """Klub Slavia Praha je"""

input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")

output = model.generate(input_ids, max_new_tokens=200, num_return_sequences=1, do_sample=True, temperature=0.9)

print(tokenizer.batch_decode(output, skip_special_tokens=False)[0])