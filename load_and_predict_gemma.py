import torch
from transformers import AutoTokenizer
from GemmaModule import Gemma2Finetuner

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
model = Gemma2Finetuner(warmup_steps=500, total_steps=10000)

#model.load_state_dict(torch.load('my_gemma2_cswiki_checkpoints/cp-epoch=0-step=10000-v1.ckpt/pytorch_model.bin', weights_only=True))
model.load_state_dict(torch.load('adam_gemma2_cswiki_checkpoints/cp-epoch=1-step=20000.ckpt/pytorch_model.bin', weights_only=True))
model.eval()

model.gemma2.to("cuda")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    
    with torch.no_grad():
        output = model.gemma2.generate(
            **inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            )
    return tokenizer.decode(output[0], skip_special_tokens=False)

prompt = """Klub Slavia Praha je"""

generated_text = generate_text(prompt)

print("\033[94m" + "Generated Text:" + "\033[0m")

print(generated_text)
