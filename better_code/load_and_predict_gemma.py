import torch
import torch.nn.functional as F
from collections import OrderedDict
from transformers import AutoTokenizer
from Model import LightningGemma2Module
from litgpt import Config
import warnings
from train_gemma2 import load_yaml_config, convert_values

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')

checkpoint_path = "gemma-2-2b-checkpoints/cp-epoch=0-step=35000.ckpt.consolidated"
# checkpoint_path = "/mnt/proj2/open-29-45/poludmik/czech-llm/learn_lightning/pth_models/adam_gemma2_cswiki_block2048.pth"

# Load the checkpoint
if ".ckpt.consolidated" in checkpoint_path:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
        checkpoint = torch.load(checkpoint_path)
        model_config = Config.from_name("gemma-2-2b")
        hp_config = checkpoint["hyper_parameters"]["hp_config"]
        model = LightningGemma2Module(model_config=model_config, hp_config=hp_config)
        model.configure_model()

        # Remove 'module.' prefix from the state_dict keys
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Strip the 'module.' prefix
            new_state_dict[new_key] = v

        model.module.load_state_dict(new_state_dict)
elif ".pth" in checkpoint_path: # doesn't work
    state_dict = torch.load(checkpoint_path)

    hp_config_path = "config.yaml"
    hp_config = load_yaml_config(hp_config_path)
    hp_config = convert_values(hp_config)
    
    # Initialize model configuration and model
    model_config = Config.from_name("gemma-2-2b")
    model = LightningGemma2Module(model_config=model_config, hp_config=hp_config)
    model.configure_model()

    # Load the state dictionary
    model.module.load_state_dict(state_dict)

model.module.eval()
model.module.to("cuda")

def generate_text(model, input_ids, max_new_tokens=10, temperature=0.5):
    model.eval()  # Set the model to evaluation mode
    generated_sequence = input_ids.clone()  # Keep track of token indices (not embeddings)
    new_tokens = []  # Store new tokens here

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated_sequence)            
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            next_token_logits = next_token_logits / temperature
            probabilities = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)

            # Append the new token to the list of new tokens
            new_tokens.append(next_token_id.item())

            # Add the new token to the generated sequence
            generated_sequence = torch.cat((generated_sequence, next_token_id), dim=1)
            
            if len(new_tokens) >= max_new_tokens:
                break

    # Convert the list of new tokens into a tensor
    return torch.tensor(new_tokens).unsqueeze(0).to(input_ids.device)


prompt = """to j"""
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to("cuda")

print("Generating text...")
generated_token_ids = generate_text(model.module, input_ids, max_new_tokens=50)

generated_text = tokenizer.decode(generated_token_ids[0].tolist(), skip_special_tokens=False)

print("\033[94m" + prompt + "\033[0m" + generated_text)
