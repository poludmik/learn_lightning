from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lightning as L
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
import litgpt  # Ensure litgpt is installed and accessible
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.callbacks.progress import RichProgressBar


class GPT2Finetuner(L.LightningModule):
    def __init__(self, warmup_steps=0, total_steps=0):
        super().__init__()

        if warmup_steps == 0:
            print("\033[91m" + "Warning: warmup_steps is set to 0." + "\033[0m")

        if total_steps == 0:
            print("\033[91m" + "Warning: total_steps is set to 0." + "\033[0m")

        self.save_hyperparameters()  # Saves hyperparameters for checkpointing

        self.gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        # self.gpt2.train()

    def training_step(self, batch, batch_idx):
        targets = batch["labels"]
        outputs = self.gpt2(batch["input_ids"])

        # Compute loss using chunked cross-entropy for efficiency
        loss = litgpt.utils.chunked_cross_entropy(outputs.logits, targets)
        self.log("train_loss", loss, prog_bar=True)

        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("current_lr", current_lr)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)

        total_steps = self.hparams.total_steps
        warmup_steps = self.hparams.warmup_steps

        warmup_scheduler = LinearLR(
            optimizer, start_factor=1e-10, end_factor=1.0, total_iters=warmup_steps
        )

        constant_scheduler = ConstantLR(
            optimizer, factor=1.0, total_iters=total_steps - warmup_steps
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[warmup_steps]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',       # Update the scheduler after each step
                'frequency': 1,
            }
        }

tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
# model = AutoModelForCausalLM.from_pretrained('gpt2')

# # run model for prediction:
model = GPT2Finetuner()

# load from the checkpoint
# model.load_state_dict(torch.load('my_gpt2_big_checkpoints/cp-epoch=0-step=130000.ckpt', weights_only=True)['state_dict'])
model.load_state_dict(torch.load('pth_models/model_gpt2_big.pth', weights_only=True))

# model.load_state_dict(torch.load('model_gpt2_cznews.pth', weights_only=True))
model.eval()
# input_text = """Německý kancléř Gerhard"""
input_text = """Kontext: \"Klub Sparta vyhral klub Slavia Praha 3:1\".
Otázka: \"Kdo vyhral zápas Sparta nebo Slavia?\".
Odpověď: \"Sparta\".

Kontext: \"Největší město České republiky je Praha\".
Otázka: \"Jaké je největší město České republiky?\".
Odpověď: \"Praha\".

Kontext: \"Nejvyšší hora Sněžka je z České republiky\".
Otázka: \"Jaká je nejvyšší hora České republiky?\".
Odpověď: \"Sněžka\".

Kontext: \"Prezident Německa je Frank-Walter Steinmeier\".
Otázka: \"Kdo je prezidentem Německa?\".
Odpověď: \"Steinmeier\".

Kontext: \"Brno je druhé největší město České republiky\".
Otázka: \"Jaké je 2. největší město ČR?\".
Odpověď: \""""
input_ids = tokenizer.encode(input_text, return_tensors='pt')

attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

# generate output and ignore the generated special tokens
output = model.gpt2.generate(input_ids, attention_mask=attention_mask, 
                             max_new_tokens=100,
                             num_return_sequences=1, 
                             no_repeat_ngram_size=5, 
                            #  top_k=50,
                            #  top_p=0.95,
                            #  do_sample=True,
                            #  temperature=0.9,
                             eos_token_id=None
                            )

for i in range(0, len(output)):
    print("\033[94m>>>>>>\033[0m")
    print(tokenizer.decode(output[i], skip_special_tokens=False))
