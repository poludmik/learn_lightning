from datasets import load_dataset
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
import torch
from torch import nn
import torch.nn.functional as F
# from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import litgpt
import lightning as L

from transformers import AutoModelForSequenceClassification, AutoTokenizer,GPT2Model, GPT2LMHeadModel
import evaluate
import numpy as np
from datasets import load_dataset

class TokenizedDataset(Dataset):
    def __init__(self, data_file, block_size):
        self.data_file = data_file
        self.block_size = block_size
        self.data = np.memmap(self.data_file, dtype=np.uint32, mode='r')
        self.data_len = len(self.data) // self.block_size 
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx*self.block_size:idx*self.block_size+self.block_size].astype(np.uint32)).to(torch.long)
        y = torch.from_numpy(self.data[idx*self.block_size+1:idx*self.block_size+1+self.block_size].astype(np.uint32)).to(torch.long)
        # pad the last sequence if it's shorter than block_size
        if y.size(0) < self.block_size:
            pad = torch.zeros(self.block_size - y.size(0), dtype=torch.long)
            y = torch.cat([y, pad])
        # print({'input_ids': x, 'labels': y})
        return {'input_ids': x, 'labels': y}

from torch.utils.data import DataLoader

# Parameters
data_file = 'large_text_dataset/binary_tokenized_dataset.bin'  # Or 'tokenized_data.npy'
block_size = 5  # Adjust based on model's context length
batch_size = 1

# Choose Dataset type
dataset = TokenizedDataset(data_file, block_size)
print("Dataset_Length:", len(dataset), end="\n\n")  # Print the length of the dataset

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=1,
)

class GPT2Finetuner(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.gpt2.train()

    def training_step(self, batch, batch_idx):
        # print("Batch: ", batch)
        # print("input_ids: ", batch["input_ids"])
        # print("labels: ", batch["labels"])
        targets = batch["labels"]
        
        outputs = self.gpt2(batch["input_ids"])
        # print("Outputs: ", outputs.last_hidden_state.shape) # (batch_size, seq_len, hidden_size)
        # print("Outputs: ", outputs.last_hidden_state.shape)
        # print("Targets: ", targets.shape)
        # print("Outputs: ", outputs)

        loss = litgpt.utils.chunked_cross_entropy(outputs.logits, targets)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

model = GPT2Finetuner()
trainer = L.Trainer(
    max_epochs=50,
    # callbacks=[checkpoint_callback]
)

# trainer.fit(model,
#             DataLoader(dataset, batch_size=1),
#             # DataLoader(small_train_dataset, batch_size=1),
#             # DataLoader(small_eval_dataset, batch_size=1),
#             # LTDataModule("large_text_dataset/large_text_dataset.jsonl"),
#             # ckpt_path="my_bert_checkpoints/sample-mnist-epoch=1-step=1500.ckpt"
#             )
# torch.save(model.state_dict(), 'model_gpt2.pth')


# run model for prediction:
model = GPT2Finetuner()
model.load_state_dict(torch.load('model_gpt2.pth'))
model.eval()
input_text = "to eat or not to be overfitted on what? what have you done! <no> text to be"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.gpt2.generate(input_ids, 
                             max_new_tokens=10, 
                             num_return_sequences=1, 
                             do_sample=True, 
                             top_k=1)
print(tokenizer.decode(output[0], skip_special_tokens=False))
